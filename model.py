import torch
import torch.nn as nn
from modules import MLP, Conv1d, SelfAttention, Block, AttentionPooling, DoubleConv1d
import torch.nn.functional as F

# ----------------------------
# Model Config
# ----------------------------
class GestureModelConfig:
    def __init__(
        self,
        num_gestures: int = 16,
        n_embd: int = 64,
        seq_len: int = 128,
        n_layer: int = 2,
        n_heads: int = 2,
        dout: float = 0.1,
        device: str = "cuda"
    ):
        self.num_gestures = num_gestures
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.dout = dout
        self.device = device
        self.seq_len = seq_len


# ----------------------------
# Model
# ----------------------------
class GestureModel(nn.Module):
    def __init__(self, config: GestureModelConfig, in_dim: int):
        super().__init__()
        self.cfg = config
        self.in_dim = in_dim
        n_embd = self.cfg.n_embd
        NJ = in_dim // 3
        JCDC = (NJ*(NJ-1))//2 # 325 if num_joints==26
        fusion_dim = n_embd

        self.feature_blocks = nn.ModuleDict({
            "jcd": nn.Sequential(
                MLP(JCDC, n_embd),
                nn.LayerNorm(n_embd),
                Block(n_embd, n_heads=self.cfg.n_heads)
            ),
            "raw": DoubleConv1d(in_dim, n_embd),
            "slow_diff": DoubleConv1d(in_dim, n_embd),
            "fast_diff": DoubleConv1d(in_dim, n_embd),
            "slow_acc": DoubleConv1d(in_dim, n_embd),
            "fast_acc": DoubleConv1d(in_dim, n_embd),
        })
        self.dropout = nn.Dropout(self.cfg.dout)

        self.feature_emb = nn.Parameter(torch.randn(len(self.feature_blocks), self.cfg.n_embd))
        self.pos_emb = nn.Parameter(torch.randn(1, self.cfg.seq_len, 1, n_embd))

        self.cross_attn = nn.MultiheadAttention(embed_dim=self.cfg.n_embd, num_heads=8, batch_first=True, dropout=0.2)
        self.cross_gate_mlp = MLP(n_embd, n_embd)
        self.ln_cross_feat_pre  = nn.LayerNorm(self.cfg.n_embd)  # pre-attention
        self.ln_cross_feat_post = nn.LayerNorm(self.cfg.n_embd)  # post-attention/residual
        self.ln_cross_time      = nn.LayerNorm(self.cfg.n_embd)  # after feature aggregation
        self.feature_gate = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.Sigmoid()
        )

        self.attn_blocks = nn.Sequential(
            *[Block(n_embd=fusion_dim, n_heads=self.cfg.n_heads) for _ in range(self.cfg.n_layer)]
        )

        self.head_dropout = nn.Dropout(self.cfg.dout)

        self.boundary_conv = nn.Conv1d(
            fusion_dim, fusion_dim, 
            kernel_size=5, padding=2
        ) # helps detect peaks instead of noise for start/end
        self.bnonlin = nn.GELU()

        self.gesture_conv = nn.Conv1d(
            fusion_dim, fusion_dim, 
            kernel_size=5, padding=2
        )
        self.gnonlin = nn.GELU()

        self.gesture_head = nn.Linear(fusion_dim*2, self.cfg.num_gestures)
        self.start_head   = nn.Linear(fusion_dim, 1)
        self.end_head     = nn.Linear(fusion_dim, 1)

    def forward(self, x):
        """
        x: input sequences, can be either:
           - dict of views, e.g., {
                 "raw":         [B, T, C],
                 "jcd":         [B, T, (C*(C-1))/2],
                 "slow_diff":   [B, T, C],
                 "fast_diff":   [B, T, C],
                 "slow_acc":    [B, T, C],
                 "fast_acc":    [B, T, C]
             }

        Returns three tensors (placeholders for now):
        - gesture_logits: [B, T, num_gestures] (per-frame gesture classification)
        - start_logits:   [B, T] (per-frame start probability/logits)
        - end_logits:     [B, T] (per-frame end probability/logits)
        """
        B,T,C = x["raw"].shape

        #feature_keys = self.feature_blocks.keys()
        #x = torch.cat([out[k] if k == "jcd" else out[k]*0.75 for k in feature_keys], dim=2)

        feature_names = list(self.feature_blocks.keys())
        N = len(feature_names)

        # feature extraction + embedding
        out = {k: self.dropout(self.feature_blocks[k](x[k])) for k in feature_names}  # [B,T,C] per feature
        ft = torch.stack([out[k] for k in feature_names], dim=2)  # [B,T,N,C]
        ft = ft + self.feature_emb[None, None, :, :]             # add learnable feature embedding
        ft = ft + self.pos_emb                                   # add positional encoding

        # cross-attention over T*N
        ft = ft.view(B, T*N, self.cfg.n_embd)               # [B, T*N, C]
        ft_res = ft

        ft = self.ln_cross_feat_pre(ft)  
        ca_out, _ = self.cross_attn(ft, ft, ft)

        ca_gate = torch.sigmoid(self.cross_gate_mlp(ca_out)) # [B, T*N, C]
        x = ca_gate * ca_out + (1 - ca_gate) *  ft_res       # interpolated residual gating
        x = self.ln_cross_feat_post(x)

        # dynamic feature gating & aggregation
        x = x.view(B, T, N, self.cfg.n_embd)                     # [B, T, N, C]
        weights = torch.softmax(self.feature_gate(x), dim=2)     # [B, T, N, C]
        x = (x * weights).sum(dim=2)                             # [B, T, C]
        x = self.ln_cross_time(x)

        # attention blocks
        x = self.attn_blocks(x) # [B,T,fusion_dim]
        x = self.head_dropout(x)
        x = x.transpose(1,2)

        b = self.boundary_conv(x).transpose(1,2)
        b = self.bnonlin(b)

        g = self.gesture_conv(x).transpose(1,2)
        g = self.gnonlin(g)

        gesture_logits = self.gesture_head(torch.cat([b,g], dim=-1))
        start_logits   = self.start_head(b).squeeze(-1)
        end_logits     = self.end_head(b).squeeze(-1)

        return gesture_logits, start_logits, end_logits

    def print_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")