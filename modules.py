import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        self.in_dim = in_dim
        out_dim = in_dim if out_dim==None else out_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim*4)
       # self.gelu = nn.GELU()
        self.mlp_proj = nn.Linear(out_dim*4, out_dim)
    
    def forward(self, x):
        y = self.mlp_proj(self.mlp(x))
        return x+y if self.in_dim == self.out_dim else y

    

class Conv1d(nn.Module):
    def __init__(self, in_dim, n_embd, kernel_size=3, padding=1):
        super().__init__()
        self.conv1d = nn.Conv1d(in_dim, n_embd, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv1d(x))



class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads=2):
        super().__init__()

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = n_heads
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True) 
        y = y.transpose(1,2).contiguous().view(B,T,C) 
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, attn_dropout=0.1, mlp_dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_heads)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.mlp_dropout = nn.Dropout(mlp_dropout)

    def forward(self, x):
        x = x + self.attn_dropout(self.attn(self.ln1(x)))  # residual + dropout
        x = x + self.mlp_dropout(self.mlp(self.ln2(x)))    # residual + dropout
        return x

class AttentionPooling(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.q = nn.Parameter(torch.randn(1, 1, n_embd) / n_embd**0.5)
        self.W_q = nn.Linear(n_embd, n_embd)
        self.W_kv = nn.Linear(n_embd, n_embd*2)

    def forward(self, x):
        q = self.W_q(self.q).expand(x.size(0), -1, -1)
        k,v = self.W_kv(x).split(self.n_embd, dim=2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return attn
    
class DoubleConv1d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim*2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_dim*2, out_dim, kernel_size=5, padding=2)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1,2)
        x = self.ln(x)
        return x