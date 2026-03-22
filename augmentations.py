import torch

class SequenceAugmenter:
    def __init__(self, jitter_std=0.01, scale_range=(0.9, 1.1), max_shift=2, prob=0.5):
        self.jitter_std = jitter_std
        self.smin, self.smax = scale_range
        self.max_shift = max_shift
        self.prob = prob

    def jitter(self, x):
        if torch.rand(1, device=x.device) < self.prob:
            x = x + torch.randn_like(x) * self.jitter_std
        return x

    def scale(self, x):
        if torch.rand(1, device=x.device) < self.prob:
            B, T, C = x.shape
            s = torch.rand(B, 1, 1, device=x.device) * (self.smax - self.smin) + self.smin
            x = x * s
        return x

    def temporal_shift(self, x):
        if torch.rand(1, device=x.device) < self.prob:
            B, T, C = x.shape
            shifts = torch.randint(-self.max_shift, self.max_shift + 1, (B,), device=x.device)
            base_idx = torch.arange(T, device=x.device).expand(B, T)
            idx = (base_idx - shifts[:, None]).clamp(0, T - 1)
            idx = idx.unsqueeze(-1).expand(B, T, C)
            x = x.gather(1, idx)
        return x

    def __call__(self, seqs):
        seqs["raw"] = self.jitter(self.scale(self.temporal_shift(seqs["raw"])))
        """seqs["jcd"] = self.jitter(self.temporal_shift(seqs["jcd"]))
        seqs["slow_diff"] = self.jitter(seqs["slow_diff"])
        seqs["fast_diff"] = self.jitter(seqs["fast_diff"])"""

        return seqs