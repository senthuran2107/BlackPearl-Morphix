# ---------- MODEL: knockout-conditioned flow matching over a cell neighbourhood ----------
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_dist(p):
    diff = p[:, :, None, :] - p[:, None, :, :]
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)


class TimeEmbedding(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        half = self.d // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = 1000.0 * t[:, None] * freqs[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=-1))


class DistanceBias(nn.Module):
    def __init__(self, n_heads, n_rbf=16, max_dist=4.0):
        super().__init__()
        self.register_buffer('centres', torch.linspace(0, max_dist, n_rbf))
        self.gamma = (n_rbf / max_dist) ** 2
        self.proj = nn.Linear(n_rbf, n_heads)

    def forward(self, pos):
        d = pairwise_dist(pos)
        rbf = torch.exp(-self.gamma * (d[..., None] - self.centres) ** 2)
        return self.proj(rbf).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.bias = DistanceBias(n_heads)
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, h, c, pos):
        B, N, _ = h.shape
        s1, b1, g1, s2, b2, g2 = self.ada(c)[:, None].chunk(6, dim=-1)
        mask = self.bias(pos).reshape(B * self.n_heads, N, N)
        x = self.norm1(h) * (1 + s1) + b1
        h = h + g1 * self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]
        x = self.norm2(h) * (1 + s2) + b2
        return h + g2 * self.mlp(x)


class ConditionalTissueFlow(nn.Module):
    """v_theta(x_t, t | knockout). The focal (knocked-out) cell is a fixed token at the origin; the 63 neighbours are
    generated around it. The knockout enters through a learned embedding added to the time embedding,
    so it modulates every block. Condition 0 = 'unknown', which doubles as the unconditional model."""
    def __init__(self, n_types, n_cond, d=128, n_layers=4, n_heads=4):
        super().__init__()
        self.dim = 2 + n_types
        self.inp = nn.Linear(self.dim, d)
        self.time = TimeEmbedding(d)
        self.cond = nn.Embedding(n_cond, d)
        nn.init.normal_(self.cond.weight, std=0.02)
        self.focal = nn.Parameter(torch.randn(1, 1, d) * 0.02)   # fixed token for the focal cell at the origin
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.dim)

    def forward(self, x, t, cond):
        B = x.shape[0]
        c = self.time(t) + self.cond(cond)
        # prepend the focal cell as a fixed token at (0, 0): neighbours can 'see' it and keep a realistic distance
        h = torch.cat([self.focal.expand(B, 1, -1), self.inp(x)], dim=1) + c[:, None]
        pos = torch.cat([torch.zeros(B, 1, 2, device=x.device, dtype=x.dtype), x[..., :2]], dim=1)
        for blk in self.blocks:
            h = blk(h, c, pos)
        return self.out(self.out_norm(h))[:, 1:]               # velocities for the 63 neighbours only


def random_rotate(pos):
    """Rotate/reflect neighbourhoods about the focal cell (origin)."""
    B = pos.shape[0]
    th = torch.rand(B, device=pos.device) * 2 * math.pi
    c, s = th.cos(), th.sin()
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    pos = pos @ R.transpose(1, 2)
    flip = (torch.rand(B, 1, device=pos.device) < 0.5).float() * 2 - 1
    return torch.cat([pos[..., :1] * flip[..., None], pos[..., 1:]], dim=-1)


def cond_flow_loss(model, pos, typ, cond, n_types, pos_scale, p_drop=0.1):
    B = pos.shape[0]
    x1 = torch.cat([pos / pos_scale, F.one_hot(typ, n_types).float()], dim=-1)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=pos.device)
    tt = t[:, None, None]
    xt = (1 - tt) * x0 + tt * x1
    drop = torch.rand(B, device=pos.device) < p_drop          # hide the knockout -> learn the unconditional model too
    cond = torch.where(drop, torch.zeros_like(cond), cond)
    v = model(xt, t, cond)
    err = (v - (x1 - x0)) ** 2
    lp, lt = err[..., :2].mean(), err[..., 2:].mean()
    return lp + lt, {'pos': lp.item(), 'type': lt.item()}


@torch.no_grad()
def sample_cond(model, cond_id, n, n_cells, n_types, pos_scale, steps=100, batch=250, guidance=1.0, device='cpu', seed=0):
    """Generate n neighbourhoods for one condition. guidance=1: plain conditional model;
    guidance>1: classifier-free guidance (exaggerates the knockout-specific signal)."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    P, T = [], []
    for i in range(0, n, batch):
        m = min(batch, n - i)
        x = torch.randn(m, n_cells, 2 + n_types, device=device, generator=g)
        c = torch.full((m,), cond_id, device=device, dtype=torch.long)
        c0 = torch.zeros_like(c)
        for k in range(steps):
            t = torch.full((m,), k / steps, device=device)
            v = model(x, t, c)
            if guidance != 1.0:
                v0 = model(x, t, c0)
                v = v0 + guidance * (v - v0)
            x = x + v / steps
        P.append((x[..., :2] * pos_scale).cpu().numpy())
        T.append(x[..., 2:].argmax(-1).cpu().numpy())
    model.train()
    return np.concatenate(P), np.concatenate(T)
