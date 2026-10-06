# ---------- MODEL CELL (PyTorch) ----------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_dist(p):
    """Pairwise distances with a safe gradient at zero (torch.cdist can give NaN gradients there)."""
    diff = p[:, :, None, :] - p[:, None, :, :]
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)


class TimeEmbedding(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):                       # t: [B] in [0, 1]
        half = self.d // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = 1000.0 * t[:, None] * freqs[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=-1))


class DistanceBias(nn.Module):
    """Turns pairwise cell distances into per-head attention biases: nearby cells 'talk' more.
    This makes each block a message-passing layer on a fully connected, distance-weighted cell graph."""
    def __init__(self, n_heads, n_rbf=16, max_dist=4.0):
        super().__init__()
        self.register_buffer('centres', torch.linspace(0, max_dist, n_rbf))
        self.gamma = (n_rbf / max_dist) ** 2
        self.proj = nn.Linear(n_rbf, n_heads)

    def forward(self, pos):                     # pos: [B, N, 2]
        d = pairwise_dist(pos)                  # [B, N, N]
        rbf = torch.exp(-self.gamma * (d[..., None] - self.centres) ** 2)
        return self.proj(rbf).permute(0, 3, 1, 2)   # [B, H, N, N]


class Block(nn.Module):
    """Transformer block with time-conditioned adaptive LayerNorm (as in DiT)."""
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


class TissueFlow(nn.Module):
    """Velocity field v_theta(x_t, t) over a SET of cells. Each cell's state = [x, y, soft one-hot type].
    No positional encoding over cell index, so the model is permutation-equivariant (cell order is meaningless)."""
    def __init__(self, n_types, d=128, n_layers=4, n_heads=4):
        super().__init__()
        self.dim = 2 + n_types
        self.inp = nn.Linear(self.dim, d)
        self.time = TimeEmbedding(d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.dim)

    def forward(self, x, t):                    # x: [B, N, 2+K], t: [B]
        c = self.time(t)
        h = self.inp(x) + c[:, None]
        for blk in self.blocks:
            h = blk(h, c, x[..., :2])
        return self.out(self.out_norm(h))


def random_rotate(pos):
    """Tissue has no preferred orientation: random rotation + reflection as augmentation."""
    B = pos.shape[0]
    th = torch.rand(B, device=pos.device) * 2 * math.pi
    c, s = th.cos(), th.sin()
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)   # [B, 2, 2]
    pos = pos @ R.transpose(1, 2)
    flip = (torch.rand(B, 1, device=pos.device) < 0.5).float() * 2 - 1
    return torch.cat([pos[..., :1] * flip[..., None], pos[..., 1:]], dim=-1)


def flow_matching_loss(model, pos, typ, n_types, pos_scale, d_min, type_weight=1.0, overlap_weight=0.0):
    """Conditional flow matching (straight paths from Gaussian noise x0 to real tissue x1),
    plus an optional physics term: penalise predicted final cells that sit closer than d_min."""
    B, N, _ = pos.shape
    x1 = torch.cat([pos / pos_scale, F.one_hot(typ, n_types).float()], dim=-1)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=pos.device)
    tt = t[:, None, None]
    xt = (1 - tt) * x0 + tt * x1
    v = model(xt, t)
    err = (v - (x1 - x0)) ** 2
    loss_pos, loss_type = err[..., :2].mean(), err[..., 2:].mean()
    loss = loss_pos + type_weight * loss_type
    loss_ov = torch.zeros((), device=pos.device)
    if overlap_weight > 0:
        x1_hat = (xt[..., :2] + (1 - tt) * v[..., :2]) * pos_scale      # predicted final positions, spacing units
        d = pairwise_dist(x1_hat) + torch.eye(N, device=pos.device) * 1e3
        viol = F.relu(d_min - d) ** 2
        loss_ov = (viol.sum((1, 2)) / N * t).mean()       # energy per cell; trust the estimate more near t=1
        loss = loss + overlap_weight * loss_ov
    return loss, {'pos': loss_pos.item(), 'type': loss_type.item(), 'overlap': loss_ov.item()}


@torch.no_grad()
def sample(model, n, n_cells, n_types, pos_scale, steps=100, batch=250, device='cpu', keep_traj=False):
    """Integrate dx/dt = v_theta(x, t) from noise (t=0) to tissue (t=1) with Euler steps."""
    model.eval()
    all_pos, all_typ, traj = [], [], []
    for i in range(0, n, batch):
        m = min(batch, n - i)
        x = torch.randn(m, n_cells, 2 + n_types, device=device)
        for k in range(steps):
            if keep_traj and i == 0 and k % (steps // 4) == 0:
                traj.append(x[0].clone().cpu())
            t = torch.full((m,), k / steps, device=device)
            x = x + model(x, t) / steps
        if keep_traj and i == 0:
            traj.append(x[0].clone().cpu())
        pos = x[..., :2] * pos_scale
        pos = pos - pos.mean(1, keepdim=True)
        all_pos.append(pos.cpu().numpy())
        all_typ.append(x[..., 2:].argmax(-1).cpu().numpy())
    model.train()
    out = (np.concatenate(all_pos), np.concatenate(all_typ))
    return (*out, traj) if keep_traj else out
