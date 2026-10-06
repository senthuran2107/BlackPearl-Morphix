# ---------- MODEL: neighbourhood generation conditioned on the focal cell's expression ----------
class ExprTissueFlow(nn.Module):
    """v_theta(x_t, t | z), z = PCA summary of the focal cell's gene expression.
    A learned 'null' embedding replaces z when it is hidden (10-15% of training), giving the unconditional
    model in the same network. The focal cell is a fixed token at the origin."""
    def __init__(self, n_types, z_dim, d=128, n_layers=4, n_heads=4):
        super().__init__()
        self.dim = 2 + n_types
        self.inp = nn.Linear(self.dim, d)
        self.time = TimeEmbedding(d)
        self.zproj = nn.Sequential(nn.Linear(z_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.zeros(1, d))
        self.focal = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.dim)

    def forward(self, x, t, z, keep):
        B = x.shape[0]
        k = keep[:, None].float()
        c = self.time(t) + k * self.zproj(z) + (1 - k) * self.null
        h = torch.cat([self.focal.expand(B, 1, -1), self.inp(x)], dim=1) + c[:, None]
        pos = torch.cat([torch.zeros(B, 1, 2, device=x.device, dtype=x.dtype), x[..., :2]], dim=1)
        for blk in self.blocks:
            h = blk(h, c, pos)
        return self.out(self.out_norm(h))[:, 1:]


def expr_flow_loss(model, pos, typ, z, n_types, pos_scale, p_drop=0.15, keep=None):
    B = pos.shape[0]
    x1 = torch.cat([pos / pos_scale, F.one_hot(typ, n_types).float()], dim=-1)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=pos.device)
    tt = t[:, None, None]
    xt = (1 - tt) * x0 + tt * x1
    if keep is None:
        keep = torch.rand(B, device=pos.device) >= p_drop
    v = model(xt, t, z, keep)
    err = (v - (x1 - x0)) ** 2
    lp, lt = err[..., :2].mean(), err[..., 2:].mean()
    return lp + lt, {'pos': lp.item(), 'type': lt.item()}


@torch.no_grad()
def sample_expr(model, z, n_types, n_cells, pos_scale, steps=50, batch=1000, keep=True, device='cpu', seed=0):
    """One generated neighbourhood per row of z (repeat rows of z to get several samples per cell)."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    P, T = [], []
    for i in range(0, len(z), batch):
        zb = torch.as_tensor(z[i:i + batch], device=device)
        m = len(zb)
        x = torch.randn(m, n_cells, 2 + n_types, device=device, generator=g)
        kb = torch.full((m,), bool(keep), device=device)
        for k in range(steps):
            t = torch.full((m,), k / steps, device=device)
            x = x + model(x, t, zb, kb) / steps
        P.append((x[..., :2] * pos_scale).cpu().numpy())
        T.append(x[..., 2:].argmax(-1).cpu().numpy())
    model.train()
    return np.concatenate(P), np.concatenate(T)
