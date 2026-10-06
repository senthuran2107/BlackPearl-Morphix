# ---------- STAGE 3: simulated closed-loop experiment design ----------
import os, glob, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

FEATS = ['frac_tcell', 'frac_host', 'dist_to_tcell']


def load_cells():
    cands = sorted(glob.glob('morphix_pfish_cells*.csv.gz'), key=os.path.getmtime)
    if not cands:
        from google.colab import files
        print('Upload morphix_pfish_cells (1).csv.gz (from the successful Step 2 run):')
        up = files.upload()
        cands = [n for n in up if n.startswith('morphix_pfish_cells')]
    cells = pd.read_csv(cands[-1], index_col='cell_id')
    if not (cells.cell_class == 'tumour').any():
        raise ValueError('This is the old file from the failed run; upload the newer one.')
    return cells


def real_neighbourhood_features(cells, n_focal=40000, k=63, seed=0):
    """Neighbourhood features of real tumour-like cells (same definitions as Stage 2). Standardised, so every
    simulated effect is in SD units of real cell-to-cell variation."""
    rng = np.random.default_rng(seed)
    tc = np.log10(cells.total_counts + 1)
    thr = np.quantile(tc[cells.cell_class == 'tumour'], 0.05)
    cls = np.where(cells.cell_class == 'tcell', 1, np.where((cells.cell_class == 'tumour') | (tc >= thr), 0, 2))
    XY = cells[['x', 'y']].values
    tree = cKDTree(XY)
    unit = float(np.median(tree.query(XY, k=2)[0][:, 1]))
    focal = np.flatnonzero(cls == 0)
    focal = rng.choice(focal, min(n_focal, len(focal)), replace=False)
    d, nb = tree.query(XY[focal], k=k + 1)
    d, nb = d[:, 1:] / unit, nb[:, 1:]
    c = cls[nb]
    dt = np.where(c == 1, d, np.inf).min(1)
    dt = np.where(np.isfinite(dt), dt, d.max(1))
    F = np.c_[(c == 1).mean(1), (c == 2).mean(1), dt]
    mu, sd = F.mean(0), F.std(0)
    return (F - mu) / sd, XY[focal], unit


class VirtualLab:
    """A lab that can run experiments. Experiment = measure n cells carrying knockout g; each cell's readout is a
    REAL tumour cell's neighbourhood (real variance and spatial structure) plus the knockout's TRUE effect.
    design='randomised': cells drawn at random positions (arrayed, randomised layout).
    design='mosaic': cells drawn from one contiguous patch (a clone), like the pooled Perturb-FISH screen, so
    local tissue differences leak into every measurement."""
    def __init__(self, F, XY, effects, design, rng, n_per_exp=50):
        self.F, self.XY, self.effects, self.design, self.rng, self.n = F, XY, effects, design, rng, n_per_exp
        self.tree = cKDTree(XY) if design == 'mosaic' else None

    def run(self, g):
        if self.design == 'randomised':
            idx = self.rng.integers(0, len(self.F), self.n)      # random cells anywhere in the tissue
        else:
            centre = self.XY[self.rng.integers(len(self.XY))]
            _, idx = self.tree.query(centre, k=self.n)
        return self.F[idx] + self.effects[g]          # [n, n_feats]


def make_world(n_genes, rng, sizes=(0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 1.0)):
    """True effects: most knockouts do nothing; a few shift ONE neighbourhood feature by a measured-scale amount
    (Stage 2 found ~0.2-0.6 SD effects and one ~1 SD effect, in about a quarter of knockouts)."""
    E = np.zeros((n_genes, len(FEATS)))
    hits = rng.choice(n_genes, len(sizes), replace=False)
    for g, s in zip(hits, sizes):
        E[g, rng.integers(len(FEATS))] = s * rng.choice([-1, 1])
    return E, set(hits.tolist())


class Posterior:
    """Per-knockout, per-feature Gaussian posterior on the effect (prior N(0, tau^2), unit cell-level noise).
    Bayesian calls stay valid when the next experiment depends on earlier results."""
    def __init__(self, n_genes, n_feats, tau=0.5):
        self.s = np.zeros((n_genes, n_feats)); self.n = np.zeros(n_genes); self.tau2 = tau ** 2

    def add(self, g, Y):
        self.s[g] += Y.sum(0); self.n[g] += len(Y)

    def stats(self):
        prec = 1 / self.tau2 + self.n[:, None]
        return self.s / prec, 1 / prec                       # posterior mean, variance

    def p_real(self, delta):
        m, v = self.stats(); sd = np.sqrt(v)
        return norm.sf((delta - m) / sd) + norm.cdf((-delta - m) / sd)   # P(|effect| > delta)


def confirmed(post, delta, threshold, min_cells):
    """A knockout counts as a hit only if P(|effect| > delta) > threshold AND it has been measured in at least
    min_cells cells (i.e. confirmed by repeat experiments). Same rule for every strategy."""
    return (post.p_real(delta).max(1) > threshold) & (post.n >= min_cells)


def choose_next(strategy, post, step, n_genes, rng, delta, threshold, min_cells):
    if strategy == 'round_robin' or step < n_genes:          # everyone starts with one screening experiment
        return step % n_genes
    if strategy == 'random':
        return int(rng.integers(n_genes))
    if strategy == 'active':
        # Test the most promising knockout that is not yet a confirmed hit; ignore knockouts that already look
        # confidently null (P(real) < 5%). This spends experiments where they can still change the answer.
        p = post.p_real(delta).max(1)
        open_ = ~confirmed(post, delta, threshold, min_cells)
        cand = np.flatnonzero(open_ & (p > 0.05))
        if len(cand) == 0:
            cand = np.flatnonzero(open_)
        return int(cand[p[cand].argmax()]) if len(cand) else int(rng.integers(n_genes))
    raise ValueError(strategy)


def calls(post, delta, threshold, min_cells):
    return set(np.flatnonzero(confirmed(post, delta, threshold, min_cells)).tolist())


def run_campaign(F, XY, strategy, design, n_genes, budget, rng, n_per_exp=50, delta=0.1, threshold=0.99,
                 min_cells=150, checkpoints=None, size_scale=1.0, world=0):
    E, truth = make_world(n_genes, rng)
    E = E * size_scale
    lab = VirtualLab(F, XY, E, design, rng, n_per_exp)
    post = Posterior(n_genes, F.shape[1])
    out = []
    for step in range(budget):
        g = choose_next(strategy, post, step, n_genes, rng, delta, threshold, min_cells)
        post.add(g, lab.run(g))
        if checkpoints is None or (step + 1) in checkpoints:
            c = calls(post, delta, threshold, min_cells)
            tp = len(c & truth)
            m, _ = post.stats()
            err = np.mean([np.abs(m[g] - E[g]).max() for g in truth])
            out.append(dict(experiments=step + 1, cells=(step + 1) * n_per_exp, recall=tp / len(truth),
                            false_discoveries=len(c - truth), fdr=(len(c - truth) / max(len(c), 1)),
                            effect_error=err, strategy=strategy, design=design, world=world,
                            size_scale=size_scale))
    return out
