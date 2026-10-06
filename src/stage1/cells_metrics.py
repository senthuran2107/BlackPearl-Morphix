# ---------- METRICS CELL (numpy / scipy only) ----------
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance


def nn_distances(pos):
    """Distance from every cell to its nearest neighbour, pooled over patches (cell-spacing units)."""
    out = []
    for p in pos:
        d, _ = cKDTree(p).query(p, k=2)
        out.append(d[:, 1])
    return np.concatenate(out)


def pair_distances(pos, max_patches=300):
    return np.concatenate([pdist(p) for p in pos[:max_patches]])


def patch_radius(pos):
    """Radius of gyration of each patch: how spread out / compact it is."""
    return np.sqrt((pos ** 2).sum(-1).mean(-1))


def composition(typ, n_types):
    c = np.bincount(typ.ravel(), minlength=n_types).astype(float)
    return c / c.sum()


def cooccurrence(pos, typ, n_types, k=5):
    """P(neighbour type = b | cell type = a) over each cell's k nearest neighbours."""
    C = np.zeros((n_types, n_types))
    for p, t in zip(pos, typ):
        _, idx = cKDTree(p).query(p, k=k + 1)
        nb = t[idx[:, 1:]]
        np.add.at(C, (np.repeat(t, k), nb.ravel()), 1)
    rows = C.sum(1, keepdims=True)
    return C / np.maximum(rows, 1), rows.ravel()


def evaluate(gen_pos, gen_typ, ref_pos, ref_typ, n_types, d_min):
    """Compare a set of patches against reference (held-out real) patches. Lower = closer to real tissue."""
    nn_g, nn_r = nn_distances(gen_pos), nn_distances(ref_pos)
    P_g, _ = cooccurrence(gen_pos, gen_typ, n_types)
    P_r, rows_r = cooccurrence(ref_pos, ref_typ, n_types)
    w = rows_r / rows_r.sum()
    return {
        'NN-distance W1': float(wasserstein_distance(nn_g, nn_r)),
        'Pair-distance W1': float(wasserstein_distance(pair_distances(gen_pos), pair_distances(ref_pos))),
        'Patch-radius W1': float(wasserstein_distance(patch_radius(gen_pos), patch_radius(ref_pos))),
        'Composition TVD': float(0.5 * np.abs(composition(gen_typ, n_types) - composition(ref_typ, n_types)).sum()),
        'Neighbourhood TVD': float((w * 0.5 * np.abs(P_g - P_r).sum(1)).sum()),
        'Overlap rate': float((nn_g < d_min).mean()),
    }


# ---------- BASELINES ----------
def random_baseline(n, n_cells, ref_pos, type_freq, rng):
    """Cells scattered uniformly in a disc of realistic size, types drawn independently."""
    R = np.median(np.sqrt((ref_pos ** 2).sum(-1)).max(-1))
    r = R * np.sqrt(rng.random((n, n_cells)))
    a = rng.random((n, n_cells)) * 2 * np.pi
    pos = np.stack([r * np.cos(a), r * np.sin(a)], -1)
    pos -= pos.mean(1, keepdims=True)
    typ = rng.choice(len(type_freq), size=(n, n_cells), p=type_freq)
    return pos.astype(np.float32), typ


def shuffled_types_baseline(pos, typ, rng):
    """Real cell positions, but cell types shuffled within each patch: tests if arrangement of types is learned."""
    return pos.copy(), rng.permuted(typ, axis=1)
