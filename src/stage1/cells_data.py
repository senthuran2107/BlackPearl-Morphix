# ---------- DATA CELL (numpy / scipy only) ----------
import os, sys, subprocess, urllib.request
import numpy as np
from scipy.spatial import cKDTree

DATASETS = {
    # Mouse embryo seqFISH (Lohoff et al. 2021), ~19k cells, ~22 cell types
    'seqfish': dict(url='https://ndownloader.figshare.com/files/26098364', file='seqfish.h5ad',
                    key='celltype_mapped_refined', sq='seqfish'),
    # Human breast cancer imaging mass cytometry (Jackson et al. 2020), ~4.7k cells
    'imc': dict(url='https://ndownloader.figshare.com/files/26098406', file='imc.h5ad',
                key='cell type', sq='imc'),
}


def load_anndata(name, data_dir='data'):
    """Download a public spatial dataset as AnnData. Falls back to squidpy if the direct link fails."""
    import anndata as ad
    info = DATASETS[name]
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, info['file'])
    if os.path.exists(path):
        return ad.read_h5ad(path)
    try:
        print(f'Downloading {name} ...')
        urllib.request.urlretrieve(info['url'], path)
        return ad.read_h5ad(path)
    except Exception as e:
        print(f'Direct download failed ({e}). Falling back to squidpy (slower install)...')
        if os.path.exists(path):
            os.remove(path)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'squidpy'], check=True)
        import squidpy as sq
        adata = getattr(sq.datasets, info['sq'])()
        adata.write_h5ad(path)
        return adata


def extract_cells(adata, preferred_key=None):
    """Return 2D coordinates, integer cell types, type names and the obs column used."""
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)[:, :2]
    key = preferred_key if preferred_key in adata.obs.columns else None
    if key is None:
        cands = [c for c in adata.obs.columns
                 if (str(adata.obs[c].dtype) in ('category', 'object'))
                 and 3 <= adata.obs[c].nunique() <= 60]
        if not cands:
            raise ValueError(f'No cell-type column found. obs columns: {list(adata.obs.columns)}')
        key = cands[0]
    labels = adata.obs[key].astype(str).values
    type_names, types = np.unique(labels, return_inverse=True)
    return coords, types.astype(np.int64), [str(t) for t in type_names], key


def normalise_spacing(coords):
    """Rescale so the median nearest-neighbour distance between cells is 1 ('cell spacing units')."""
    d, _ = cKDTree(coords).query(coords, k=2)
    nn = d[:, 1]
    d_nn = float(np.median(nn[nn > 0]))
    return coords / d_nn, d_nn


def block_split(coords, test_frac=0.2, n_blocks=4, seed=0):
    """Cut the tissue into an n_blocks x n_blocks grid and hold out random blocks as test tissue.
    Unlike a single strip, test blocks are scattered across the tissue, so train and test
    contain similar anatomical regions and cell-type mixes. Returns a boolean test mask per cell."""
    rng = np.random.default_rng(seed)
    lo, hi = coords.min(0), coords.max(0)
    b = np.minimum(((coords - lo) / (hi - lo + 1e-9) * n_blocks).astype(int), n_blocks - 1)
    block_id = b[:, 0] * n_blocks + b[:, 1]
    ids = rng.permutation(np.unique(block_id))
    counts = np.bincount(block_id, minlength=n_blocks ** 2)
    test_blocks, n_test = [], 0
    for i in ids:
        if n_test >= test_frac * len(coords):
            break
        test_blocks.append(i)
        n_test += counts[i]
    return np.isin(block_id, test_blocks)


def make_patches(coords, types, in_split, n_patches, n_cells, rng):
    """A patch = the n_cells nearest cells to a random centre cell, centred at their centroid.
    Neighbours are searched in the WHOLE tissue and a patch is kept only if every one of its cells
    belongs to this split, so patches are never cut off by a block edge and train/test never share cells."""
    tree = cKDTree(coords)
    pool = np.flatnonzero(in_split)
    _, idx_all = tree.query(coords[pool], k=n_cells)
    clean = in_split[idx_all].all(1)
    if clean.sum() == 0:
        raise ValueError('No clean patches: use fewer/larger blocks or fewer cells per patch.')
    idx = idx_all[clean][rng.integers(0, clean.sum(), n_patches)]
    idx = np.take_along_axis(idx, rng.permuted(np.tile(np.arange(n_cells), (n_patches, 1)), axis=1), axis=1)
    pos = coords[idx]
    pos = pos - pos.mean(axis=1, keepdims=True)
    return pos.astype(np.float32), types[idx].astype(np.int64)
