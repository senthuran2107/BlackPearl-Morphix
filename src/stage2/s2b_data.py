# ---------- DATA: expression of every cell + neighbourhoods around tumour-like focal cells ----------
import urllib.request

BASE = 'https://download.brainimagelibrary.org/0c/bd/0cbd479c521afff9/extras/tumors/processed/finaltables/'


def fetch(name, folder='pfish', tries=6):
    """Download with retries: the Brain Image Library server sometimes times out briefly."""
    import time as _t
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    for k in range(tries):
        try:
            print(f'downloading {name} (attempt {k + 1}/{tries}) ...')
            req = urllib.request.Request(BASE + name, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as r, open(path + '.part', 'wb') as f:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(path + '.part', path)
            return path
        except Exception as e:
            wait = 15 * (k + 1)
            print(f'  failed ({e}); retrying in {wait}s')
            _t.sleep(wait)
    raise RuntimeError(f'Could not download {name} after {tries} attempts. The data server may be down; try again later.')


def load_expression(n_cells_expected, gene_cols=(3, 503)):
    """500-gene counts for every cell (rows aligned with the cells file). Step 2 found the genes in columns 3..502
    of merfishcounttable.csv; gene names come from tumorMerfish.csv. Verified here by exact row matching."""
    def read_tm(nrows):
        # read like Step 2: no forced index column (the header may be one name shorter than the data rows)
        df = pd.read_csv(fetch('tumorMerfish.csv'), nrows=nrows)
        if str(df.columns[0]).startswith('Unnamed'):
            df = df.iloc[:, 1:]
        return df
    tm_df = read_tm(400)
    names = [str(c) for c in tm_df.columns]
    full = pd.read_csv(fetch('merfishcounttable.csv'), header=None, dtype=np.float32).values
    if full.shape[0] != n_cells_expected:
        raise ValueError(f'count table has {full.shape[0]} rows, cells file has {n_cells_expected}')
    X = full[:, gene_cols[0]:gene_cols[1]]
    del full
    # verification: sample tumour rows from tumorMerfish.csv must appear exactly in X
    if len(names) != X.shape[1]:
        raise ValueError(f'tumorMerfish.csv has {len(names)} gene columns but the count table slice has {X.shape[1]}')
    tm = tm_df.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
    w = np.random.default_rng(0).integers(1, 1_000_003, size=X.shape[1]).astype(np.float64)
    sig = set(np.round(X.astype(np.float64) @ w).astype(np.int64))
    ok = np.mean([s in sig for s in np.round(tm[tm.sum(1) >= 5] @ w).astype(np.int64)])
    print(f'gene columns verified: {100 * ok:.1f}% of sample tumour rows found exactly')
    if ok < 0.9:
        raise ValueError('gene columns do not match tumorMerfish.csv; the column layout differs from Step 2')
    return X, names


def lognorm(X):
    tot = X.sum(1, keepdims=True)
    return np.log1p(X / np.maximum(tot, 1) * np.median(tot[tot > 0]))


def leaky_genes(L, cells, names, ratio=2.0, eps=0.05):
    """Pre-specified rule: drop genes whose mean (log-normalised) expression in T cells or in host cells is more than
    `ratio` x that in labelled tumour cells. These can 'leak' into a tumour cell from touching neighbours."""
    tum = (cells.cell_class == 'tumour').values
    tc = (cells.ntype == TYPES.index('tcell')).values
    host = (cells.ntype == TYPES.index('host')).values
    m_t, m_c, m_h = L[tum].mean(0), L[tc].mean(0), L[host].mean(0)
    leak = ((m_c + eps) / (m_t + eps) > ratio) | ((m_h + eps) / (m_t + eps) > ratio)
    tab = pd.DataFrame({'gene': names, 'tumour': m_t, 'tcell': m_c, 'host': m_h, 'dropped': leak})
    return leak, tab


def pca_fit(Z, k, max_rows=60000, seed=0):
    rng = np.random.default_rng(seed)
    S = Z[rng.choice(len(Z), min(max_rows, len(Z)), replace=False)]
    mu, sd = S.mean(0), S.std(0) + 1e-6
    _, _, Vt = np.linalg.svd((S - mu) / sd, full_matrices=False)
    W = Vt[:k].T
    proj = ((S - mu) / sd) @ W
    return dict(mu=mu, sd=sd, W=W, psd=proj.std(0) + 1e-6)


def std_fit(Z):
    return dict(mu=Z.mean(0), sd=Z.std(0) + 1e-3)


def std_apply(Z, P, clip=5.0):
    return np.clip((Z - P['mu']) / P['sd'], -clip, clip).astype(np.float32)


def pca_apply(Z, P):
    return (((Z - P['mu']) / P['sd']) @ P['W'] / P['psd']).astype(np.float32)


def build_expr_patches(cells, cfg, rng):
    XY = cells[['x', 'y']].values
    tree = cKDTree(XY)
    d2, _ = tree.query(XY, k=2)
    unit = float(np.median(d2[:, 1]))
    is_test = block_split_xy(XY, cfg['test_frac'], cfg['n_blocks'], cfg['seed'])
    focal_all = np.flatnonzero(cells.ntype.values == TYPES.index('tumour_like'))
    _, nb = tree.query(XY[focal_all], k=cfg['n_neighbours'] + 1)
    nb = nb[:, 1:]
    clean = (is_test[nb] == is_test[focal_all][:, None]).all(1)
    focal_all, nb = focal_all[clean], nb[clean]
    te = is_test[focal_all]
    keep = np.zeros(len(focal_all), bool)
    for split, cap in [(False, cfg['max_train']), (True, cfg['max_test'])]:
        idx = np.flatnonzero(te == split)
        keep[rng.choice(idx, min(cap, len(idx)), replace=False)] = True
    focal, nb = focal_all[keep], nb[keep]
    pos = ((XY[nb] - XY[focal][:, None, :]) / unit).astype(np.float32)
    typ = cells.ntype.values[nb].astype(np.int64)
    return dict(pos=pos, typ=typ, focal=focal, test=is_test[focal]), unit
