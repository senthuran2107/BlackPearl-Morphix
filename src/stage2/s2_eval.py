# ---------- EVALUATION: does the model reproduce knockout effects on held-out tissue? ----------
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance, pearsonr

FEATS = ['frac_tcell', 'frac_host', 'mean_dist', 'nn_dist', 'dist_to_tcell']


def patch_features(pos, typ):
    """Per-neighbourhood summaries, all relative to the focal cell at the origin (cell-spacing units)."""
    r = np.sqrt((pos ** 2).sum(-1))                      # distance of each neighbour to the focal cell
    is_t = typ == 1
    dt = np.where(is_t, r, np.inf).min(1)
    dt = np.where(np.isfinite(dt), dt, r.max(1))         # no T cell in the patch -> patch edge
    return pd.DataFrame({'frac_tcell': is_t.mean(1), 'frac_host': (typ == 2).mean(1),
                         'mean_dist': r.mean(1), 'nn_dist': r.min(1), 'dist_to_tcell': dt})


def shifts(feat_df, cond, genes_ids, control_id):
    """Mean feature of each knockout minus mean of Control (same set of cells)."""
    ctrl = feat_df[cond == control_id]
    out = {}
    for g in genes_ids:
        sel = feat_df[cond == g]
        out[g] = (sel.mean() - ctrl.mean()).values, len(sel)
    return out, ctrl


def matched_shifts(feat_df, cond, block, genes_ids, control_id):
    """Location-matched effect: for each knockout, compare its cells with Control cells from the SAME tissue
    block (as in the Step 3 signal check), then average over blocks weighted by the knockout's cells."""
    X = feat_df.values
    out = {}
    ctrl = cond == control_id
    for g in genes_ids:
        isg = cond == g
        diffs, w, var_g, var_c = [], [], [], []
        for b in np.unique(block[isg]):
            gi, ci = isg & (block == b), ctrl & (block == b)
            if ci.sum() == 0:
                continue
            diffs.append(X[gi].mean(0) - X[ci].mean(0)); w.append(gi.sum())
            var_c.append(X[ci].var(0) / ci.sum() * gi.sum() ** 2)
        if not w:
            out[g] = (np.full(X.shape[1], np.nan), 0, np.full(X.shape[1], np.nan)); continue
        w = np.array(w, float); diffs = np.array(diffs)
        n_used = int(w.sum())
        sel = X[isg & np.isin(block, [b for b in np.unique(block[isg]) if (ctrl & (block == b)).any()])]
        se = np.sqrt(sel.var(0) / n_used + np.sum(var_c, 0) / w.sum() ** 2)
        out[g] = ((w[:, None] * diffs).sum(0) / w.sum(), n_used, se)
    return out


def evaluate_matched(train_df, train_c, train_b, test_df, test_c, test_b, gen, cond_names, min_test=10):
    ctrl_id = cond_names.index('Control')
    genes = [i for i in range(2, len(cond_names)) if (test_c == i).sum() >= min_test and (train_c == i).sum() >= min_test]
    te = matched_shifts(test_df, test_c, test_b, genes, ctrl_id)
    trn = matched_shifts(train_df, train_c, train_b, genes, ctrl_id)
    sd = test_df[test_c == ctrl_id].std().values + 1e-9
    rows = []
    for g in genes:
        real, n_g, se = te[g]
        if n_g < min_test or not np.all(np.isfinite(trn[g][0])):
            continue
        model = (gen[g].mean() - gen[ctrl_id].mean()).values
        for j, f in enumerate(FEATS):
            rows.append(dict(gene=cond_names[g], feature=f, n_test_matched=n_g, real_test=real[j] / sd[j],
                             test_se=se[j] / sd[j], lookup_train=trn[g][0][j] / sd[j], model=model[j] / sd[j]))
    tab = pd.DataFrame(rows)
    y = tab.real_test.values
    summ = {'n_knockouts_evaluated': int(tab.gene.nunique()), 'n_gene_feature_pairs': len(tab),
            'noise_floor_mse': float(np.mean(tab.test_se ** 2)),
            'mse_zero': float(np.mean(y ** 2)),
            'mse_lookup': float(np.mean((tab.lookup_train - y) ** 2)),
            'mse_model': float(np.mean((tab.model - y) ** 2)),
            'r_lookup_vs_test': float(pearsonr(tab.lookup_train, y)[0]),
            'r_model_vs_test': float(pearsonr(tab.model, y)[0]),
            'r_model_vs_train': float(pearsonr(tab.model, tab.lookup_train)[0])}
    return tab, summ


def evaluate_effects(train_df, train_c, test_df, test_c, gen, cond_names, min_test=10):
    """gen: {cond_id: features DataFrame of generated neighbourhoods}. Returns long table + summary."""
    ctrl_id = cond_names.index('Control')
    genes = [i for i in range(2, len(cond_names)) if (test_c == i).sum() >= min_test and (train_c == i).sum() >= min_test]
    test_sh, test_ctrl = shifts(test_df, test_c, genes, ctrl_id)
    train_sh, _ = shifts(train_df, train_c, genes, ctrl_id)
    sd = test_ctrl.std().values + 1e-9
    rows = []
    for g in genes:
        real, n_g = test_sh[g]
        sel = test_df[test_c == g]
        se = np.sqrt(sel.var().values / n_g + test_ctrl.var().values / len(test_ctrl)) / sd
        model = (gen[g].mean() - gen[ctrl_id].mean()).values
        for j, f in enumerate(FEATS):
            rows.append(dict(gene=cond_names[g], feature=f, n_test=n_g,
                             real_test=real[j] / sd[j], test_se=se[j],
                             lookup_train=train_sh[g][0][j] / sd[j], model=model[j] / sd[j]))
    tab = pd.DataFrame(rows)
    y = tab.real_test.values
    summ = {
        'n_knockouts_evaluated': len(genes), 'n_gene_feature_pairs': len(tab),
        'noise_floor_mse (unavoidable, from test sampling error)': float(np.mean(tab.test_se ** 2)),
        'mse_zero (no-effect baseline)': float(np.mean(y ** 2)),
        'mse_lookup (train shifts as prediction)': float(np.mean((tab.lookup_train - y) ** 2)),
        'mse_model': float(np.mean((tab.model - y) ** 2)),
        'r_lookup_vs_test': float(pearsonr(tab.lookup_train, y)[0]),
        'r_model_vs_test': float(pearsonr(tab.model, y)[0]),
        'r_model_vs_train (did it learn the training effects?)': float(pearsonr(tab.model, tab.lookup_train)[0]),
    }
    return tab, summ


def realism(gen_df, test_df, train_df):
    """Per-feature W1 distance to held-out Control neighbourhoods, vs. real training Control (reference)."""
    return pd.DataFrame({'generated vs test': [wasserstein_distance(gen_df[f], test_df[f]) for f in FEATS],
                         'real train vs test (reference)': [wasserstein_distance(train_df[f], test_df[f]) for f in FEATS]},
                        index=FEATS)
