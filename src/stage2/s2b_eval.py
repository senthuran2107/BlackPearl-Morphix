# ---------- EVALUATION: per-cell prediction of the real neighbourhood on held-out tissue ----------
from scipy.stats import spearmanr


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def knn_predict(z_tr, Y_tr, z_te, k=50):
    _, idx = cKDTree(z_tr).query(z_te, k=k)
    return Y_tr[idx].mean(1)


def ridge_predict(z_tr, Y_tr, z_te, lam=1.0):
    A = np.c_[z_tr, np.ones(len(z_tr))]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y_tr)
    return np.c_[z_te, np.ones(len(z_te))] @ W


def score_table(Y, preds):
    rows = []
    for name, P in preds.items():
        for j, f in enumerate(FEATS):
            rows.append(dict(method=name, feature=f, R2=r2(Y[:, j], P[:, j]),
                             spearman=float(spearmanr(Y[:, j], P[:, j])[0]) if np.std(P[:, j]) > 0 else 0.0))
    return pd.DataFrame(rows)
