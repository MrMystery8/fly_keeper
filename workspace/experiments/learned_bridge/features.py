"""Offline feature construction + numpy linear models for the learned bridge.

All of this operates on the saved dataset (raw per-window selected-neuron spike
counts + teacher labels), so the expensive brain simulation runs ONCE and every
alignment / feature-count / window-order variant is built offline.

Feature construction (causal): at decision step t, the feature vector is the
concatenation of the selected neurons' spike counts over the last `n_windows`
20 ms windows, most-recent-first by default:

    x_t = [ counts[t], counts[t-1], ..., counts[t-(n_windows-1)] ]

Windows before the episode start are zero-padded (matches the runtime ring
buffer). Only PAST/PRESENT windows are used.

Target alignment (Section 15): the label for features-at-t may be the teacher
command at t + offset (offset in windows). Only features available at or before
t are ever used; the offset shifts WHICH teacher command we predict, chosen on
train/val only.

Models are plain numpy (no sklearn): closed-form ridge regression and
gradient-descent regularized logistic regression, so the trainable parameter
count is explicit and the weights are inspectable.
"""
from __future__ import annotations
import numpy as np

DECISION_MS = 20.0


# ----------------------------------------------------------------- features
def episode_features(counts, n_windows=4, order="newest_first", feature_idx=None,
                     window_mask=None):
    """Build per-step temporal features for one episode.

    counts: (n_steps, n_sel) int array of per-window spike counts.
    n_windows: number of causal 20 ms windows per feature vector.
    order: 'newest_first' | 'oldest_first' | 'reversed' (reversed == flip of
           newest_first, used for the temporal-order ablation).
    feature_idx: optional indices into the n_sel neuron axis (feature-count
                 subset / population ablation). None = all neurons.
    window_mask: optional boolean length n_windows selecting which windows to
                 KEEP (for remove-oldest / remove-newest / single-window
                 ablations). Applied in newest-first indexing (0 = newest).

    Returns X: (n_steps, dim).
    """
    counts = np.asarray(counts, dtype=np.float64)
    n_steps, n_sel = counts.shape
    if feature_idx is not None:
        counts = counts[:, feature_idx]
        n_sel = counts.shape[1]
    # build the stack of windows, newest-first: win[k] = counts shifted by k
    wins = []
    for k in range(n_windows):
        if k == 0:
            wins.append(counts)
        else:
            shifted = np.zeros_like(counts)
            shifted[k:] = counts[:-k]
            wins.append(shifted)                 # counts[t-k], zero-padded
    # wins is newest-first: wins[0]=t, wins[1]=t-1, ...
    if window_mask is not None:
        wins = [w for w, keep in zip(wins, window_mask) if keep]
    if order == "oldest_first":
        wins = wins[::-1]
    elif order == "reversed":
        wins = wins[::-1]
    # newest_first == as built
    return np.concatenate(wins, axis=1)


def build_matrix(dataset, ep_indices, *, n_windows=4, order="newest_first",
                 feature_idx=None, window_mask=None, target_offset=0,
                 label_kind="continuous"):
    """Stack per-step features + labels across the given episodes.

    target_offset: predict teacher u at (t + target_offset) windows using
                   features at t. Steps without a valid target are dropped.
    label_kind: 'continuous' (teacher u), 'sign' (LEFT/RIGHT class from group),
                or 'sign_u' (sign of teacher u).

    Returns X, y, ep_id (episode index per row), meta.
    """
    counts_all = dataset["counts"]
    u_all = dataset["u_teacher"]
    groups = dataset["groups"]
    Xs, ys, eids = [], [], []
    for ei in ep_indices:
        c = counts_all[ei]
        u = np.asarray(u_all[ei], dtype=np.float64)
        n = c.shape[0]
        X = episode_features(c, n_windows=n_windows, order=order,
                             feature_idx=feature_idx, window_mask=window_mask)
        # target index t + offset
        idx = np.arange(n)
        tgt = idx + target_offset
        valid = tgt < n
        idx = idx[valid]; tgt = tgt[valid]
        if len(idx) == 0:
            continue
        Xe = X[idx]
        if label_kind == "continuous":
            ye = u[tgt]
        elif label_kind == "sign_u":
            ye = np.sign(u[tgt])
        elif label_kind == "sign":
            g = str(groups[ei])
            if g == "left":
                ye = np.full(len(idx), -1.0)
            elif g == "right":
                ye = np.full(len(idx), +1.0)
            else:
                ye = np.zeros(len(idx))          # center -> 0 (dropped by callers)
        else:
            raise ValueError(label_kind)
        Xs.append(Xe); ys.append(ye); eids.append(np.full(len(idx), ei))
    if not Xs:
        return (np.zeros((0, 0)), np.zeros(0), np.zeros(0, dtype=int))
    return (np.concatenate(Xs, 0), np.concatenate(ys, 0),
            np.concatenate(eids, 0))


# ----------------------------------------------------------------- normalize
def fit_normalizer(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd


def apply_normalizer(X, mu, sd):
    return (X - mu) / sd


# ------------------------------------------------------------ ridge (closed form)
def ridge_fit(X, y, alpha=1.0):
    """Closed-form ridge: w = (X^T X + alpha I)^-1 X^T y, with a bias column.

    Returns (w, b) where w has X.shape[1] entries. Regularizes weights only
    (not the bias).
    """
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    A = Xb.T @ Xb
    reg = alpha * np.eye(d + 1)
    reg[d, d] = 0.0                              # do not regularize bias
    coef = np.linalg.solve(A + reg, Xb.T @ y)
    return coef[:d], float(coef[d])


def ridge_predict(X, w, b):
    return X @ w + b


# ---------------------------------------------------- logistic (gradient descent)
def logistic_fit(X, y01, alpha=1.0, lr=0.1, iters=500):
    """L2-regularized logistic regression via gradient descent (numpy).

    y01 in {0,1}. Returns (w, b). alpha regularizes weights only.
    """
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        gw = X.T @ (p - y01) / n + alpha * w / n
        gb = float((p - y01).mean())
        w -= lr * gw; b -= lr * gb
    return w, b


def logistic_predict_proba(X, w, b):
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))


# ----------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    err = y_pred - y_true
    mae = float(np.abs(err).mean())
    mse = float((err ** 2).mean())
    # Pearson correlation
    if y_true.std() > 1e-9 and y_pred.std() > 1e-9:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = 0.0
    # signed direction accuracy on non-neutral targets
    nz = np.abs(y_true) > 0.1
    if nz.any():
        dir_acc = float((np.sign(y_pred[nz]) == np.sign(y_true[nz])).mean())
    else:
        dir_acc = float("nan")
    return dict(mae=round(mae, 4), mse=round(mse, 4), corr=round(corr, 4),
                signed_dir_acc=round(dir_acc, 4), n=int(len(y_true)))
