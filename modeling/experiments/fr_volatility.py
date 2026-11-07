"""
experiments/fr_volatility.py - Why a distributional forecast helps: what the input says about RTT volatility.

Question
--------
Does the 6 s input tell a forecaster how far RTT will move in the next 0.2-3 s,
and does a quantile forecast (DS2S) use that while a point forecast with a fixed
conformal band (S2S) cannot? Two candidate mechanisms are separated:
  (a) volatility clustering INSIDE a measurement run (calm follows calm, bursts
      follow bursts, the effect GARCH models capture in returns), and
  (b) noise level that differs BETWEEN runs (a calm-looking input window identifies
      a calm run), which the input reveals without any clustering.
Every window-level correlation is therefore also split into a within-session part
and a between-session part (see corr_ci).
Note on direction: a near-zero autocorrelation of the RTT change beyond 0.2 s does
NOT mean the direction of the next move is unpredictable. RTT that is noise around
a level (5G) reverts to that level, so the sign of the next change is predictable
from the current deviation. Item 5 measures this directly.

What it computes (no training, CPU only, nothing fitted on test data)
--------------------------------------------------------------------
1. ACF on TRAINING sessions per link, lags 1..50 rows (0.2 s .. 10 s), of
       RTT     the level r[t],
       dRTT    the row-to-row change d[t] = r[t] - r[t-1],
       |dRTT|  the change magnitude.
   Only pairs of rows inside one session are used (never across a session
   boundary) and a change needs two consecutive rows of the same session.
   Primary estimate ("measured", "within"): the measured RTT that the models are
   scored on, restricted to real pings (rows the Starlink parser interpolated
   are dropped, so a change needs two consecutive real pings), with every
   session centred on its own mean. Own-mean centring removes differences in
   level and volatility BETWEEN sessions, so a positive |dRTT| ACF is clustering
   inside a session, not "some sessions are noisier than others".
   95% CI: session bootstrap (training sessions resampled with replacement,
   B = --n_boot, fixed seed).
   Lag 1 (0.2 s) is partly mechanical: one spike makes two consecutive large
   changes, so even memoryless noise gives ACF(dRTT) = -0.5 and ACF(|dRTT|) > 0
   at lag 1 and exactly 0 beyond. The "shuffle" null measures this per link:
   inside every session the real pings are permuted (same values, no time
   order). Memory is therefore read from lags >= 0.4 s.
   Per-session ACFs at 0.2 s, 1 s and 5 s show whether a few sessions drive it.
   Robustness variants (CSV only): pooled centring (one global mean), and the
   model-input view of RTT (interpolated pings held at the last real ping).
2. TEST windows: Pearson and Spearman correlation between the RTT standard
   deviation of the 6 s input window (model-input view, 30 rows) and the
   absolute 1 s-ahead (h = 5) RTT error of S2S-E (prev_seq2seq, mean of the
   seed forecasts). Also with the input mean |dRTT|, for persistence and
   DS2S-E errors, and for the DS2S-E raw 95% width (q0.975 - q0.025).
   Window-level volatility persistence: correlation between the realised
   volatility (mean |dRTT|) of the 6 s input and of the 3 s horizon (14
   changes between the 15 measured target rows).
   95% CIs of the main correlations: bootstrap over TEST sessions.
   Within/between split of every correlation: "within" = correlation of
   within-session percentile ranks (does it hold inside a run, session-bootstrap
   CI), "between" = Spearman of the per-session medians (one point per test session).
3. TEST windows binned by input-SD decile. Decile edges come from VALIDATION
   windows only. Per decile: mean |error| of persistence, S2S-E and DS2S-E
   (median), DS2S-E raw 95% width, and coverage and width of calibrated 95%
   intervals (CQR on DS2S-E, split-conformal |residual| on S2S-E). Conformal
   offsets come from all validation windows at h = 5.
4. Volatility transition matrix: P(horizon-volatility decile | input-volatility
   decile) on TEST windows, both decile edge sets from VALIDATION windows.
   Under no persistence every cell is 0.1.
5. Direction of the 1 s change: share of TEST windows where the predicted change
   from the last input value (persistence) has the sign of the true change, for
   S2S-E, DS2S-E (median) and a no-learning baseline that expects a return to the
   6 s input mean. Balanced version = mean hit rate on up and on down moves (0.5
   for any constant guess). True ties are left out, predicted ties count as misses.
   Session-bootstrap CIs. Nothing is fitted.

Outputs
-------
modeling/results_final/analysis/volatility/
    acf_<link>.csv          every lag and variant, bootstrap CIs, pair counts, shuffle null
    acf_sessions_<link>.csv per-training-session ACF at 0.2 s, 1 s and 5 s
    meta_<link>.json        data facts (sessions, rows, row spacing, interpolated share)
    cache/win_<link>.npz    per-window input and horizon statistics (re-run without reloading data)
    windows_<link>.json     items 2-5 for one link
    acf_summary.csv         ACF at 0.2 s, 1 s and 5 s per link (primary and robustness)
    window_corr.csv         items 2 and 5, one row per link (with the within/between split)
    deciles.csv             item 3, one row per link and decile
    transition.csv          item 4, one row per link and cell
    summary.json            everything above in one file
reports/final_report/data/fr_volatility_acf_<link>.dat     ACF figure data (50 rows, incl. shuffle null dnull/absdnull)
reports/final_report/data/fr_volatility_band_<link>.dat    95% CI bands as closed polygons (100 rows)
(the figure data is rewritten by the summary step, so --summary_only refreshes it)
Figure: reports/final_report/figs/fr_volatility.tex

Usage
-----
LSN_DEVICE=cpu python experiments/fr_volatility.py                 # all links
LSN_DEVICE=cpu python experiments/fr_volatility.py --links starlink_dl
LSN_DEVICE=cpu python experiments/fr_volatility.py --summary_only  # rebuild summaries
The run is deterministic. A link whose ACF CSV and window cache exist is not
reloaded unless --recompute is given (items 2-5 are always recomputed from the
cache and the saved predictions; a cache written before the input mean was
stored skips the return-to-mean baseline with a note). The summary step re-reads every per-link
file present, so links can be (re)run one at a time. Missing prediction files
(a method still training) are skipped with a note.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LSN_DEVICE", "cpu")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))

LINKS = ["oneweb_dl", "oneweb_ul", "starlink_dl", "starlink_ul", "quectel_dl", "quectel_ul"]
L, H = 30, 15
ROW_S = 0.2                                     # nominal row spacing (5 Hz)
REPORT_LAGS = {"0p2s": 1, "1s": 5, "5s": 25}    # lag in rows
SERIES = ["rtt", "d", "absd"]


# ---------------------------------------------------------------------------
# ACF with session boundaries
# ---------------------------------------------------------------------------

def acf_stats(x, valid, sid, n_ses, max_lag, center):
    """Per-session sufficient statistics of the ACF of x at lags 1..max_lag.

    Only pairs (t, t+k) with both rows valid and in the same session count.
    center = "within": every session centred on its own mean (over valid rows)
    center = "pooled": one global mean.
    Returns S (n_ses, max_lag) sums of products, N (n_ses, max_lag) pair counts,
    Q (n_ses,) sums of squares, C (n_ses,) valid-row counts.
    """
    vf = valid.astype(np.float64)
    x = np.where(valid, x, 0.0).astype(np.float64)
    C = np.bincount(sid, weights=vf, minlength=n_ses)
    if center == "within":
        mu = np.bincount(sid, weights=x, minlength=n_ses) / np.maximum(C, 1)
        c = (x - mu[sid]) * vf
    else:
        c = (x - x[valid].mean()) * vf
    Q = np.bincount(sid, weights=c * c, minlength=n_ses)
    S = np.zeros((n_ses, max_lag)); N = np.zeros((n_ses, max_lag))
    for k in range(1, max_lag + 1):
        m = valid[:-k] & valid[k:] & (sid[:-k] == sid[k:])
        S[:, k - 1] = np.bincount(sid[:-k], weights=c[:-k] * c[k:] * m, minlength=n_ses)
        N[:, k - 1] = np.bincount(sid[:-k], weights=m.astype(np.float64), minlength=n_ses)
    return S, N, Q, C


def acf_from_stats(S, N, Q, C, w=None):
    """ACF from (optionally weighted) per-session statistics. w: (B, n_ses) or None."""
    if w is None:
        return (S.sum(0) / N.sum(0)) / (Q.sum() / C.sum())
    return ((w @ S) / (w @ N)) / ((w @ Q) / (w @ C))[:, None]


def series_rows(r, ok, sid):
    """Level, change and change magnitude with their validity masks.

    d[t] = r[t] - r[t-1] is valid only if rows t-1 and t are valid and in the same session.
    """
    same = np.r_[False, sid[1:] == sid[:-1]]
    d = np.r_[0.0, np.diff(r)]
    dv = same & ok & np.r_[False, ok[:-1]]
    return {"rtt": (r, ok), "d": (d, dv), "absd": (np.abs(d), dv)}


def to_seconds(ts):
    """SplitData.ts is datetime int64 / 1e9. The int64 unit depends on the pandas
    datetime resolution (ns -> s, us -> ks). Rescale to unix seconds."""
    t = ts[np.isfinite(ts) & (ts > 0)]
    if not len(t):
        return ts
    p = np.round(np.log10(1.7e9 / np.median(t)) / 3) * 3          # nearest power of 1000
    return ts * 10.0 ** p


# ---------------------------------------------------------------------------
# Stage A: load one link, ACF on training sessions, per-window statistics
# ---------------------------------------------------------------------------

def window_stats(split, ridx, tmean, tstd):
    """RTT (ms, model-input view) of every input window: SD, mean |dRTT|, session id, mean."""
    st = split.starts.numpy()
    xr = tmean + tstd * split.X[:, ridx].numpy().astype(np.float64)
    w = xr[st[:, None] + np.arange(L)[None]]
    return w.std(1, ddof=1), np.abs(np.diff(w, axis=1)).mean(1), split.sid[st + L - 1], w.mean(1)


def stage_a(ds, a, out):
    import torch
    torch.set_num_threads(2)
    import final_benchmark as B

    prov, dirn = ds.split("_")
    data, meta = B.load_splits(prov, dirn, "base", L, H, torch.device("cpu"), a.data_root)
    ridx = meta["target_idx"][0]
    tmean, tstd = float(meta["tmean"][0]), float(meta["tstd"][0])

    trn = data["train"]
    sid = trn.sid.astype(np.int64)
    n_ses = int(sid.max()) + 1
    r_meas = tmean + tstd * trn.Y[:, 0].numpy().astype(np.float64)       # measured target RTT
    r_in = tmean + tstd * trn.X[:, ridx].numpy().astype(np.float64)      # model-input RTT
    interp = trn.M[:, 0].numpy() > 0.5
    ts = to_seconds(trn.ts)
    dt = np.diff(ts)[sid[1:] == sid[:-1]]
    dt = dt[np.isfinite(dt) & (dt > 0)]

    # Shuffle null: inside every session the real pings are permuted among the real-ping rows.
    # Same values (same spikes, same marginal distribution), no time order. Its ACF at lag 1 is
    # the purely mechanical part (one spike makes two consecutive large changes), at lags >= 2 it is 0.
    rng_sh = np.random.default_rng(a.seed + 1)
    r_shuf = r_meas.copy()
    for j in range(n_ses):
        ii = np.flatnonzero((sid == j) & ~interp)
        r_shuf[ii] = r_shuf[rng_sh.permutation(ii)]
    views = {"measured": series_rows(r_meas, ~interp, sid),
             "input": series_rows(r_in, np.ones_like(interp), sid),
             "shuffle": series_rows(r_shuf, ~interp, sid)}
    rng = np.random.default_rng(a.seed)
    W = rng.multinomial(n_ses, np.full(n_ses, 1.0 / n_ses), size=a.n_boot).astype(np.float64)
    lags = np.arange(1, a.max_lag + 1)
    acf = pd.DataFrame({"lag_rows": lags, "lag_s": lags * ROW_S})
    ses_rows = []
    for vname, sers in views.items():
        for center in ("within", "pooled"):
            if vname == "shuffle" and center == "pooled":
                continue
            for s in SERIES:
                x, v = sers[s]
                S, N, Q, C = acf_stats(x, v, sid, n_ses, a.max_lag, center)
                col = f"{s}_{vname}_{center}"
                acf[col] = acf_from_stats(S, N, Q, C)
                if vname == "measured" and center == "within":
                    bt = acf_from_stats(S, N, Q, C, W)
                    acf[col + "_lo"] = np.nanquantile(bt, 0.025, axis=0)
                    acf[col + "_hi"] = np.nanquantile(bt, 0.975, axis=0)
                    acf[f"npairs_{s}_measured"] = N.sum(0).astype(np.int64)
                    # per-session ACF (each session on its own) at the reported lags
                    with np.errstate(invalid="ignore", divide="ignore"):
                        per = (S / N) / (Q / C)[:, None]
                    for tag, k in REPORT_LAGS.items():
                        for j in range(n_ses):
                            ses_rows.append({"link": ds, "session": j, "n_rows": int(C[j]), "series": s,
                                             "lag": tag, "acf": float(per[j, k - 1])})
    acf.to_csv(out / f"acf_{ds}.csv", index=False)
    pd.DataFrame(ses_rows).to_csv(out / f"acf_sessions_{ds}.csv", index=False)

    meta_out = {"link": ds, "n_train_sessions": n_ses, "n_train_rows": int(len(sid)),
                "n_test_sessions": int(meta["n_sessions"]["test"]), "n_val_sessions": int(meta["n_sessions"]["val"]),
                "train_interp_share": float(interp.mean()),
                "row_dt_median_s": float(np.median(dt)), "row_dt_p90_s": float(np.quantile(dt, 0.9)),
                "row_dt_share_over_0p5s": float((dt > 0.5).mean()),
                "rtt_train_mean_ms": tmean, "rtt_train_sd_ms": tstd,
                "n_pairs_d_lag1_measured": int(acf["npairs_d_measured"].iloc[0]),
                "n_boot": a.n_boot, "boot_seed": a.seed}
    json.dump(meta_out, open(out / f"meta_{ds}.json", "w"), indent=1)

    sd_te, mad_te, sid_te, mu_te = window_stats(data["test"], ridx, tmean, tstd)
    sd_va, mad_va, _, mu_va = window_stats(data["val"], ridx, tmean, tstd)
    (out / "cache").mkdir(exist_ok=True)
    np.savez_compressed(out / "cache" / f"win_{ds}.npz", sd_te=sd_te.astype(np.float32),
                        mad_te=mad_te.astype(np.float32), sid_te=sid_te.astype(np.int32),
                        sd_va=sd_va.astype(np.float32), mad_va=mad_va.astype(np.float32),
                        mu_te=mu_te.astype(np.float32), mu_va=mu_va.astype(np.float32))
    print(f"[{ds}] sessions={n_ses} rows={len(sid)} interp={interp.mean():.3f} dt_med={np.median(dt):.3f}s "
          f"ACF(|d|) 1s={acf['absd_measured_within'].iloc[4]:.3f} ACF(d) 1s={acf['d_measured_within'].iloc[4]:.3f}",
          flush=True)


# ---------------------------------------------------------------------------
# Stage B: test windows conditioned on input volatility (cache + saved predictions)
# ---------------------------------------------------------------------------

def order_stat(s, alpha):
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def _boot_pearson(x, y, inv, W):
    """Pearson correlation for every row of group weights W (B, G), from per-group sums."""
    G = W.shape[1]
    s = [np.bincount(inv, weights=v, minlength=G) for v in (np.ones_like(x), x, y, x * x, y * y, x * y)]
    n, sx, sy, sxx, syy, sxy = (W @ v for v in s)
    cov = sxy / n - (sx / n) * (sy / n)
    return cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))


def corr_ci(a_, b_, groups, n_boot, seed):
    """Pearson and Spearman with 95% CIs from a bootstrap over groups (test sessions),
    plus the Spearman split into a within-session and a between-session part.

    Windows of one session overlap and are strongly dependent, so whole sessions
    are resampled. The bootstrap Pearson is exact (per-session sums). The bootstrap
    Spearman is the Pearson of the full-sample ranks of the resample (a close and
    standard approximation that avoids re-ranking ~10^5 windows per draw).

    within  = Pearson correlation of within-session percentile ranks (the windows of
              every session ranked on their own): does the association hold INSIDE a
              measurement run? Its session-bootstrap CI is exact (ranks do not change
              when sessions are resampled).
    between = Spearman correlation of the per-session medians (one point per test
              session): do runs with a larger a also have a larger b? CI: bootstrap
              of the session medians (separate generator, so the CIs above do not change).
    A large pooled value with within ~ 0 means the input identifies the run, not a
    burst inside the run.
    """
    p = float(pearsonr(a_, b_)[0]); s = float(spearmanr(a_, b_).correlation)
    ug, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(ug), np.full(len(ug), 1.0 / len(ug)), size=n_boot).astype(np.float64)
    bp = _boot_pearson(a_.astype(np.float64), b_.astype(np.float64), inv, W)
    bs = _boot_pearson(rankdata(a_), rankdata(b_), inv, W)
    q = lambda v: ([float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
                   if np.isfinite(v).any() else [float("nan"), float("nan")])
    df = pd.DataFrame({"a": a_, "b": b_, "g": inv})
    ra = df.groupby("g").a.rank(pct=True).to_numpy(np.float64)
    rb = df.groupby("g").b.rank(pct=True).to_numpy(np.float64)
    wi = float(np.corrcoef(ra, rb)[0, 1])
    bw = _boot_pearson(ra, rb, inv, W)
    med = df.groupby("g")[["a", "b"]].median()
    ma, mb = med.a.to_numpy(np.float64), med.b.to_numpy(np.float64)
    if np.ptp(ma) == 0 or np.ptp(mb) == 0:
        # e.g. the true change: its session median is 0 in every session, so no between part exists
        be, bb = float("nan"), np.full(n_boot, np.nan)
    else:
        be = float(spearmanr(ma, mb).correlation)
        idx = np.random.default_rng(seed + 7).integers(0, len(ma), size=(n_boot, len(ma)))
        ka, kb = rankdata(ma[idx], axis=1), rankdata(mb[idx], axis=1)
        ka -= ka.mean(1, keepdims=True); kb -= kb.mean(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            bb = (ka * kb).sum(1) / np.sqrt((ka * ka).sum(1) * (kb * kb).sum(1))
    return {"pearson": p, "spearman": s, "pearson_ci": q(bp), "spearman_ci": q(bs),
            "within": wi, "within_ci": q(bw), "between": be, "between_ci": q(bb),
            "n": int(len(a_)), "n_groups": int(len(ug)), "n_boot": int(n_boot)}


def sign_ci(pc, tc, groups, n_boot, seed):
    """Direction of the change from the last input value: share of windows where the
    predicted change pc has the sign of the true change tc, and the balanced share
    (mean of the hit rates on up and on down moves, 0.5 for any constant guess).
    True ties (|tc| < 1e-6 ms) are left out, predicted ties count as misses.
    95% CI from a bootstrap over test sessions (exact, per-session counts)."""
    m = np.abs(tc) > 1e-6
    hit = m & (np.sign(pc) == np.sign(tc))
    up, dn = m & (tc > 0), m & (tc < 0)
    ug, inv = np.unique(groups, return_inverse=True)
    G = len(ug)
    W = np.random.default_rng(seed).multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(np.float64)
    cnt = lambda v: np.bincount(inv, weights=v.astype(np.float64), minlength=G)
    h, n, hu, nu, hd, nd = (cnt(v) for v in (hit, m, hit & up, up, hit & dn, dn))
    q = lambda v: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
    with np.errstate(invalid="ignore", divide="ignore"):
        b_acc = (W @ h) / (W @ n)
        b_bal = 0.5 * ((W @ hu) / (W @ nu) + (W @ hd) / (W @ nd))
    return {"acc": float(h.sum() / n.sum()), "acc_ci": q(b_acc),
            "balanced": float(0.5 * (hu.sum() / nu.sum() + hd.sum() / nd.sum())), "balanced_ci": q(b_bal),
            "share_down": float(nd.sum() / n.sum()), "share_tie": float(1 - m.mean())}


def decile_bins(ref, x, n_bins):
    """Decile index of x with edges from ref (validation windows)."""
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    return np.digitize(x, edges[1:-1]), edges


def stage_b(ds, a, out):
    from final_extras import load_ens, cqr_offsets

    z = np.load(out / "cache" / f"win_{ds}.npz")
    sd_te, mad_te, sid_te = z["sd_te"].astype(np.float64), z["mad_te"].astype(np.float64), z["sid_te"]
    sd_va, mad_va = z["sd_va"].astype(np.float64), z["mad_va"].astype(np.float64)
    pdir = Path(a.res_dir) / ds / "preds"
    tr = np.load(pdir / "truth.npz")
    y_te, y_va = tr["y_te"], tr["y_va"]
    assert len(sd_te) == len(y_te) and len(sd_va) == len(y_va), ds
    hi_ = a.h - 1
    y = y_te[:, hi_, 0].astype(np.float64)
    nb = a.n_bins

    res = {"link": ds, "h": a.h, "ref_method": a.ref_method, "fr_method": a.fr_method,
           "n_test_windows": int(len(y)), "n_val_windows": int(len(y_va)),
           "n_test_sessions": int(len(np.unique(sid_te))),
           "input_sd_test_median_ms": float(np.median(sd_te)), "notes": []}
    cc = lambda u, v: corr_ci(u, v, sid_te, a.n_boot_corr, a.seed)
    last = tr["pers_te"][:, hi_, 0].astype(np.float64)      # last input value (model-input view)
    dy = y - last                                           # true change over the next h steps

    def put_sign(name, pred):
        r = sign_ci(pred - last, dy, sid_te, a.n_boot_corr, a.seed)
        res[f"sign_{name}_acc"], res[f"sign_{name}_balanced"] = r["acc"], r["balanced"]
        res[f"sign_{name}_acc_ci_lo"], res[f"sign_{name}_acc_ci_hi"] = r["acc_ci"]
        res[f"sign_{name}_balanced_ci_lo"], res[f"sign_{name}_balanced_ci_hi"] = r["balanced_ci"]
        res["share_true_change_down"], res["share_true_change_tie"] = r["share_down"], r["share_tie"]

    if "mu_te" in z.files:   # no-learning baseline: expect a return to the 6 s input mean
        put_sign("revert_to_input_mean", z["mu_te"].astype(np.float64))
    else:
        res["notes"].append("cache has no input mean (rerun with --recompute): return-to-mean baseline skipped")

    # window-level volatility persistence (realised volatility: mean |dRTT|)
    rv_out_te = np.abs(np.diff(y_te[:, :, 0].astype(np.float64), axis=1)).mean(1)
    rv_out_va = np.abs(np.diff(y_va[:, :, 0].astype(np.float64), axis=1)).mean(1)
    res["corr_input_rv_horizon_rv"] = cc(mad_te, rv_out_te)
    res["corr_input_sd_horizon_rv"] = cc(sd_te, rv_out_te)

    errs = {"persistence": np.abs(y - tr["pers_te"][:, hi_, 0])}
    res["corr_inputsd_abs_err_persistence"] = cc(sd_te, errs["persistence"])
    s2 = load_ens(pdir, a.ref_method)
    d3 = load_ens(pdir, a.fr_method)
    if s2 is None:
        res["notes"].append(f"{a.ref_method} predictions missing")
    else:
        res["ref_n_seeds"] = s2["n"]
        errs["s2s_e"] = np.abs(y - s2["p_te"][:, hi_, 0])
        res["corr_inputsd_abs_err_s2s_e"] = cc(sd_te, errs["s2s_e"])
        res["corr_inputrv_abs_err_s2s_e"] = cc(mad_te, errs["s2s_e"])
        q = order_stat(np.abs(y_va[:, hi_, 0] - s2["p_va"][:, hi_, 0]), 0.05)
        s2lo, s2hi = s2["p_te"][:, hi_, 0] - q, s2["p_te"][:, hi_, 0] + q
        res["s2s_e_conformal_halfwidth_ms"] = q
        res["s2s_e_mae_ms"] = float(errs["s2s_e"].mean())
        put_sign("s2s_e", s2["p_te"][:, hi_, 0].astype(np.float64))
        res["corr_predchange_truechange_s2s_e"] = cc(s2["p_te"][:, hi_, 0] - last, dy)
    has_d = d3 is not None and "d_te" in d3
    if not has_d:
        res["notes"].append(f"{a.fr_method} distributional predictions missing")
    else:
        res["fr_n_seeds"] = d3["n"]
        dq = d3["d_te"][:, hi_, 0, :].astype(np.float64)
        errs["ds2s_e"] = np.abs(y - dq[:, 3])
        wraw = dq[:, 5] - dq[:, 1]
        c = float(cqr_offsets(d3["d_va"], y_va)[hi_, 0])
        dlo, dhi = dq[:, 1] - c, dq[:, 5] + c
        res["ds2s_e_cqr_offset_ms"] = c
        res["ds2s_e_mae_ms"] = float(errs["ds2s_e"].mean())
        res["corr_inputsd_abs_err_ds2s_e"] = cc(sd_te, errs["ds2s_e"])
        res["corr_inputsd_rawwidth_ds2s_e"] = cc(sd_te, wraw)
        res["corr_inputrv_rawwidth_ds2s_e"] = cc(mad_te, wraw)
        res["corr_rawwidth_abs_err_ds2s_e"] = cc(wraw, errs["ds2s_e"])
        res["corr_rawwidth_horizon_rv_ds2s_e"] = cc(wraw, rv_out_te)
        put_sign("ds2s_e", dq[:, 3])
        res["corr_predchange_truechange_ds2s_e"] = cc(dq[:, 3] - last, dy)

    # deciles of input SD (edges from validation windows)
    b, edges = decile_bins(sd_va, sd_te, nb)
    rows = []
    for j in range(nb):
        m = b == j
        row = {"link": ds, "decile": j + 1, "edge_lo_ms": float(edges[j]), "edge_hi_ms": float(edges[j + 1]),
               "n_test": int(m.sum()), "test_share": float(m.mean()),
               "input_sd_mean_ms": float(sd_te[m].mean()) if m.any() else np.nan,
               "horizon_rv_mean_ms": float(rv_out_te[m].mean()) if m.any() else np.nan}
        for k_, e in errs.items():
            row[f"mae_{k_}_ms"] = float(e[m].mean()) if m.any() else np.nan
        if s2 is not None and m.any():
            row["cov95_s2s_e_conformal"] = float(((y[m] >= s2lo[m]) & (y[m] <= s2hi[m])).mean())
            row["width95_s2s_e_conformal_ms"] = float((s2hi - s2lo)[m].mean())
        if has_d and m.any():
            row["rawwidth95_ds2s_e_ms"] = float(wraw[m].mean())
            row["cov95_ds2s_e_cqr"] = float(((y[m] >= dlo[m]) & (y[m] <= dhi[m])).mean())
            row["width95_ds2s_e_cqr_ms"] = float((dhi - dlo)[m].mean())
        rows.append(row)
    res["deciles"] = rows
    top, bot = b == nb - 1, b == 0
    if s2 is not None:
        e = errs["s2s_e"]
        res["s2s_e_mae_top_over_bottom_decile"] = float(e[top].mean() / e[bot].mean())
    res["persistence_mae_top_over_bottom_decile"] = float(errs["persistence"][top].mean() / errs["persistence"][bot].mean())
    if has_d:
        res["ds2s_e_rawwidth_top_over_bottom_decile"] = float(wraw[top].mean() / wraw[bot].mean())

    # volatility transition matrix (input RV decile -> horizon RV decile)
    bi, ei = decile_bins(mad_va, mad_te, nb)
    bo, eo = decile_bins(rv_out_va, rv_out_te, nb)
    T = np.zeros((nb, nb)); np.add.at(T, (bi, bo), 1)
    P = T / np.maximum(T.sum(1, keepdims=True), 1)
    res["transition"] = {"P": P.tolist(), "counts": T.astype(int).tolist(),
                         "input_rv_edges_ms": ei.tolist(), "horizon_rv_edges_ms": eo.tolist(),
                         "diag_mean": float(np.mean(np.diag(P))),
                         "p_calm_to_calm": float(P[0, 0]), "p_volatile_to_volatile": float(P[-1, -1]),
                         "p_calm_to_volatile": float(P[0, -1]), "p_volatile_to_calm": float(P[-1, 0]),
                         "p_within_one_decile": float(np.mean([P[i, max(0, i - 1):i + 2].sum() for i in range(nb)])),
                         "test_share_per_input_decile": (T.sum(1) / T.sum()).tolist(),
                         "test_share_per_horizon_decile": (T.sum(0) / T.sum()).tolist()}
    json.dump(res, open(out / f"windows_{ds}.json", "w"), indent=1)


def fig_data(ds, out, fig_dir):
    """Small text tables for pgfplots (read from the per-link results)."""
    acf = pd.read_csv(out / f"acf_{ds}.csv")
    fig = acf[["lag_s"]].copy()
    for s in SERIES:
        base = f"{s}_measured_within"
        fig[s] = acf[base]; fig[s + "lo"] = acf[base + "_lo"]; fig[s + "hi"] = acf[base + "_hi"]
    for s in ("d", "absd"):   # memoryless (shuffled pings) null, drawn at 0.2 s in the figure
        fig[s + "null"] = acf[f"{s}_shuffle_within"]
    fig.to_csv(fig_dir / f"fr_volatility_acf_{ds}.dat", sep=" ", index=False, float_format="%.4f")
    # 95% CI bands as closed polygons (upper bound left to right, lower bound back)
    band = pd.DataFrame({"x": np.r_[fig.lag_s.values, fig.lag_s.values[::-1]]})
    for s in SERIES:
        band[s] = np.r_[fig[s + "hi"].values, fig[s + "lo"].values[::-1]]
    band.to_csv(fig_dir / f"fr_volatility_band_{ds}.dat", sep=" ", index=False, float_format="%.4f")


# ---------------------------------------------------------------------------
# Summary across links
# ---------------------------------------------------------------------------

def summarise(out, fig_dir):
    acf_rows, corr_rows, dec_rows, tr_rows, full = [], [], [], [], {}
    for ds in LINKS:
        f = out / f"acf_{ds}.csv"
        if not f.exists():
            continue
        fig_data(ds, out, fig_dir)
        acf = pd.read_csv(f)
        meta = json.load(open(out / f"meta_{ds}.json"))
        row = {"link": ds, **{k: v for k, v in meta.items() if k != "link"}}
        for tag, k in REPORT_LAGS.items():
            r = acf[acf.lag_rows == k].iloc[0]
            for s in SERIES:
                for v in ("measured_within", "measured_pooled", "input_within", "input_pooled"):
                    row[f"{s}_{tag}_{v}"] = float(r[f"{s}_{v}"])
                row[f"{s}_{tag}_ci_lo"] = float(r[f"{s}_measured_within_lo"])
                row[f"{s}_{tag}_ci_hi"] = float(r[f"{s}_measured_within_hi"])
        for s in ("d", "absd"):
            for tag, k in REPORT_LAGS.items():
                row[f"{s}_{tag}_shuffle_null"] = float(acf[f"{s}_shuffle_within"].iloc[k - 1])
        sf = out / f"acf_sessions_{ds}.csv"
        if sf.exists():
            ses = pd.read_csv(sf)
            for s in SERIES:
                for tag in REPORT_LAGS:
                    v = ses[(ses.series == s) & (ses.lag == tag)].acf.dropna()
                    row[f"{s}_{tag}_sessions_median"] = float(v.median())
                    row[f"{s}_{tag}_sessions_min"] = float(v.min())
                    row[f"{s}_{tag}_sessions_max"] = float(v.max())
                    row[f"{s}_{tag}_sessions_share_pos"] = float((v > 0).mean())
        row["d_max_abs_lag0p4to10s"] = float(acf.d_measured_within.iloc[1:].abs().max())
        pos = acf.absd_measured_within_lo > 0
        row["absd_last_lag_s_ci_above_0"] = float(acf.lag_s[pos].max()) if pos.any() else 0.0
        first_out = np.flatnonzero(~pos.values)
        row["absd_first_lag_s_ci_includes_0"] = float(acf.lag_s.iloc[first_out[0]]) if len(first_out) else np.nan
        row["absd_over_absd_lag_ratio_1s"] = float(acf.absd_measured_within.iloc[4] / max(abs(acf.d_measured_within.iloc[4]), 1e-9))
        acf_rows.append(row)
        full[ds] = {"meta": meta, "acf_report_lags": row}
        wf = out / f"windows_{ds}.json"
        if wf.exists():
            w = json.load(open(wf))
            full[ds]["windows"] = w
            c = {"link": ds, "ref_method": w.get("ref_method"), "fr_method": w.get("fr_method"),
                 "n_test_windows": w["n_test_windows"], "n_test_sessions": w["n_test_sessions"]}
            for key, val in w.items():
                if key.startswith("corr_"):
                    c[key + "_pearson"] = val["pearson"]; c[key + "_spearman"] = val["spearman"]
                    c[key + "_spearman_ci_lo"], c[key + "_spearman_ci_hi"] = val["spearman_ci"]
                    if "within" in val:
                        c[key + "_within"] = val["within"]
                        c[key + "_within_ci_lo"], c[key + "_within_ci_hi"] = val["within_ci"]
                        c[key + "_between"] = val["between"]
                        c[key + "_between_ci_lo"], c[key + "_between_ci_hi"] = val["between_ci"]
                elif isinstance(val, (int, float)) and key not in ("h",):
                    c[key] = val
            corr_rows.append(c)
            dec_rows += w["deciles"]
            t = w["transition"]
            P = np.array(t["P"])
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    tr_rows.append({"link": ds, "input_rv_decile": i + 1, "horizon_rv_decile": j + 1,
                                    "p": P[i, j], "count": t["counts"][i][j]})
    pd.DataFrame(acf_rows).to_csv(out / "acf_summary.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out / "window_corr.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(out / "deciles.csv", index=False)
    pd.DataFrame(tr_rows).to_csv(out / "transition.csv", index=False)
    json.dump(full, open(out / "summary.json", "w"), indent=1)
    if acf_rows:
        cols = ["link"] + [f"{s}_{t}_measured_within" for t in REPORT_LAGS for s in ("d", "absd", "rtt")]
        print(pd.DataFrame(acf_rows)[cols].round(3).to_string(index=False))
    if corr_rows:
        print(pd.DataFrame(corr_rows).round(3).T.to_string())


def main():
    ap = argparse.ArgumentParser(description="RTT volatility clustering analysis (see module docstring)")
    ap.add_argument("--links", nargs="+", default=LINKS)
    ap.add_argument("--fr_method", default="fr_dist", help="distributional model (DS2S) whose quantiles are used")
    ap.add_argument("--ref_method", default="prev_seq2seq", help="point model (S2S) whose error is correlated")
    ap.add_argument("--h", type=int, default=5, help="horizon step (5 = 1 s)")
    ap.add_argument("--max_lag", type=int, default=50)
    ap.add_argument("--n_boot", type=int, default=2000, help="session bootstrap draws for the ACF CIs")
    ap.add_argument("--n_boot_corr", type=int, default=2000, help="test-session bootstrap draws for correlation CIs")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true", help="reload data even if the cache exists")
    ap.add_argument("--summary_only", action="store_true")
    ap.add_argument("--data_root", default=str(ROOT / "data/lsn-data"))
    ap.add_argument("--res_dir", default=str(ROOT / "results_final/main"))
    ap.add_argument("--out_dir", default=str(ROOT / "results_final/analysis/volatility"))
    ap.add_argument("--fig_dir", default=str(ROOT.parent / "reports/final_report/data"))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    Path(a.fig_dir).mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        for ds in a.links:
            if a.recompute or not ((out / f"acf_{ds}.csv").exists() and (out / "cache" / f"win_{ds}.npz").exists()):
                stage_a(ds, a, out)
            stage_b(ds, a, out)
    summarise(out, Path(a.fig_dir))


if __name__ == "__main__":
    main()
