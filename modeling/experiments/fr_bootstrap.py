"""
experiments/fr_bootstrap.py - How sure are the RTT improvements? A session-level bootstrap.

Why
---
The final model (DS2S-E, 3-seed ensemble of --fr_method, default fr_dist) beats the model of the
previous report (S2S-E, 3-seed ensemble of prev_seq2seq) on the pooled test windows. Consecutive test
windows of one session share 29 of their 30 input rows and the same link conditions, so they are not
independent samples: the independent unit is the test SESSION (9 to 11 per satellite link, 58 per 5G
link). A gain carried by one or two favourable sessions looks solid when windows are counted and
disappears when sessions are counted. The bootstrap below resamples whole sessions, so its interval
says how much the gain could move on another draw of measurement sessions from the same links.

What it computes (RTT only, target 0, ms)
-----------------------------------------
For every test window i and step h = 1..15 (0.2 s to 3 s ahead):
  S2S-E   point forecast = ensemble mean (used for MAE and RMSE)
          95% interval = mean -/+ q_h, q_h = split-conformal quantile of |y - mean| on validation windows
  DS2S-E  MAE uses the ensemble median, RMSE the ensemble mean
          95% interval = [q.025 - c_h, q.975 + c_h], c_h = CQR offset of the score max(q_lo - y, y - q_hi)
          on validation windows (final_extras.cqr_offsets)
  interval score IS = (hi - lo) + (2/a)(lo - y)+ + (2/a)(y - hi)+, a = 0.05 (lower is better)
Window values: |e| and IS averaged over the 15 steps, the squared error kept per step. Their sums per
session are sufficient statistics. A replicate draws the S test sessions with replacement (w_s = number
of times session s is drawn) and evaluates, on exactly the windows of the drawn sessions,
  MAE  = sum_s w_s A_s / sum_s w_s n_s            (A_s: sum of window |e| in s, n_s: windows in s)
  IS   = sum_s w_s I_s / sum_s w_s n_s
  RMSE = mean_h sqrt(sum_s w_s Q_sh / sum_s w_s n_s)   (per-step RMSE averaged over the steps, the
                                                         convention of the report tables)
and the relative change 100 (DS2S-E - S2S-E) / S2S-E, with the same draw for both models (paired).
B = 2000 replicates, generator seed 0, re-seeded for every link so that a link's interval does not
depend on which other links or methods are run.

Reported per link and metric: the full-test-set change (equal to the report tables), the bootstrap
median and 2.5 / 97.5 percentiles (percentile interval), the fraction of replicates in which DS2S-E is
not better, the number of sessions in which it is better, the range of the change when one session is
left out, and, for contrast, the percentile interval of a naive bootstrap that resamples windows as if
they were independent (seed 1).

No test leakage: q_h and c_h are learned on validation windows (the calibration sessions) only and are
held fixed in every replicate, as in deployment. The intervals are therefore conditional on the trained
models and on the calibration offsets.

Other methods
-------------
--extra (default fr_dist_w4 fr_gauss): the same comparison for other distributional variants. They are
added to the figure and get their own table body when all seeds are present.
--ablation (default fr_resid fr_full lqr_fr ridge_fr): the same comparison against S2S-E, CSV only.
Methods without quantile outputs use their mean for MAE and a split-conformal interval. Missing or
incomplete methods (for example still training) are skipped with a note, so the script can simply be
re-run when they finish.

Outputs
-------
modeling/results_final/analysis/bootstrap/
  bootstrap_summary.csv   one row per (link, method, metric): values of both models, full-set change,
                          bootstrap median / percentiles, naive window interval, leave-one-out range
  bootstrap_summary.json  the same rows plus settings, calibration offsets and coverage per link
  session_stats.csv       per (link, method, test session): n_s, A_s, I_s, Q_s1..Q_s15 (enough to
                          recompute every replicate), session start time in unix seconds
  is_decomposition.csv    where the interval-score gain comes from (mechanism_checks): five 95%
                          intervals from the S2S-E one to the DS2S-E one, one ingredient at a time
                          (asymmetric band, median centre, input-dependent width); IS, PICP, width
  point_and_jumps.csv     MAE of the DS2S-E median and of its mean against S2S-E, and coverage / width
                          of both calibrated intervals on calm steps and on large jumps (|y - last
                          input| below the validation median / above the validation 99th percentile)
  main_vs_others.csv      the final model against every other available method (ablations, linear
                          quantile model, variants) with the same paired session draws: change of
                          the final model in % (negative = final model better), percentile interval
  replicates_<link>.npz   the B replicate changes (B x 3: MAE, RMSE, IS) for every method
  sid_<link>.npz          cached test-session id and start time (unix s) per test window (load_splits)
  log.txt
reports/final_report/data/
  bootstrap_table.tex             table body for --fr_method: Link & MAE & RMSE & IS, each cell
                                  "median [lo, hi]" in %, bold when the interval lies below 0
                                  (significant gain), italic when it lies above 0 (significant loss)
  bootstrap_table_<method>.tex    the same for each available --extra method
  bootstrap_forest_<method>.dat   forest-plot data (one file per plotted method)
reports/final_report/figs/fr_bootstrap.tex
                                  forest plot, generated here (edit this script, not the .tex)

Usage
-----
cd modeling && LSN_DEVICE=cpu python experiments/fr_bootstrap.py [--fr_method fr_dist] [--links ...]
  --report_only   rewrite the table bodies, forest data and figure from bootstrap_summary.csv
The first run reads the test-session ids with load_splits (about 1 min per 5G link) and caches them.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LSN_DEVICE", "cpu")              # analysis only: never touch the GPU / MPS
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")                      # keep CPU use moderate (training runs alongside)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # modeling/
REPO = ROOT.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from final_extras import cqr_offsets  # noqa: E402  (same CQR offsets as the report)

LINKS = ["oneweb_dl", "oneweb_ul", "starlink_dl", "starlink_ul", "quectel_dl", "quectel_ul"]
NICE = {"oneweb_dl": "OneWeb DL", "oneweb_ul": "OneWeb UL", "starlink_dl": "Starlink DL",
        "starlink_ul": "Starlink UL", "quectel_dl": "5G DL", "quectel_ul": "5G UL"}
LABEL = {"prev_seq2seq": "S2S-E", "fr_dist": "DS2S-E", "fr_dist_w4": "DS2S-E (pinball weight 4)",
         "fr_gauss": "DS2S-E (Gaussian head)", "fr_resid": "residual decoder (3-seed ens.)",
         "fr_full": "+ data pipeline (3-seed ens.)", "lqr_fr": "linear quantile model",
         "ridge_fr": "ridge, extended features"}
# Plot style per method in the forest plot: (color, filled mark, open mark)
STYLE = {"fr_dist": ("cFR", "square*", "square"), "fr_dist_w4": ("cFR!45!black", "diamond*", "diamond"),
         "fr_gauss": ("cResid", "triangle*", "triangle")}
STYLE_FALLBACK = [("cRidge", "pentagon*", "pentagon"), ("cSnaive", "*", "o")]
SINGLE_SEED = {"persistence", "snaive", "ridge", "ridge_fr", "lqr_fr"}   # one run by design in final_benchmark.py
BASE = "prev_seq2seq"
METRICS = ["MAE", "RMSE", "IS"]
L, H, K_RTT = 30, 15, 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_rtt(pdir: Path, method: str, seeds, allow_partial=False):
    """Seed ensemble of one method, RTT channel only. Same arithmetic as final_extras.load_ens
    (mean over seeds of the mean output and of every quantile) but over exactly the requested
    seeds, so a run that is still training cannot change an ensemble silently.
    Returns (dict or None, note)."""
    want = [0] if method in SINGLE_SEED else list(seeds)
    files = [pdir / f"{method}_s{s}.npz" for s in want]
    have = [f for f in files if f.exists()]
    if not have:
        return None, "no prediction files"
    if len(have) < len(files) and not allow_partial:
        return None, f"incomplete: {len(have)}/{len(files)} seeds"
    parts = {"p_te": [], "p_va": [], "d_te": [], "d_va": []}
    try:
        for f in have:
            with np.load(f) as z:
                parts["p_te"].append(np.array(z["p_te"][:, :, K_RTT], np.float32))
                parts["p_va"].append(np.array(z["p_va"][:, :, K_RTT], np.float32))
                if "d_te" in z.files:
                    parts["d_te"].append(z["d_te"][:, :, K_RTT].astype(np.float32))
                    parts["d_va"].append(z["d_va"][:, :, K_RTT].astype(np.float32))
    except Exception as e:  # noqa: BLE001  e.g. a file still being written by a training job
        return None, f"unreadable ({type(e).__name__}: {e})"
    out = {"p_te": np.mean(parts["p_te"], 0), "p_va": np.mean(parts["p_va"], 0),
           "seeds": [int(f.stem.rsplit("_s", 1)[1]) for f in have]}
    if parts["d_te"] and len(parts["d_te"]) == len(have):
        out["d_te"] = np.mean(parts["d_te"], 0)       # (N, H, 6): mean, q.025, q.25, q.5, q.75, q.975
        out["d_va"] = np.mean(parts["d_va"], 0)
    return out, f"{len(have)} seed(s)"


def unix_seconds(ts):
    """load_splits stores ts = datetime64 as int64 / 1e9, which is unix seconds only when pandas parses
    the time stamps at ns resolution. pandas >= 2 may parse at us (or ms, s) resolution, which makes the
    values 1e3 (1e6, 1e9) times too small (e.g. 1756390.9 instead of 1756390913 = 2025-08-28).
    Rescale by powers of 1000 into the plausible range of unix seconds (1e9 to 1e10, years 2001-2286).
    Used for the session start times only (session_stats.csv), never for a metric."""
    ts = np.asarray(ts, np.float64)
    pos = ts[np.isfinite(ts) & (ts > 0)]
    if pos.size == 0:
        return ts
    med, k = float(np.median(pos)), 0
    while med * 1000.0 ** k < 1e9 and k < 3:
        k += 1
    return ts * 1000.0 ** k


def test_sessions(link: str, n_expect: int, data_root: str, cache_dir: Path, refresh=False, log=print):
    """Test-session id and start time (unix s) of every test window, in the order of the saved
    predictions. Read once with final_benchmark.load_splits (base variant) and cached."""
    cache = cache_dir / f"sid_{link}.npz"
    if cache.exists() and not refresh:
        z = np.load(cache)
        if len(z["sid"]) == n_expect:
            sid, ts = z["sid"], z["ts"]
            ts_s = unix_seconds(ts)
            if not np.array_equal(ts_s, ts):                     # cache written before the unit fix
                np.savez(cache, sid=sid, ts=ts_s)
                log(f"  [cache] {cache.name}: time stamps rescaled to unix seconds")
            return sid, ts_s
        log(f"  [cache] {cache.name} has {len(z['sid'])} windows, expected {n_expect}: reloading")
    import torch
    torch.set_num_threads(2)
    import final_benchmark as B
    t0 = time.time()
    prov, dirn = link.split("_")
    data, _ = B.load_splits(prov, dirn, "base", L, H, torch.device("cpu"), data_root)
    te = data["test"]
    st = te.starts.cpu().numpy()
    assert len(st) == n_expect, (link, len(st), n_expect)
    sid = np.asarray(te.sid)[st].astype(np.int32)
    assert (np.asarray(te.sid)[st + L + H - 1] == sid).all(), "a window crosses a session boundary"
    ts = unix_seconds(np.asarray(te.ts)[st])
    np.savez(cache, sid=sid, ts=ts)
    del data, te
    gc.collect()
    log(f"  [sessions] {link}: {len(np.unique(sid))} test sessions, {len(sid)} windows "
        f"({time.time() - t0:.0f}s)")
    return sid, ts


# ---------------------------------------------------------------------------
# Window values, session sums, metrics
# ---------------------------------------------------------------------------

def conformal_level(n, alpha):
    return min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)


def window_values(m, y_te, y_va, alpha):
    """Per-window RTT values of one model (all arrays (N, H) in ms).
    abs: |y - point| averaged over steps (point = median when the model has quantiles, else mean)
    sq:  (y - mean)^2 per step
    is:  interval score of the calibrated 95% interval averaged over steps
    The calibration offsets come from validation windows only."""
    dist = "d_te" in m
    point = m["d_te"][..., 3] if dist else m["p_te"]
    if dist:
        off = cqr_offsets(m["d_va"][:, :, None, :], y_va[:, :, None], alpha)[:, 0]          # (H,)
        lo, hi = m["d_te"][..., 1] - off, m["d_te"][..., 5] + off
        kind = "CQR"
    else:
        off = np.quantile(np.abs(y_va - m["p_va"]), conformal_level(len(y_va), alpha), axis=0, method="higher")
        lo, hi = m["p_te"] - off, m["p_te"] + off
        kind = "split conformal"
    IS = (hi - lo) + (2 / alpha) * np.maximum(lo - y_te, 0) + (2 / alpha) * np.maximum(y_te - hi, 0)
    vals = {"abs": np.abs(y_te - point).mean(1).astype(np.float64),
            "sq": ((y_te - m["p_te"]) ** 2).astype(np.float64),
            "is": IS.mean(1).astype(np.float64)}
    cal = {"kind": kind, "offset_h_ms": [float(v) for v in off],
           "PICP": float(((y_te >= lo) & (y_te <= hi)).mean()), "width_ms": float((hi - lo).mean())}
    return vals, cal


IS_VARIANTS = [  # (key, description) of the interval-score decomposition, from the S2S-E to the DS2S-E interval
    ("s2s_sym", "S2S-E mean -/+ split-conformal |r| (constant width, symmetric; S2S-E interval of the tables)"),
    ("s2s_asym", "S2S-E mean + split-conformal quantiles of the signed residual, a/2 per tail (constant width, asymmetric)"),
    ("ds2s_med_sym", "DS2S-E median -/+ split-conformal |r| (constant width, symmetric)"),
    ("ds2s_med_asym", "DS2S-E median + split-conformal quantiles of the signed residual (constant width, asymmetric)"),
    ("ds2s_cqr", "DS2S-E [q.025 - c, q.975 + c], CQR (input-dependent width, asymmetric; DS2S-E interval of the tables)"),
]


def mechanism_checks(base, fr, y_te, y_va, pers_te, pers_va, alpha):
    """Where the RTT gains of the final model come from (full test set, RTT in ms). Every offset and
    threshold is learned on VALIDATION windows and applied to test windows, per horizon step.
    1) Interval-score decomposition (IS_VARIANTS): change one ingredient at a time. s2s_asym - s2s_sym
       isolates the asymmetry of the band (RTT jumps up far more than down), ds2s_med_* the better
       centre, and ds2s_cqr - ds2s_med_asym the input-dependent width (the band follows the recent
       history of the link).
    2) Point forecast behind the MAE gain: MAE of the DS2S-E mean next to that of its median.
    3) Coverage of the two calibrated intervals on calm steps and on large jumps:
       |y - last input RTT| at most the validation median / above the validation 99th percentile.
    base, fr: RTT ensembles (dicts of load_rtt). pers_*: last input RTT per window, (N, H)."""
    off = cqr_offsets(fr["d_va"][:, :, None, :], y_va[:, :, None], alpha)[:, 0]   # as in window_values
    y_te, y_va = y_te.astype(np.float64), y_va.astype(np.float64)
    n = len(y_va)
    lv_abs, lv_tail = conformal_level(n, alpha), conformal_level(n, alpha / 2)

    def sym(c_te, c_va):
        q = np.quantile(np.abs(y_va - c_va), lv_abs, axis=0, method="higher")
        return c_te - q, c_te + q

    def asym(c_te, c_va):
        r = y_va - c_va
        return (c_te + np.quantile(r, 1 - lv_tail, axis=0, method="lower"),
                c_te + np.quantile(r, lv_tail, axis=0, method="higher"))

    med_te, med_va = fr["d_te"][..., 3].astype(np.float64), fr["d_va"][..., 3].astype(np.float64)
    b_te, b_va = base["p_te"].astype(np.float64), base["p_va"].astype(np.float64)
    bands = {"s2s_sym": sym(b_te, b_va), "s2s_asym": asym(b_te, b_va),
             "ds2s_med_sym": sym(med_te, med_va), "ds2s_med_asym": asym(med_te, med_va),
             "ds2s_cqr": (fr["d_te"][..., 1] - off, fr["d_te"][..., 5] + off)}
    dec = []
    for key, desc in IS_VARIANTS:
        lo, hi = bands[key]
        IS = (hi - lo) + (2 / alpha) * np.maximum(lo - y_te, 0) + (2 / alpha) * np.maximum(y_te - hi, 0)
        dec.append({"variant": key, "IS_ms": float(IS.mean()), "PICP": float(((y_te >= lo) & (y_te <= hi)).mean()),
                    "width_ms": float((hi - lo).mean()), "description": desc})
    for r in dec:
        r["change_vs_s2s_pct"] = float(rel(r["IS_ms"], dec[0]["IS_ms"]))
    pj = {"MAE_s2s_mean_ms": float(np.abs(y_te - b_te).mean()),
          "MAE_ds2s_median_ms": float(np.abs(y_te - med_te).mean()),
          "MAE_ds2s_mean_ms": float(np.abs(y_te - fr["p_te"]).mean())}
    pj["MAE_change_median_pct"] = float(rel(pj["MAE_ds2s_median_ms"], pj["MAE_s2s_mean_ms"]))
    pj["MAE_change_mean_pct"] = float(rel(pj["MAE_ds2s_mean_ms"], pj["MAE_s2s_mean_ms"]))
    if pers_va is not None:
        jv, jt = np.abs(y_va - pers_va), np.abs(y_te - pers_te)
        thr = {"calm": float(np.quantile(jv, 0.5)), "large_jump": float(np.quantile(jv, 0.99))}
        masks = {"calm": jt <= thr["calm"], "large_jump": jt > thr["large_jump"]}
        for sub, mk in masks.items():
            pj[f"{sub}_threshold_ms"] = thr[sub]
            pj[f"{sub}_share"] = float(mk.mean())
            for name, key in (("s2s", "s2s_sym"), ("ds2s", "ds2s_cqr")):
                lo, hi = bands[key]
                pj[f"{sub}_PICP_{name}"] = float(((y_te >= lo) & (y_te <= hi))[mk].mean())
                pj[f"{sub}_width_{name}_ms"] = float((hi - lo)[mk].mean())
    return {"is_decomposition": dec, "point_and_jumps": pj}


def session_sums(vals, inv, S):
    """Sufficient statistics per session: n (S,), A (S,), I (S,), Q (S, H)."""
    n = np.bincount(inv, minlength=S).astype(np.float64)
    A = np.bincount(inv, weights=vals["abs"], minlength=S)
    I_ = np.bincount(inv, weights=vals["is"], minlength=S)
    Q = np.stack([np.bincount(inv, weights=vals["sq"][:, h], minlength=S) for h in range(vals["sq"].shape[1])], 1)
    return {"n": n, "A": A, "I": I_, "Q": Q}


def weighted_metrics(W, st):
    """W: (B, S) session weights. Returns (B, 3) = MAE, RMSE, IS of the windows of the drawn sessions."""
    nn_ = W @ st["n"]
    mae = (W @ st["A"]) / nn_
    rmse = np.sqrt((W @ st["Q"]) / nn_[:, None]).mean(1)
    is_ = (W @ st["I"]) / nn_
    return np.stack([mae, rmse, is_], 1)


def counts_matrix(draws, S):
    """(B, S) draws of indices -> (B, S) counts (how often each index is drawn in each replicate)."""
    B = draws.shape[0]
    return np.bincount((draws + S * np.arange(B)[:, None]).ravel(), minlength=B * S).reshape(B, S).astype(np.float64)


def naive_window_bootstrap(vals_by_m, B, seed, chunk=40):
    """Window-level bootstrap (windows treated as independent), paired across methods.
    Returns {method: (B, 3) metrics}."""
    rng = np.random.default_rng(seed)
    N = len(next(iter(vals_by_m.values()))["abs"])
    out = {m: [] for m in vals_by_m}
    for b0 in range(0, B, chunk):
        b = min(chunk, B - b0)
        C = counts_matrix(rng.integers(0, N, size=(b, N)), N)
        nn_ = C.sum(1)
        for m, v in vals_by_m.items():
            mae = (C @ v["abs"]) / nn_
            rmse = np.sqrt((C @ v["sq"]) / nn_[:, None]).mean(1)
            is_ = (C @ v["is"]) / nn_
            out[m].append(np.stack([mae, rmse, is_], 1))
        del C
    return {m: np.concatenate(v, 0) for m, v in out.items()}


def rel(a, b):
    """Relative change of a against b in %."""
    return 100.0 * (a - b) / b


# ---------------------------------------------------------------------------
# One link
# ---------------------------------------------------------------------------

def run_link(link, a, methods, roles, log):
    pdir = Path(a.res_dir) / link / "preds"
    tr_path = pdir / "truth.npz"
    if not tr_path.exists():
        log(f"[skip] {link}: no truth.npz")
        return None
    with np.load(tr_path) as tr:
        y_te = np.array(tr["y_te"][:, :, K_RTT], np.float32)
        y_va = np.array(tr["y_va"][:, :, K_RTT], np.float32)
        pers_te = np.array(tr["pers_te"][:, :, K_RTT], np.float32)          # last input RTT, repeated over h
    pers_va = None
    if (pdir / "persistence_s0.npz").exists():                              # same quantity on validation
        with np.load(pdir / "persistence_s0.npz") as z:
            pers_va = np.array(z["p_va"][:, :, K_RTT], np.float32)
    N = len(y_te)
    sid, ts = test_sessions(link, N, a.data_root, Path(a.out_dir), a.refresh_sid, log)
    sess, inv = np.unique(sid, return_inverse=True)
    S = len(sess)

    models, notes = {}, {}
    for m in [BASE] + methods:
        d, note = load_rtt(pdir, m, a.seeds, a.allow_partial)
        notes[m] = note
        if d is None:
            log(f"  [skip] {link} {m}: {note}")
            continue
        assert d["p_te"].shape == y_te.shape and d["p_va"].shape == y_va.shape, (link, m, d["p_te"].shape)
        models[m] = d
    if BASE not in models:
        log(f"[skip] {link}: baseline {BASE} missing ({notes[BASE]})")
        return None

    vals, cal, stats = {}, {}, {}
    for m, d in models.items():
        vals[m], cal[m] = window_values(d, y_te, y_va, a.alpha)
        stats[m] = session_sums(vals[m], inv, S)
        cal[m]["seeds"] = d["seeds"]
    mech = None
    if a.fr_method in models and "d_te" in models[a.fr_method]:
        mech = mechanism_checks(models[BASE], models[a.fr_method], y_te, y_va, pers_te, pers_va, a.alpha)
        mech["method"] = a.fr_method
        for v, m in ((mech["is_decomposition"][0], BASE), (mech["is_decomposition"][-1], a.fr_method)):
            assert np.isclose(v["IS_ms"], vals[m]["is"].mean(), rtol=1e-5), (link, v, vals[m]["is"].mean())
    del models
    gc.collect()

    # Session-level bootstrap: one set of draws per link, shared by all methods (paired).
    rng = np.random.default_rng(a.seed)
    W = counts_matrix(rng.integers(0, S, size=(a.B, S)), S)
    boot = {m: weighted_metrics(W, st) for m, st in stats.items()}
    full = {m: weighted_metrics(np.ones((1, S)), st)[0] for m, st in stats.items()}
    per_sess = {m: weighted_metrics(np.eye(S), st) for m, st in stats.items()}          # (S, 3)
    loo = {m: weighted_metrics(1.0 - np.eye(S), st) for m, st in stats.items()} if S > 1 else None
    naive = naive_window_bootstrap(vals, a.naive_B, a.seed + 1) if a.naive_B > 0 else None

    # Sanity: the full-set values equal a direct computation on the windows.
    for m, v in vals.items():
        direct = [v["abs"].mean(), np.sqrt(v["sq"].mean(0)).mean(), v["is"].mean()]
        assert np.allclose(full[m], direct, rtol=1e-9), (link, m, full[m], direct)

    t_start = np.full(S, np.inf)
    np.minimum.at(t_start, inv, ts)
    rows, reps, srows = [], {}, []
    for m in vals:
        st = stats[m]
        for s in range(S):
            r = {"link": link, "method": m, "session": int(sess[s]),
                 "t_start_unix": float(t_start[s]), "n_windows": int(st["n"][s]),
                 "sum_abs": st["A"][s], "sum_is": st["I"][s]}
            r.update({f"sum_sq_h{h + 1}": st["Q"][s, h] for h in range(H)})
            srows.append(r)
        if m == BASE:
            continue
        ch = rel(boot[m], boot[BASE])                                       # (B, 3)
        reps[m] = ch.astype(np.float32)
        ch_full = rel(full[m], full[BASE])
        ch_sess = rel(per_sess[m], per_sess[BASE])                          # (S, 3)
        ch_loo = rel(loo[m], loo[BASE]) if loo is not None else None
        ch_naive = rel(naive[m], naive[BASE]) if naive is not None else None
        for j, met in enumerate(METRICS):
            lo, med, hi = np.percentile(ch[:, j], [2.5, 50, 97.5])
            r = {"link": link, "method": m, "label": LABEL.get(m, m), "role": roles[m], "metric": met,
                 "n_seeds": len(cal[m]["seeds"]), "n_sessions": S, "n_windows": N,
                 "base_value": full[BASE][j], "method_value": full[m][j],
                 "change_full_pct": ch_full[j], "boot_median_pct": med, "boot_lo_pct": lo, "boot_hi_pct": hi,
                 "boot_mean_pct": ch[:, j].mean(), "boot_sd_pct": ch[:, j].std(ddof=1),
                 "frac_not_better": float((ch[:, j] >= 0).mean()),
                 "ci_excludes_zero": bool(hi < 0 or lo > 0),
                 "verdict": "better" if hi < 0 else ("worse" if lo > 0 else "inconclusive"),
                 "sessions_better": int((ch_sess[:, j] < 0).sum()),
                 "session_change_median_pct": float(np.median(ch_sess[:, j])),
                 "loo_min_pct": float(ch_loo[:, j].min()) if ch_loo is not None else np.nan,
                 "loo_max_pct": float(ch_loo[:, j].max()) if ch_loo is not None else np.nan}
            if ch_naive is not None:
                nlo, nhi = np.percentile(ch_naive[:, j], [2.5, 97.5])
                r.update({"naive_lo_pct": nlo, "naive_hi_pct": nhi,
                          "width_ratio_session_vs_naive": (hi - lo) / (nhi - nlo) if nhi > nlo else np.nan})
            rows.append(r)
    np.savez_compressed(Path(a.out_dir) / f"replicates_{link}.npz",
                        metrics=np.array(METRICS), **{m: v for m, v in reps.items()})
    info = {"link": link, "n_sessions": S, "n_windows": N, "n_val_windows": len(y_va),
            "methods": {m: {"label": LABEL.get(m, m), "note": notes[m], **cal[m],
                            "MAE_ms": full[m][0], "RMSE_ms": full[m][1], "IS_ms": full[m][2]}
                        for m in vals},
            "skipped": {m: notes[m] for m in [BASE] + methods if m not in vals}}
    if mech is not None:
        info["mechanism"] = mech
    # The final model against every other method (not only S2S-E), same paired session draws:
    # is it better than the ablations and than the linear quantile model?
    f, vs = a.fr_method, []
    if f in boot:
        for m in boot:
            if m in (f, BASE):
                continue
            ch, ch_full, ch_sess = rel(boot[f], boot[m]), rel(full[f], full[m]), rel(per_sess[f], per_sess[m])
            for j, met in enumerate(METRICS):
                lo, med, hi = np.percentile(ch[:, j], [2.5, 50, 97.5])
                vs.append({"link": link, "method": f, "against": m, "against_label": LABEL.get(m, m),
                           "against_role": roles.get(m, ""), "against_seeds": len(cal[m]["seeds"]), "metric": met,
                           "n_sessions": S, "method_value": float(full[f][j]), "against_value": float(full[m][j]),
                           "change_full_pct": float(ch_full[j]), "boot_median_pct": float(med),
                           "boot_lo_pct": float(lo), "boot_hi_pct": float(hi),
                           "frac_not_better": float((ch[:, j] >= 0).mean()),
                           "verdict": "better" if hi < 0 else ("worse" if lo > 0 else "inconclusive"),
                           "sessions_better": int((ch_sess[:, j] < 0).sum())})
    info["vs_others"] = vs
    return rows, srows, info


# ---------------------------------------------------------------------------
# Table, forest data, figure
# ---------------------------------------------------------------------------

def _fmt(v, nd):
    s = f"{v:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def table_cell(med, lo, hi):
    nd = 2 if min(abs(med), abs(lo), abs(hi)) < 0.05 else 1          # one decimal unless a value would read 0.0
    body = f"${_fmt(med, nd)}$ [${_fmt(lo, nd)}$, ${_fmt(hi, nd)}$]"
    if hi < 0:
        return "\\textbf{\\boldmath " + body + "}"         # significant improvement
    if lo > 0:
        return "\\textit{" + body + "}"                    # significant degradation
    return body


def write_table(df, method, path, links):
    lines = []
    for link in links:
        s = df[(df.link == link) & (df.method == method)].set_index("metric")
        if s.empty:
            continue
        cells = [table_cell(s.loc[m, "boot_median_pct"], s.loc[m, "boot_lo_pct"], s.loc[m, "boot_hi_pct"])
                 for m in METRICS]
        lines.append(f"{NICE[link]} & " + " & ".join(cells) + " \\\\")
    path.write_text("\n".join(lines) + "\n")


def write_forest_dat(df, method, path, links, offset):
    recs = []
    for i, link in enumerate(links, 1):
        r = {"i": i, "pos": i + offset}
        s = df[(df.link == link) & (df.method == method)].set_index("metric")
        for met in METRICS:
            k = met.lower()
            if s.empty or met not in s.index:
                r.update({f"{k}{c}": np.nan for c in ("med", "lo", "hi", "ep", "em", "sig", "non", "full")})
                continue
            med, lo, hi = s.loc[met, ["boot_median_pct", "boot_lo_pct", "boot_hi_pct"]]
            sig = bool(hi < 0 or lo > 0)
            r.update({f"{k}med": med, f"{k}lo": lo, f"{k}hi": hi, f"{k}ep": hi - med, f"{k}em": med - lo,
                      f"{k}sig": med if sig else np.nan, f"{k}non": np.nan if sig else med,
                      f"{k}full": s.loc[met, "change_full_pct"]})
        recs.append(r)
    pd.DataFrame(recs).to_csv(path, sep=" ", index=False, float_format="%.4f", na_rep="nan")


def nice_limits(lo, hi):
    """Axis range that contains 0 and [lo, hi] with a small margin, and a tick step giving 5 to 7 ticks."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = 0.07 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad
    rnd = 0.5 if hi - lo <= 4 else (1 if hi - lo <= 12 else (2 if hi - lo <= 30 else 5))
    lo, hi = math.floor(lo / rnd) * rnd, math.ceil(hi / rnd) * rnd
    tick = next(t for t in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100) if (hi - lo) / t <= 7)
    return lo, hi, tick


def write_figure(path, plotted, links, xlims, fr_method, nsess):
    """plotted: list of (method, data path relative to reports/final_report, (color, mark, open mark)).
    nsess: number of test sessions per link (second line of the y tick label)."""
    titles = {"MAE": "(a) RTT MAE", "RMSE": "(b) RTT RMSE", "IS": "(c) RTT interval score"}
    ylabels = ", ".join(f"{{{NICE[x]}\\\\\\textcolor{{black!60}}{{{nsess[x]} sessions}}}}" for x in links)
    n = len(links)
    out = [
        f"% fr_bootstrap.tex - generated by modeling/experiments/fr_bootstrap.py (--fr_method {fr_method}).",
        "% Edit the script, not this file. Session-level bootstrap of the RTT change against S2S-E:",
        "% marker = bootstrap median, bar = 95% percentile interval, filled = interval excludes 0.",
        "\\begin{tikzpicture}",
        "\\pgfplotsset{bootstrapbar/.style={only marks, mark size=1.4pt, forget plot,",
        "  error bars/x dir=both, error bars/x explicit, error bars/error bar style={line width=0.7pt},",
        "  error bars/error mark options={rotate=90, mark size=2.4pt, line width=0.7pt}}}",
        "\\begin{groupplot}[group style={group size=3 by 1, horizontal sep=0.8cm, y descriptions at=edge left},",
        f"  frbase, scale only axis, width=4.8cm, height={0.78 * n + 0.2:.2f}cm, ymin=0.45, ymax={n + 0.55}, y dir=reverse,",
        f"  ytick={{{','.join(str(i) for i in range(1, n + 1))}}}, yticklabels={{{ylabels}}},",
        "  yticklabel style={align=right, font=\\scriptsize, inner sep=1pt}, ymajorgrids=false, ytick style={draw=none},",
        f"  minor ytick={{{','.join(f'{i + 0.5:g}' for i in range(1, n))}}}, yminorgrids, minor grid style={{gray!35}},",
        "  xmajorgrids, xlabel={change against S2S-E (\\%)}, xlabel style={yshift=0.5ex}]",
    ]
    for j, met in enumerate(METRICS):
        k = met.lower()
        lo, hi, tick = xlims[met]
        opts = f"title={{{titles[met]}}}, xmin={lo:g}, xmax={hi:g}, xtick distance={tick:g}"
        if j == 0:
            opts += (", legend to name=legboot, legend columns=-1, legend style={draw=black!60, fill=white,"
                     " /tikz/every even column/.append style={column sep=0.35cm}}")
        out.append(f"\\nextgroupplot[{opts}]")
        if j == 0:
            for m, _, (col, mk, _) in plotted:
                out.append(f"\\addlegendimage{{only marks, mark={mk}, color={col}, mark size=1.6pt}}"
                           f"\\addlegendentry{{{LABEL.get(m, m)}}}")
            # filled / open entries in the plotted marker shape when one method is shown (else circles)
            fmk, omk = (plotted[0][2][1], plotted[0][2][2]) if len(plotted) == 1 else ("*", "o")
            out.append(f"\\addlegendimage{{only marks, mark={fmk}, color=black, mark size=1.5pt}}"
                       "\\addlegendentry{95\\% interval excludes 0}")
            out.append(f"\\addlegendimage{{only marks, mark={omk}, color=black, mark size=1.5pt, mark options={{fill=white}}}}"
                       "\\addlegendentry{interval includes 0}")
        out.append("\\draw[black!75, densely dashed, line width=0.6pt] "
                   "({axis cs:0,0}|-{rel axis cs:0,0}) -- ({axis cs:0,0}|-{rel axis cs:0,1});")
        for m, dat, (col, mk, mko) in plotted:
            tab = f"table[x={k}sig, y=pos, x error plus={k}ep, x error minus={k}em] {{{dat}}};"
            out.append(f"\\addplot[bootstrapbar, color={col}, mark={mk}] {tab}")
            tab = f"table[x={k}non, y=pos, x error plus={k}ep, x error minus={k}em] {{{dat}}};"
            out.append(f"\\addplot[bootstrapbar, color={col}, mark={mko}, mark options={{fill=white}}] {tab}")
    out += ["\\end{groupplot}",
            "\\node at ($(group c2r1.north)+(0,0.95cm)$) {\\pgfplotslegendfromname{legboot}};",
            "\\end{tikzpicture}"]
    path.write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_report(df, a, extra, rep, log):
    """Table bodies, forest data and figure for the main method and the available extras,
    plus a console summary. Works on the summary table, so it can be re-run alone (--report_only)."""
    links = [l for l in LINKS if l in set(df.link)]
    plotted_methods = [m for m in [a.fr_method] + extra if (df.method == m).any()]
    write_table(df, a.fr_method, rep / "data" / "bootstrap_table.tex", links)
    for m in plotted_methods[1:]:
        write_table(df, m, rep / "data" / f"bootstrap_table_{m}.tex", links)
    k = len(plotted_methods)
    offsets = [0.0] if k == 1 else list(np.linspace(-0.26, 0.26, k)) if k <= 3 else list(np.linspace(-0.33, 0.33, k))
    plotted = []
    for j, m in enumerate(plotted_methods):
        dat = rep / "data" / f"bootstrap_forest_{m}.dat"
        write_forest_dat(df, m, dat, links, offsets[j])
        plotted.append((m, f"data/{dat.name}", STYLE.get(m, STYLE_FALLBACK[j % len(STYLE_FALLBACK)])))
    sel = df[df.method.isin(plotted_methods)]
    xlims = {met: nice_limits(sel[sel.metric == met].boot_lo_pct.min(), sel[sel.metric == met].boot_hi_pct.max())
             for met in METRICS}
    nsess = df.groupby("link").n_sessions.first().astype(int).to_dict()
    write_figure(rep / "figs" / "fr_bootstrap.tex", plotted, links, xlims, a.fr_method, nsess)

    # Console summary
    show = df[df.role != "ablation"][["link", "method", "metric", "n_sessions", "change_full_pct", "boot_median_pct",
                                      "boot_lo_pct", "boot_hi_pct", "frac_not_better", "sessions_better", "verdict"]
                                     + [c for c in ("naive_lo_pct", "naive_hi_pct") if c in df.columns]]
    log("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    ab = df[df.role == "ablation"]
    if not ab.empty:
        log("\n[ablation, CSV only]\n" + ab[["link", "method", "metric", "change_full_pct", "boot_lo_pct",
                                            "boot_hi_pct", "verdict"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    log(f"wrote {rep / 'data' / 'bootstrap_table.tex'} and {rep / 'figs' / 'fr_bootstrap.tex'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fr_method", default="fr_dist", help="final model compared with S2S-E (table + figure)")
    ap.add_argument("--extra", nargs="*", default=["fr_dist_w4", "fr_gauss"],
                    help="other distributional variants added to the figure (skipped if missing)")
    ap.add_argument("--ablation", nargs="*", default=["fr_resid", "fr_full", "lqr_fr", "ridge_fr"],
                    help="further methods compared with S2S-E, CSV only (skipped if missing)")
    ap.add_argument("--links", nargs="+", default=LINKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="seeds of every ensemble")
    ap.add_argument("--allow_partial", action="store_true", help="ensemble whatever seeds exist")
    ap.add_argument("--B", type=int, default=2000, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap generator seed (naive bootstrap uses seed + 1)")
    ap.add_argument("--naive_B", type=int, default=2000, help="replicates of the naive window bootstrap (0 = off)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--res_dir", default=str(ROOT / "results_final/main"))
    ap.add_argument("--out_dir", default=str(ROOT / "results_final/analysis/bootstrap"))
    ap.add_argument("--data_root", default=str(ROOT / "data/lsn-data"))
    ap.add_argument("--report_dir", default=str(REPO / "reports/final_report"))
    ap.add_argument("--refresh_sid", action="store_true", help="re-read the test-session ids from the data")
    ap.add_argument("--report_only", action="store_true",
                    help="only rewrite the table bodies and the figure from bootstrap_summary.csv")
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rep = Path(a.report_dir); (rep / "data").mkdir(parents=True, exist_ok=True); (rep / "figs").mkdir(exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    extra = [m for m in a.extra if m != a.fr_method]
    ablation = [m for m in a.ablation if m not in extra and m != a.fr_method]
    methods = [a.fr_method] + extra + ablation
    roles = {a.fr_method: "main", **{m: "extra" for m in extra}, **{m: "ablation" for m in ablation}}
    log(f"\n=== fr_bootstrap {time.strftime('%Y-%m-%d %H:%M:%S')} | main {a.fr_method} | extra {extra} | "
        f"ablation {ablation} | B={a.B} seed={a.seed} naive_B={a.naive_B} alpha={a.alpha}")

    f_sum, f_ses, f_js = out / "bootstrap_summary.csv", out / "session_stats.csv", out / "bootstrap_summary.json"
    if a.report_only:
        write_report(pd.read_csv(f_sum), a, extra, rep, log)
        return

    rows, srows, infos = [], [], {}
    for link in a.links:
        t0 = time.time()
        res = run_link(link, a, methods, roles, log)
        if res is None:
            continue
        r, s, info = res
        rows += r; srows += s; infos[link] = info
        log(f"[done] {link}: {info['n_sessions']} sessions, {info['n_windows']} windows, "
            f"methods {list(info['methods'])} ({time.time() - t0:.0f}s)")

    # Merge with earlier results of links not re-run in this call.
    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(srows)
    if f_sum.exists() and set(a.links) != set(LINKS):
        old = pd.read_csv(f_sum)
        df = pd.concat([old[~old.link.isin(a.links)], df], ignore_index=True)
        if f_ses.exists():
            olds = pd.read_csv(f_ses)
            sdf = pd.concat([olds[~olds.link.isin(a.links)], sdf], ignore_index=True)
        if f_js.exists():
            oldj = json.load(open(f_js)).get("links", {})
            infos = {**{k: v for k, v in oldj.items() if k not in a.links}, **infos}
    if df.empty:
        log("nothing to report")
        return
    keys = {"_l": df.link.map({l: i for i, l in enumerate(LINKS)}),
            "_r": df.role.map({"main": 0, "extra": 1, "ablation": 2}),
            "_m": df.metric.map({m: i for i, m in enumerate(METRICS)})}
    df = df.assign(**keys).sort_values(["_l", "_r", "method", "_m"]).drop(columns=list(keys)).reset_index(drop=True)
    df.to_csv(f_sum, index=False, float_format="%.6g")
    sdf.to_csv(f_ses, index=False, float_format="%.10g")
    json.dump({"settings": {k: v for k, v in vars(a).items()}, "baseline": BASE, "baseline_label": LABEL[BASE],
               "metrics": {"MAE": "mean |e| over windows and 15 steps (median of distributional models)",
                           "RMSE": "per-step RMSE of the mean forecast, averaged over the 15 steps",
                           "IS": "interval score of the calibrated 95% interval, averaged over windows and steps"},
               "change": "100 (method - S2S-E) / S2S-E in %, negative = method better",
               "links": infos, "rows": df.to_dict(orient="records")},
              open(f_js, "w"), indent=1, default=float)

    write_mechanism(infos, out, log)
    write_report(df, a, extra, rep, log)
    log(f"\nwrote {f_sum}, {f_ses}, {f_js}")


def write_mechanism(infos, out, log):
    """main_vs_others.csv, is_decomposition.csv and point_and_jumps.csv from the 'vs_others' and
    'mechanism' entries of every link (also of links kept from an earlier run)."""
    dec, pj = [], []
    for link in LINKS:
        mech = infos.get(link, {}).get("mechanism")
        if not mech:
            continue
        dec += [{"link": link, "method": mech.get("method", ""), **r} for r in mech["is_decomposition"]]
        pj.append({"link": link, "method": mech.get("method", ""), **mech["point_and_jumps"]})
    vs = [r for link in LINKS for r in infos.get(link, {}).get("vs_others", [])]
    if vs:
        vdf = pd.DataFrame(vs)
        vdf.to_csv(out / "main_vs_others.csv", index=False, float_format="%.6g")
        log("\n[final model against the other methods, paired session bootstrap, change in %]\n"
            + vdf[["link", "against", "metric", "change_full_pct", "boot_lo_pct", "boot_hi_pct", "sessions_better",
                   "verdict"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if not dec:
        return
    dd = pd.DataFrame(dec)
    dd.to_csv(out / "is_decomposition.csv", index=False, float_format="%.6g")
    pd.DataFrame(pj).to_csv(out / "point_and_jumps.csv", index=False, float_format="%.6g")
    piv = dd.pivot(index="link", columns="variant", values="change_vs_s2s_pct").reindex(
        [l for l in LINKS if l in set(dd.link)])[[k for k, _ in IS_VARIANTS]]
    log("\n[interval-score decomposition, change vs S2S-E interval in %]\n"
        + piv.to_string(float_format=lambda v: f"{v:.1f}"))
    log("[point forecast and jumps]\n" + pd.DataFrame(pj).drop(columns="method").set_index("link").T.to_string(
        float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
