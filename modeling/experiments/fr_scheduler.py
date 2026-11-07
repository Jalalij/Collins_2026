"""
experiments/fr_scheduler.py - What the forecasts buy a multipath scheduler.

Trace-driven emulation of DEADLINE-AWARE PACKET DUPLICATION on the test sessions.

Setting
-------
A sender has a satellite path (primary) and a terrestrial 5G path (backup). At
time t (the end of a 6 s input window) it plans the packets it will send at
t+h, h = 1..15 steps of 200 ms (0.2 s to 3 s ahead). A packet sent on the
primary at t+h misses its deadline D when the measured primary RTT at t+h
exceeds D. The scheduler may DUPLICATE the packet on the backup, which costs
backup capacity but (assumed here, checked below) always meets D. Per link we
count over all test windows and all steps h:
    duplication rate = share of packets duplicated               (cost)
    miss rate        = share of packets NOT duplicated whose primary RTT at
                       t+h exceeds D                             (failure)
Primary links: starlink_dl, starlink_ul, oneweb_dl, oneweb_ul (RTT target).
Deadline D per link = 90th percentile of the VALIDATION RTT (all validation
windows and steps, the same spike threshold as final_extras.py). The 95th
percentile is reported as well.

Policies (each has one knob; sweeping the knob traces a duplication-vs-miss curve)
-------------------------------------------------------------------------------
reactive   duplicate if  last measured RTT + m > D                 (no forecast)
s2s        duplicate if  S2S-E mean + m > D                        (point forecast)
s2s_adapt  duplicate if  S2S-E mean + m * s(x) > D, s(x) = input RTT volatility,
           chosen on VALIDATION among {SD of the 6 s window, mean |dRTT| of the
           window, SD of the last 2 s} (locally adaptive margin, the strongest
           heuristic we could give a point forecast)
ds2s       duplicate if  q0.5 + lam * (q0.975 - q0.5) > D             (DS2S-E)
gauss      same rule with the Gaussian mean-variance heads (fr_gauss), if present
frfull     duplicate if  fr_full-E mean + m > D: the point forecaster with the SAME
           inputs, encoder and anchored decoder as DS2S-E, trained with squared error.
           DS2S-E vs frfull isolates the value of the predictive distribution, S2S-E
           vs frfull the value of the new inputs/decoder (control, not in the figure)
frresid    fr_resid-E mean + m (anchored decoder, S2S-E inputs), ridgefr = ridge_fr mean + m
           (linear point model with the DS2S-E inputs, the point twin of lqr)
oracle     duplicate exactly when y > D (lower bound: miss = max(0, p - dup))
random     duplicate a random share (no information: miss = p (1 - dup))
Each knob rule is "score > threshold" for one score per (window, step):
last, mean, (mean - D)/s(x), (q0.5 - D)/(q0.975 - q0.5). The curve is therefore
computed exactly over ALL knob values (every distinct score is a threshold),
the limit of a fine knob grid. The DS2S score is a monotone proxy of the
predicted exceedance probability P(RTT > D | x) under a location-scale model.

Calibrated operating points (deployable, knob set on VALIDATION only)
---------------------------------------------------------------------
One-sided split-conformal upper bound at level 1 - a, per step h:
    U = base + c_h,   c_h = conformal quantile of (y - base) on validation
base = q0.975 (DS2S-E, Gauss-E), mean (S2S-E), last value (reactive); for
s2s_adapt U = mean + c_h s(x) with c_h from (y - mean)/s(x). Duplicate if U > D.
If the bound covers y with probability >= 1 - a, the miss rate is <= a by
construction (a miss needs y > D >= U), whatever the model. The model only
decides how much duplication this guarantee costs. Levels 0.8 .. 0.99.
The two-sided 95% bounds of the paper tables (S2S-E: mean + conformal |residual|
quantile, DS2S-E: q0.975 + CQR offset) are reported as well ("paper95").
Conformal offset = np.quantile(scores, min(1, ceil((n+1)(1-a))/n), method="higher").

Headline comparison (per link and deadline)
-------------------------------------------
ref = the S2S-E one-sided 95% calibrated bound (dup_S, miss_S). On the DS2S-E
curve: the duplication needed to reach miss_S, and the miss rate at dup_S.
The same for the S2S-E curve (its best knob in hindsight) and for the paper95
bound. Reading a curve at miss_S picks the knob on test (like a point on an ROC
curve), so the deployable twin is reported as well ("valmatch"): the DS2S-E knob
(and the frfull knob) is set on VALIDATION so that its validation miss rate equals
the validation miss rate of the S2S-E reference bound, then applied to test.
95% CIs from a paired bootstrap over TEST SESSIONS (B resamples, fixed
seed): windows of one session are correlated, so sessions are the unit.
Also: per-step (h) breakdown, a deadline sweep (validation percentiles 80..99),
duplication at fixed recall (share of late packets protected), a decomposition
of the DS2S-E gain into "new inputs/decoder" (frfull, a point forecast), "better
location" (its own mean + m) and "spread", a risk
map (test violation rate by S2S-E mean bin x DS2S-E spread quintile inside the
bin, edges from validation: how much risk a point forecast cannot see), and the backup
check: share of 5G DL / 5G UL test RTT values above D (DL primary pairs with
5G DL, UL with 5G UL), with the miss rate corrected under independence.

Nothing is fitted on test data: deadlines, conformal offsets, the s(x) choice
and its floor come from validation windows. No training, CPU only.

Outputs
-------
modeling/results_final/analysis/scheduler/[<fr_method>/ if not fr_dist]
    inputs_<link>.npz         cache of input statistics per window (last value,
                              volatility candidates, session id), test and val
    curves_<link>.csv         every policy, D90 and D95: miss rate on a 0.5% grid
                              of duplication rates (plus knob value)
    calibrated_<link>.csv     calibrated operating points per policy and level
    per_h_<link>.csv          per-step comparison at the S2S-E 95% bound
    sweep_<link>.csv          deadline sweep x step, duplication at 90% recall
    riskmap_<link>.csv        violation rate per (S2S-E mean bin, DS2S-E spread quintile) cell
    riskmap_frfull_<link>.csv the same with fr_full-E mean bins (same-inputs point forecast, control)
    link_<link>.json          every number for one link (incl. bootstrap CIs)
    summary.json, headline.csv, backup_check.csv, fixed_budget.csv
reports/final_report/data/ (default method only)
    fr_scheduler_curve_<link>.dat   figure curves (D90, 201 rows)
    fr_scheduler_cal_<link>.dat     calibrated points (D90, 5 levels)
    fr_scheduler_heat_<link>.dat    deadline x step heat map data (saving; savingc clipped at -50)
    fr_scheduler_risk_<link>.dat    risk map data (10 mean deciles x 5 spread quintiles, log colour)
reports/final_report/figs/ (generated pgfplots, default method only)
    fr_scheduler.tex          1 x 4: duplication vs miss rate, insets on the low-duplication region
    fr_scheduler_heat.tex     1 x 4 heat maps: duplication saved, deadline percentile x lead time
    fr_scheduler_risk.tex     1 x 4 heat maps: violation rate, S2S-E mean bin x DS2S-E spread quintile

Usage
-----
LSN_DEVICE=cpu python experiments/fr_scheduler.py                    # all four links
LSN_DEVICE=cpu python experiments/fr_scheduler.py --links starlink_dl
LSN_DEVICE=cpu python experiments/fr_scheduler.py --fr_method fr_dist_w4
LSN_DEVICE=cpu python experiments/fr_scheduler.py --summary_only     # rebuild summaries
Deterministic. Each link writes its own files and the summary re-reads every
per-link file present, so links can be (re)run one at a time. Methods that are
still training (fr_gauss, fr_dist_w4, lqr_fr) are skipped with a note and picked
up on a re-run (CSV/JSON). The Gauss curve enters fr_scheduler.tex only once every
link has a 3-seed ensemble (it is then a "-E" model like the others). The input
statistics are cached (inputs_<link>.npz), so a re-run does not reload the raw sessions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LSN_DEVICE", "cpu")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))

from final_extras import load_ens  # noqa: E402

LINKS = ["starlink_dl", "starlink_ul", "oneweb_dl", "oneweb_ul"]
BACKUP = {"dl": "quectel_dl", "ul": "quectel_ul"}
NICE = {"starlink_dl": "Starlink DL", "starlink_ul": "Starlink UL", "oneweb_dl": "OneWeb DL",
        "oneweb_ul": "OneWeb UL", "quectel_dl": "5G DL", "quectel_ul": "5G UL"}
L, H = 30, 15
K_RTT = 0
LEVELS = (0.8, 0.9, 0.95, 0.975, 0.99)
DEADLINE_PCTS = (90.0, 95.0)
SWEEP_PCTS = (80.0, 85.0, 90.0, 95.0, 97.5, 99.0)
GRID = np.round(np.arange(0, 100.0001, 0.5), 4) / 100.0       # duplication rates for the curves
BUDGETS = (0.05, 0.10, 0.20, 0.30)                            # fixed duplication budgets
RECALLS = (0.8, 0.9, 0.95)                                    # share of late packets protected
VOL_CANDIDATES = ("sd30", "mad30", "sd10")
RISK_MIN = 200                                                # min decisions per risk-map cell
RISK_MEAN_BINS = 10                                           # S2S-E mean bins (deciles) of the risk map
HEAT_CLIP = -50.0                                             # heat map colour floor (values below are saturated)
EPS_W = 1e-3                                                  # ms, floor of q0.975 - q0.5
# optional extra distributional methods: key -> label (scored like DS2S-E)
EXTRA_DIST = {"fr_gauss": "gauss", "fr_dist_w4": "ds2s_w4", "lqr_fr": "lqr"}
# control point forecasters: key -> label (scored like S2S-E, mean + m > D). fr_full has the inputs,
# encoder and anchored decoder of DS2S-E but one squared-error output, so DS2S-E vs frfull is the
# value of the distribution at equal inputs. ridge_fr is the linear point twin of lqr_fr.
POINT_EXTRA = {"fr_full": "frfull", "fr_resid": "frresid", "ridge_fr": "ridgefr"}
MIN_SEEDS_FIG = 3                                             # seeds needed before an extra curve is drawn as "-E"


# ---------------------------------------------------------------------------
# Conformal helpers
# ---------------------------------------------------------------------------

def conformal_q(scores, alpha):
    """Split-conformal quantile along axis 0 (validation windows)."""
    n = scores.shape[0]
    lev = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, lev, axis=0, method="higher")


# ---------------------------------------------------------------------------
# Trade-off curves
# ---------------------------------------------------------------------------

class Curve:
    """Policy 'duplicate iff score > tau' for every tau (exact, ties kept together).

    Point j duplicates the j highest-scored groups of tied scores. dup and miss
    are fractions of all decisions. thr[j] is a threshold that realises point j.
    """

    def __init__(self, score, ev):
        s = score.ravel().astype(np.float64)
        e = ev.ravel()
        self.order = np.argsort(-s, kind="stable")
        ss = s[self.order]
        self.ev_sorted = e[self.order]
        # end index of every group of tied scores in the sorted order
        self.last = np.r_[np.nonzero(np.diff(ss) != 0)[0], len(ss) - 1]
        nxt = np.r_[ss[self.last[:-1] + 1], -np.inf]
        self.thr = np.r_[np.inf, 0.5 * (ss[self.last] + np.where(np.isfinite(nxt), nxt, ss[self.last] - 1.0))]
        self.n = len(s)
        self.n_ev = int(e.sum())
        ce = np.cumsum(self.ev_sorted)[self.last]
        self.dup = np.r_[0.0, (self.last + 1) / self.n]
        self.miss = np.r_[self.n_ev / self.n, (self.n_ev - ce) / self.n]

    def weighted(self, w):
        """(dup, miss) of the same thresholds with per-decision weights w (bootstrap)."""
        ws = w.ravel()[self.order].astype(np.float64)
        cw = np.cumsum(ws)[self.last]
        ce = np.cumsum(ws * self.ev_sorted)[self.last]
        W, E = cw[-1], ce[-1]
        return np.r_[0.0, cw / W], np.r_[E / W, (E - ce) / W]


def miss_at_dup(dup, miss, x):
    """Miss rate at duplication rate x (linear between thresholds = randomised tie-break)."""
    return float(np.interp(x, dup, miss))


def dup_at_miss(dup, miss, y):
    """Smallest duplication rate that reaches miss rate <= y (linear interpolation)."""
    i = int(np.searchsorted(-miss, -y, side="left"))   # first point with miss <= y
    if i == 0:
        return float(dup[0])
    if i >= len(miss):
        return float("nan")
    m0, m1, d0, d1 = miss[i - 1], miss[i], dup[i - 1], dup[i]
    return float(d0 + (m0 - y) / (m0 - m1) * (d1 - d0))


def knob_at_dup(curve, x, kind, D):
    """Knob value (m or lambda) of the threshold closest to duplication rate x."""
    j = int(np.clip(np.searchsorted(curve.dup, x), 0, len(curve.thr) - 1))
    t = curve.thr[j]
    if not np.isfinite(t):
        return float("nan")
    return float(D - t) if kind == "shift" else float(-t)


def op_point(U, y, ev, D, w=None):
    """(dup, miss, coverage of U) of the rule 'duplicate iff U > D' (ev = y > D)."""
    dupm = U > D
    if w is None:
        return float(dupm.mean()), float((~dupm & ev).mean()), float((y <= U).mean())
    W = w.sum()
    return float((w * dupm).sum() / W), float((w * (~dupm & ev)).sum() / W), float((w * (y <= U)).sum() / W)


_BACKUP_CACHE = {}


def backup_rtt(res_dir, link):
    """5G test RTT, each test row once (first step of every window); None if missing."""
    if link not in _BACKUP_CACHE:
        f = Path(res_dir) / link / "preds" / "truth.npz"
        _BACKUP_CACHE[link] = np.load(f)["y_te"][:, 0, K_RTT].astype(np.float64) if f.exists() else None
    return _BACKUP_CACHE[link]


# ---------------------------------------------------------------------------
# Inputs (cached): last value, input volatility, session id per window
# ---------------------------------------------------------------------------

def input_stats(link, cache, data_root):
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import torch
    torch.set_num_threads(2)
    from final_benchmark import load_splits
    prov, dirn = link.split("_")
    t0 = time.time()
    data, meta = load_splits(prov, dirn, "base", L, H, torch.device("cpu"), data_root)
    j = meta["target_idx"][K_RTT]
    out = {"n_sessions": np.array([meta["n_sessions"]["train"], meta["n_sessions"]["val"],
                                   meta["n_sessions"]["test"]])}
    for split, tag in (("test", "te"), ("val", "va")):
        sp = data[split]
        r = sp.X[:, j].numpy().astype(np.float64) * float(meta["tstd"][K_RTT]) + float(meta["tmean"][K_RTT])
        st = sp.starts.numpy()
        win = r[st[:, None] + np.arange(L)[None]]                      # (N, 30) input RTT, ms
        out[f"last_{tag}"] = win[:, -1].astype(np.float32)
        out[f"sd30_{tag}"] = win.std(1).astype(np.float32)
        out[f"mad30_{tag}"] = np.abs(np.diff(win, axis=1)).mean(1).astype(np.float32)
        out[f"sd10_{tag}"] = win[:, -10:].std(1).astype(np.float32)
        out[f"sid_{tag}"] = np.asarray(sp.sid)[st].astype(np.int32)
        out[f"ts_{tag}"] = np.asarray(sp.ts)[st + L - 1].astype(np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f"  [{link}] inputs loaded and cached ({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# One link
# ---------------------------------------------------------------------------

def dist_score(q, D):
    """DS2S rule q0.5 + lam (q0.975 - q0.5) > D  <=>  (q0.5 - D)/(q0.975 - q0.5) > -lam."""
    return (q[..., 3] - D) / np.maximum(q[..., 5] - q[..., 3], EPS_W)


def partial_area(curve, xmax=0.5):
    """Mean miss rate over duplication rates 0..xmax (lower is better)."""
    xs = np.linspace(0, xmax, 101)
    return float(np.mean(np.interp(xs, curve.dup, curve.miss)))


def analyse_link(link, a, out):
    t0 = time.time()
    pdir = Path(a.res_dir) / link / "preds"
    tr = np.load(pdir / "truth.npz")
    y, yv = tr["y_te"][:, :, K_RTT].astype(np.float64), tr["y_va"][:, :, K_RTT].astype(np.float64)
    N, Hh = y.shape
    note = []

    inp = input_stats(link, out / f"inputs_{link}.npz", a.data_root)
    assert len(inp["last_te"]) == N and len(inp["last_va"]) == len(yv), "window count mismatch"

    pers = np.load(pdir / "persistence_s0.npz")
    last, lastv = pers["p_te"][:, :, K_RTT].astype(np.float64), pers["p_va"][:, :, K_RTT].astype(np.float64)
    gap = float(np.abs(inp["last_te"] - last[:, 0]).max())
    assert gap < 1e-2, f"input last value != persistence ({gap})"

    s2s = load_ens(pdir, "prev_seq2seq")
    ds = load_ens(pdir, a.fr_method)
    if s2s is None or ds is None or "d_te" not in ds:
        raise SystemExit(f"{link}: prev_seq2seq or {a.fr_method} predictions missing")
    mu, muv = s2s["p_te"][:, :, K_RTT].astype(np.float64), s2s["p_va"][:, :, K_RTT].astype(np.float64)
    q, qv = ds["d_te"][:, :, K_RTT].astype(np.float64), ds["d_va"][:, :, K_RTT].astype(np.float64)
    extras = {}
    for key, lab in EXTRA_DIST.items():
        if key == a.fr_method:
            continue
        e = load_ens(pdir, key)
        if e is None or "d_te" not in e:
            note.append(f"{key} not available (skipped)")
            continue
        extras[lab] = (e["d_te"][:, :, K_RTT].astype(np.float64), e["d_va"][:, :, K_RTT].astype(np.float64), e["n"])
        note.append(f"{key} included ({e['n']} seeds)")
    points = {}
    for key, lab in POINT_EXTRA.items():
        e = load_ens(pdir, key)
        if e is None:
            note.append(f"{key} not available (skipped)")
            continue
        points[lab] = (e["p_te"][:, :, K_RTT].astype(np.float64), e["p_va"][:, :, K_RTT].astype(np.float64), e["n"])
        note.append(f"{key} included ({e['n']} seeds)")
    n_seeds = {"s2s": s2s["n"], "ds2s": ds["n"], **{lab: v[2] for lab, v in extras.items()},
               **{lab: v[2] for lab, v in points.items()}}

    # input volatility candidates, floored at their validation 5th percentile
    vol = {}
    for c in VOL_CANDIDATES:
        fl = float(np.quantile(inp[f"{c}_va"], 0.05))
        fl = max(fl, 1e-3)
        vol[c] = (np.maximum(inp[f"{c}_te"].astype(np.float64), fl)[:, None],
                  np.maximum(inp[f"{c}_va"].astype(np.float64), fl)[:, None], fl)

    sid = np.repeat(inp["sid_te"][:, None], Hh, 1)
    ses = np.unique(inp["sid_te"])
    sid_idx = np.searchsorted(ses, sid)
    rng = np.random.default_rng(a.seed)
    boot_counts = np.stack([np.bincount(rng.integers(0, len(ses), len(ses)), minlength=len(ses))
                            for _ in range(a.n_boot)]).astype(np.float64)

    res = {"link": link, "fr_method": a.fr_method, "n_test_windows": int(N), "n_val_windows": int(len(yv)),
           "n_steps": int(Hh), "n_decisions": int(N * Hh), "n_test_sessions": int(len(ses)),
           "n_sessions_train_val_test": inp["n_sessions"].tolist(),
           "n_seeds": n_seeds, "notes": note, "deadlines": {}}
    curve_rows, cal_rows, perh_rows = [], [], []

    for pct in DEADLINE_PCTS:
        D = float(np.quantile(yv, pct / 100))
        ev, evv = y > D, yv > D
        p = float(ev.mean())
        tag = f"D{pct:g}"
        dres = {"pct": pct, "D_ms": D, "p_test": p, "p_val": float(evv.mean())}

        # s(x) for s2s_adapt: pick the candidate with the lowest validation partial area
        va_area = {c: partial_area(Curve((muv - D) / vol[c][1], evv)) for c in VOL_CANDIDATES}
        vbest = min(va_area, key=va_area.get)
        s_te, s_va, s_floor = vol[vbest]
        dres["s2s_adapt_choice"] = {"chosen": vbest, "floor_ms": s_floor, "val_partial_area": va_area}

        scores = {"reactive": (last, lastv, "shift"), "s2s": (mu, muv, "shift"),
                  "s2s_adapt": ((mu - D) / s_te, (muv - D) / s_va, "norm"),
                  "ds2s": (dist_score(q, D), dist_score(qv, D), "norm"),
                  # diagnostics: which part of the DS2S output carries the gain
                  "ds2s_mean": (q[..., 0], qv[..., 0], "shift"),
                  "ds2s_q975": (q[..., 5], qv[..., 5], "shift")}
        for lab, (et, evd, _) in extras.items():
            scores[lab] = (dist_score(et, D), dist_score(evd, D), "norm")
        for lab, (pt, pv, _) in points.items():
            scores[lab] = (pt, pv, "shift")
        curves = {k: Curve(v[0], ev) for k, v in scores.items()}

        # ---- curves on the grid (figure + CSV)
        for k, cv in curves.items():
            for x in GRID:
                curve_rows.append({"deadline": tag, "policy": k, "dup": x, "miss": miss_at_dup(cv.dup, cv.miss, x),
                                   "knob": knob_at_dup(cv, x, scores[k][2], D)})
        for x in GRID:
            curve_rows.append({"deadline": tag, "policy": "oracle", "dup": x, "miss": max(0.0, p - x), "knob": np.nan})
            curve_rows.append({"deadline": tag, "policy": "random", "dup": x, "miss": p * (1 - x), "knob": np.nan})

        # ---- fixed budgets / fixed recall (curve vs curve, knob chosen in hindsight)
        dres["at_budget"] = {k: {f"{b:g}": miss_at_dup(cv.dup, cv.miss, b) for b in BUDGETS} for k, cv in curves.items()}
        dres["at_recall"] = {k: {f"{r:g}": dup_at_miss(cv.dup, cv.miss, (1 - r) * p) for r in RECALLS}
                             for k, cv in curves.items()}
        dres["at_recall"]["oracle"] = {f"{r:g}": r * p for r in RECALLS}
        dres["at_recall"]["random"] = {f"{r:g}": r for r in RECALLS}
        dres["partial_area_0_50"] = {k: partial_area(cv) for k, cv in curves.items()}

        # ---- calibrated bounds (validation offsets per step)
        bases = {"reactive": (last, lastv, None), "s2s": (mu, muv, None), "s2s_adapt": (mu, muv, (s_te, s_va)),
                 "ds2s": (q[..., 5], qv[..., 5], None)}
        for lab, (et, evd, _) in extras.items():
            bases[lab] = (et[..., 5], evd[..., 5], None)
        for lab, (pt, pv, _) in points.items():
            bases[lab] = (pt, pv, None)
        bounds = {}
        for k, (bt, bv, sc) in bases.items():
            for lev in LEVELS:
                if sc is None:
                    U = bt + conformal_q(yv - bv, 1 - lev)[None]
                else:
                    U = bt + conformal_q((yv - bv) / sc[1], 1 - lev)[None] * sc[0]
                bounds[(k, f"one{lev:g}")] = U
        # paper two-sided 95% bounds
        bounds[("s2s", "paper95")] = mu + conformal_q(np.abs(yv - muv), 0.05)[None]
        bounds[("ds2s", "paper95")] = q[..., 5] + conformal_q(np.maximum(qv[..., 1] - yv, yv - qv[..., 5]), 0.05)[None]
        cal = {}
        for (k, lv), U in bounds.items():
            d_, m_, c_ = op_point(U, y, ev, D)
            cal[f"{k}|{lv}"] = {"dup": d_, "miss": m_, "coverage": c_}
            cal_rows.append({"deadline": tag, "policy": k, "level": lv, "dup": d_, "miss": m_, "coverage_of_U": c_,
                             "miss_bound": (1 - float(lv[3:])) if lv.startswith("one") else 0.05})
        dres["calibrated"] = cal
        # the S2S-E reference bounds evaluated on the VALIDATION windows (same offsets as above): their
        # validation miss rate is the target of the deployable matched comparison ("valmatch")
        ref_val = {("s2s", "one0.95"): muv + conformal_q(yv - muv, 1 - 0.95)[None],
                   ("s2s", "paper95"): muv + conformal_q(np.abs(yv - muv), 0.05)[None]}

        # ---- headline: S2S-E 95% bound vs the DS2S-E curve (+ paired session bootstrap)
        head = {}
        for refname, refkey, ownkey in (("one95", ("s2s", "one0.95"), ("ds2s", "one0.95")),
                                        ("paper95", ("s2s", "paper95"), ("ds2s", "paper95"))):
            U_S, U_D = bounds[refkey], bounds[ownkey]
            dS, mS, _ = op_point(U_S, y, ev, D)
            dD, mD, _ = op_point(U_D, y, ev, D)
            # deployable twin of "the DS2S-E curve at the S2S-E miss rate": the knob is the threshold that
            # reaches the validation miss rate of the S2S-E reference bound on the VALIDATION curve, then
            # it is applied unchanged to test (no test information). Same for the frfull control.
            mSv = float(((ref_val[refkey] <= D) & evv).mean())
            vmask, vm = {}, {}
            for k in ("ds2s", "frfull"):
                if k in scores:
                    cvv = Curve(scores[k][1], evv)
                    thr = float(cvv.thr[int(np.argmax(cvv.miss <= mSv))])   # first point with val. miss <= target
                    vmask[k] = scores[k][0] > thr
                    vm[k] = {"dup": float(vmask[k].mean()), "miss": float((~vmask[k] & ev).mean()),
                             "knob_threshold": thr, "val_miss_target": mSv}
            h = {"s2s_point": {"dup": dS, "miss": mS}, "ds2s_point": {"dup": dD, "miss": mD},
                 "ds2s_curve_dup_at_s2s_miss": dup_at_miss(curves["ds2s"].dup, curves["ds2s"].miss, mS),
                 "s2s_curve_dup_at_s2s_miss": dup_at_miss(curves["s2s"].dup, curves["s2s"].miss, mS),
                 "s2s_adapt_curve_dup_at_s2s_miss": dup_at_miss(curves["s2s_adapt"].dup, curves["s2s_adapt"].miss, mS),
                 "reactive_curve_dup_at_s2s_miss": dup_at_miss(curves["reactive"].dup, curves["reactive"].miss, mS),
                 "oracle_dup_at_s2s_miss": max(0.0, p - mS),
                 "ds2s_curve_miss_at_s2s_dup": miss_at_dup(curves["ds2s"].dup, curves["ds2s"].miss, dS),
                 "s2s_curve_miss_at_s2s_dup": miss_at_dup(curves["s2s"].dup, curves["s2s"].miss, dS),
                 # vice versa: the S2S-E curve at the DS2S-E calibrated point
                 "s2s_curve_dup_at_ds2s_miss": dup_at_miss(curves["s2s"].dup, curves["s2s"].miss, mD),
                 "s2s_curve_miss_at_ds2s_dup": miss_at_dup(curves["s2s"].dup, curves["s2s"].miss, dD)}
            h["dup_saving_vs_s2s_point"] = 1 - h["ds2s_curve_dup_at_s2s_miss"] / dS if dS > 0 else float("nan")
            h["dup_saving_vs_s2s_curve"] = (1 - h["ds2s_curve_dup_at_s2s_miss"] / h["s2s_curve_dup_at_s2s_miss"]
                                            if h["s2s_curve_dup_at_s2s_miss"] > 0 else float("nan"))
            h["miss_reduction_at_s2s_dup"] = 1 - h["ds2s_curve_miss_at_s2s_dup"] / mS if mS > 0 else float("nan")
            # control: the point forecaster with the DS2S-E inputs and decoder (value of the distribution itself)
            nan = float("nan")
            h["frfull_curve_dup_at_s2s_miss"] = (dup_at_miss(curves["frfull"].dup, curves["frfull"].miss, mS)
                                                 if "frfull" in curves else nan)
            h["dup_saving_vs_frfull_curve"] = (1 - h["ds2s_curve_dup_at_s2s_miss"] / h["frfull_curve_dup_at_s2s_miss"]
                                               if h["frfull_curve_dup_at_s2s_miss"] > 0 else nan)
            # deployable matched comparison (knobs from validation only)
            h["valmatch"] = vm
            if "ds2s" in vm:
                h["valmatch_saving_vs_s2s_point"] = 1 - vm["ds2s"]["dup"] / dS if dS > 0 else nan
                h["valmatch_ds2s_dominates"] = bool(vm["ds2s"]["dup"] <= dS and vm["ds2s"]["miss"] <= mS)
            if "ds2s" in vm and "frfull" in vm:
                h["valmatch_saving_vs_frfull"] = 1 - vm["ds2s"]["dup"] / vm["frfull"]["dup"] if vm["frfull"]["dup"] > 0 else nan
            # bootstrap over test sessions (paired: same resample for every policy)
            B = []
            for cnt in boot_counts:
                w = cnt[sid_idx]
                if w.sum() == 0:
                    continue
                dS_b, mS_b, _ = op_point(U_S, y, ev, D, w)
                dD_b, mD_b, _ = op_point(U_D, y, ev, D, w)
                cd, cm = curves["ds2s"].weighted(w)
                sd_, sm_ = curves["s2s"].weighted(w)
                dDc = dup_at_miss(cd, cm, mS_b)
                dSc = dup_at_miss(sd_, sm_, mS_b)
                mDc = miss_at_dup(cd, cm, dS_b)
                if "frfull" in curves:
                    fd_, fm_ = curves["frfull"].weighted(w)
                    dFc = dup_at_miss(fd_, fm_, mS_b)
                else:
                    dFc = np.nan
                Wt = w.sum()
                vb = {}
                for k, msk in vmask.items():
                    vb[k] = (float((w * msk).sum() / Wt), float((w * (~msk & ev)).sum() / Wt))
                dV_b, mV_b = vb.get("ds2s", (np.nan, np.nan))
                dVF_b = vb.get("frfull", (np.nan, np.nan))[0]
                B.append([1 - dDc / dS_b if dS_b > 0 else np.nan,
                          1 - dDc / dSc if dSc > 0 else np.nan,
                          1 - mDc / mS_b if mS_b > 0 else np.nan,
                          dS_b, mS_b, dD_b, mD_b, dDc, mDc,
                          1 - dDc / dFc if dFc > 0 else np.nan,
                          dV_b, mV_b,
                          1 - dV_b / dS_b if dS_b > 0 else np.nan,
                          mV_b - mS_b,
                          1 - dV_b / dVF_b if dVF_b > 0 else np.nan])
            B = np.array(B)
            names = ["dup_saving_vs_s2s_point", "dup_saving_vs_s2s_curve", "miss_reduction_at_s2s_dup",
                     "s2s_point_dup", "s2s_point_miss", "ds2s_point_dup", "ds2s_point_miss",
                     "ds2s_curve_dup_at_s2s_miss", "ds2s_curve_miss_at_s2s_dup",
                     "dup_saving_vs_frfull_curve", "valmatch_ds2s_dup", "valmatch_ds2s_miss",
                     "valmatch_saving_vs_s2s_point", "valmatch_miss_change_vs_s2s_point", "valmatch_saving_vs_frfull"]
            bs_ = {}
            for i, nm in enumerate(names):
                col_ = B[:, i]
                ok_ = np.isfinite(col_)
                bs_[nm] = ({"lo": float(np.percentile(col_[ok_], 2.5)), "hi": float(np.percentile(col_[ok_], 97.5)),
                            "p_positive": float(np.mean(col_[ok_] > 0))} if ok_.any() else
                           {"lo": nan, "hi": nan, "p_positive": nan})
            h["bootstrap"] = bs_
            h["bootstrap"]["n"] = int(len(B))
            # the DS2S-E calibrated point dominates the S2S-E one (less dup AND fewer misses)?
            h["ds2s_point_dominates"] = bool(dD <= dS and mD <= mS)
            h["p_ds2s_point_dominates"] = float(np.mean((B[:, 5] <= B[:, 3]) & (B[:, 6] <= B[:, 4])))
            h["p_valmatch_dominates"] = float(np.mean((B[:, 10] <= B[:, 3]) & (B[:, 11] <= B[:, 4])))
            head[refname] = h
        dres["headline"] = head

        # ---- decomposition: location vs spread (at the S2S-E one-sided 95% miss rate)
        mS = head["one95"]["s2s_point"]["miss"]
        # frfull (same inputs as DS2S-E, point output) separates "new inputs/decoder" from "distribution"
        dres["decomposition_at_s2s95_miss"] = {k: dup_at_miss(curves[k].dup, curves[k].miss, mS)
                                               for k in ("reactive", "s2s", "s2s_adapt", "frresid", "frfull", "ridgefr",
                                                         "ds2s_mean", "ds2s_q975", "ds2s", "lqr", "gauss", "ds2s_w4")
                                               if k in curves}

        # ---- per step h (primary deadline only and D95)
        for hh in range(Hh):
            evh = ev[:, hh]
            ph = float(evh.mean())
            ch = {k: Curve(v[0][:, hh], evh) for k, v in scores.items() if k in ("reactive", "s2s", "s2s_adapt", "ds2s")}
            US = bounds[("s2s", "one0.95")][:, hh]
            UD = bounds[("ds2s", "one0.95")][:, hh]
            dS_h = float((US > D).mean()); mS_h = float(((US <= D) & evh).mean())
            dD_h = float((UD > D).mean()); mD_h = float(((UD <= D) & evh).mean())
            row = {"deadline": tag, "h": hh + 1, "t_s": round(0.2 * (hh + 1), 1), "p": ph,
                   "s2s95_dup": dS_h, "s2s95_miss": mS_h, "ds2s95_dup": dD_h, "ds2s95_miss": mD_h}
            for k, cv in ch.items():
                row[f"{k}_dup_at_s2s95_miss"] = dup_at_miss(cv.dup, cv.miss, mS_h)
                row[f"{k}_dup_at_recall90"] = dup_at_miss(cv.dup, cv.miss, 0.1 * ph)
            row["saving_vs_s2s_point"] = 1 - row["ds2s_dup_at_s2s95_miss"] / dS_h if dS_h > 0 else np.nan
            row["saving_vs_s2s_curve_recall90"] = 1 - row["ds2s_dup_at_recall90"] / row["s2s_dup_at_recall90"]
            # spread of the DS2S-E forecast, and how often each forecast crosses D at this step
            row["ds2s_mean_spread_ms"] = float(np.mean(q[:, hh, 5] - q[:, hh, 3]))
            row["s2s_mean_sd_ms"] = float(np.std(mu[:, hh]))
            row["frac_s2s_mean_above_D"] = float((mu[:, hh] > D).mean())
            row["frac_ds2s_q975_above_D"] = float((q[:, hh, 5] > D).mean())
            perh_rows.append(row)
        # how often the point forecast and the upper quantile cross the deadline (all steps)
        dres["frac_s2s_mean_above_D"] = float((mu > D).mean())
        dres["frac_ds2s_median_above_D"] = float((q[..., 3] > D).mean())
        dres["frac_ds2s_q975_above_D"] = float((q[..., 5] > D).mean())
        # burst structure of the late packets: does "late now" predict "late at t+h"?
        late_now = last[:, 0] > D
        dres["burst"] = {
            "p_late_given_late_now": [float(ev[late_now, k].mean()) for k in range(Hh)],
            "p_late_given_ontime_now": [float(ev[~late_now, k].mean()) for k in range(Hh)],
            "share_late_now": float(late_now.mean()),
            # what each forecast says about the packet at t+h when the RTT is late NOW
            "after_late_now": [{"h": k + 1, "late_rate": float(ev[late_now, k].mean()),
                                "s2s_mean_ms": float(mu[late_now, k].mean()),
                                "frac_s2s_mean_above_D": float((mu[late_now, k] > D).mean()),
                                "ds2s_mean_ms": float(q[late_now, k, 0].mean()),
                                "ds2s_median_ms": float(q[late_now, k, 3].mean()),
                                "frac_ds2s_median_above_D": float((q[late_now, k, 3] > D).mean()),
                                "ds2s_q975_ms": float(q[late_now, k, 5].mean()),
                                "frac_ds2s_q975_above_D": float((q[late_now, k, 5] > D).mean()),
                                # control: point forecast with the DS2S-E inputs and decoder
                                **({"frfull_mean_ms": float(points["frfull"][0][late_now, k].mean()),
                                    "frac_frfull_mean_above_D": float((points["frfull"][0][late_now, k] > D).mean())}
                                   if "frfull" in points else {})}
                               for k in range(Hh)]}
        if "frfull" in points:
            dres["frac_frfull_mean_above_D"] = float((points["frfull"][0] > D).mean())
        runs = []
        for sv in np.unique(inp["sid_te"]):          # runs of late rows (each test row once: step 1)
            x_ = (y[inp["sid_te"] == sv, 0] > D).astype(np.int8)
            dx = np.diff(np.r_[0, x_, 0])
            runs += list(np.nonzero(dx == -1)[0] - np.nonzero(dx == 1)[0])
        runs = np.asarray(runs, dtype=float)
        if len(runs):
            dres["burst"].update({"n_runs": int(len(runs)), "run_mean_s": float(runs.mean() * 0.2),
                                  "run_median_s": float(np.median(runs) * 0.2),
                                  "share_single_sample_runs": float((runs == 1).mean()),
                                  "share_late_rows_in_runs_ge_5s": float(runs[runs >= 25].sum() / runs.sum())})
        res["deadlines"][tag] = dres
        print(f"  [{link}] {tag}: D={D:.1f} ms p={p:.4f} | S2S-E 95%: dup {head['one95']['s2s_point']['dup']:.3f} "
              f"miss {mS:.4f} | DS2S-E curve dup at that miss {head['one95']['ds2s_curve_dup_at_s2s_miss']:.3f} "
              f"(S2S-E curve {head['one95']['s2s_curve_dup_at_s2s_miss']:.3f}) | saving vs point "
              f"{head['one95']['dup_saving_vs_s2s_point']:+.3f} [{head['one95']['bootstrap']['dup_saving_vs_s2s_point']['lo']:+.3f},"
              f"{head['one95']['bootstrap']['dup_saving_vs_s2s_point']['hi']:+.3f}]", flush=True)

    # ---- risk map (D90): "same mean, different risk". Decisions are binned by the S2S-E mean forecast
    # (deciles, validation edges) and, inside each mean bin, by the DS2S-E spread q0.975 - q0.5
    # (quintiles of that bin's validation decisions). A point forecast gives a whole column the same
    # risk. The test violation rate per cell shows how much risk the spread separates at a fixed mean.
    D = res["deadlines"]["D90"]["D_ms"]
    ev = y > D
    spread, spreadv = q[..., 5] - q[..., 3], qv[..., 5] - qv[..., 3]

    def risk_map(mref, mrefv):
        """Cells (mean bin of the reference point forecast x DS2S-E spread quintile); all edges from validation."""
        em = np.quantile(mrefv, np.linspace(0, 1, RISK_MEAN_BINS + 1)[1:-1])
        bm, bmv = np.searchsorted(em, mref, side="right"), np.searchsorted(em, mrefv, side="right")
        bs = np.zeros_like(bm)
        for i in range(RISK_MEAN_BINS):
            es = np.quantile(spreadv[bmv == i], [0.2, 0.4, 0.6, 0.8])
            bs[bm == i] = np.searchsorted(es, spread[bm == i], side="right")
        cells = []
        for j in range(5):                       # rows (y): spread quintile inside the mean bin
            for i in range(RISK_MEAN_BINS):      # columns (x, varies fastest): mean bin
                m = (bm == i) & (bs == j)
                n_ = int(m.sum())
                cells.append({"mean_bin": i + 1, "spread_q": j + 1, "n": n_, "ok": bool(n_ >= RISK_MIN),
                              "rate": float(ev[m].mean()) if n_ else float("nan"),
                              "ref_mean_ms": float(mref[m].mean()) if n_ else float("nan"),
                              "ds2s_median_ms": float(q[..., 3][m].mean()) if n_ else float("nan"),
                              "ds2s_spread_ms": float(spread[m].mean()) if n_ else float("nan"),
                              "bin_center_ms": float(np.median(mrefv[bmv == i]))})
        rm_ = pd.DataFrame(cells)
        col = {}
        for i in range(1, RISK_MEAN_BINS + 1):
            c = rm_[(rm_.mean_bin == i) & rm_.ok]
            col[str(i)] = {"rate_min": float(c.rate.min()) if len(c) else None,
                           "rate_max": float(c.rate.max()) if len(c) else None,
                           "ref_mean_range_ms": float(c.ref_mean_ms.max() - c.ref_mean_ms.min()) if len(c) else None,
                           "rate_all": float(ev[bm == i - 1].mean())}
        return rm_, {"deadline": "D90", "edges_ref_mean_ms": em.tolist(), "min_cell": RISK_MIN,
                     "mean_bins": RISK_MEAN_BINS, "per_mean_bin": col}

    rmap, res["riskmap"] = risk_map(mu, muv)                 # S2S-E mean (figure)
    res["riskmap"]["reference"] = "s2s"
    rmap.rename(columns={"ref_mean_ms": "s2s_mean_ms"}).to_csv(out / f"riskmap_{link}.csv", index=False,
                                                               float_format="%.6g")
    if "frfull" in points:                                   # control: same-inputs point forecast as the mean
        rmf, res["riskmap_frfull"] = risk_map(points["frfull"][0], points["frfull"][1])
        res["riskmap_frfull"]["reference"] = "frfull"
        rmf.rename(columns={"ref_mean_ms": "frfull_mean_ms"}).to_csv(out / f"riskmap_frfull_{link}.csv", index=False,
                                                                     float_format="%.6g")

    # ---- deadline sweep x step: duplication at 90% recall, DS2S-E vs S2S-E curves
    sweep_rows = []
    for pct in SWEEP_PCTS:
        D = float(np.quantile(yv, pct / 100))
        ev = y > D
        sc = {"s2s": mu, "ds2s": dist_score(q, D), "reactive": last}
        for hh in [None] + list(range(Hh)):
            sl = slice(None) if hh is None else hh
            e_ = ev[:, sl]
            ph = float(e_.mean())
            row = {"pct": pct, "D_ms": D, "h": 0 if hh is None else hh + 1, "p": ph}
            for k, s_ in sc.items():
                cv = Curve(s_[:, sl], e_)
                for r in RECALLS:
                    row[f"{k}_dup_r{r:g}"] = dup_at_miss(cv.dup, cv.miss, (1 - r) * ph)
            for r in RECALLS:
                row[f"saving_r{r:g}"] = 1 - row[f"ds2s_dup_r{r:g}"] / row[f"s2s_dup_r{r:g}"]
            sweep_rows.append(row)

    # ---- backup check: 5G RTT above D (paired direction first)
    bk = []
    for b in ("quectel_dl", "quectel_ul"):
        yb = backup_rtt(a.res_dir, b)
        if yb is None:
            note.append(f"{b} truth missing (backup check skipped)")
            continue
        for tag, dres in res["deadlines"].items():
            frac = float((yb > dres["D_ms"]).mean())
            row = {"primary": link, "backup": b, "paired": b == BACKUP[link.split("_")[1]], "deadline": tag,
                   "D_ms": dres["D_ms"], "backup_frac_above_D": frac, "backup_rtt_p50": float(np.median(yb)),
                   "backup_rtt_p99": float(np.quantile(yb, 0.99))}
            # miss rate if the backup also misses D with this probability, independently
            for nm, key in (("s2s95", "s2s|one0.95"), ("ds2s95", "ds2s|one0.95")):
                pt = dres["calibrated"][key]
                caught = dres["p_test"] - pt["miss"]          # late packets that were duplicated
                row[f"{nm}_miss"] = pt["miss"]
                row[f"{nm}_miss_corrected"] = pt["miss"] + frac * caught
            bk.append(row)
    res["backup_check"] = bk
    res["seconds"] = round(time.time() - t0, 1)

    pd.DataFrame(curve_rows).to_csv(out / f"curves_{link}.csv", index=False, float_format="%.6g")
    pd.DataFrame(cal_rows).to_csv(out / f"calibrated_{link}.csv", index=False, float_format="%.6g")
    pd.DataFrame(perh_rows).to_csv(out / f"per_h_{link}.csv", index=False, float_format="%.6g")
    pd.DataFrame(sweep_rows).to_csv(out / f"sweep_{link}.csv", index=False, float_format="%.6g")
    json.dump(res, open(out / f"link_{link}.json", "w"), indent=1, default=float)
    print(f"  [{link}] done ({res['seconds']}s) {'; '.join(note)}", flush=True)
    return res


# ---------------------------------------------------------------------------
# Summary and figure data
# ---------------------------------------------------------------------------

def summarise(a, out, fig):
    links = [l for l in LINKS if (out / f"link_{l}.json").exists()]
    allres = {l: json.load(open(out / f"link_{l}.json")) for l in links}
    head_rows, budget_rows, bk_rows = [], [], []
    for l, r in allres.items():
        for tag, d in r["deadlines"].items():
            for ref, h in d["headline"].items():
                bs = h["bootstrap"]
                head_rows.append({
                    "link": l, "deadline": tag, "D_ms": d["D_ms"], "p_test": d["p_test"], "ref_bound": ref,
                    "s2s_dup": h["s2s_point"]["dup"], "s2s_miss": h["s2s_point"]["miss"],
                    "ds2s_dup": h["ds2s_point"]["dup"], "ds2s_miss": h["ds2s_point"]["miss"],
                    "ds2s_curve_dup_at_s2s_miss": h["ds2s_curve_dup_at_s2s_miss"],
                    "s2s_curve_dup_at_s2s_miss": h["s2s_curve_dup_at_s2s_miss"],
                    "s2s_adapt_curve_dup_at_s2s_miss": h["s2s_adapt_curve_dup_at_s2s_miss"],
                    "reactive_curve_dup_at_s2s_miss": h["reactive_curve_dup_at_s2s_miss"],
                    "oracle_dup_at_s2s_miss": h["oracle_dup_at_s2s_miss"],
                    "dup_saving_vs_s2s_point": h["dup_saving_vs_s2s_point"],
                    "dup_saving_vs_s2s_point_lo": bs["dup_saving_vs_s2s_point"]["lo"],
                    "dup_saving_vs_s2s_point_hi": bs["dup_saving_vs_s2s_point"]["hi"],
                    "dup_saving_vs_s2s_curve": h["dup_saving_vs_s2s_curve"],
                    "dup_saving_vs_s2s_curve_lo": bs["dup_saving_vs_s2s_curve"]["lo"],
                    "dup_saving_vs_s2s_curve_hi": bs["dup_saving_vs_s2s_curve"]["hi"],
                    "p_saving_vs_curve_positive": bs["dup_saving_vs_s2s_curve"]["p_positive"],
                    "ds2s_curve_miss_at_s2s_dup": h["ds2s_curve_miss_at_s2s_dup"],
                    "s2s_curve_miss_at_s2s_dup": h["s2s_curve_miss_at_s2s_dup"],
                    "miss_reduction_at_s2s_dup": h["miss_reduction_at_s2s_dup"],
                    "miss_reduction_lo": bs["miss_reduction_at_s2s_dup"]["lo"],
                    "miss_reduction_hi": bs["miss_reduction_at_s2s_dup"]["hi"],
                    "s2s_curve_dup_at_ds2s_miss": h["s2s_curve_dup_at_ds2s_miss"],
                    "s2s_curve_miss_at_ds2s_dup": h["s2s_curve_miss_at_ds2s_dup"],
                    "ds2s_point_dominates": h["ds2s_point_dominates"],
                    "p_ds2s_point_dominates": h["p_ds2s_point_dominates"],
                    # control: point forecaster with the DS2S-E inputs and decoder (fr_full-E)
                    "frfull_curve_dup_at_s2s_miss": h.get("frfull_curve_dup_at_s2s_miss", np.nan),
                    "dup_saving_vs_frfull_curve": h.get("dup_saving_vs_frfull_curve", np.nan),
                    "dup_saving_vs_frfull_curve_lo": bs.get("dup_saving_vs_frfull_curve", {}).get("lo", np.nan),
                    "dup_saving_vs_frfull_curve_hi": bs.get("dup_saving_vs_frfull_curve", {}).get("hi", np.nan),
                    # deployable matched comparison: DS2S-E / fr_full-E knob set on validation
                    "valmatch_val_miss_target": h.get("valmatch", {}).get("ds2s", {}).get("val_miss_target", np.nan),
                    "valmatch_ds2s_dup": h.get("valmatch", {}).get("ds2s", {}).get("dup", np.nan),
                    "valmatch_ds2s_miss": h.get("valmatch", {}).get("ds2s", {}).get("miss", np.nan),
                    "valmatch_frfull_dup": h.get("valmatch", {}).get("frfull", {}).get("dup", np.nan),
                    "valmatch_frfull_miss": h.get("valmatch", {}).get("frfull", {}).get("miss", np.nan),
                    "valmatch_saving_vs_s2s_point": h.get("valmatch_saving_vs_s2s_point", np.nan),
                    "valmatch_saving_vs_s2s_point_lo": bs.get("valmatch_saving_vs_s2s_point", {}).get("lo", np.nan),
                    "valmatch_saving_vs_s2s_point_hi": bs.get("valmatch_saving_vs_s2s_point", {}).get("hi", np.nan),
                    "valmatch_miss_change_lo": bs.get("valmatch_miss_change_vs_s2s_point", {}).get("lo", np.nan),
                    "valmatch_miss_change_hi": bs.get("valmatch_miss_change_vs_s2s_point", {}).get("hi", np.nan),
                    "p_valmatch_dominates": h.get("p_valmatch_dominates", np.nan),
                    "valmatch_saving_vs_frfull": h.get("valmatch_saving_vs_frfull", np.nan),
                    "valmatch_saving_vs_frfull_lo": bs.get("valmatch_saving_vs_frfull", {}).get("lo", np.nan),
                    "valmatch_saving_vs_frfull_hi": bs.get("valmatch_saving_vs_frfull", {}).get("hi", np.nan),
                    "n_test_sessions": r["n_test_sessions"]})
            for k, v in d["at_budget"].items():
                budget_rows.append({"link": l, "deadline": tag, "policy": k, "kind": "miss_at_dup_budget",
                                    **{f"b{b}": v[b] for b in v}})
            for k, v in d["at_recall"].items():
                budget_rows.append({"link": l, "deadline": tag, "policy": k, "kind": "dup_at_recall",
                                    **{f"r{rr}": v[rr] for rr in v}})
        bk_rows += r["backup_check"]
    pd.DataFrame(head_rows).to_csv(out / "headline.csv", index=False, float_format="%.5g")
    pd.DataFrame(budget_rows).to_csv(out / "fixed_budget.csv", index=False, float_format="%.5g")
    pd.DataFrame(bk_rows).to_csv(out / "backup_check.csv", index=False, float_format="%.5g")
    json.dump({"settings": {k: v for k, v in vars(a).items()}, "links": allres}, open(out / "summary.json", "w"),
              indent=1, default=float)

    if a.fr_method != "fr_dist" and not a.write_fig:
        return
    fig.mkdir(parents=True, exist_ok=True)
    for l, r in allres.items():
        cv = pd.read_csv(out / f"curves_{l}.csv")
        cv = cv[cv.deadline == "D90"]
        wide = cv.pivot(index="dup", columns="policy", values="miss").reset_index()
        wide = wide.rename(columns=lambda c: c.replace("_", ""))
        wide.iloc[:, :] = wide.values * 100.0                     # percent
        wide.to_csv(fig / f"fr_scheduler_curve_{l}.dat", sep=" ", index=False, float_format="%.4f")
        cal = pd.read_csv(out / f"calibrated_{l}.csv")
        cal = cal[(cal.deadline == "D90") & cal.level.str.startswith("one")]
        rows = []
        for lv in [f"one{x:g}" for x in LEVELS]:
            row = {"level": float(lv[3:]) * 100}
            for k in ("s2s", "ds2s", "reactive", "s2s_adapt", "gauss"):
                m = cal[(cal.policy == k) & (cal.level == lv)]
                if len(m):
                    row[f"{k.replace('_', '')}dup"] = float(m.dup.iloc[0]) * 100
                    row[f"{k.replace('_', '')}miss"] = float(m.miss.iloc[0]) * 100
            rows.append(row)
        pd.DataFrame(rows).to_csv(fig / f"fr_scheduler_cal_{l}.dat", sep=" ", index=False, float_format="%.4f")
        sw = pd.read_csv(out / f"sweep_{l}.csv")
        sw = sw[sw.h > 0][["pct", "h", "saving_r0.9"]].copy()
        sw["t"] = sw.h * 0.2
        sw["row"] = sw.pct.map({p: i for i, p in enumerate(SWEEP_PCTS)})
        sw["saving"] = sw["saving_r0.9"] * 100
        sw["savingc"] = sw["saving"].clip(lower=HEAT_CLIP)
        sw.sort_values(["row", "h"])[["h", "t", "row", "pct", "saving", "savingc"]].to_csv(
            fig / f"fr_scheduler_heat_{l}.dat", sep=" ", index=False, float_format="%.4f")
        if (out / f"riskmap_{l}.csv").exists():
            rm = pd.read_csv(out / f"riskmap_{l}.csv")
            rm["rate"] = np.where(rm.ok, rm.rate * 100, np.nan)
            rm["lrate"] = np.log10(np.clip(rm.rate, 0.1, 100))          # colour: log10 of the rate (%)
            rm[["mean_bin", "spread_q", "rate", "lrate", "n", "s2s_mean_ms", "bin_center_ms"]].to_csv(
                fig / f"fr_scheduler_risk_{l}.dat", sep=" ", index=False, float_format="%.4f", na_rep="nan")
    print(f"figure data written to {fig}")
    write_figures(allres, fig, Path(a.fig_tex_dir))


# ---------------------------------------------------------------------------
# pgfplots figures (generated, so a re-run with more methods updates them)
# ---------------------------------------------------------------------------

PANEL = {"starlink_dl": "a", "starlink_ul": "b", "oneweb_dl": "c", "oneweb_ul": "d"}
# zoom window (duplication %, low-duplication region) per link for the insets
ZOOM_X = {"starlink_dl": (0.0, 20.0), "starlink_ul": (2.0, 20.0), "oneweb_dl": (2.0, 20.0), "oneweb_ul": (4.0, 16.0)}
STYLES = r"""\pgfplotsset{
  schOracle/.style={black, dashed, thick, mark=none},
  schRandom/.style={black!45, thin, mark=none},
  schReact/.style={cPers, densely dotted, thick, mark=x, mark size=1.7pt, mark options={solid}, mark repeat=20, mark phase=11},
  schS2S/.style={mPrev, mark repeat=20, mark phase=11},
  schAdapt/.style={cResid, thick, dashed, mark=triangle, mark size=1.6pt, mark options={solid}, mark repeat=20, mark phase=11},
  schDS2S/.style={mFR, mark repeat=20, mark phase=11},
  schGauss/.style={cRidge, thick, dashdotted, mark=diamond, mark size=1.6pt, mark options={solid}, mark repeat=20, mark phase=11},
  schCalS2S/.style={only marks, mark=triangle*, mark size=2.4pt, mark options={draw=black, fill=cPrev, thin}},
  schCalDS2S/.style={only marks, mark=pentagon*, mark size=2.4pt, mark options={draw=black, fill=cFR, thin}},
  schRing/.style={only marks, mark=o, mark size=3.9pt, black, semithick, forget plot},
  schInset/.style={scale only axis, width=1.6cm, height=1.15cm, axis background/.style={fill=white},
    tick label style={font=\tiny}, xtick pos=bottom, ytick pos=left, grid=major, grid style={gray!20},
    tick align=inside, major tick length=1.5pt, clip=true},
}
"""


def _nice_ceil(v):
    for step in (0.5, 1, 2, 5, 10):
        if v / step <= 12:
            return math.ceil(v / step) * step
    return math.ceil(v)


def write_figures(allres, fig, texdir):
    links = [l for l in LINKS if l in allres and (fig / f"fr_scheduler_curve_{l}.dat").exists()]
    if not links:
        return
    texdir.mkdir(parents=True, exist_ok=True)
    # the Gauss curve is drawn (as "Gauss-E") only when every link has a full seed ensemble of it
    gauss = all("gauss" in pd.read_csv(fig / f"fr_scheduler_curve_{l}.dat", sep=" ", nrows=1).columns
                and allres[l].get("n_seeds", {}).get("gauss", 0) >= MIN_SEEDS_FIG for l in links)
    curves = [("schOracle", "oracle", "Oracle"), ("schRandom", "random", "Random"),
              ("schReact", "reactive", "Reactive"), ("schS2S", "s2s", "S2S-E"),
              ("schAdapt", "s2sadapt", "S2S-E adapt."), ]
    if gauss:
        curves.append(("schGauss", "gauss", "Gauss-E"))
    curves.append(("schDS2S", "ds2s", "DS2S-E"))
    cals = [("schCalS2S", "s2s", "S2S-E bound"), ("schCalDS2S", "ds2s", "DS2S-E bound")]
    i95 = LEVELS.index(0.95)
    ring = f"skip coords between index={{0}}{{{i95}}}, skip coords between index={{{i95 + 1}}}{{{len(LEVELS)}}}"
    ncol = len(curves) + len(cals) if len(curves) + len(cals) <= 8 else math.ceil((len(curves) + len(cals)) / 2)

    def plots(l, legend):
        s = []
        for sty, col, lab in curves:
            s.append(f"\\addplot[{sty}] table[x=dup, y={col}] {{data/fr_scheduler_curve_{l}.dat}};"
                     + (f" \\addlegendentry{{{lab}}}" if legend else ""))
        for sty, col, lab in cals:
            s.append(f"\\addplot[{sty}] table[x={col}dup, y={col}miss] {{data/fr_scheduler_cal_{l}.dat}};"
                     + (f" \\addlegendentry{{{lab}}}" if legend else ""))
        for sty, col, lab in cals:   # ring around the 95% level
            s.append(f"\\addplot[schRing, {ring}] table[x={col}dup, y={col}miss] {{data/fr_scheduler_cal_{l}.dat}};")
        return "\n".join(s)

    out = ["% fr_scheduler.tex - generated by modeling/experiments/fr_scheduler.py (edit the script, not this file).",
           "% Deadline-aware duplication on the test sessions: duplication rate vs deadline-miss rate, D = validation p90.",
           STYLES.strip(),
           "\\begin{tikzpicture}",
           "\\begin{groupplot}[group style={group size=4 by 1, horizontal sep=1.15cm}, frbase, scale only axis,",
           "  width=3.45cm, height=3.0cm, xmin=0, xmax=100, xtick={0,25,50,75,100}, ymin=0,",
           "  xlabel={duplication rate (\\%)}, title style={font=\\footnotesize, yshift=-0.6ex}]"]
    zooms = {}
    for i, l in enumerate(links):
        r = allres[l]["deadlines"]["D90"]
        p = r["p_test"] * 100
        ymax = _nice_ceil(1.06 * p)
        cv = pd.read_csv(fig / f"fr_scheduler_curve_{l}.dat", sep=" ")
        xa, xb = ZOOM_X[l]
        fc = [c for c in ("reactive", "s2s", "s2sadapt", "ds2s") + (("gauss",) if gauss else ()) if c in cv.columns]
        win = cv[(cv.dup >= xa) & (cv.dup <= xb)][fc]
        lo, hi = float(win.min().min()), float(win.max().max())
        pad = 0.06 * (hi - lo)
        lo, hi = max(0.0, lo - pad), min(ymax, hi + pad)
        zooms[l] = (xa, xb, lo, hi, ymax)
        opts = [f"title={{({PANEL[l]}) {NICE[l]}, $D={r['D_ms']:.0f}$\\,ms}}", f"ymax={ymax:g}"]
        if i == 0:
            opts += ["ylabel={deadline-miss rate (\\%)}", "legend to name=legsched", f"legend columns={ncol}",
                     "legend cell align=left",
                     "legend style={font=\\scriptsize, draw=black!60, /tikz/every even column/.append style={column sep=0.22cm}}"]
        out.append(f"\\nextgroupplot[{', '.join(opts)}]")
        out.append(plots(l, legend=(i == 0)))
        ym = 0.5 * (lo + hi)
        out.append(f"\\draw[black!65, thin] (axis cs:{xa:g},{lo:.3f}) rectangle (axis cs:{xb:g},{hi:.3f});")
        out.append(f"\\coordinate (zsrc{PANEL[l]}) at (axis cs:{xb:g},{ym:.3f});")
    out.append("\\end{groupplot}")
    for i, l in enumerate(links):
        xa, xb, lo, hi, ymax = zooms[l]
        c = f"c{i + 1}r1"
        out.append(f"\\begin{{axis}}[schInset, name=ins{PANEL[l]}, at={{($(group {c}.north east)+(-0.08cm,-0.08cm)$)}}, "
                   f"anchor=north east, xmin={xa:g}, xmax={xb:g}, ymin={lo:.3f}, ymax={hi:.3f}]")
        out.append(plots(l, legend=False))
        out.append("\\end{axis}")
        out.append(f"\\draw[-latex, black!65, thin] (zsrc{PANEL[l]}) -- (ins{PANEL[l]}.south west);")
    out.append("\\path ($(group c1r1.north west)!0.5!(group c4r1.north east)$) ++(0,0.62cm) "
               "node[anchor=south] {\\pgfplotslegendfromname{legsched}};")
    out.append("\\end{tikzpicture}")
    (texdir / "fr_scheduler.tex").write_text("\n".join(out) + "\n")

    # heat map: duplication saved by DS2S-E against S2S-E at 90% recall, deadline percentile x step
    vals = pd.concat([pd.read_csv(fig / f"fr_scheduler_heat_{l}.dat", sep=" ") for l in links])
    vmin, vmax = math.floor(min(0.0, vals.savingc.min()) / 10) * 10, math.ceil(vals.savingc.max() / 10) * 10
    h = ["% fr_scheduler_heat.tex - generated by modeling/experiments/fr_scheduler.py (edit the script, not this file).",
         "% Duplication saved by DS2S-E against S2S-E (equal miss rate, 90% of late packets protected).",
         "\\begin{tikzpicture}",
         "\\begin{groupplot}[group style={group size=4 by 1, horizontal sep=0.35cm, y descriptions at=edge left},",
         "  frbase, scale only axis, width=3.45cm, height=2.2cm, grid=none, enlargelimits=false,",
         "  xmin=0.1, xmax=3.1, ymin=-0.5, ymax=5.5, xtick={0.2,1,2,3}, xlabel={lead time $h$ (s)},",
         "  ytick={0,1,2,3,4,5}, yticklabels={80,85,90,95,97.5,99}, ylabel={deadline (val.\\ pct.)},",
         f"  colormap/viridis, point meta min={vmin:g}, point meta max={vmax:g},",
         "  title style={font=\\footnotesize, yshift=-0.6ex}]"]
    for i, l in enumerate(links):
        opts = [f"title={{({PANEL[l]}) {NICE[l]}}}"]
        if i == len(links) - 1:
            opts += ["colorbar", "colorbar style={width=0.18cm, ytick distance=20, tick label style={font=\\scriptsize}, "
                     "ylabel={duplication saved (\\%)}, ylabel style={font=\\scriptsize}}"]
        h.append(f"\\nextgroupplot[{', '.join(opts)}]")
        h.append(f"\\addplot[matrix plot*, mesh/cols=15, point meta=explicit] table[x=t, y=row, meta=savingc] "
                 f"{{data/fr_scheduler_heat_{l}.dat}};")
    h += ["\\end{groupplot}", "\\end{tikzpicture}"]
    (texdir / "fr_scheduler_heat.tex").write_text("\n".join(h) + "\n")

    # risk map: violation rate by S2S-E mean bin (x) and DS2S-E spread quintile inside the bin (y)
    rlinks = [l for l in links if (fig / f"fr_scheduler_risk_{l}.dat").exists()]
    if rlinks:
        rv = pd.concat([pd.read_csv(fig / f"fr_scheduler_risk_{l}.dat", sep=" ") for l in rlinks])
        nb = RISK_MEAN_BINS
        g = ["% fr_scheduler_risk.tex - generated by modeling/experiments/fr_scheduler.py (edit the script, not this file).",
             "% Test deadline-violation rate (D = validation p90) per S2S-E mean bin (x) and DS2S-E spread quintile",
             f"% inside that bin (y). Gray: fewer than {RISK_MIN} decisions.",
             "\\begin{tikzpicture}",
             "\\begin{groupplot}[group style={group size=4 by 1, horizontal sep=0.45cm, y descriptions at=edge left},",
             "  frbase, scale only axis, width=3.3cm, height=2.0cm, grid=none, enlargelimits=false,",
             "  axis background/.style={fill=gray!25}, tick style={draw=none},",
             f"  xmin=0.5, xmax={nb + 0.5}, ymin=0.5, ymax=5.5, ytick={{1,3,5}}, yticklabels={{low,mid,high}},",
             "  xlabel={S2S-E mean forecast (ms)}, ylabel={DS2S-E spread},",
             "  colormap/viridis, point meta min=-1, point meta max=2,",
             "  title style={font=\\footnotesize, yshift=-0.6ex}]"]
        for i, l in enumerate(rlinks):
            r = allres[l]["deadlines"]["D90"]
            rm = pd.read_csv(fig / f"fr_scheduler_risk_{l}.dat", sep=" ")
            ctr = rm.groupby("mean_bin").bin_center_ms.first()
            ticks = [1, (nb + 1) // 2, nb]
            lab = ",".join(f"{ctr[t]:.0f}" for t in ticks)
            opts = [f"title={{({PANEL[l]}) {NICE[l]}}}",
                    f"xtick={{{','.join(str(t) for t in ticks)}}}", f"xticklabels={{{lab}}}"]
            if i == len(rlinks) - 1:
                opts += ["colorbar", "colorbar style={width=0.18cm, tick label style={font=\\scriptsize}, "
                         "ytick={-1,0,1,2}, yticklabels={0.1,1,10,100}, "
                         "ylabel={late packets (\\%)}, ylabel style={font=\\scriptsize}}"]
            g.append(f"\\nextgroupplot[{', '.join(opts)}]")
            g.append(f"\\addplot[matrix plot*, mesh/cols={nb}, point meta=explicit] table[x=mean_bin, y=spread_q, meta=lrate] "
                     f"{{data/fr_scheduler_risk_{l}.dat}};")
        g += ["\\end{groupplot}", "\\end{tikzpicture}"]
        (texdir / "fr_scheduler_risk.tex").write_text("\n".join(g) + "\n")
    print(f"figures written to {texdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--links", nargs="+", default=LINKS)
    ap.add_argument("--fr_method", default="fr_dist", help="distributional model scored as DS2S-E")
    ap.add_argument("--res_dir", default=str(ROOT / "results_final/main"))
    ap.add_argument("--data_root", default=str(ROOT / "data/lsn-data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results_final/analysis/scheduler"))
    ap.add_argument("--fig_dir", default=str(ROOT.parent / "reports/final_report/data"))
    ap.add_argument("--fig_tex_dir", default=str(ROOT.parent / "reports/final_report/figs"))
    ap.add_argument("--n_boot", type=int, default=1000, help="session bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary_only", action="store_true")
    ap.add_argument("--write_fig", action="store_true", help="write figure data for a non-default --fr_method")
    a = ap.parse_args()
    out = Path(a.out_dir) if a.fr_method == "fr_dist" else Path(a.out_dir) / a.fr_method
    out.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        for l in a.links:
            print(f"=== {l} ({a.fr_method}) ===", flush=True)
            analyse_link(l, a, out)
    summarise(a, out, Path(a.fig_dir))


if __name__ == "__main__":
    main()
