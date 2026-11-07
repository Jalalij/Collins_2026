"""
experiments/fr_condcov.py - Conditional calibration of the 95% RTT intervals.

Question
--------
A 95% interval that covers 95% of ALL test windows can still fail the windows that
matter. If its width ignores the state of the link it covers almost every calm
window (the band is wider than needed) and too few volatile windows (the band is
too narrow exactly when the application needs a warning). Average coverage hides
this. We therefore bin the test windows by how volatile their 6 s input was and
measure coverage and width inside every bin and at every horizon step.

Conditioning variable
---------------------
vol(x) = sample standard deviation (ddof = 1) of the 30 input RTT values of the
window, in ms, model-input view (load_splits(..., "base"): interpolated Starlink /
5G pings held at the last real ping). Same definition as fr_volatility.py.
Decile edges come from the VALIDATION windows only (np.quantile over all
validation windows, bins by np.digitize on the 9 inner edges), then are applied
to the TEST windows. Decile 1 = calmest tenth, decile 10 = most volatile tenth.

Intervals (RTT, alpha = 0.05, every offset learned on validation windows only)
---------------------------------------------------------------------------
s2s_const   S2S-E point forecast (mean of the prev_seq2seq seed forecasts) +
            split conformal: half-width q_h = conformal quantile of |y - yhat|
            per horizon step h. Constant width for every window.
ds2s_cqr    DS2S-E (mean of the --fr_method seed quantiles) + CQR (Romano et al.
            2019): score max(q.025 - y, y - q.975), interval [q.025 - c_h, q.975 + c_h].
s2s_norm    S2S-E + locally adaptive (normalised) split conformal: score
            |y - yhat| / s(x), s(x) = vol(x) + eps, interval yhat +- c_h s(x).
            eps = --eps_frac x median(validation vol), a small floor that keeps a
            flat input window from getting a zero-width band. A cheap
            volatility-scaled baseline: it is told the binning variable.
s2s_normtuned  the same with eps picked on validation from EPS_GRID x median(vol)
            by the validation interval score (strongest version of the baseline).
gauss_cqr   fr_gauss ensemble (Gaussian mean-variance heads) + CQR, if trained.
w4_cqr      fr_dist_w4 ensemble + CQR, if trained.
lqr_cqr     lqr_fr (linear quantile model, one fit) + CQR, if trained.
ds2s_cqrr   DS2S-E + CQR-r (Sesia and Candes 2020): score max(q.025 - y, y - q.975) / w,
            w = q.975 - q.025, interval [q.025 - c_h w, q.975 + c_h w] (width-scaled offset).
ds2s_raw    DS2S-E raw [q.025, q.975], no conformal step (reference only).
s2s_mondrian S2S-E + Mondrian (per-volatility-decile) split conformal: one
            half-width per (decile, h) from the validation windows of that decile.
            Equalises decile coverage BY CONSTRUCTION (reference only; it needs the
            binning variable to be chosen by hand).
Decomposition of the DS2S-E interval-score gain (reference bands, CSV/JSON only):
ds2s_meanconst  constant split-conformal band around the DS2S-E MEAN forecast: a better
            centre only, no adaptive width.
s2s_ds2swidth   S2S-E centre +- (DS2S-E half-width (q.975 - q.025)/2 + c_h), score
            |y - yhat| - half-width: the DS2S-E width per window, but symmetric and around
            the S2S-E centre. ds2s_cqr minus s2s_ds2swidth = asymmetric placement of the band.
summary.json links.<link>.info.ds2s_cqr.asym_h<h_plot>: median upper / lower half-width of the
            raw DS2S-E band around its median at --h_plot (q.975 - q.5 vs q.5 - q.025), test windows.
Conformal quantile: np.quantile(scores, min(1, ceil((n+1)(1-alpha))/n), method="higher").

Per (method, decile, h): n, coverage, mean width, interval score
  IS = (hi - lo) + (2/alpha)(lo - y)+ + (2/alpha)(y - hi)+,
and the miss rates below / above the band.
Per (link, method): overall coverage, width and interval score; decile coverage
(pooled over h, which equals the mean over h because every window has all 15
steps); max and mean absolute deviation of decile coverage from 0.95; calm (d1)
and volatile (d10) coverage and width; worst (decile, h) cell. 95% CIs from a
session bootstrap of the TEST sessions (sessions resampled with replacement,
--n_boot, fixed seed; deciles and offsets stay fixed) and paired differences
between methods on the same resamples.

Outputs
-------
modeling/results_final/analysis/condcov/
    inputs_<link>.npz   cache of per-window input statistics (vol, last, sid) for
                        val and test; rebuilt with --recompute_inputs
    cells_<link>.csv    per (method, decile, h) numbers
    summary.csv         per (link, method) numbers incl. bootstrap CIs
    pairs.csv           paired bootstrap differences between methods
    summary.json        everything above + decile edges, eps, offsets, seeds used
reports/final_report/data/
    fr_condcov_heat_<link>.dat   150 rows: h, dec, cov_<m>, wid_<m> (heatmaps)
    fr_condcov_wid_<link>.dat    10 rows: dec, vol_med, wid_<m>, cov_<m> at --h_plot
    fr_condcov_tab.tex           LaTeX tabular: coverage gap and interval score per link and band
reports/final_report/figs/ (generated, so methods that finish training later appear after a re-run)
    fr_condcov.tex               figure*: coverage heatmaps (rows = band type, columns = --heat_links)
    fr_condwidth.tex             figure*: band width at --h_plot vs input-SD decile, 2 x 3 links, insets

Usage
-----
LSN_DEVICE=cpu python3 experiments/fr_condcov.py                    # all links
LSN_DEVICE=cpu python3 experiments/fr_condcov.py --links starlink_dl
LSN_DEVICE=cpu python3 experiments/fr_condcov.py --fr_method fr_dist_w4
LSN_DEVICE=cpu python3 experiments/fr_condcov.py --tex_only          # rebuild table and figures
Deterministic. No training, CPU only. Methods whose prediction files are missing
(still training) are skipped and picked up on the next run. Loading the 5G input
windows takes a few minutes on an idle machine the first time; afterwards the
cache inputs_<link>.npz is used and a full run takes about a minute.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("LSN_DEVICE", "cpu")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

LINKS = ["oneweb_dl", "oneweb_ul", "starlink_dl", "starlink_ul", "quectel_dl", "quectel_ul"]
NAMES = {"oneweb_dl": "OneWeb DL", "oneweb_ul": "OneWeb UL", "starlink_dl": "Starlink DL",
         "starlink_ul": "Starlink UL", "quectel_dl": "5G DL", "quectel_ul": "5G UL"}
L_IN, H_OUT, K_RTT, NDEC = 30, 15, 0, 10
# method key -> label used in summary.csv
LABELS = {"s2s_const": "S2S-E + conformal", "ds2s_cqr": "DS2S-E + CQR",
          "s2s_asym": "S2S-E + asymmetric conformal (constant width, alpha/2 per tail)",
          "frfull_const": "S2S-AE-E (new inputs + anchored decoder, scalar output) + conformal",
          "s2s_norm": "S2S-E + normalised conformal",
          "s2s_normtuned": "S2S-E + normalised conformal (tuned eps)", "gauss_cqr": "Gauss-E + CQR",
          "w4_cqr": "DS2S-E (w4) + CQR", "lqr_cqr": "LQR + CQR",
          "ds2s_cqrr": "DS2S-E + CQR-r (width-scaled offset)",
          "ds2s_raw": "DS2S-E raw quantiles", "s2s_mondrian": "S2S-E + Mondrian conformal",
          "ds2s_meanconst": "DS2S-E mean + conformal (constant band, decomposition)",
          "s2s_ds2swidth": "S2S-E centre + DS2S-E width, symmetric (decomposition)"}
ORDER = list(LABELS)
EPS_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)   # tuned-eps grid, x median(vol_va)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def conf_q(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Split-conformal quantile along axis 0 (finite-sample corrected, 'higher')."""
    n = scores.shape[0]
    return np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), axis=0, method="higher")


def load_preds(pdir: Path, method: str, min_seeds: int, max_seeds: int = 3):
    """Seed-averaged forecasts (mean and each quantile) of one method, or None.

    Only exact file names <method>_s<k>.npz, k < max_seeds, are used, so
    fr_dist never picks up fr_dist_w4 files. A file that cannot be read (for
    example one a training job is still writing) makes the method count as
    missing for this run."""
    fs = [pdir / f"{method}_s{k}.npz" for k in range(max_seeds)]
    fs = [f for f in fs if f.exists()]
    if len(fs) < min_seeds:
        return None
    try:
        zs = [np.load(f) for f in fs]
        out = {"p_te": np.mean([z["p_te"][:, :, K_RTT] for z in zs], 0).astype(np.float64),
               "p_va": np.mean([z["p_va"][:, :, K_RTT] for z in zs], 0).astype(np.float64),
               "n": len(fs), "files": [f.name for f in fs]}
        if "d_te" in zs[0].files:
            out["d_te"] = np.mean([z["d_te"][:, :, K_RTT].astype(np.float64) for z in zs], 0)
            out["d_va"] = np.mean([z["d_va"][:, :, K_RTT].astype(np.float64) for z in zs], 0)
    except Exception as exc:  # noqa: BLE001 - partially written file of a running job
        print(f"  [skip] {method}: could not read ({exc})")
        return None
    return out


def input_stats(link: str, cache: Path, data_root: str, recompute: bool) -> dict:
    """Per-window input statistics for the validation and test windows (cached)."""
    f = cache / f"inputs_{link}.npz"
    if f.exists() and not recompute:
        return dict(np.load(f))
    import torch
    torch.set_num_threads(2)
    from final_benchmark import load_splits
    prov, d = link.split("_")
    t0 = time.time()
    data, meta = load_splits(prov, d, "base", L_IN, H_OUT, torch.device("cpu"), data_root)
    ti = meta["target_idx"][K_RTT]
    out = {}
    for sp, tag in (("val", "va"), ("test", "te")):
        S = data[sp]
        X = S.X.cpu().numpy()
        st = S.starts.cpu().numpy()
        rtt = meta["tmean"][K_RTT] + meta["tstd"][K_RTT] * X[:, ti].astype(np.float64)
        W = rtt[st[:, None] + np.arange(L_IN)[None]]
        out[f"vol_{tag}"] = W.std(1, ddof=1)
        out[f"last_{tag}"] = W[:, -1]
        out[f"sid_{tag}"] = np.asarray(S.sid)[st].astype(np.int64)
        assert (np.asarray(S.sid)[st] == np.asarray(S.sid)[st + L_IN + H_OUT - 1]).all(), \
            "a window crosses a session boundary"
    del data
    np.savez_compressed(f, **out)
    print(f"  [inputs] {link}: loaded and cached in {time.time() - t0:.0f}s")
    return out


def band_metrics(lo, hi, y, alpha):
    """Per window and step: hit, width, interval score, miss below, miss above."""
    below, above = y < lo, y > hi
    w = hi - lo
    isc = w + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
    return {"hit": (~below & ~above).astype(np.float64), "wid": w, "is": isc,
            "mlo": below.astype(np.float64), "mhi": above.astype(np.float64)}


def per_session_sums(m: dict, sess: np.ndarray, dec: np.ndarray, S: int):
    """Sum every per-window quantity over (session, decile): arrays (S, NDEC, H)."""
    idx = sess * NDEC + dec
    out = {}
    for k, v in m.items():
        a = np.zeros((S * NDEC, v.shape[1]))
        np.add.at(a, idx, v)
        out[k] = a.reshape(S, NDEC, v.shape[1])
    out["n"] = np.bincount(idx, minlength=S * NDEC).reshape(S, NDEC).astype(np.float64)
    return out


def stats_from_sums(sums: dict, w: np.ndarray | None = None, target=0.95):
    """Summary statistics from session sums; w (S,) are bootstrap session weights."""
    if w is None:
        w = np.ones(sums["n"].shape[0])
    n_d = w @ sums["n"]                                       # (NDEC,)
    hit_dh = np.einsum("s,sdh->dh", w, sums["hit"])
    wid_dh = np.einsum("s,sdh->dh", w, sums["wid"])
    is_dh = np.einsum("s,sdh->dh", w, sums["is"])
    H = hit_dh.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov_d = hit_dh.sum(1) / (n_d * H)
        wid_d = wid_dh.sum(1) / (n_d * H)
        ntot = n_d.sum() * H
        dev = np.abs(cov_d - target)
        return {"picp": hit_dh.sum() / ntot, "width": wid_dh.sum() / ntot, "mis": is_dh.sum() / ntot,
                "maxdev": np.nanmax(dev), "meandev": np.nanmean(dev),
                "cov_d1": cov_d[0], "cov_d10": cov_d[-1], "wid_d1": wid_d[0], "wid_d10": wid_d[-1],
                "cov_d": cov_d, "wid_d": wid_d}


# ---------------------------------------------------------------------------
# One link
# ---------------------------------------------------------------------------

def run_link(link: str, a, res_dir: Path, out_dir: Path, fig_dir: Path) -> dict | None:
    pdir = res_dir / link / "preds"
    if not (pdir / "truth.npz").exists():
        print(f"[{link}] no truth.npz, skipped")
        return None
    t0 = time.time()
    tr = np.load(pdir / "truth.npz")
    y_te = tr["y_te"][:, :, K_RTT].astype(np.float64)
    y_va = tr["y_va"][:, :, K_RTT].astype(np.float64)
    inp = input_stats(link, out_dir, a.data_root, a.recompute_inputs)
    assert len(inp["vol_te"]) == len(y_te) and len(inp["vol_va"]) == len(y_va), "window count mismatch"
    # The cached last input value must equal the saved persistence forecast: same windows, same order.
    assert np.allclose(inp["last_te"], tr["pers_te"][:, 0, K_RTT], atol=1e-2), "window order mismatch"

    vol_va, vol_te = inp["vol_va"], inp["vol_te"]
    edges = np.quantile(vol_va, np.linspace(0, 1, NDEC + 1))
    if len(np.unique(edges[1:-1])) < NDEC - 1:
        print(f"  [warn] {link}: tied decile edges, some deciles are empty")
    dec_va = np.digitize(vol_va, edges[1:-1])                 # 0..9, validation edges only
    dec_te = np.digitize(vol_te, edges[1:-1])
    eps = a.eps_frac * float(np.median(vol_va))
    s_va, s_te = vol_va + eps, vol_te + eps

    alpha = a.alpha
    bands, info = {}, {}
    s2s = load_preds(pdir, "prev_seq2seq", a.min_seeds)
    if s2s is not None:
        q = conf_q(np.abs(y_va - s2s["p_va"]), alpha)                         # (H,)
        bands["s2s_const"] = (s2s["p_te"] - q, s2s["p_te"] + q)
        info["s2s_const"] = {"offset_h": q.tolist(), "seeds": s2s["files"]}
        # Asymmetric constant band: split-conformal quantiles of the signed residual, alpha/2 per tail.
        r_va = y_va - s2s["p_va"]
        qlo, qhi = conf_q(-r_va, alpha / 2), conf_q(r_va, alpha / 2)
        bands["s2s_asym"] = (s2s["p_te"] - qlo, s2s["p_te"] + qhi)
        info["s2s_asym"] = {"offset_lo_h": qlo.tolist(), "offset_hi_h": qhi.tolist(), "seeds": s2s["files"]}
        c = conf_q(np.abs(y_va - s2s["p_va"]) / s_va[:, None], alpha)
        bands["s2s_norm"] = (s2s["p_te"] - c * s_te[:, None], s2s["p_te"] + c * s_te[:, None])
        info["s2s_norm"] = {"offset_h": c.tolist(), "eps_ms": eps, "seeds": s2s["files"]}
        # Strongest version of the same baseline: eps picked from EPS_GRID x median(vol_va) by the
        # validation interval score (offset and choice both on validation windows).
        med, best = float(np.median(vol_va)), None
        for f in EPS_GRID:
            sv = vol_va[:, None] + f * med
            cc = conf_q(np.abs(y_va - s2s["p_va"]) / sv, alpha)
            isv = band_metrics(s2s["p_va"] - cc * sv, s2s["p_va"] + cc * sv, y_va, alpha)["is"].mean()
            if best is None or isv < best[0]:
                best = (isv, f, cc)
        _, f, cc = best
        st_ = vol_te[:, None] + f * med
        bands["s2s_normtuned"] = (s2s["p_te"] - cc * st_, s2s["p_te"] + cc * st_)
        info["s2s_normtuned"] = {"offset_h": cc.tolist(), "eps_frac": f, "eps_ms": f * med, "seeds": s2s["files"]}
        qm = np.zeros((NDEC, H_OUT))
        for d in range(NDEC):
            md = dec_va == d
            qm[d] = conf_q(np.abs(y_va[md] - s2s["p_va"][md]), alpha) if md.sum() > 0 else np.nan
        hw = qm[dec_te]                                                        # (N, H)
        bands["s2s_mondrian"] = (s2s["p_te"] - hw, s2s["p_te"] + hw)
        info["s2s_mondrian"] = {"offset_dh": qm.tolist(), "seeds": s2s["files"]}

    # Same inputs and anchored decoder as DS2S, scalar squared-error output (ablation S2S-AE),
    # with the constant conformal band: separates the new inputs/decoder from the distribution.
    ff = load_preds(pdir, "fr_full", a.min_seeds)
    if ff is not None:
        q = conf_q(np.abs(y_va - ff["p_va"]), alpha)
        bands["frfull_const"] = (ff["p_te"] - q, ff["p_te"] + q)
        info["frfull_const"] = {"offset_h": q.tolist(), "seeds": ff["files"]}

    def cqr(key, method, min_seeds):
        m = load_preds(pdir, method, min_seeds)
        if m is None or "d_te" not in m:
            return None
        lo_va, hi_va = m["d_va"][..., 1], m["d_va"][..., 5]
        c = conf_q(np.maximum(lo_va - y_va, y_va - hi_va), alpha)
        bands[key] = (m["d_te"][..., 1] - c, m["d_te"][..., 5] + c)
        info[key] = {"offset_h": c.tolist(), "seeds": m["files"], "method": method}
        return m

    dm = cqr("ds2s_cqr", a.fr_method, a.min_seeds)
    if dm is not None:
        bands["ds2s_raw"] = (dm["d_te"][..., 1], dm["d_te"][..., 5])
        info["ds2s_raw"] = {"seeds": dm["files"], "method": a.fr_method}
        # CQR-r (Sesia and Candes 2020): the offset is a multiple of the model's own band width,
        # so one number per step can widen a wide band a lot and a narrow band a little.
        lo_va, hi_va = dm["d_va"][..., 1], dm["d_va"][..., 5]
        w_va = np.maximum(hi_va - lo_va, 1e-6)
        c = conf_q(np.maximum(lo_va - y_va, y_va - hi_va) / w_va, alpha)
        lo_te, hi_te = dm["d_te"][..., 1], dm["d_te"][..., 5]
        w_te = np.maximum(hi_te - lo_te, 1e-6)
        bands["ds2s_cqrr"] = (lo_te - c * w_te, hi_te + c * w_te)
        info["ds2s_cqrr"] = {"offset_h": c.tolist(), "seeds": dm["files"], "method": a.fr_method}
        # Decomposition of the DS2S-E gain (offsets from validation windows only).
        # (1) better centre only: constant conformal band around the DS2S-E mean forecast.
        q = conf_q(np.abs(y_va - dm["p_va"]), alpha)
        bands["ds2s_meanconst"] = (dm["p_te"] - q, dm["p_te"] + q)
        info["ds2s_meanconst"] = {"offset_h": q.tolist(), "seeds": dm["files"], "method": a.fr_method}
        # (2) adaptive width only: DS2S-E half-width per window, symmetric around the S2S-E forecast.
        if s2s is not None:
            hv = (hi_va - lo_va) / 2
            ht = (hi_te - lo_te) / 2
            c = conf_q(np.abs(y_va - s2s["p_va"]) - hv, alpha)
            hw = np.maximum(ht + c, 0.0)          # never an inverted band (count stored below)
            bands["s2s_ds2swidth"] = (s2s["p_te"] - hw, s2s["p_te"] + hw)
            info["s2s_ds2swidth"] = {"offset_h": c.tolist(), "seeds": s2s["files"] + dm["files"],
                                     "method": a.fr_method, "n_cells_clamped_to_zero": int((ht + c < 0).sum())}
        # Shape of the raw DS2S-E band at h_plot: how far it reaches above vs below its median.
        dp = dm["d_te"][:, a.h_plot - 1]
        up, dn = dp[:, 5] - dp[:, 3], dp[:, 3] - dp[:, 1]
        info["ds2s_cqr"]["asym_h%d" % a.h_plot] = {
            "upper_halfwidth_median_ms": float(np.median(up)), "lower_halfwidth_median_ms": float(np.median(dn)),
            "upper_over_lower_median": float(np.median(up / np.maximum(dn, 1e-6)))}
    cqr("gauss_cqr", "fr_gauss", a.min_seeds)
    if a.fr_method != "fr_dist_w4":
        cqr("w4_cqr", "fr_dist_w4", a.min_seeds)
    cqr("lqr_cqr", "lqr_fr", 1)
    if not bands:
        print(f"[{link}] no methods available")
        return None

    # session index of each test window (0..S-1) for the bootstrap
    _, sess = np.unique(inp["sid_te"], return_inverse=True)
    S = int(sess.max()) + 1
    rng = np.random.default_rng(a.seed)
    Wb = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(a.n_boot)]).astype(np.float64)

    cells, summ, sums_all, boot_all = [], [], {}, {}
    keys = [k for k in ORDER if k in bands]
    for k in keys:
        lo, hi = bands[k]
        m = band_metrics(lo, hi, y_te, alpha)
        sums = per_session_sums(m, sess, dec_te, S)
        sums_all[k] = sums
        n_d = sums["n"].sum(0)
        for d in range(NDEC):
            for h in range(H_OUT):
                nd = n_d[d]
                cells.append({"link": link, "method": k, "decile": d + 1, "h": h + 1, "n": int(nd),
                              "coverage": sums["hit"][:, d, h].sum() / nd if nd else np.nan,
                              "width": sums["wid"][:, d, h].sum() / nd if nd else np.nan,
                              "interval_score": sums["is"][:, d, h].sum() / nd if nd else np.nan,
                              "miss_below": sums["mlo"][:, d, h].sum() / nd if nd else np.nan,
                              "miss_above": sums["mhi"][:, d, h].sum() / nd if nd else np.nan})
        st = stats_from_sums(sums, target=1 - alpha)
        boot = [stats_from_sums(sums, w, target=1 - alpha) for w in Wb]
        boot_all[k] = (st, boot)
        row = {"link": link, "method": k, "label": LABELS[k], "n_test": len(y_te), "n_sessions": S}
        for s in ("picp", "width", "mis", "maxdev", "meandev", "cov_d1", "cov_d10", "wid_d1", "wid_d10"):
            row[s] = float(st[s])
            bs = np.array([b[s] for b in boot])
            row[s + "_lo"], row[s + "_hi"] = (float(v) for v in np.nanquantile(bs, [0.025, 0.975]))
        row["wid_ratio_d10_d1"] = float(st["wid_d10"] / st["wid_d1"])
        cdh = np.array([[c["coverage"] for c in cells if c["method"] == k and c["decile"] == d + 1]
                        for d in range(NDEC)])
        row["worst_cell_cov"] = float(np.nanmin(cdh))
        row["frac_cells_below_0.90"] = float(np.mean(cdh < 0.90))
        for d in range(NDEC):
            row[f"cov_dec{d + 1}"] = float(st["cov_d"][d])
            row[f"wid_dec{d + 1}"] = float(st["wid_d"][d])
        summ.append(row)

    # Paired bootstrap differences (same session resamples for both methods)
    pairs = []
    for ka, kb in (("s2s_const", "ds2s_cqr"), ("s2s_norm", "ds2s_cqr"), ("s2s_const", "s2s_norm"),
                   ("s2s_asym", "ds2s_cqr"), ("frfull_const", "ds2s_cqr"), ("w4_cqr", "ds2s_cqr"),
                   ("s2s_normtuned", "ds2s_cqr"),
                   ("ds2s_cqr", "gauss_cqr"), ("ds2s_cqr", "w4_cqr"), ("s2s_mondrian", "ds2s_cqr"),
                   ("ds2s_cqr", "lqr_cqr"), ("ds2s_cqr", "ds2s_cqrr"),
                   # decomposition: centre only, width only, asymmetric placement
                   ("s2s_const", "ds2s_meanconst"), ("s2s_const", "s2s_ds2swidth"), ("s2s_ds2swidth", "ds2s_cqr")):
        if ka not in boot_all or kb not in boot_all:
            continue
        (sa, ba), (sb, bb) = boot_all[ka], boot_all[kb]
        for s in ("maxdev", "meandev", "mis", "cov_d10", "cov_d1", "width"):
            pa, pb = sa[s], sb[s]
            diff = np.array([x[s] - z[s] for x, z in zip(ba, bb)])
            pairs.append({"link": link, "a": ka, "b": kb, "stat": s, "a_value": float(pa), "b_value": float(pb),
                          "diff_a_minus_b": float(pa - pb),
                          "diff_lo": float(np.nanquantile(diff, 0.025)), "diff_hi": float(np.nanquantile(diff, 0.975)),
                          "frac_boot_a_gt_b": float(np.mean(diff > 0))})

    cdf = pd.DataFrame(cells)
    cdf.to_csv(out_dir / f"cells_{link}.csv", index=False)

    # Figure data: heatmap (dec outer, h inner -> mesh/cols = 15) and width at h_plot
    heat = pd.DataFrame({"h": np.tile(np.arange(1, H_OUT + 1), NDEC), "dec": np.repeat(np.arange(1, NDEC + 1), H_OUT)})
    for k in keys:
        sub = cdf[cdf.method == k].sort_values(["decile", "h"])
        heat["cov" + k.replace("_", "")] = sub.coverage.values
        heat["wid" + k.replace("_", "")] = sub.width.values
    heat.to_csv(fig_dir / f"fr_condcov_heat_{link}.dat", sep=" ", index=False, float_format="%.4f")
    hp = a.h_plot
    wdf = pd.DataFrame({"dec": np.arange(1, NDEC + 1),
                        "volmed": [float(np.median(vol_va[dec_va == d])) for d in range(NDEC)]})
    for k in keys:
        sub = cdf[(cdf.method == k) & (cdf.h == hp)].sort_values("decile")
        wdf["wid" + k.replace("_", "")] = sub.width.values
        wdf["cov" + k.replace("_", "")] = sub.coverage.values
    wdf.to_csv(fig_dir / f"fr_condcov_wid_{link}.dat", sep=" ", index=False, float_format="%.4f")

    print(f"[{link}] {len(y_te)} test windows, {S} test sessions, eps {eps:.3f} ms, "
          f"methods {keys} ({time.time() - t0:.0f}s)")
    print(pd.DataFrame(summ)[["method", "picp", "width", "mis", "maxdev", "cov_d1", "cov_d10",
                              "wid_d1", "wid_d10"]].round(3).to_string(index=False))
    return {"summary": summ, "pairs": pairs,
            "edges_ms": edges.tolist(), "eps_ms": eps,
            "val_decile_counts": np.bincount(dec_va, minlength=NDEC).tolist(),
            "test_decile_counts": np.bincount(dec_te, minlength=NDEC).tolist(),
            "test_decile_vol_median_ms": [float(np.median(vol_te[dec_te == d])) if (dec_te == d).any() else None
                                          for d in range(NDEC)],
            "info": info}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Table and figure .tex (regenerated on every run, so methods that finish
# training later appear after a re-run without hand edits)
# ---------------------------------------------------------------------------

# Table columns: (key, header, family). Family 0 = S2S-E point forecast + conformal band,
# family 1 = quantile forecast + CQR. Missing methods (still training) are left out.
TAB_COLS = [("s2s_const", "const.", 0), ("s2s_norm", "SD", 0), ("s2s_normtuned", "SD-tuned", 0),
            ("s2s_mondrian", r"per-dec.$^\dagger$", 0),
            ("lqr_cqr", "LQR", 1), ("gauss_cqr", "Gauss-E", 1), ("ds2s_cqr", "DS2S-E", 1)]


def write_table(summary: pd.DataFrame, fig_dir: Path):
    """Full tabular, two row blocks (coverage gap, interval score), one row per link and one
    column per band. Bold = best per row among the bands that do not use the binning variable
    by hand (the per-decile band is a reference)."""
    cols = [c for c in TAB_COLS if c[0] in set(summary.method)]
    n0 = sum(1 for c in cols if c[2] == 0)
    n1 = len(cols) - n0
    nc = len(cols)
    lines = [r"% generated by modeling/experiments/fr_condcov.py (RTT, 95% bands, TEST windows)",
             r"% gap = max over input-SD deciles (edges from validation) of |coverage - 0.95|, coverage pooled over the 15 steps",
             r"% SD = band scaled by the input SD (small eps), SD-tuned = eps picked on validation,",
             r"% per-dec. = Mondrian conformal per input-SD decile (uses the binning variable, reference only)",
             r"\begin{tabular}{l" + "c" * nc + "}", r"\toprule",
             r" & \multicolumn{%d}{c}{S2S-E point forecast + conformal} & \multicolumn{%d}{c}{quantile forecast + CQR} \\"
             % (n0, n1),
             r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (n0 + 1, n0 + 2, nc + 1),
             r"Link & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for stat, title, fmt in (("maxdev", r"coverage gap $\max_d|\mathrm{cov}_d-0.95|$", "{:.3f}"),
                             ("mis", r"interval score (ms)", None)):
        lines.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (nc + 1, title))
        for ln in LINKS:
            sub = summary[summary.link == ln].set_index("method")
            if sub.empty:
                continue
            fair = [c[0] for c in cols if c[0] in sub.index and c[0] != "s2s_mondrian"]
            best = sub.loc[fair, stat].min()
            cells = []
            for k, _, _ in cols:
                if k not in sub.index:
                    cells.append("--")
                    continue
                v = sub.loc[k, stat]
                t = fmt.format(v) if fmt else (f"{v:.1f}" if v < 1000 else f"{v:.0f}")
                if k != "s2s_mondrian" and np.isclose(v, best):
                    t = r"\textbf{" + t + "}"
                cells.append(t)
            lines.append(NAMES[ln] + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (fig_dir / "fr_condcov_tab.tex").write_text("\n".join(lines) + "\n")


HEAT_ROWS = [("s2s_const", r"\textbf{S2S-E}\\constant band"),
             ("ds2s_cqr", r"\textbf{DS2S-E}\\CQR band"),
             ("s2s_norm", r"\textbf{S2S-E}\\SD-scaled band")]


def write_heat_tex(fig_dir: Path, tex_dir: Path, links: list[str]):
    """figure* heatmaps: rows = band type, columns = links, cell = coverage per (decile, step)."""
    dats = {ln: fig_dir / f"fr_condcov_heat_{ln}.dat" for ln in links}
    if not all(f.exists() for f in dats.values()):
        print("[tex] fr_condcov.tex not written: heatmap data missing for", [ln for ln, f in dats.items() if not f.exists()])
        return
    have = {ln: set(pd.read_csv(f, sep=" ", nrows=1).columns) for ln, f in dats.items()}
    rows = [(k, lab) for k, lab in HEAT_ROWS if all("cov" + k.replace("_", "") in have[ln] for ln in links)]
    nl, nr = len(links), len(rows)
    o = [r"% fr_condcov.tex - generated by modeling/experiments/fr_condcov.py, do not edit by hand.",
         r"% Conditional coverage of the calibrated 95% RTT bands (figure*, full text width).",
         r"% Cell = share of TEST windows whose RTT at that horizon step fell inside the band, among the",
         r"% windows whose 6 s input lies in that input-volatility decile (edges from validation windows).",
         r"% White = nominal 0.95, blue = over-coverage (band wider than needed), red = under-coverage.",
         r"% Values below 0.85 share the darkest red.",
         r"\begin{tikzpicture}",
         r"\begin{groupplot}[",
         r"  group style={group size=%d by %d, horizontal sep=0.28cm, vertical sep=0.62cm," % (nl, nr),
         r"    xticklabels at=edge bottom, yticklabels at=edge left, xlabels at=edge bottom, ylabels at=edge left},",
         r"  scale only axis, width=4.75cm, height=2.2cm, enlargelimits=false, axis on top,",
         r"  xmin=0.5, xmax=%g, ymin=0.5, ymax=%g," % (H_OUT + 0.5, NDEC + 0.5),
         r"  xtick={1,5,10,15}, xticklabels={0.2,1,2,3}, ytick={1,5,10}, minor ytick={2,3,4,6,7,8,9},",
         r"  tick label style={font=\scriptsize}, label style={font=\scriptsize},",
         r"  title style={font=\footnotesize, yshift=-0.9ex}, ylabel style={align=center, yshift=-0.5ex},",
         r"  xlabel={horizon (s)},",
         r"  colormap={covdiv}{rgb255(0cm)=(103,0,31); rgb255(1cm)=(178,24,43); rgb255(1.5cm)=(239,138,98);",
         r"    rgb255(2cm)=(247,247,247); rgb255(2.5cm)=(146,197,222); rgb255(3cm)=(33,102,172)},",
         r"  point meta min=0.85, point meta max=1.0,",
         r"]"]
    letters = "abcdefghijklmnopqrstuvwxyz"
    for r, (k, lab) in enumerate(rows):
        for c, ln in enumerate(links):
            opts = [r"title={(%s) %s}" % (letters[r * nl + c], NAMES[ln])]
            if c == 0:
                opts.append(r"ylabel={input-SD decile}")
            if r == 0 and c == 0:
                opts.append(r"colorbar horizontal, colorbar to name=cbcondcov, colorbar style={width=6.2cm, height=0.2cm, "
                            r"xtick={0.85,0.9,0.95,1}, tick label style={font=\scriptsize}, xticklabel style={/pgf/number format/fixed, "
                            r"/pgf/number format/fixed zerofill, /pgf/number format/precision=2}}")
            o.append(r"\nextgroupplot[" + ", ".join(opts) + "]")
            o.append(r"\addplot[matrix plot*, mesh/cols=%d, point meta=explicit] table[x=h, y=dec, meta=cov%s] {data/fr_condcov_heat_%s.dat};"
                     % (H_OUT, k.replace("_", ""), ln))
    mid = (nl + 1) // 2
    o.append(r"\end{groupplot}")
    for r, (k, lab) in enumerate(rows):   # row label: how the band is built
        o.append(r"\node[rotate=90, anchor=south, align=center, text width=2.4cm, font=\scriptsize] "
                 r"at ($(group c1r%d.west)+(-1.05cm,0)$) {%s};" % (r + 1, lab))
    o += [
          r"\node[draw=black!50, fill=white, inner sep=3pt, anchor=south] at ($(group c%dr1.north)+(0,0.62cm)$)" % mid,
          r"  {\scriptsize coverage of the nominal 95\% band\quad\raisebox{-0.5ex}{\pgfplotscolorbarfromname{cbcondcov}}\quad",
          r"   \begin{tabular}{@{}l@{}}red: band too narrow\\[-1pt] blue: band too wide\end{tabular}};",
          r"\end{tikzpicture}"]
    (tex_dir / "fr_condcov.tex").write_text("\n".join(o) + "\n")
    print(f"[tex] {tex_dir / 'fr_condcov.tex'} ({nr} rows x {nl} links)")


# key -> (pgfplots style, legend entry). Drawn in this order (DS2S-E last, on top).
WIDTH_LINES = [
    ("s2s_mondrian", r"black!55, thin, densely dashdotted, mark=x, mark size=1.6pt, mark options={solid}",
     r"S2S-E, per-decile conformal (ref.)"),
    ("s2s_norm", r"cPrev, densely dashed, thick, mark=triangle, mark size=1.6pt, mark options={solid}",
     r"S2S-E, SD-scaled"),
    ("s2s_const", r"mPrev", r"S2S-E, constant"),
    ("gauss_cqr", r"mGauss", r"Gauss-E, CQR"),      # shared Gauss-E style of plotstyle.tex
    ("ds2s_cqr", r"mFR", r"DS2S-E, CQR"),
]
# Panels that get a zoomed inset: (first decile, last decile, headroom factor on ymax). The bands
# other than the SD-scaled one overlap there; the inset sits in the empty top left of the panel.
WIDTH_INSETS = {"oneweb_dl": (1, 9, 2.6), "starlink_ul": (1, 10, 1.0)}


def _lin_ticks(lo: float, hi: float) -> list[float]:
    """Two to four round ticks for a short linear axis."""
    for step in (1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500):
        t = [k * step for k in range(int(np.ceil(lo / step)), int(np.floor(hi / step)) + 1)]
        if 2 <= len(t) <= 4:
            return t
    return [lo, hi]


def _log_ticks(lo: float, hi: float) -> list[float]:
    """Readable ticks for a log axis between lo and hi (at least two)."""
    for mant in ((1, 2, 5), (1, 1.5, 2, 3, 5, 7), (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8)):
        t = [m * 10 ** e for e in range(-1, 6) for m in mant if lo <= m * 10 ** e <= hi]
        if len(t) >= 3:
            return t
    return t


def write_width_tex(fig_dir: Path, tex_dir: Path, links: list[str], h_plot: int):
    """2 x 3 groupplot: mean 95% band width at h_plot vs input-SD decile, all links, log y.
    A zoomed inset is added where the bands other than the SD-scaled one overlap closely."""
    dats = {ln: fig_dir / f"fr_condcov_wid_{ln}.dat" for ln in links}
    if not all(f.exists() for f in dats.values()):
        print("[tex] fr_condwidth.tex not written: width data missing for", [ln for ln, f in dats.items() if not f.exists()])
        return
    dfs = {ln: pd.read_csv(f, sep=" ") for ln, f in dats.items()}
    lines = [(k, st, lg) for k, st, lg in WIDTH_LINES if all("wid" + k.replace("_", "") in dfs[ln] for ln in links)]
    sec = h_plot * 0.2
    o = [r"% fr_condwidth.tex - generated by modeling/experiments/fr_condcov.py, do not edit by hand.",
         r"%% Mean width of the calibrated 95%% RTT band at %g s ahead (step %d) per input-volatility decile," % (sec, h_plot),
         r"% TEST windows, decile edges from validation windows. Log y axis.",
         r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=3 by 2, horizontal sep=1.15cm, vertical sep=1.35cm},",
         r"  frbase, width=5.9cm, height=4.3cm, ymode=log, log ticks with fixed point,",
         r"  yticklabel style={/pgf/number format/1000 sep={}}, xmin=0.6, xmax=10.4, xtick={1,...,10}]"]
    letters = "abcdefghijklmnopqrstuvwxyz"
    insets = []
    for i, ln in enumerate(links):
        df = dfs[ln]
        allv = np.concatenate([df["wid" + k.replace("_", "")].values for k, _, _ in lines])
        lo, hi = float(allv.min()) / 1.15, float(allv.max()) * 1.15
        if ln in WIDTH_INSETS:
            hi *= WIDTH_INSETS[ln][2]
        opts = [r"title={(%s) %s}" % (letters[i], NAMES[ln]), "ymin=%.4g, ymax=%.4g" % (lo, hi),
                "ytick={%s}" % ",".join("%g" % t for t in _log_ticks(lo, hi))]
        if i % 3 == 0:
            opts.append(r"ylabel={95\%% band width at %g s (ms)}" % sec)
        if i >= 3:
            opts.append(r"xlabel={input-SD decile (1 = calmest)}")
        if i == 0:
            opts.append(r"legend to name=legcondwidth, legend columns=%d" % len(lines))
        o.append(r"\nextgroupplot[" + ", ".join(opts) + "]")
        for k, st, lg in lines:
            o.append(r"\addplot[%s] table[x=dec, y=wid%s] {data/fr_condcov_wid_%s.dat};" % (st, k.replace("_", ""), ln)
                     + (r" \addlegendentry{%s}" % lg if i == 0 else ""))
        if ln in WIDTH_INSETS:
            x0, x1, _ = WIDTH_INSETS[ln]
            sel = (df["dec"] >= x0) & (df["dec"] <= x1)
            core = np.concatenate([df.loc[sel, "wid" + k.replace("_", "")].values for k, _, _ in lines if k != "s2s_norm"])
            pad = (core.max() - core.min()) * 0.08
            ylo, yhi = float(core.min() - pad), float(core.max() + pad)
            o.append(r"\draw[black!70] (axis cs:%g,%.4g) rectangle (axis cs:%g,%.4g);" % (x0 - 0.3, ylo, x1 + 0.3, yhi))
            o.append(r"\coordinate (cwz%d) at (axis cs:%g,%.4g);" % (i, x0 + 0.3 * (x1 - x0), yhi))
            insets.append((i, ln, x0, x1, ylo, yhi))
    o.append(r"\end{groupplot}")
    for i, ln, x0, x1, ylo, yhi in insets:
        c, r = i % 3 + 1, i // 3 + 1
        xt = "{%d,%d,%d}" % (x0, (x0 + x1 + 1) // 2, x1)
        yt = "{%s}" % ",".join("%g" % t for t in _lin_ticks(ylo, yhi))
        o.append(r"\begin{axis}[at={($(group c%dr%d.north west)+(0.48cm,-0.14cm)$)}, anchor=north west, scale only axis, "
                 r"width=2.45cm, height=1.15cm, xmin=%g, xmax=%g, ymin=%.4g, ymax=%.4g, xtick=%s, ytick=%s, "
                 r"tick label style={font=\tiny}, xticklabel style={yshift=0.5ex}, yticklabel style={xshift=0.5ex}, "
                 r"axis background/.style={fill=white}, grid=major, grid style={gray!20}, name=cwins%d]"
                 % (c, r, x0 - 0.3, x1 + 0.3, ylo, yhi, xt, yt, i))
        for k, st, _ in lines:
            if k == "s2s_norm":
                continue
            o.append(r"\addplot[%s, mark size=0.9pt, restrict x to domain=%g:%g] table[x=dec, y=wid%s] {data/fr_condcov_wid_%s.dat};"
                     % (st, x0, x1, k.replace("_", ""), ln))
        o.append(r"\end{axis}")
        o.append(r"\draw[->, black!70] (cwz%d) -- ($(cwins%d.south west)!0.3!(cwins%d.south east)$);" % (i, i, i))
    o += [r"\node at ($(group c2r1.north)+(0,0.95cm)$) {\pgfplotslegendfromname{legcondwidth}};", r"\end{tikzpicture}"]
    (tex_dir / "fr_condwidth.tex").write_text("\n".join(o) + "\n")
    print(f"[tex] {tex_dir / 'fr_condwidth.tex'} ({len(lines)} bands, insets on {[x[1] for x in insets]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--links", nargs="+", default=LINKS)
    ap.add_argument("--fr_method", default="fr_dist", help="distributional model behind DS2S-E")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--eps_frac", type=float, default=0.05,
                    help="eps of the normalised score = eps_frac x median validation input SD")
    ap.add_argument("--min_seeds", type=int, default=3, help="seed files required for an ensemble")
    ap.add_argument("--h_plot", type=int, default=5, help="horizon step of the width figure (5 = 1 s)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute_inputs", action="store_true")
    ap.add_argument("--tex_only", action="store_true",
                    help="only rebuild the table and figure .tex from the saved results")
    ap.add_argument("--res_dir", default=str(ROOT / "results_final/main"))
    ap.add_argument("--out_dir", default=str(ROOT / "results_final/analysis/condcov"))
    ap.add_argument("--fig_dir", default=str(ROOT.parent / "reports/final_report/data"))
    ap.add_argument("--data_root", default=str(ROOT / "data/lsn-data"))
    ap.add_argument("--tex_dir", default=str(ROOT.parent / "reports/final_report/figs"))
    ap.add_argument("--heat_links", nargs="+", default=["starlink_dl", "oneweb_dl", "quectel_ul"],
                    help="columns of the heatmap figure")
    ap.add_argument("--width_links", nargs="+",
                    default=["starlink_dl", "oneweb_dl", "quectel_dl", "starlink_ul", "oneweb_ul", "quectel_ul"],
                    help="panels of the width figure (2 x 3)")
    a = ap.parse_args()
    res_dir, out_dir, fig_dir = Path(a.res_dir), Path(a.out_dir), Path(a.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    js_path = out_dir / "summary.json"
    allres = json.load(open(js_path)) if js_path.exists() else {}
    allres.setdefault("links", {})
    for link in ([] if a.tex_only else a.links):
        r = run_link(link, a, res_dir, out_dir, fig_dir)
        if r is not None:
            allres["links"][link] = r
    if not a.tex_only:
        allres["settings"] = {k: v for k, v in vars(a).items()
                              if k not in ("links", "heat_links", "width_links", "tex_only")}
        allres["settings"]["definition"] = ("vol = sample SD (ddof=1) of the 30 input RTT values (ms), model-input "
                                            "view; decile edges from validation windows; alpha=%g" % a.alpha)
    json.dump(allres, open(js_path, "w"), indent=1, default=float)

    # Summary tables from every link present in the JSON (links can be run one at a time)
    summ = pd.DataFrame([r for ln in LINKS if ln in allres["links"] for r in allres["links"][ln]["summary"]])
    pairs = pd.DataFrame([r for ln in LINKS if ln in allres["links"] for r in allres["links"][ln]["pairs"]])
    summ.to_csv(out_dir / "summary.csv", index=False)
    pairs.to_csv(out_dir / "pairs.csv", index=False)
    if not summ.empty:
        write_table(summ, fig_dir)
    tex_dir = Path(a.tex_dir)
    tex_dir.mkdir(parents=True, exist_ok=True)
    write_heat_tex(fig_dir, tex_dir, a.heat_links)
    write_width_tex(fig_dir, tex_dir, a.width_links, a.h_plot)
    if not summ.empty:
        piv = summ.pivot(index="link", columns="method", values="maxdev").reindex(
            [ln for ln in LINKS if ln in set(summ.link)])
        print("\nmax |decile coverage - 0.95| (coverage pooled over h):")
        print(piv[[k for k in ORDER if k in piv.columns]].round(3).to_string())
        piv = summ.pivot(index="link", columns="method", values="mis").reindex(
            [ln for ln in LINKS if ln in set(summ.link)])
        print("\ninterval score (ms):")
        print(piv[[k for k in ORDER if k in piv.columns]].round(2).to_string())


if __name__ == "__main__":
    main()
