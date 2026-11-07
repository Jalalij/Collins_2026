"""
experiments/final_analysis.py - Further analyses of the saved test predictions
(no training). For every link, with the 3-seed ensembles S2S-E (prev_seq2seq)
and DS2S-E (fr_dist):

  reliability   empirical P(y <= q_tau) of the raw DS2S-E quantiles
  coverage      coverage and width of calibrated 50% and 95% intervals
                (CQR for DS2S-E, split conformal for S2S-E)
  adaptivity    Spearman correlation of DS2S-E interval width with |error|
  sessions      per-test-session RTT MAE, wins, Wilcoxon signed-rank p-value
  events        RTT MAE with / without a handover (or Starlink 15 s
                reconfiguration instant) inside the forecast horizon
  hour          RTT MAE by hour of day (4 h bins) of the forecast origin
  scatter       1,000 random test windows at h = 5 (1 s) for scatter plots

Writes results_final/summary/analysis_*.csv and reports/final_report/data/{rel,hour,scat,sess}_<link>.dat
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import final_benchmark as B
from final_extras import load_ens

RES = ROOT / "results_final/main"; OUT = ROOT / "results_final/summary"; FIG = ROOT.parent / "reports/final_report/data"
LINKS = ["oneweb_dl", "oneweb_ul", "starlink_dl", "starlink_ul", "quectel_dl", "quectel_ul"]
TAUS = [0.025, 0.25, 0.5, 0.75, 0.975]
L, H = 30, 15


def order_stat(s, alpha):
    n = len(s)
    return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), axis=0, method="higher")


rel_rows, cov_rows, adap_rows, sess_rows, ev_rows, hour_rows = [], [], [], [], [], []
for ds in LINKS:
    pdir = RES / ds / "preds"
    tr = np.load(pdir / "truth.npz")
    y_te, y_va = tr["y_te"], tr["y_va"]
    s2 = load_ens(pdir, "prev_seq2seq"); d3 = load_ens(pdir, "fr_dist")
    tnames = ["rtt", "jitter", "loss"]
    K = 1 if ds == "oneweb_ul" else 3
    # --- reliability of raw quantiles
    for k in range(K):
        for j, tau in enumerate(TAUS):
            rel_rows.append({"dataset": ds, "target": tnames[k], "tau": tau,
                             "empirical": float((y_te[:, :, k] <= d3["d_te"][:, :, k, j + 1]).mean())})
    # --- calibrated coverage and width at 50% and 95%
    for k in range(K):
        for alpha, lo_i, hi_i in [(0.5, 2, 4), (0.05, 1, 5)]:
            c = order_stat(np.maximum(d3["d_va"][..., k, lo_i] - y_va[..., k], y_va[..., k] - d3["d_va"][..., k, hi_i]), alpha)
            lo, hi = d3["d_te"][..., k, lo_i] - c, d3["d_te"][..., k, hi_i] + c
            q = order_stat(np.abs(y_va[..., k] - s2["p_va"][..., k]), alpha)
            lo2, hi2 = s2["p_te"][..., k] - q, s2["p_te"][..., k] + q
            yk = y_te[..., k]
            cov_rows.append({"dataset": ds, "target": tnames[k], "level": 1 - alpha,
                             "DS2S_cov": float(((yk >= lo) & (yk <= hi)).mean()), "DS2S_width": float((hi - lo).mean()),
                             "S2S_cov": float(((yk >= lo2) & (yk <= hi2)).mean()), "S2S_width": float((hi2 - lo2).mean())})
            if alpha == 0.05 and k == 0:
                w = (hi - lo).ravel(); e = np.abs(yk - d3["d_te"][..., k, 3]).ravel()
                rho = spearmanr(w, e).correlation
                adap_rows.append({"dataset": ds, "spearman_width_abs_err": float(rho),
                                  "width_p10": float(np.quantile(w, 0.1)), "width_p50": float(np.median(w)),
                                  "width_p90": float(np.quantile(w, 0.9)), "S2S_width": float((hi2 - lo2).mean()),
                                  "S2S_median_signed_err_h5": float(np.median(s2["p_te"][:, 4, 0] - y_te[:, 4, 0])),
                                  "DS2S_median_signed_err_h5": float(np.median(d3["d_te"][:, 4, 0, 3] - y_te[:, 4, 0]))})
    # --- metadata of the test windows
    prov, dirn = ds.split("_")
    data, meta = B.load_splits(prov, dirn, "base", L, H, torch.device("cpu"), str(ROOT / "data/lsn-data"))
    te = data["test"]; st = te.starts.numpy(); origin = st + L - 1
    assert len(st) == len(y_te), (ds, len(st), len(y_te))
    sid = te.sid[origin]; hour = te.hour_raw[origin] if hasattr(te, "hour_raw") else te.H.numpy()[origin]
    tgt_rows = st[:, None] + L + np.arange(H)[None]
    ev = te.ev[tgt_rows].max(1) > 0
    if prov == "starlink":
        sec = np.floor(np.mod(te.ts[tgt_rows], 60)).astype(int)
        recon = np.isin(sec, [12, 27, 42, 57]).any(1)
    else:
        recon = np.zeros(len(st), bool)
    e2 = np.abs(y_te[..., 0] - s2["p_te"][..., 0]).mean(1)
    e3 = np.abs(y_te[..., 0] - d3["d_te"][..., 0, 3]).mean(1)
    # sessions
    sess = pd.DataFrame({"sid": sid, "e2": e2, "e3": e3}).groupby("sid").mean()
    try:
        p = float(wilcoxon(sess.e3, sess.e2).pvalue)
    except ValueError:
        p = float("nan")
    sess_rows.append({"dataset": ds, "n_sessions": len(sess), "ds2s_wins": int((sess.e3 < sess.e2).sum()),
                      "median_rel_change_pct": float(100 * np.median((sess.e3 - sess.e2) / sess.e2)), "wilcoxon_p": p})
    sess.assign(rel=100 * (sess.e3 - sess.e2) / sess.e2).sort_values("rel").reset_index(drop=True).rename_axis("i") \
        .reset_index()[["i", "rel"]].to_csv(FIG / f"sess_{ds}.dat", sep=" ", index=False, float_format="%.3f")
    # events
    for name, mask in [("handover", ev), ("reconfig15s", recon)]:
        if mask.any() and (~mask).any():
            ev_rows.append({"dataset": ds, "event": name, "frac_windows": float(mask.mean()),
                            "S2S_MAE_event": float(e2[mask].mean()), "DS2S_MAE_event": float(e3[mask].mean()),
                            "S2S_MAE_none": float(e2[~mask].mean()), "DS2S_MAE_none": float(e3[~mask].mean())})
    # hour of day
    hb = (hour // 4).astype(int) * 4
    hdf = pd.DataFrame({"hour": hb, "e2": e2, "e3": e3}).groupby("hour").agg(S2S=("e2", "mean"), DS2S=("e3", "mean"), n=("e2", "size")).reset_index()
    hdf.to_csv(FIG / f"hour_{ds}.dat", sep=" ", index=False, float_format="%.4f")
    for r in hdf.itertuples():
        hour_rows.append({"dataset": ds, "hour_bin": r.hour, "S2S_MAE": r.S2S, "DS2S_MAE": r.DS2S, "n": r.n})
    # scatter at h = 5
    rng = np.random.default_rng(0); idx = rng.choice(len(y_te), size=min(400, len(y_te)), replace=False)
    pd.DataFrame({"y": y_te[idx, 4, 0], "s2s": s2["p_te"][idx, 4, 0], "ds2s": d3["d_te"][idx, 4, 0, 3]}) \
        .to_csv(FIG / f"scat_{ds}.dat", sep=" ", index=False, float_format="%.3f")
    print("done", ds, flush=True)

rel = pd.DataFrame(rel_rows); rel.to_csv(OUT / "analysis_reliability.csv", index=False)
for ds in LINKS:
    piv = rel[rel.dataset == ds].pivot(index="tau", columns="target", values="empirical").reset_index()
    piv.to_csv(FIG / f"rel_{ds}.dat", sep=" ", index=False, float_format="%.4f")
for name, rows in [("coverage", cov_rows), ("adaptivity", adap_rows), ("sessions", sess_rows),
                   ("events", ev_rows), ("hour", hour_rows)]:
    df = pd.DataFrame(rows); df.to_csv(OUT / f"analysis_{name}.csv", index=False)
    print(f"\n== {name}\n" + df.round(4).to_string())
