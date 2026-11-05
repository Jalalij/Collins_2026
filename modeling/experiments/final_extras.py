"""
experiments/final_extras.py - Analyses on saved predictions (no training).

1. RTT spike warning: can a forecast flag that RTT at t+h will exceed the
   validation 90th percentile? AUROC per horizon, using the initial mean, the final
   mean, the final upper quantile (q0.975), ridge and persistence as scores.
2. Starlink "real pings only": RTT metrics restricted to targets that were
   measured (not parser-interpolated), audit item A3.
3. Example trace for the report figure: one test session stretch with the
   truth, the initial forecast and the Final ensemble median and conformal band at
   a fixed horizon.
4. Deployment constants: CQR offsets c[h, k] of the Final ensemble, saved as JSON
   next to the models so live inference can produce calibrated intervals.

Writes results_final/summary/{spike_auroc.csv, real_ping_rtt.csv,
cqr_offsets_<link>.json} and <fig_dir>/trace_<link>.dat, spike_<link>.dat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATASETS = ["oneweb_dl", "oneweb_ul", "starlink_dl", "starlink_ul", "quectel_dl", "quectel_ul"]


def load_ens(pdir: Path, method: str):
    fs = sorted(pdir.glob(f"{method}_s*.npz"))
    if not fs:
        return None
    zs = [np.load(f) for f in fs]
    out = {"p_te": np.mean([z["p_te"] for z in zs], 0), "p_va": np.mean([z["p_va"] for z in zs], 0), "n": len(fs)}
    if "d_te" in zs[0].files:
        out["d_te"] = np.mean([z["d_te"].astype(np.float32) for z in zs], 0)
        out["d_va"] = np.mean([z["d_va"].astype(np.float32) for z in zs], 0)
    return out


def cqr_offsets(d_va, y_va, alpha=0.05):
    n = y_va.shape[0]
    s = np.maximum(d_va[..., 1] - y_va, y_va - d_va[..., 5])          # (N, H, K)
    k = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(s, k, axis=0, method="higher")                  # (H, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default=str(HERE.parent / "results_final/main"))
    ap.add_argument("--out_dir", default=str(HERE.parent / "results_final/summary"))
    ap.add_argument("--fig_dir", default=str(HERE.parent.parent / "reports/final_report/data"))
    ap.add_argument("--trace_h", type=int, default=5, help="horizon step for the trace figure (5 = 1 s)")
    a = ap.parse_args()
    res, out, fig = Path(a.res_dir), Path(a.out_dir), Path(a.fig_dir)
    out.mkdir(parents=True, exist_ok=True); fig.mkdir(parents=True, exist_ok=True)

    spike_rows, real_rows = [], []
    for ds in DATASETS:
        pdir = res / ds / "preds"
        if not (pdir / "truth.npz").exists():
            continue
        tr = np.load(pdir / "truth.npz")
        y_te, y_va = tr["y_te"], tr["y_va"]
        k = 0  # RTT is target 0
        thr = np.quantile(y_va[:, :, k], 0.9)
        meth = {m: load_ens(pdir, m) for m in ["persistence", "ridge_y3", "y2_seq2seq", "y3_dist"]}
        scores = {}
        if meth["persistence"]: scores["persistence"] = meth["persistence"]["p_te"]
        if meth["ridge_y3"]: scores["ridge_y3"] = meth["ridge_y3"]["p_te"]
        if meth["y2_seq2seq"]: scores["y2_seq2seq_ens"] = meth["y2_seq2seq"]["p_te"]
        if meth["y3_dist"]:
            scores["y3_dist_ens_mean"] = meth["y3_dist"]["p_te"]
            scores["y3_dist_ens_q975"] = meth["y3_dist"]["d_te"][..., 5]
        H = y_te.shape[1]
        for name, sc in scores.items():
            for h in range(H):
                ev = y_te[:, h, k] > thr
                if ev.all() or not ev.any():
                    continue
                spike_rows.append({"dataset": ds, "score": name, "h": h + 1, "threshold_ms": thr,
                                   "event_rate": ev.mean(), "auroc": roc_auc_score(ev, sc[:, h, k])})

        # Real-ping-only RTT metrics (only differs where the parser interpolated)
        if "interp_te" in tr.files and tr["interp_te"][:, :, k].any():
            real = tr["interp_te"][:, :, k] == 0
            for name, m in meth.items():
                if m is None:
                    continue
                p = m["p_te"][:, :, k]
                med = m["d_te"][..., 3][:, :, k] if "d_te" in m else p
                e, em = (y_te[:, :, k] - p), (y_te[:, :, k] - med)
                real_rows.append({"dataset": ds, "method": name + ("_ens" if m["n"] > 1 else ""),
                                  "interp_frac": 1 - real.mean(),
                                  "RMSE_all": np.sqrt((e ** 2).mean()), "RMSE_real": np.sqrt((e[real] ** 2).mean()),
                                  "MAE_all": np.abs(em).mean(), "MAE_real": np.abs(em[real]).mean()})

        if meth["y3_dist"]:
            d = meth["y3_dist"]
            c = cqr_offsets(d["d_va"], y_va)
            json.dump({"link": ds, "alpha": 0.05, "targets": [str(t) for t in tr["targets"]],
                       "n_seeds": d["n"], "offsets_hk": c.tolist(),
                       "note": "95% interval = [q0.025 - c, q0.975 + c] of the seed-averaged quantiles"},
                      open(out / f"cqr_offsets_{ds}.json", "w"), indent=1)
            # Trace: 300 consecutive windows (60 s) from the middle of the test set
            h = a.trace_h - 1
            n = len(y_te); s0 = n // 2; idx = np.arange(s0, s0 + 300)
            lo = d["d_te"][idx, h, k, 1] - c[h, k]; hi = d["d_te"][idx, h, k, 5] + c[h, k]
            tdf = pd.DataFrame({"t": np.arange(300) * 0.2, "truth": y_te[idx, h, k],
                                "y3med": d["d_te"][idx, h, k, 3], "y3lo": lo, "y3hi": hi,
                                "y2": meth["y2_seq2seq"]["p_te"][idx, h, k] if meth["y2_seq2seq"] else np.nan,
                                "pers": meth["persistence"]["p_te"][idx, h, k] if meth["persistence"] else np.nan})
            tdf.to_csv(fig / f"trace_{ds}.dat", sep=" ", index=False, float_format="%.4g")

    sp = pd.DataFrame(spike_rows)
    sp.to_csv(out / "spike_auroc.csv", index=False)
    if not sp.empty:
        for ds in sp.dataset.unique():
            piv = sp[sp.dataset == ds].pivot(index="h", columns="score", values="auroc")
            piv.columns = [c.replace("_", "") for c in piv.columns]
            piv.reset_index().to_csv(fig / f"spike_{ds}.dat", sep=" ", index=False, float_format="%.4f")
        print(sp.groupby(["dataset", "score"]).auroc.mean().unstack().round(3).to_string())
    rr = pd.DataFrame(real_rows)
    rr.to_csv(out / "real_ping_rtt.csv", index=False)
    if not rr.empty:
        print(rr.round(3).to_string())


if __name__ == "__main__":
    main()
