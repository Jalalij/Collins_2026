"""
experiments/y3_aggregate.py - Collect y3_benchmark results into tables and figure data.

Reads   results_y3/main/<provider>_<direction>/{<method>_s<seed>.csv, preds/*.npz}
Writes  results_y3/summary/*.csv            (machine-readable summaries)
        <fig_dir>/*.dat                     (pgfplots data for the report and paper)
        <fig_dir>/*.tex                     (booktabs table bodies)

Point-forecast convention for the distributional model (y3_dist): RMSE and R2
use its mean output, MAE uses its median output (the optimal point forecast for
each loss, Gneiting 2011). Every other method has a single output used for all.

Ensembles ("<method>_ens") average the per-seed outputs (mean and each
quantile) and are scored with the same metric code as single runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from y3_benchmark import metrics  # noqa: E402

DATASETS = ["oneweb_dl", "oneweb_ul", "starlink_dl", "starlink_ul", "quectel_dl", "quectel_ul"]
NICE = {"oneweb_dl": "OneWeb DL", "oneweb_ul": "OneWeb UL", "starlink_dl": "Starlink DL",
        "starlink_ul": "Starlink UL", "quectel_dl": "5G DL", "quectel_ul": "5G UL"}
METHODS = ["persistence", "snaive", "ridge", "ridge_y3", "y2_seq2seq", "y3_resid", "y3_full", "y3_dist"]
MNICE = {"persistence": "Persistence", "ridge": "Ridge (direct)", "y2_seq2seq": "Y2 Seq2Seq-GRU",
         "y3_resid": "+ residual decoder", "y3_full": "+ data pipeline",
         "y3_dist": "+ distributional heads", "y3_dist_ens": "Y3 (3-seed ensemble)",
         "y2_seq2seq_ens": "Y2 (3-seed ensemble)"}
TARGETS = ["rtt", "jitter", "loss"]


def point_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MAE_pt"] = df["MAE_median"] if "MAE_median" in df and df["MAE_median"].notna().all() else df["MAE"]
    df["MIS_best"] = df["MIS_cqr"] if "MIS_cqr" in df and df["MIS_cqr"].notna().all() else df["MIS"]
    df["PICP_best"] = df["PICP_cqr"] if "PICP_cqr" in df and df["PICP_cqr"].notna().all() else df["PICP"]
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Year-2-style aggregate (mean over all target x horizon keys) + per target."""
    df = point_cols(df)
    out = {"AGG_MAE": df.MAE_pt.mean(), "AGG_RMSE": df.RMSE.mean(), "AGG_R2": df.R2.mean(skipna=True),
           "AGG_MIS": df.MIS_best.mean(), "AGG_PICP": df.PICP_best.mean(),
           "AGG_MIS_y2style": df.MIS_test_quantile.mean()}
    for t in TARGETS:
        s = df[df.target == t]
        if s.empty:
            continue
        out.update({f"{t}_MAE": s.MAE_pt.mean(), f"{t}_RMSE": s.RMSE.mean(), f"{t}_R2": s.R2.mean(),
                    f"{t}_MIS": s.MIS_best.mean(), f"{t}_PICP": s.PICP_best.mean(),
                    f"{t}_skill": 1 - s.RMSE.mean() / s.RMSE_persist.mean(),
                    f"{t}_MIS_y2style": s.MIS_test_quantile.mean()})
    return out


def ensemble_rows(ddir: Path, method: str):
    files = sorted((ddir / "preds").glob(f"{method}_s*.npz"))
    if len(files) < 2:
        return None, 0
    tr = np.load(ddir / "preds" / "truth.npz")
    zs = [np.load(f) for f in files]
    p_te = np.mean([z["p_te"] for z in zs], 0)
    p_va = np.mean([z["p_va"] for z in zs], 0)
    d_te = d_va = None
    if "d_te" in zs[0].files:
        d_te = np.mean([z["d_te"].astype(np.float32) for z in zs], 0)
        d_va = np.mean([z["d_va"].astype(np.float32) for z in zs], 0)
        # averaging keeps quantile order; the mean channel is p_te
    meta = {"targets": list(tr["targets"])}
    df = metrics(p_te, tr["y_te"], p_va, tr["y_va"], tr["pers_te"], meta, dist_te=d_te, dist_va=d_va)
    return df, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default=str(HERE.parent / "results_y3/main"))
    ap.add_argument("--out_dir", default=str(HERE.parent / "results_y3/summary"))
    ap.add_argument("--fig_dir", default=str(HERE.parent.parent / "reports/year3/data"))
    a = ap.parse_args()
    res, out, fig = Path(a.res_dir), Path(a.out_dir), Path(a.fig_dir)
    out.mkdir(parents=True, exist_ok=True); fig.mkdir(parents=True, exist_ok=True)

    rows, horizon = [], []
    for ds in DATASETS:
        ddir = res / ds
        if not ddir.exists():
            continue
        for m in METHODS + ["y2_seq2seq_ens", "y3_dist_ens"]:
            if m.endswith("_ens"):
                df, n = ensemble_rows(ddir, m[:-4])
                if df is None:
                    continue
                runs = [(f"ens{n}", df)]
            else:
                runs = [(int(f.stem.split("_s")[-1]), pd.read_csv(f)) for f in sorted(ddir.glob(f"{m}_s*.csv"))]
            for seed, df in runs:
                rows.append({"dataset": ds, "method": m, "seed": seed, **summarise(df)})
                d = point_cols(df)
                for (t, h), s in d.groupby(["target", "h"]):
                    horizon.append({"dataset": ds, "method": m, "seed": seed, "target": t, "h": h,
                                    "MAE": s.MAE_pt.iloc[0], "RMSE": s.RMSE.iloc[0], "R2": s.R2.iloc[0],
                                    "RMSE_persist": s.RMSE_persist.iloc[0], "MIS": s.MIS_best.iloc[0],
                                    "PICP": s.PICP_best.iloc[0]})
    runs = pd.DataFrame(rows)
    hz = pd.DataFrame(horizon)
    runs.to_csv(out / "runs.csv", index=False)
    hz.to_csv(out / "horizon_runs.csv", index=False)

    num = [c for c in runs.columns if c not in ("dataset", "method", "seed")]
    single = runs[~runs.method.str.endswith("_ens")]
    mean = single.groupby(["dataset", "method"])[num].mean()
    std = single.groupby(["dataset", "method"])[num].std()
    n = single.groupby(["dataset", "method"]).size().rename("n_seeds")
    summ = mean.join(std, rsuffix="_std").join(n).reset_index()
    ens = runs[runs.method.str.endswith("_ens")].drop(columns="seed")
    summ = pd.concat([summ, ens.assign(n_seeds=1)], ignore_index=True)
    summ.to_csv(out / "summary.csv", index=False)

    # Relative change of Y3 vs Y2 (both averaged over seeds, and ensembles)
    rel = []
    for ds in summ.dataset.unique():
        s = summ[summ.dataset == ds].set_index("method")
        for a_, b_ in [("y2_seq2seq", "y3_dist"), ("y2_seq2seq", "y3_dist_ens"), ("y2_seq2seq_ens", "y3_dist_ens"),
                       ("y2_seq2seq", "y3_resid"), ("y2_seq2seq", "y3_full")]:
            if a_ in s.index and b_ in s.index:
                r = {"dataset": ds, "baseline": a_, "method": b_}
                for c in ["AGG_MAE", "AGG_RMSE", "AGG_MIS", "rtt_MAE", "rtt_RMSE", "rtt_MIS",
                          "jitter_MAE", "jitter_RMSE", "loss_MAE", "loss_RMSE"]:
                    if c in s.columns:
                        r[c + "_pct"] = 100 * (s.loc[b_, c] - s.loc[a_, c]) / s.loc[a_, c]
                r["AGG_R2_delta"] = s.loc[b_, "AGG_R2"] - s.loc[a_, "AGG_R2"]
                rel.append(r)
    pd.DataFrame(rel).to_csv(out / "relative_change.csv", index=False)

    # Horizon curves for figures: seed-mean RTT RMSE / MAE / skill per method
    hs = hz[~hz.method.str.endswith("_ens") | hz.method.isin(["y3_dist_ens", "y2_seq2seq_ens"])]
    hm = hs.groupby(["dataset", "method", "target", "h"])[["MAE", "RMSE", "R2", "RMSE_persist", "MIS", "PICP"]].mean().reset_index()
    hm["skill"] = 1 - hm.RMSE / hm.RMSE_persist
    hm.to_csv(out / "horizon_mean.csv", index=False)
    for ds in hm.dataset.unique():
        for t in TARGETS:
            w = hm[(hm.dataset == ds) & (hm.target == t)]
            if w.empty:
                continue
            for metric in ["RMSE", "MAE", "MIS", "skill", "PICP"]:
                piv = w.pivot(index="h", columns="method", values=metric)
                piv.columns = [c.replace("_", "") for c in piv.columns]
                piv.reset_index().to_csv(fig / f"h_{ds}_{t}_{metric}.dat", sep=" ", index=False,
                                         float_format="%.5g", na_rep="nan")
    print(summ[["dataset", "method", "n_seeds", "AGG_MAE", "AGG_RMSE", "AGG_R2", "AGG_MIS", "AGG_PICP",
                "rtt_MAE", "rtt_RMSE", "rtt_skill"]].to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
