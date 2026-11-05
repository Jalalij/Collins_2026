"""
experiments/final_tables.py - Write the LaTeX tables and number macros of the
final report and the TWC paper from results_final/summary/*.csv.

Every number printed in the documents comes from this script (or from the
pgfplots .dat files written by final_aggregate.py / final_extras.py, or from
final_audit_numbers.py for the audit and reproduction macros).

Outputs (in --out_dir, default reports/final_report/data):
  numbers_results.tex    \newcommand macros used in the text
  tab_main.tex           RTT results per link (seed mean +- std, ensembles)
  tab_y2compare.tex      Year-2-style aggregate vs the Modeling Dynamics report
  tab_ablation.tex       relative change of each final report step vs initial study
  tab_baselines.tex      non-neural references
  tab_targets.tex        jitter and loss
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LINKS = [("oneweb_dl", "OneWeb DL"), ("oneweb_ul", "OneWeb UL"), ("starlink_dl", "Starlink DL"),
         ("starlink_ul", "Starlink UL"), ("quectel_dl", "5G DL"), ("quectel_ul", "5G UL")]
# Seq2Seq-GRU rows of the Modeling Dynamics report (Tables 1-3): MAE, RMSE, R2, MIS
Y2_REPORT = {"oneweb_dl": (6.12, 23.73, 0.217, 108.69), "oneweb_ul": (22.81, 49.40, 0.918, 337.12),
             "quectel_dl": (1.10, 1.66, 0.273, 8.94), "quectel_ul": (4.40, 7.91, 0.114, 39.35),
             "starlink_dl": (8.55, 14.95, 0.406, 98.37), "starlink_ul": (1.48, 3.18, 0.199, 15.26)}
K = {"oneweb_dl": "OWDL", "oneweb_ul": "OWUL", "starlink_dl": "SLDL", "starlink_ul": "SLUL",
     "quectel_dl": "QDL", "quectel_ul": "QUL"}


def f(v, d=2):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def pm(m, s, d=2):
    if s is None or not np.isfinite(s):
        return f(m, d)
    return f"{m:.{d}f}\\,$\\pm$\\,{s:.{d}f}"


def pct(new, old):
    return 100.0 * (new - old) / old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results_final/summary"))
    ap.add_argument("--out_dir", default=str(ROOT.parent / "reports/final_report/data"))
    a = ap.parse_args()
    S = Path(a.summary)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(S / "summary.csv").set_index(["dataset", "method"])
    spike = pd.read_csv(S / "spike_auroc.csv")
    real = pd.read_csv(S / "real_ping_rtt.csv")

    def g(ds, m, c):
        try:
            return float(summ.loc[(ds, m), c])
        except KeyError:
            return float("nan")

    def gs(ds, m, c):
        try:
            return float(summ.loc[(ds, m), c + "_std"])
        except KeyError:
            return float("nan")

    rows = []
    for ds, nm in LINKS:
        block = [("y2_seq2seq", "Initial (retrained)", True), ("y3_dist", "Final", True),
                 ("y2_seq2seq_ens", "Initial ensemble", False), ("y3_dist_ens", "Final ensemble", False)]
        for i, (m, lab, has_std) in enumerate(block):
            cells = [pm(g(ds, m, c), gs(ds, m, c) if has_std else None, d)
                     for c, d in [("rtt_MAE", 2), ("rtt_RMSE", 2), ("rtt_R2", 3), ("rtt_MIS", 1), ("rtt_PICP", 3)]]
            first = f"\\multirow{{4}}{{*}}{{{nm}}}" if i == 0 else ""
            rows.append(f"{first} & {lab} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    (out / "tab_main.tex").write_text("\n".join(rows[:-1]) + "\n")

    rows = []
    for ds, nm in LINKS:
        r = Y2_REPORT[ds]
        rows.append(f"\\multirow{{3}}{{*}}{{{nm}}} & MD report & {r[0]:.2f} & {r[1]:.2f} & {r[2]:.3f} & {r[3]:.1f} & -- \\\\")
        for m, lab in [("y2_seq2seq", "Initial (retrained)"), ("y3_dist_ens", "Final ensemble")]:
            rows.append(f" & {lab} & {f(g(ds, m, 'AGG_MAE'))} & {f(g(ds, m, 'AGG_RMSE'))} & {f(g(ds, m, 'AGG_R2'), 3)} & "
                        f"{f(g(ds, m, 'AGG_MIS_y2style'), 1)} & {f(g(ds, m, 'AGG_MIS'), 1)} \\\\")
        rows.append("\\midrule")
    (out / "tab_y2compare.tex").write_text("\n".join(rows[:-1]) + "\n")

    rows = []
    steps = [("y3_resid", "1: residual decoder"), ("y3_full", "1+2: + data pipeline"),
             ("y3_dist", "1+2+3: + distrib. heads"), ("y3_dist_ens", "1--4: + 3-seed ensemble")]
    for m, lab in steps:
        cells = []
        for ds, _ in LINKS:
            for c in ["rtt_MAE", "rtt_RMSE", "rtt_MIS"]:
                cells.append(f"{pct(g(ds, m, c), g(ds, 'y2_seq2seq', c)):+.1f}")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    (out / "tab_ablation.tex").write_text("\n".join(rows) + "\n")

    rows = []
    for m, lab in [("persistence", "Persistence"), ("snaive", "Seasonal repeat (1\\,s)"), ("ridge", "Ridge"),
                   ("ridge_y3", "Ridge-E"), ("y2_seq2seq", "Initial Seq2Seq-GRU"), ("y3_dist", "Final"),
                   ("y3_dist_ens", "Final ensemble")]:
        cells = []
        for ds, _ in LINKS:
            v = g(ds, m, "rtt_MAE")
            cells += [f(v, 2 if v < 100 else 1), f(g(ds, m, "rtt_skill") * 100, 1)]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    (out / "tab_baselines.tex").write_text("\n".join(rows) + "\n")

    rows = []
    for ds, nm in LINKS:
        if ds == "oneweb_ul":
            continue  # constant zero targets
        for i, (m, lab) in enumerate([("y2_seq2seq", "Initial"), ("y3_dist", "Final"), ("y3_dist_ens", "Final ens.")]):
            first = f"\\multirow{{3}}{{*}}{{{nm}}}" if i == 0 else ""
            cells = [f(g(ds, m, f"{t}_{c}"), 3) for t in ["jitter", "loss"] for c in ["MAE", "RMSE", "MIS"]]
            rows.append(f"{first} & {lab} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    (out / "tab_targets.tex").write_text("\n".join(rows[:-1]) + "\n")

    rows = []
    for ds, nm in LINKS:
        a2, b2 = g(ds, "y2_seq2seq", "rtt_MIS_y2style"), g(ds, "y3_dist", "rtt_MIS_y2style")
        a3, b3 = g(ds, "y2_seq2seq", "rtt_MIS"), g(ds, "y3_dist", "rtt_MIS")
        rows.append(f"{nm} & {a2:.1f} & {b2:.1f} & {pct(b2, a2):+.1f} & {a3:.1f} & {b3:.1f} & {pct(b3, a3):+.1f} \\\\")
    (out / "tab_isproto.tex").write_text("\n".join(rows) + "\n")

    mac = {}
    acc = {k: [] for k in ["MAE", "RMSE", "MIS", "EMAE", "ERMSE", "EMIS"]}
    for ds, _ in LINKS:
        k = K[ds]
        for c, nm in [("rtt_MAE", "MAE"), ("rtt_RMSE", "RMSE"), ("rtt_MIS", "MIS")]:
            v = pct(g(ds, "y3_dist", c), g(ds, "y2_seq2seq", c))
            ve = pct(g(ds, "y3_dist_ens", c), g(ds, "y2_seq2seq_ens", c))
            mac[f"Chg{nm}{k}"] = f"{v:.1f}"
            mac[f"ChgEns{nm}{k}"] = f"{ve:.1f}"
            acc[nm].append(v); acc["E" + nm].append(ve)
        mac[f"PICP{k}"] = f(g(ds, "y3_dist", "rtt_PICP"), 3)
        mac[f"PICPEns{k}"] = f(g(ds, "y3_dist_ens", "rtt_PICP"), 3)
        mac[f"PICPYtwo{k}"] = f(g(ds, "y2_seq2seq", "rtt_PICP"), 3)
        mac[f"SkillYthreeEns{k}"] = f(100 * g(ds, "y3_dist_ens", "rtt_skill"), 1)
        mac[f"SkillRidge{k}"] = f(100 * g(ds, "ridge_y3", "rtt_skill"), 1)
    for nm in ["MAE", "RMSE", "MIS"]:
        arr, arre = acc[nm], acc["E" + nm]
        mac[f"Chg{nm}Best"], mac[f"Chg{nm}Worst"], mac[f"Chg{nm}Mean"] = f"{min(arr):.1f}", f"{max(arr):.1f}", f"{np.mean(arr):.1f}"
        mac[f"ChgEns{nm}Best"], mac[f"ChgEns{nm}Worst"], mac[f"ChgEns{nm}Mean"] = f"{min(arre):.1f}", f"{max(arre):.1f}", f"{np.mean(arre):.1f}"
    picps = [g(ds, "y3_dist", "rtt_PICP") for ds, _ in LINKS]
    mac["PICPMin"], mac["PICPMax"] = f"{min(picps):.3f}", f"{max(picps):.3f}"
    picps = [g(ds, "y3_dist_ens", "rtt_PICP") for ds, _ in LINKS]
    mac["PICPEnsMin"], mac["PICPEnsMax"] = f"{min(picps):.3f}", f"{max(picps):.3f}"
    ratio = [g(ds, "y2_seq2seq", "rtt_RMSE") / g(ds, "y2_seq2seq", "rtt_MAE") for ds, _ in LINKS]
    mac["RatioRMSEMAEMin"], mac["RatioRMSEMAEMax"] = f"{min(ratio):.1f}", f"{max(ratio):.1f}"
    sp = spike.groupby(["dataset", "score"]).auroc.mean()
    for ds, _ in LINKS:
        k = K[ds]
        mac[f"AUCYtwo{k}"] = f"{sp.loc[(ds, 'y2_seq2seq_ens')]:.3f}"
        mac[f"AUCYthree{k}"] = f"{sp.loc[(ds, 'y3_dist_ens_q975')]:.3f}"
        mac[f"AUCRidge{k}"] = f"{sp.loc[(ds, 'ridge_y3')]:.3f}"
        mac[f"AUCPers{k}"] = f"{sp.loc[(ds, 'persistence')]:.3f}"
    rd = real[real.dataset == "starlink_dl"].set_index("method")
    mac["InterpFracSLDL"] = f"{100 * rd.loc['persistence', 'interp_frac']:.1f}"
    ru = real[real.dataset == "starlink_ul"].set_index("method")
    mac["InterpFracSLUL"] = f"{100 * ru.loc['persistence', 'interp_frac']:.1f}"
    for m, nm in [("y2_seq2seq_ens", "Ytwo"), ("y3_dist_ens", "Ythree")]:
        mac[f"RealMAE{nm}SLDL"] = f"{rd.loc[m, 'MAE_real']:.2f}"
        mac[f"RealRMSE{nm}SLDL"] = f"{rd.loc[m, 'RMSE_real']:.2f}"
        mac[f"AllMAE{nm}SLDL"] = f"{rd.loc[m, 'MAE_all']:.2f}"
        mac[f"AllRMSE{nm}SLDL"] = f"{rd.loc[m, 'RMSE_all']:.2f}"
    tm = json.load(open(S / "timing.json"))
    mac["TimeSSCPU"], mac["TimeDSCPU"] = f"{tm['S2S_cpu_1thread']:.1f}", f"{tm['DS2S_cpu_1thread']:.1f}"
    mac["JitISSLDLDS"] = f(g("starlink_dl", "y3_dist", "jitter_MIS"), 1)
    mac["JitISSLDLS"] = f(g("starlink_dl", "y2_seq2seq", "jitter_MIS"), 1)
    eps = [json.load(open(js))["epochs"] for ds, _ in LINKS
           for js in (ROOT / "results_final/main" / ds).glob("y*_s*.json")]
    mac["EpochsMin"], mac["EpochsMax"], mac["EpochsMedian"] = str(min(eps)), str(max(eps)), f"{np.median(eps):.0f}"
    mac["NRuns"] = str(len(eps))
    mac["BestEpochMax"] = str(max(eps) - 15)
    lines = ["% Generated by modeling/experiments/final_tables.py from results_final/summary. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    (out / "numbers_results.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
