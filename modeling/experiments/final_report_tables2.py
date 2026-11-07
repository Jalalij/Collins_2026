"""
experiments/final_report_tables2.py - LaTeX tables for the extended analyses
(analysis_*.csv from final_analysis.py, per-horizon results, training cost).
Writes reports/final_report/data/tab2_*.tex and numbers_analysis.tex.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "results_final/summary"; OUT = ROOT.parent / "reports/final_report/data"
NICE = {"oneweb_dl": "OneWeb DL", "oneweb_ul": "OneWeb UL", "starlink_dl": "Starlink DL",
        "starlink_ul": "Starlink UL", "quectel_dl": "5G DL", "quectel_ul": "5G UL"}
LINKS = list(NICE)
mac = {}
K = {"oneweb_dl": "OWDL", "oneweb_ul": "OWUL", "starlink_dl": "SLDL", "starlink_ul": "SLUL", "quectel_dl": "QDL", "quectel_ul": "QUL"}

cov = pd.read_csv(S / "analysis_coverage.csv")
rows = []
for ds in LINKS:
    for t in ["rtt", "jitter", "loss"]:
        c = cov[(cov.dataset == ds) & (cov.target == t)]
        if c.empty: continue
        a, b = c[c.level == 0.5].iloc[0], c[c.level == 0.95].iloc[0]
        rows.append(f"{NICE[ds]} & {t} & {a.S2S_cov:.3f} & {a.DS2S_cov:.3f} & {a.S2S_width:.3g} & {a.DS2S_width:.3g} & "
                    f"{b.S2S_cov:.3f} & {b.DS2S_cov:.3f} & {b.S2S_width:.3g} & {b.DS2S_width:.3g} \\\\")
(OUT / "tab2_coverage.tex").write_text("\n".join(rows) + "\n")

ad = pd.read_csv(S / "analysis_adaptivity.csv")
rows = [f"{NICE[r.dataset]} & {r.spearman_width_abs_err:.2f} & {r.width_p10:.1f} & {r.width_p50:.1f} & {r.width_p90:.1f} & {r.S2S_width:.1f} \\\\" for r in ad.itertuples()]
(OUT / "tab2_adaptivity.tex").write_text("\n".join(rows) + "\n")
for r in ad.itertuples():
    mac[f"BiasPrev{K[r.dataset]}"] = f"{r.S2S_median_signed_err_h5:+.1f}"
    mac[f"BiasFR{K[r.dataset]}"] = f"{r.DS2S_median_signed_err_h5:+.2f}"
mac["RhoMin"], mac["RhoMax"] = f"{ad.spearman_width_abs_err.min():.2f}", f"{ad.spearman_width_abs_err.max():.2f}"

se = pd.read_csv(S / "analysis_sessions.csv")
rows = []
for r in se.itertuples():
    p = "$<10^{-4}$" if r.wilcoxon_p < 1e-4 else f"{r.wilcoxon_p:.4f}"
    rows.append(f"{NICE[r.dataset]} & {r.n_sessions} & {r.ds2s_wins} & {r.median_rel_change_pct:+.1f} & {p} \\\\")
    mac[f"SessWins{K[r.dataset]}"] = str(r.ds2s_wins); mac[f"SessN{K[r.dataset]}"] = str(r.n_sessions)
    mac[f"SessP{K[r.dataset]}"] = p.replace("$", "") if r.wilcoxon_p >= 1e-4 else "10^{-4}"
(OUT / "tab2_sessions.tex").write_text("\n".join(rows) + "\n")
mac["SessWinsTotal"], mac["SessNTotal"] = str(se.ds2s_wins.sum()), str(se.n_sessions.sum())

ev = pd.read_csv(S / "analysis_events.csv"); ev = ev[ev.event == "handover"]
rows = [f"{NICE[r.dataset]} & {100*r.frac_windows:.1f} & {r.S2S_MAE_event:.2f} & {r.DS2S_MAE_event:.2f} & "
        f"{100*(r.DS2S_MAE_event/r.S2S_MAE_event-1):+.1f} & {r.S2S_MAE_none:.2f} & {r.DS2S_MAE_none:.2f} & "
        f"{100*(r.DS2S_MAE_none/r.S2S_MAE_none-1):+.1f} \\\\" for r in ev.itertuples()]
(OUT / "tab2_events.tex").write_text("\n".join(rows) + "\n")

# per-horizon appendix tables (seed means; RTT/jitter/loss; MAE, RMSE, MIS)
hz = pd.read_csv(S / "horizon_mean.csv")
MN = [("persistence", "Pers."), ("snaive", "Seas."), ("ridge_fr", "Ridge-E"), ("prev_seq2seq", "Initial"),
      ("fr_resid", "+res."), ("fr_full", "+data"), ("fr_dist", "Final"), ("fr_dist_ens", "Final ens.")]
for ds in LINKS:
    for t in (["rtt"] if ds == "oneweb_ul" else ["rtt", "jitter", "loss"]):
        rows = []
        for metric in ["MAE", "RMSE", "MIS"]:
            for i, (m, lab) in enumerate(MN):
                w = hz[(hz.dataset == ds) & (hz.target == t) & (hz.method == m)].sort_values("h")
                if w.empty: continue
                vals = w[metric].values
                d = 1 if vals.mean() >= 10 else (2 if vals.mean() >= 1 else 3)
                first = f"\\multirow{{{len(MN)}}}{{*}}{{{metric}}}" if i == 0 else ""
                rows.append(f"{first} & {lab} & " + " & ".join(f"{v:.{d}f}" for v in vals[[0, 2, 4, 7, 9, 14]]) + " \\\\")
            rows.append("\\midrule")
        (OUT / f"tab2_h_{ds}_{t}.tex").write_text("\n".join(rows[:-1]) + "\n")

# training cost
rows = []
for ds in LINKS:
    for m, lab in [("prev_seq2seq", "Initial"), ("fr_resid", "+residual"), ("fr_full", "+data"), ("fr_dist", "Final")]:
        js = [json.load(open(f)) for f in sorted((ROOT / "results_final/main" / ds).glob(f"{m}_s*.json"))]
        ep = [j["epochs"] for j in js]; sec = [j["seconds"] for j in js]
        rows.append(f"{NICE[ds]} & {lab} & {js[0]['params']:,} & {np.mean(ep):.1f} & {np.mean(sec)/60:.1f} \\\\".replace(",", "{,}"))
    rows.append("\\midrule")
(OUT / "tab2_cost.tex").write_text("\n".join(rows[:-1]) + "\n")

rows = []
for r in se.itertuples():
    p = "$<10^{-4}$" if r.wilcoxon_p < 1e-4 else f"{r.wilcoxon_p:.4f}"
    rho = ad[ad.dataset == r.dataset].spearman_width_abs_err.iloc[0]
    rows.append(f"{NICE[r.dataset]} & {r.n_sessions} & {r.ds2s_wins} & {r.median_rel_change_pct:+.1f} & {p} & {rho:.2f} \\\\")
(OUT / "tab_sessadapt.tex").write_text("\n".join(rows) + "\n")

lines = ["% Generated by modeling/experiments/final_report_tables2.py. Do not edit."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
(OUT / "numbers_analysis.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
