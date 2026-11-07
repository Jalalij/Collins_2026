"""
experiments/final_sweeps.py - Summaries of the sensitivity sweeps in results_final/sweeps/<cfg>/
(input window 15/60 rows, 30-step horizon, hidden size 64/256, pinball weight 0.25/4),
compared with the base setting (L=30, H=15, hidden 128, weight 1) from results_final/main
restricted to the same seeds (0, 1) and links.
Writes results_final/summary/sweeps.csv, reports/final_report/data/tab3_sweeps*.tex, sweep_*.dat, numbers_sweeps.tex
"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "reports/final_report/data"
LINKS = ["oneweb_dl", "oneweb_ul", "starlink_dl", "starlink_ul"]
NICE = {"oneweb_dl": "OneWeb DL", "oneweb_ul": "OneWeb UL", "starlink_dl": "Starlink DL", "starlink_ul": "Starlink UL"}
CFGS = [("base", "default (6\\,s input, 3\\,s horizon)"), ("L15", "input 15 rows (3 s)"), ("L60", "input 60 rows (12 s)"),
        ("H30", "horizon 30 steps (6 s)"), ("hid64", "hidden size 64"), ("hid256", "hidden size 256"),
        ("qw025", "pinball weight 0.25"), ("qw4", "pinball weight 4")]


def score(df, first15=False):
    d = df[df.target == "rtt"]
    if first15: d = d[d.h <= 15]
    mae = d.MAE_median.mean() if "MAE_median" in d and d.MAE_median.notna().all() else d.MAE.mean()
    mis = d.MIS_cqr.mean() if "MIS_cqr" in d and d.MIS_cqr.notna().all() else d.MIS.mean()
    picp = d.PICP_cqr.mean() if "PICP_cqr" in d and d.PICP_cqr.notna().all() else d.PICP.mean()
    return mae, d.RMSE.mean(), mis, picp


rows = []
for cfg, _ in CFGS:
    base = ROOT / ("results_final/main" if cfg == "base" else f"results_final/sweeps/{cfg}")
    for ds in LINKS:
        for m in ["prev_seq2seq", "fr_dist"]:
            for seed in [0, 1]:
                f = base / ds / f"{m}_s{seed}.csv"
                if not f.exists(): continue
                df = pd.read_csv(f)
                for first15 in ([False, True] if cfg == "H30" else [False]):
                    mae, rmse, mis, picp = score(df, first15)
                    rows.append(dict(cfg=cfg + ("_first15" if first15 else ""), dataset=ds, method=m, seed=seed,
                                     MAE=mae, RMSE=rmse, MIS=mis, PICP=picp))
r = pd.DataFrame(rows)
(ROOT / "results_final/summary").mkdir(exist_ok=True)
r.to_csv(ROOT / "results_final/summary/sweeps_runs.csv", index=False)
g = r.groupby(["cfg", "dataset", "method"])[["MAE", "RMSE", "MIS", "PICP"]].mean().reset_index()
n = r.groupby(["cfg", "dataset", "method"]).size().rename("n").reset_index()
g = g.merge(n); g.to_csv(ROOT / "results_final/summary/sweeps.csv", index=False)
print(g.round(3).to_string())

# Table A: DS2S across configs, relative to base DS2S (MAE, RMSE, MIS) per link
rows = []
for cfg, lab in CFGS:
    c = cfg + ("_first15" if cfg == "H30" else "")
    cells = []
    for ds in LINKS:
        try:
            b = g[(g.cfg == "base") & (g.dataset == ds) & (g.method == "fr_dist")].iloc[0]
            x = g[(g.cfg == c) & (g.dataset == ds) & (g.method == "fr_dist")].iloc[0]
            if cfg == "base":
                cells += [f"{x.MAE:.2f}", f"{x.RMSE:.2f}", f"{x.MIS:.1f}"]
            else:
                cells += [f"{100*(x.MAE/b.MAE-1):+.1f}", f"{100*(x.RMSE/b.RMSE-1):+.1f}", f"{100*(x.MIS/b.MIS-1):+.1f}"]
        except IndexError:
            cells += ["--"] * 3
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
(OUT / "tab3_sweeps_ds2s.tex").write_text("\n".join(rows) + "\n")

# Table B: DS2S vs S2S within each config (window and horizon sweeps), % change
rows = []
for cfg, lab in CFGS[:4]:
    c = cfg
    cells = []
    for ds in LINKS:
        try:
            a = g[(g.cfg == c) & (g.dataset == ds) & (g.method == "prev_seq2seq")].iloc[0]
            b = g[(g.cfg == c) & (g.dataset == ds) & (g.method == "fr_dist")].iloc[0]
            cells += [f"{100*(b.MAE/a.MAE-1):+.1f}", f"{100*(b.RMSE/a.RMSE-1):+.1f}", f"{100*(b.MIS/a.MIS-1):+.1f}"]
        except IndexError:
            cells += ["--"] * 3
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
(OUT / "tab3_sweeps_vs.tex").write_text("\n".join(rows) + "\n")

# Figure data: MAE vs input length per link and method
for ds in LINKS:
    lines = []
    for L, cfg in [(15, "L15"), (30, "base"), (60, "L60")]:
        vals = {}
        for m in ["prev_seq2seq", "fr_dist"]:
            x = g[(g.cfg == cfg) & (g.dataset == ds) & (g.method == m)]
            vals[m] = (x.MAE.iloc[0], x.MIS.iloc[0]) if len(x) else (np.nan, np.nan)
        lines.append(f"{L*0.2:.0f} {vals['prev_seq2seq'][0]:.4f} {vals['fr_dist'][0]:.4f} {vals['prev_seq2seq'][1]:.4f} {vals['fr_dist'][1]:.4f}")
    (OUT / f"sweepL_{ds}.dat").write_text("Ls s2smae ds2smae s2smis ds2smis\n" + "\n".join(lines) + "\n")
# horizon-30 per-step curves (seed mean)
for ds in LINKS:
    out = {}
    for m in ["prev_seq2seq", "fr_dist"]:
        fs = [ROOT / f"results_final/sweeps/H30/{ds}/{m}_s{s}.csv" for s in [0, 1]]
        fs = [f for f in fs if f.exists()]
        if not fs: continue
        dd = pd.concat([pd.read_csv(f) for f in fs]); dd = dd[dd.target == "rtt"]
        col_mae = "MAE_median" if m == "fr_dist" else "MAE"; col_mis = "MIS_cqr" if m == "fr_dist" else "MIS"
        out[m + "_mae"] = dd.groupby("h")[col_mae].mean(); out[m + "_mis"] = dd.groupby("h")[col_mis].mean()
    if out:
        pd.DataFrame(out).rename_axis("h").reset_index().to_csv(OUT / f"sweepH_{ds}.dat", sep=" ", index=False, float_format="%.4f")
print("ok")
