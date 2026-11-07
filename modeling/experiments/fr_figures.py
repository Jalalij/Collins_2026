"""
experiments/fr_figures.py - Template-style figures from the summary files:

  fr_hzoom.tex    RTT MAE (row 1) and calibrated 95% interval score (row 2) vs horizon for three
                  links, all methods in the main panels and a zoomed inset on the learned models
  fr_heat.tex     heatmaps of the change (%) of the final model's ensemble against the previous
                  model's ensemble, link x horizon step, for MAE and interval score

Reads  modeling/results_final/summary/horizon_mean.csv  (written by final_aggregate.py)
Writes <report>/data/fr_hzoom_*.dat, fr_heat_*.dat and <report>/figs/fr_hzoom.tex, fr_heat.tex
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LINKS = [("starlink_dl", "Starlink DL"), ("oneweb_dl", "OneWeb DL"), ("quectel_ul", "5G UL"),
         ("starlink_ul", "Starlink UL"), ("oneweb_ul", "OneWeb UL"), ("quectel_dl", "5G DL")]
# column, style, legend label, in inset?
LINES = [("persistence", "mPers", "Persistence", False), ("snaive", "mSnaive", "Seasonal (1\\,s)", False),
         ("ridge_fr", "mRidge", "Ridge-E", True), ("prev_seq2seq_ens", "mPrev", "S2S-E", True),
         ("fr_gauss_ens", "mGauss", "GS2S-E", True), ("fr_dist_ens", "mFR", "DS2S-E", True),
         ("fr_dist_w4_ens", "mFRw", "DS2S-E ($w{=}4$)", True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ROOT.parent / "reports/final_report"))
    ap.add_argument("--fr", default="fr_dist_ens", help="final-model column for the heatmaps")
    a = ap.parse_args()
    rep = Path(a.report); (rep / "data").mkdir(exist_ok=True); (rep / "figs").mkdir(exist_ok=True)
    hm = pd.read_csv(ROOT / "results_final/summary/horizon_mean.csv")
    hm = hm[hm.target == "rtt"]
    have = set(hm.method)
    lines = [l for l in LINES if l[0] in have]
    # ---- data per link
    for ds, _ in LINKS:
        for met in ["MAE", "MIS"]:
            w = hm[hm.dataset == ds].pivot(index="h", columns="method", values=met)
            w = w[[l[0] for l in lines if l[0] in w.columns]]
            w.columns = [c.replace("_", "") for c in w.columns]
            w.reset_index().assign(t=lambda d: d.h * 0.2).to_csv(rep / f"data/fr_hzoom_{ds}_{met}.dat", sep=" ", index=False, float_format="%.5g")
    # ---- figure: 2 rows x 3 links, insets on row 1
    show = LINKS[:3]
    out = [r"\begin{tikzpicture}",
           r"\begin{groupplot}[group style={group size=3 by 2, horizontal sep=1.25cm, vertical sep=1.45cm},",
           r"  frbase, width=5.9cm, height=4.3cm, xmin=0.2, xmax=3, xtick={0.2,1,2,3}]"]
    insets = []
    for row, met, ylab in [(1, "MAE", "RTT MAE (ms)"), (2, "MIS", "RTT interval score (ms)")]:
        for i, (ds, nm) in enumerate(show):
            letter = "abcdef"[(row - 1) * 3 + i]
            opts = [f"title={{({letter}) {nm}}}"]
            if i == 0: opts.append(f"ylabel={{{ylab}}}")
            if row == 2: opts.append("xlabel={horizon (s)}")
            if row == 1 and i == 0: opts.append("legend to name=leghz, legend columns=6")
            out.append(rf"\nextgroupplot[{', '.join(opts)}]")
            d = pd.read_csv(rep / f"data/fr_hzoom_{ds}_{met}.dat", sep=" ")
            for col, sty, lab, inz in lines:
                c = col.replace("_", "")
                if c not in d.columns: continue
                if col in ("persistence", "snaive"): continue
                out.append(rf"\addplot[{sty}] table[x=t, y={c}] {{data/fr_hzoom_{ds}_{met}.dat}};" + (rf" \addlegendentry{{{lab}}}" if row == 1 and i == 0 else ""))
            if met == "MAE":
                learned = [col.replace("_", "") for col, _, _, inz in lines if inz and col.replace("_", "") in d.columns]
                first = d[d.t <= 1.0 + 1e-9]
                lo, hi = float(first[learned].min().min()), float(first[learned].max().max())
                pad = 0.08 * (hi - lo + 1e-9); lo -= pad; hi += pad
                x0, x1 = 0.2, 1.0
                out.append(rf"\draw[black!70] (axis cs:{x0},{lo:.4g}) rectangle (axis cs:{x1},{hi:.4g});")
                out.append(rf"\coordinate (zb{i}) at (axis cs:{(x0+x1)/2:.3g},{hi:.4g});")
                allmax = float(d[[col.replace('_','') for col, _, _, _ in lines if col.replace('_','') in d.columns]].max().max())
                insets.append((i, ds, lo, hi, learned, allmax))
    out.append(r"\end{groupplot}")
    # insets placed in the upper part of each row-1 panel
    for i, ds, lo, hi, learned, allmax in insets:
        out.append(rf"\begin{{axis}}[at={{($(group c{i+1}r1.south east)+(-0.15cm,0.18cm)$)}}, anchor=south east, width=3.0cm, height=2.25cm,"
                   rf" xmin=0.2, xmax=1.0, ymin={lo:.4g}, ymax={hi:.4g}, xtick={{0.2,0.6,1}}, tick label style={{font=\tiny}}, axis background/.style={{fill=white}},"
                   rf" grid=major, grid style={{gray!20}}, name=ins{i}]")
        for col, sty, lab, inz in lines:
            c = col.replace("_", "")
            if inz and c in learned:
                out.append(rf"\addplot[{sty}, mark size=0.8pt] table[x=t, y={c}] {{data/fr_hzoom_{ds}_MAE.dat}};")
        out.append(r"\end{axis}")
        out.append(rf"\draw[->, black!70] (zb{i}) -- (ins{i}.north west);")
    out.append(r"\node at ($(group c2r1.north)+(0,0.95cm)$) {\pgfplotslegendfromname{leghz}};")
    out.append(r"\end{tikzpicture}")
    (rep / "figs/fr_hzoom.tex").write_text("\n".join(out) + "\n")

    # ---- heatmaps: change (%) of the final model vs the previous model (ensembles), link x step
    rows = []
    for k, (ds, nm) in enumerate(LINKS):
        for met in ["MAE", "MIS"]:
            w = hm[(hm.dataset == ds)].pivot(index="h", columns="method", values=met)
            if a.fr not in w.columns: continue
            ch = 100 * (w[a.fr] / w["prev_seq2seq_ens"] - 1)
            for h, v in ch.items():
                rows.append({"met": met, "x": h, "y": k, "v": v})
    hd = pd.DataFrame(rows)
    for met in ["MAE", "MIS"]:
        hd[hd.met == met][["x", "y", "v"]].sort_values(["y", "x"]).to_csv(rep / f"data/fr_heat_{met}.dat", sep=" ", index=False, float_format="%.2f")
    vmax = float(np.ceil(hd.v.abs().max() / 5) * 5)
    names = ",".join(nm for _, nm in LINKS)
    out = [r"\begin{tikzpicture}",
           r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=0.45cm, yticklabels at=edge left},",
           rf"  width=7.0cm, height=4.6cm, enlargelimits=false, xmin=0.5, xmax=15.5, ymin=-0.5, ymax=5.5, ytick={{0,...,5}}, yticklabels={{{names}}},",
           r"  xtick={1,5,10,15}, xticklabels={0.2,1,2,3}, xlabel={horizon (s)}, tick label style={font=\scriptsize}, label style={font=\scriptsize},",
           rf"  title style={{font=\footnotesize}}, colormap={{bwr}}{{rgb255=(33,102,172) rgb255=(247,247,247) rgb255=(178,24,43)}}, point meta min={-vmax}, point meta max={vmax}, y dir=reverse]"]
    for j, (met, ttl) in enumerate([("MAE", "(a) RTT MAE change (\\%)"), ("MIS", "(b) RTT interval-score change (\\%)")]):
        extra = ", colorbar right, colorbar style={width=0.18cm, ytick={" + f"{-vmax:.0f},{-vmax/2:.0f},0,{vmax/2:.0f},{vmax:.0f}" + "}, yticklabel style={font=\\scriptsize}}" if j == 1 else ""
        out.append(rf"\nextgroupplot[title={{{ttl}}}{extra}]")
        out.append(rf"\addplot[matrix plot*, mesh/cols=15, point meta=explicit] table[x=x, y=y, meta=v] {{data/fr_heat_{met}.dat}};")
    out += [r"\end{groupplot}", r"\end{tikzpicture}"]
    (rep / "figs/fr_heat.tex").write_text("\n".join(out) + "\n")

    # ---- column version (paper): panels stacked, overall change per link written right of each row
    tot = {}
    for ds, _ in LINKS:
        w = hm[hm.dataset == ds].groupby("method")[["MAE", "MIS"]].mean()
        if a.fr in w.index:
            tot[ds] = {met: 100 * (w.loc[a.fr, met] / w.loc["prev_seq2seq_ens", met] - 1) for met in ["MAE", "MIS"]}
    out = [r"% fr_heat_col.tex - generated by modeling/experiments/fr_figures.py, do not edit by hand.",
           r"\begin{tikzpicture}",
           r"\begin{groupplot}[group style={group size=1 by 2, vertical sep=0.95cm, xticklabels at=edge bottom, xlabels at=edge bottom},",
           rf"  width=6.0cm, height=3.55cm, enlargelimits=false, xmin=0.5, xmax=15.5, ymin=-0.5, ymax=5.5, ytick={{0,...,5}}, yticklabels={{{names}}},",
           r"  xtick={1,5,10,15}, xticklabels={0.2,1,2,3}, xlabel={horizon (s)}, tick label style={font=\scriptsize}, label style={font=\scriptsize},",
           rf"  title style={{font=\footnotesize, yshift=-0.6ex}}, colormap={{bwr}}{{rgb255=(33,102,172) rgb255=(247,247,247) rgb255=(178,24,43)}}, point meta min={-vmax}, point meta max={vmax}, y dir=reverse, clip=false]"]
    for j, (met, ttl) in enumerate([("MAE", "(a) RTT MAE change (\\%)"),
                                    ("MIS", "(b) RTT interval-score change (\\%)")]):
        extra = (", colorbar horizontal, colorbar style={at={(0.5,-0.62)}, anchor=north, height=0.16cm, xtick={"
                 + f"{-vmax:.0f},{-vmax/2:.0f},0,{vmax/2:.0f},{vmax:.0f}" + "}, xticklabel style={font=\\scriptsize}}") if j == 1 else ""
        out.append(rf"\nextgroupplot[title={{{ttl}}}{extra}]")
        out.append(rf"\addplot[matrix plot*, mesh/cols=15, point meta=explicit] table[x=x, y=y, meta=v] {{data/fr_heat_{met}.dat}};")
        for k, (ds, _) in enumerate(LINKS):
            if ds in tot:
                out.append(rf"\node[anchor=west, font=\scriptsize] at (axis cs:15.6,{k}) {{${tot[ds][met]:+.0f}$}};")
    out += [r"\end{groupplot}", r"\end{tikzpicture}"]
    (rep / "figs/fr_heat_col.tex").write_text("\n".join(out) + "\n")
    print("ok", [l[0] for l in lines], "vmax", vmax)


if __name__ == "__main__":
    main()
