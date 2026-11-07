"""Copy the generated report tables and data into the paper and rename the
methods to the paper's labels. Run after modeling/experiments/final_tables.py."""
import shutil, glob, os
src = "../reports/final_report/data"
dst = "figs/data"
os.makedirs(dst, exist_ok=True)
for f in glob.glob(f"{src}/*.dat") + glob.glob(f"{src}/numbers_*.tex"):
    shutil.copy(f, dst)
REN = {
    "tab_main.tex": [("Initial (retrained)", "S2S"), ("Final ensemble", "DS2S-E"), ("Initial ensemble", "S2S-E"), ("& Final &", "& DS2S &")],
    "tab_ablation.tex": [("1: residual decoder", "S2S-A"), ("1+2: + data pipeline", "S2S-AE"),
                         ("1+2+3: + distrib. heads", "DS2S"), ("1--4: + 3-seed ensemble", "DS2S-E")],
    "tab_baselines.tex": [("Ridge-E", "Ridge-E"), ("Initial Seq2Seq-GRU", "S2S"), ("Final ensemble", "DS2S-E"),
                          ("Final &", "DS2S &"), ("Seasonal repeat (1\\,s)", "Seasonal")],
    "tab_isproto.tex": [],
    "tab3_sweeps_vs.tex": [],
    "tab3_sweeps_ds2s.tex": [("L=30, H=15, d=128, w=1", "default")],
    "tab_sessadapt.tex": [],
    "tab_targets.tex": [("& Initial &", "& S2S &"), ("& Final ens. &", "& DS2S-E &"), ("& Final &", "& DS2S &")],
}
for name, reps in REN.items():
    s = open(f"{src}/{name}").read()
    for a, b in reps:
        s = s.replace(a, b)
    open(f"{dst}/{name.replace('.tex', '_paper.tex')}", "w").write(s)
print("ok")

# Figures generated for the report by the fr_*.py analysis scripts: copy them and point their
# data paths at figs/data (the .dat files were copied above).
import re
FIGS = ["fr_trace", "fr_volatility", "fr_leak", "fr_heat_col", "fr_condcov", "fr_condwidth",
        "fr_scheduler", "fr_scheduler_risk", "fr_scheduler_heat", "fr_bootstrap", "fr_attention"]
for f in FIGS:
    srcf = f"../reports/final_report/figs/{f}.tex"
    if not os.path.exists(srcf):
        print("missing", srcf); continue
    s = open(srcf).read()
    s = re.sub(r"(?<![\w/])data/", "figs/data/", s)
    open(f"figs/{f}.tex", "w").write(s)
shutil.copy("../reports/final_report/data/plotstyle.tex", "figs/plotstyle.tex")
print("figures ok")
