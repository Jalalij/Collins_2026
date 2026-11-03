"""Copy the generated report tables and data into the paper and rename the
methods to the paper's labels. Run after modeling/experiments/y3_tables.py."""
import shutil, glob, os
src = "../reports/year3/data"
dst = "figs/data"
os.makedirs(dst, exist_ok=True)
for f in glob.glob(f"{src}/*.dat") + glob.glob(f"{src}/numbers_*.tex"):
    shutil.copy(f, dst)
REN = {
    "tab_main.tex": [("Y2 retrained", "S2S"), ("Y3 ensemble", "DS2S-E"), ("Y2 ensemble", "S2S-E"), ("& Y3 &", "& DS2S &")],
    "tab_ablation.tex": [("1: residual decoder", "S2S-A"), ("1+2: + data pipeline", "S2S-AE"),
                         ("1+2+3: + distrib. heads", "DS2S"), ("1--4: + 3-seed ensemble", "DS2S-E")],
    "tab_baselines.tex": [("Ridge-Y3", "Ridge-E"), ("Y2 Seq2Seq-GRU", "S2S"), ("Y3 ensemble", "DS2S-E"),
                          ("Y3 &", "DS2S &"), ("Seasonal repeat (1\\,s)", "Seasonal")],
    "tab_isproto.tex": [],
    "tab3_sweeps_vs.tex": [],
    "tab3_sweeps_ds2s.tex": [("L=30, H=15, d=128, w=1", "default")],
    "tab_sessadapt.tex": [],
    "tab_targets.tex": [("& Y2 &", "& S2S &"), ("& Y3 ens. &", "& DS2S-E &"), ("& Y3 &", "& DS2S &")],
}
for name, reps in REN.items():
    s = open(f"{src}/{name}").read()
    for a, b in reps:
        s = s.replace(a, b)
    open(f"{dst}/{name.replace('.tex', '_paper.tex')}", "w").write(s)
print("ok")
