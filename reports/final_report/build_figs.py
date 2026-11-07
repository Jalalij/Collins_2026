"""Compile the analysis figures of the report as standalone PDFs (figs_pdf/<name>.pdf).

The report has more pgfplots figures than TeX's main memory holds in one run, so the
figures written by the fr_*.py analysis scripts are compiled one by one and included
with \\includegraphics. Their source stays in figs/<name>.tex (TikZ/pgfplots) and their
data in data/. Run from reports/final_report after the analysis scripts:

    python3 build_figs.py            # all figures in FIGS
    python3 build_figs.py fr_leak    # one figure
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = ["fr_trace", "fr_volatility", "fr_leak", "fr_heat", "fr_condcov", "fr_condwidth", "fr_bootstrap",
        "fr_scheduler", "fr_scheduler_heat", "fr_scheduler_risk", "fr_attention", "fr_attention_prof", "fr_hzoom"]
WRAP = r"""\documentclass[11pt,border=2pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc,fit,backgrounds}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots,fillbetween,colormaps}
\pgfplotsset{compat=1.18}
\input{data/plotstyle.tex}
\input{data/numbers_story.tex}
\begin{document}
\input{figs/%s.tex}
\end{document}
"""


def main():
    names = sys.argv[1:] or FIGS
    out = HERE / "figs_pdf"
    out.mkdir(exist_ok=True)
    for n in names:
        src = HERE / f"_fig_{n}.tex"
        src.write_text(WRAP % n)
        r = subprocess.run(["tectonic", "-X", "compile", src.name, "--outdir", "figs_pdf"], cwd=HERE,
                           capture_output=True, text=True)
        src.unlink()
        pdf = out / f"_fig_{n}.pdf"
        if r.returncode != 0 or not pdf.exists():
            print("FAILED", n, r.stderr[-800:])
            continue
        pdf.replace(out / f"{n}.pdf")
        print("ok", n)


if __name__ == "__main__":
    main()
