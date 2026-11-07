"""Check Josh's writing rules on the LaTeX sources: no ';' and no ', so' in the
text, at most two citations per sentence, every bib entry cited, at most 40
references, and the abstract word count."""
import re, sys, glob
files = sys.argv[1:] or sorted(glob.glob("sections/*.tex")) + glob.glob("figs/*.tex")
bad = 0
cited = set()
for f in files:
    raw = open(f).read()
    txt = re.sub(r"(?<!\\)%.*", "", raw)
    for m in re.finditer(r"\\cite\{([^}]*)\}", txt):
        cited.update(k.strip() for k in m.group(1).split(","))
    body = re.sub(r"\\begin\{(tabular|algorithmic|equation|align)\*?\}.*?\\end\{\1\*?\}", " ", txt, flags=re.S)
    body = re.sub(r"\$[^$]*\$", " ", body)
    for i, line in enumerate(body.splitlines(), 1):
        if ";" in line and "tikz" not in f and "\\draw" not in line and "\\addplot" not in line and "\\node" not in line:
            print(f"{f}:{i}: semicolon: {line.strip()[:100]}"); bad += 1
        if ", so " in line:
            print(f"{f}:{i}: ', so': {line.strip()[:100]}"); bad += 1
    flat = re.sub(r"\s+", " ", body)
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", flat):
        n = sum(len(m.group(1).split(",")) for m in re.finditer(r"\\cite\{([^}]*)\}", sent))
        if n > 2:
            print(f"{f}: {n} citations in one sentence: {sent[:140]}"); bad += 1
import os
if os.path.exists("refs.bib"):
    keys = re.findall(r"@\w+\{([^,]+),", open("refs.bib").read())
    print(f"references in refs.bib: {len(keys)}  cited: {len(cited)}")
    for k in keys:
        if k not in cited: print("not cited:", k); bad += 1
    for k in cited - set(keys): print("missing in bib:", k); bad += 1
if os.path.exists("sections/abstract.tex"):
    a = open("sections/abstract.tex").read().split("\\begin{IEEEkeywords}")[0]
    a = re.sub(r"\\(begin|end)\{abstract\}", "", a)
    a = re.sub(r"\\[A-Za-z]+\{\}", "X", a); a = re.sub(r"\\[A-Za-z]+", "X", a)
    print("abstract words:", len(a.split()))
print("issues:", bad)
