"""Build slides.pptx from slides.pdf (one full-slide image per page) with the
presenter-guide script of each slide in the speaker notes.
Usage: python3 make_pptx.py <dir with s01.png ... s30.png>"""
import re, sys, glob
from pptx import Presentation
from pptx.util import Emu

img_dir = sys.argv[1]
# number macros -> values
mac = {}
for f in glob.glob("data/numbers_*.tex"):
    for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(f).read()):
        mac[k] = v.replace("{,}", ",").replace("10^{-4}", "1e-4")

def clean(t):
    t = t.replace("\\say ", "SAY: ").replace("\\key ", "\n\nKEY NUMBERS: ").replace("\\watch ", "\n\nIF ASKED: ")
    t = re.sub(r"\\(\w+)(\{\})?", lambda m: mac.get(m.group(1), m.group(0)), t)
    for a, b in [("\\%", "%"), ("\\,", " "), ("\\ ", " "), ("``", '"'), ("''", '"'), ("--", "–"), ("~", " "),
                 ("$\\rightarrow$", "→"), ("$\\times$", "×"), ("$-", "-"), ("$+", "+"), ("\\times", "×"), ("\\pm", "±")]:
        t = t.replace(a, b)
    t = re.sub(r"\\(textbf|emph|texttt|textit|say|key|watch)\{?", "", t)
    t = re.sub(r"\\item\s*", "\n• ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?", "", t)
    t = t.replace("{", "").replace("}", "").replace("$", "")
    return re.sub(r"[ \t]+", " ", t).strip()

guide = open("presenter_guide.tex").read()
notes = {}
for m in re.finditer(r"\\slide\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}(.*?)(?=\\slide\{|\\section\*)", guide, re.S):
    nums, title, tm, body = m.groups()
    text = f"{clean(title)} (target {tm})\n\n" + clean(body)
    if "--" in nums or "–" in nums:
        a, b = re.split(r"--|–", nums)
        for n in range(int(a), int(b) + 1): notes[n] = text
    else:
        notes[int(nums)] = text

prs = Presentation()
prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)   # 16:9, 13.333 x 7.5 in
blank = prs.slide_layouts[6]
pngs = sorted(glob.glob(f"{img_dir}/s*.png"))
for i, p in enumerate(pngs, 1):
    s = prs.slides.add_slide(blank)
    s.shapes.add_picture(p, 0, 0, width=prs.slide_width, height=prs.slide_height)
    s.notes_slide.notes_text_frame.text = notes.get(i, "")
prs.save("slides.pptx")
print(len(pngs), "slides,", sum(1 for i in range(1, len(pngs) + 1) if notes.get(i)), "with notes")
