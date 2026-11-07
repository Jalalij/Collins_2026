"""experiments/fr_dataset_stats.py - Size of the measurement data set (sessions, rows, hours, dates)
used in the paper. Writes results_final/analysis/dataset_stats.json."""
import glob, json, re
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/lsn-data"
SETS = {"oneweb_dl": ("oneweb/dl", "timestamp"), "oneweb_ul": ("oneweb/ul", "timestamp"),
        "starlink_dl": ("starlink/latency_loss_throughput/dl", "timestamp_cst"),
        "starlink_ul": ("starlink/latency_loss_throughput/ul", "timestamp_cst"),
        "quectel_dl": ("quectel/latency_loss_throughput/dl", "timestamp_cst"),
        "quectel_ul": ("quectel/latency_loss_throughput/ul", "timestamp_cst")}
out = {}
for k, (sub, ts) in SETS.items():
    fs = sorted(glob.glob(str(D / sub / "*.csv")))
    rows, days = 0, set()
    for f in fs:
        n = sum(1 for _ in open(f)) - 1
        rows += n
        m = re.search(r"_(\d{8})_", Path(f).name)
        if m: days.add(m.group(1))
    out[k] = {"files": len(fs), "rows": rows, "hours": rows * 0.2 / 3600, "days": len(days),
              "first_day": min(days) if days else None, "last_day": max(days) if days else None}
out["total"] = {"files": sum(v["files"] for v in out.values()), "rows": sum(v["rows"] for v in out.values()),
                "hours": sum(v["hours"] for v in out.values())}
(ROOT / "results_final/analysis").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "results_final/analysis/dataset_stats.json", "w"), indent=1)
print(json.dumps(out, indent=1))
