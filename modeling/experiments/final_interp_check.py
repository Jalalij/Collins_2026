"""
experiments/final_interp_check.py - Evidence for audit item A3 (Starlink RTT
interpolation). On the original (legacy) Starlink DL data, for test windows
whose last input row is an interpolated ping, how often does that value lie
strictly between the previous row and the first target (the signature of an
interpolation toward the future)? Writes results_final/audit/interp_check.json.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import final_benchmark as B

out = {}
for d in ["dl", "ul"]:
    data, meta = B.load_splits("starlink", d, "y2", 30, 15, torch.device("cpu"),
                               str(HERE.parent / "data/lsn-data"), causal_ping=False)
    # same files, same order: take the interpolation mask from the causal load
    cdata, _ = B.load_splits("starlink", d, "y2", 30, 15, torch.device("cpu"),
                             str(HERE.parent / "data/lsn-data"), causal_ping=True)
    data["test"].M = cdata["test"].M
    te = data["test"]; t = meta["targets"].index("ping_server_rtt_ms"); fi = meta["target_idx"][t]
    s = te.starts.numpy(); X = te.X.numpy()[:, fi]; Y = te.Y.numpy()[:, t]; M = te.M.numpy()[:, t]
    last = X[s + 29]; prev = X[s + 28]; y1 = Y[s + 30]; m_last = M[s + 29] > 0
    between = ((last - prev) * (y1 - last) > 0) | ((last == prev) & (last == y1))
    out[d] = {"windows": int(len(s)), "frac_last_row_interpolated": float(m_last.mean()),
              "frac_between_if_interpolated": float(between[m_last].mean()),
              "frac_between_if_real": float(between[~m_last].mean())}
json.dump(out, open(HERE.parent / "results_final/audit/interp_check.json", "w"), indent=1)
print(json.dumps(out, indent=1))
