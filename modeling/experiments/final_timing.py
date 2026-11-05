"""
experiments/final_timing.py - Inference cost of the initial study (S2S) and final report (DS2S)
forecasters for one window on CPU (single thread and default threads) and on
the MPS GPU (batch of 1). Writes results_final/summary/timing.json.
"""
import json, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from models import seq2seq_prev, seq2seq_dist

F, L, H, T = 31, 30, 15, 3
def models():
    return {"S2S": seq2seq_prev.ThroughputSeq2Seq(num_features=9, embed_dim=16, hidden_size=128, input_len=L, output_len=H, num_targets=T),
            "DS2S": seq2seq_dist.ThroughputSeq2SeqDist(num_features=F, embed_dim=16, hidden_size=128, input_len=L, output_len=H,
                                                       num_targets=T, target_feature_indices=[0, 1, 2])}
out = {}
for dev, threads in [("cpu", 1), ("cpu", None), ("mps", None)]:
    if dev == "mps" and not torch.backends.mps.is_available():
        continue
    if threads:
        torch.set_num_threads(threads)
    else:
        torch.set_num_threads(torch.get_num_threads() if dev == "mps" else 8)
    for name, m in models().items():
        m = m.to(dev).eval()
        nf = 9 if name == "S2S" else F
        x = torch.randn(1, L, nf, device=dev); h = torch.zeros(1, L, device=dev); d = torch.zeros(1, L, device=dev)
        with torch.no_grad():
            for _ in range(20): m(x, h, d)
            if dev == "mps": torch.mps.synchronize()
            t0 = time.perf_counter(); n = 200
            for _ in range(n): m(x, h, d)
            if dev == "mps": torch.mps.synchronize()
            ms = (time.perf_counter() - t0) / n * 1000
        key = f"{name}_{dev}_{'1thread' if threads == 1 else 'multi'}"
        out[key] = round(ms, 3)
        print(key, f"{ms:.2f} ms", sum(p.numel() for p in m.parameters()))
json.dump(out, open(HERE.parent / "results_final/summary/timing.json", "w"), indent=1)
