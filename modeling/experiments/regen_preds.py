"""
experiments/regen_preds.py - rebuild results_final/main/<link>/preds/*.npz without retraining.

The prediction arrays are git-ignored (several GB). This script recreates them:
  - neural methods: load models/<method>_s<seed>.pt and predict val + test
  - persistence, snaive, ridge, ridge_fr: recomputed (deterministic)
  - lqr_fr: refit with the same seed (no checkpoint is saved; takes seconds)
Each regenerated method is scored with final_benchmark.metrics and compared
with the saved <method>_s<seed>.csv, so a mismatch is reported.

Usage
-----
python experiments/regen_preds.py                     # all six links
python experiments/regen_preds.py --links starlink_dl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_benchmark as fb  # noqa: E402

LINKS = ["oneweb_dl", "oneweb_ul", "starlink_dl", "starlink_ul", "quectel_dl", "quectel_ul"]
NONNEURAL = ("persistence", "snaive", "ridge", "ridge_fr", "lqr_fr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", nargs="+", default=LINKS)
    ap.add_argument("--res_dir", default=str(fb.ROOT / "results_final/main"))
    ap.add_argument("--data_root", default=str(fb.ROOT / "data/lsn-data"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    dev = fb.get_device()
    L, H = 30, 15
    report = []

    for link in a.links:
        provider, direction = link.split("_")
        ldir = Path(a.res_dir) / link
        pdir = ldir / "preds"; pdir.mkdir(exist_ok=True)
        jobs = sorted(p.stem for p in ldir.glob("*_s[0-9].csv"))
        cache = {}

        def get(variant, stride=1):
            if variant not in cache:
                t0 = time.time()
                cache[variant] = fb.load_splits(provider, direction, variant, L, H, dev, a.data_root)
                cache[variant][0]["train"].full_starts = cache[variant][0]["train"].starts
                print(f"[{link}] data {variant} loaded ({time.time()-t0:.0f}s)", flush=True)
            data, meta = cache[variant]
            # Only the fitted linear baselines depend on the training windows.
            data["train"].starts = data["train"].full_starts[::stride].contiguous()
            return data, meta

        def denorm(t, meta):
            return t.cpu().numpy() * meta["tstd"] + meta["tmean"]

        def dist_denorm(o, meta):
            sd = torch.tensor(meta["tstd"])[None, None, :, None]
            mu = torch.tensor(meta["tmean"])[None, None, :, None]
            return (o.cpu() * sd + mu).numpy()

        for job in jobs:
            method, seed = job.rsplit("_s", 1)
            seed = int(seed)
            out = pdir / f"{job}.npz"
            if out.exists() and not a.overwrite:
                print(f"[{link}] skip {job} (exists)"); continue
            variant = "fr" if method.startswith(("fr_full", "fr_dist", "fr_gauss", "ridge_fr", "lqr_fr")) else "base"
            # The 5G runs kept every k-th training window (--train_stride); the
            # stride is read from the run's json. The neural models are loaded,
            # so this only changes ridge / ridge_fr / lqr_fr.
            stride = json.load(open(ldir / f"{job}.json"))["settings"].get("train_stride", 1)
            data, meta = get(variant, stride)
            tidx = meta["target_idx"]
            t0 = time.time()
            d_te = d_va = None
            if method == "persistence":
                p_te = denorm(fb.persistence_pred(data["test"], L, H, tidx), meta)
                p_va = denorm(fb.persistence_pred(data["val"], L, H, tidx), meta)
            elif method == "snaive":
                p_te = denorm(fb.snaive_pred(data["test"], L, H, tidx), meta)
                p_va = denorm(fb.snaive_pred(data["val"], L, H, tidx), meta)
            elif method in ("ridge", "ridge_fr"):
                pr = fb.ridge_fit_predict(data, L, H, tidx)
                p_te, p_va = denorm(pr["test"], meta), denorm(pr["val"], meta)
            elif method == "lqr_fr":
                o_te, o_va = fb.lqr_fit_predict(data, L, H, tidx, seed, log=lambda s: None)
                d_te, d_va = dist_denorm(o_te, meta), dist_denorm(o_va, meta)
                p_te, p_va = d_te[..., 0], d_va[..., 0]
            else:
                ck = torch.load(ldir / "models" / f"{job}.pt", map_location="cpu", weights_only=False)
                assert ck["features"] == meta["features"], f"{job}: feature list differs"
                model = fb.build_model(method, meta, L, H)
                model.load_state_dict(ck["state_dict"])
                model.to(dev)
                o_te, o_va = fb.predict(model, data["test"], L, H), fb.predict(model, data["val"], L, H)
                if o_te.dim() == 4:
                    d_te, d_va = dist_denorm(o_te, meta), dist_denorm(o_va, meta)
                    p_te, p_va = d_te[..., 0], d_va[..., 0]
                else:
                    p_te, p_va = denorm(o_te, meta), denorm(o_va, meta)
                del model
                if dev.type == "mps":
                    torch.mps.empty_cache()

            np.savez_compressed(out, p_te=p_te.astype(np.float32), p_va=p_va.astype(np.float32),
                                **({} if d_te is None else {"d_te": d_te.astype(np.float16),
                                                            "d_va": d_va.astype(np.float16)}))
            y_te = denorm(fb.gather_targets(data["test"], L, H), meta)
            y_va = denorm(fb.gather_targets(data["val"], L, H), meta)
            pers_te = denorm(fb.persistence_pred(data["test"], L, H, tidx), meta)
            if not (pdir / "truth.npz").exists():
                tmp = pdir / f"truth_tmp_{os.getpid()}.npz"
                np.savez_compressed(tmp, y_te=y_te.astype(np.float32), y_va=y_va.astype(np.float32),
                                    pers_te=pers_te.astype(np.float32),
                                    interp_te=fb.gather_interp_mask(data["test"], L, H).cpu().numpy().astype(np.uint8),
                                    targets=np.array(meta["targets"]))
                os.replace(tmp, pdir / "truth.npz")

            # Check against the saved metrics of the original run.
            df = fb.metrics(p_te, y_te, p_va, y_va, pers_te, meta, dist_te=d_te, dist_va=d_va)
            ref = pd.read_csv(ldir / f"{job}.csv")
            mae_new, mae_ref = df.MAE.mean(), ref.MAE.mean()
            rel = abs(mae_new - mae_ref) / max(abs(mae_ref), 1e-12)
            report.append(dict(link=link, job=job, MAE_saved=mae_ref, MAE_regen=mae_new, rel_diff=rel))
            print(f"[{link}] {job:18s} MAE saved={mae_ref:.5f} regen={mae_new:.5f} "
                  f"rel diff={rel:.1e} ({time.time()-t0:.0f}s)", flush=True)
        del cache
        if dev.type == "mps":
            torch.mps.empty_cache()

    if report:
        rep = pd.DataFrame(report)
        rep.to_csv(Path(a.res_dir) / "regen_preds_check.csv", index=False)
        print(f"\n{len(rep)} prediction files written; max relative MAE difference "
              f"{rep.rel_diff.max():.1e} ({rep.loc[rep.rel_diff.idxmax(), 'link']} "
              f"{rep.loc[rep.rel_diff.idxmax(), 'job']})")


if __name__ == "__main__":
    main()
