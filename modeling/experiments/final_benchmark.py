"""
experiments/final_benchmark.py - final report benchmark: initial study baseline vs final report changes.

Runs every method on the same session-level split that initial study used
(rng seed 42, 70/15/15 by session file), for one (provider, direction) at a time,
and writes per-target / per-horizon metrics to a CSV.

Methods
-------
persistence   last observed value held over the horizon (no learning)
ridge         direct multi-horizon ridge regression on the last input row
              plus the recent target history (closed form)
y2_seq2seq    the initial study Seq2Seq-GRU (models/seq2seq_prev.py), initial study data pipeline
y3_resid      final report Seq2Seq-GRU with residual decoder (models/seq2seq.py)
y3_full       y3_resid + sentinel cleaning + causal engineered features
              + input clipping (|z| <= 5)
y3_full_huber y3_full trained with a Huber loss (delta = 1 in z units)

Windows are gathered on the GPU from contiguous per-split arrays instead of a
Python list of windows, which makes an epoch on the 5G data ~10x faster. The
values are identical to dataset.ThroughputFileDataset (same fill, same stats).

Intervals are split-conformal: the 95% half-width for each (target, horizon)
is the empirical quantile of |residual| on the validation split, then applied
to the test split. initial study took the quantile on the test residuals themselves,
which makes PICP 0.95 by construction; that number is also written
(PICP_test_quantile / MIS_test_quantile) so both can be compared.

Usage
-----
python experiments/final_benchmark.py --provider oneweb --direction dl \
    --methods persistence ridge y2_seq2seq y3_resid y3_full --seeds 0 1 2
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
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    DataConfig, find_session_files, load_csv, _validate_and_filter_df, get_device, target_source,
)
from models import seq2seq_prev  # noqa: E402
from models import seq2seq as seq2seq_y3  # noqa: E402
from models import seq2seq_dist  # noqa: E402

EXP_TYPE = {
    ("oneweb", "dl"): "dl",
    ("oneweb", "ul"): "ul",
    ("starlink", "dl"): "latency_loss_throughput_dl",
    ("starlink", "ul"): "latency_loss_throughput_ul",
    ("quectel", "dl"): "latency_loss_throughput_dl",
    ("quectel", "ul"): "latency_loss_throughput_ul",
}
TARGETS = ["rtt", "jitter", "loss"]
SHORT = {
    "rtt_mean": "rtt", "ping_server_rtt_ms": "rtt", "ping_rtt_ms": "rtt",
    "jitter_ms": "jitter", "iperf_jitter_ms": "jitter",
    "packet_loss_percent": "loss", "iperf_loss_pct": "loss",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class SplitData:
    """One split held as contiguous arrays on the device plus window starts."""

    def __init__(self, feats, hours, dows, targs, starts, device):
        self.X = torch.from_numpy(feats).to(device)
        self.H = torch.from_numpy(hours).to(device)
        self.D = torch.from_numpy(dows).to(device)
        self.Y = torch.from_numpy(targs).to(device)
        self.starts = torch.from_numpy(starts).to(device)

    def __len__(self):
        return int(self.starts.numel())

    def batch(self, idx, L, H):
        s = self.starts[idx]
        ii = s[:, None] + torch.arange(L, device=s.device)[None]
        oo = s[:, None] + L + torch.arange(H, device=s.device)[None]
        return self.X[ii], self.H[ii], self.D[ii], self.Y[oo]


EVENT_COL = {"oneweb": "satellite_change", "starlink": "satellite_handoff", "quectel": "handover"}


def load_splits(provider, direction, variant, input_len, output_len, device, data_root,
                causal_ping=True):
    """variant: 'y2' (initial study pipeline) or 'y3' (clean + engineered + clip).

    causal_ping (default on for every method): ping values that the parser
    interpolated from the next real ping are replaced in the INPUTS by the last
    real ping; targets keep the measured series (dataset.causal_ping_fix).
    """
    y3 = variant == "y3"
    cfg = DataConfig(
        provider=provider, experiment_type=EXP_TYPE[(provider, direction)],
        data_root=data_root, targets=TARGETS, input_len=input_len,
        output_len=output_len, clean=y3, engineered=y3,
        clip_z=5.0 if y3 else None, seed=42, causal_ping=causal_ping,
    )
    feats, targs = cfg.features, cfg.targets

    dfs = []
    for p in find_session_files(cfg):
        df = load_csv(p, cfg)
        if df is None:
            continue
        df = _validate_and_filter_df(df, feats, targs, cfg, p)
        if df is not None:
            dfs.append(df)

    # Identical split logic to dataset.build_dataloaders.
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(dfs))
    n_tr = max(1, int(len(dfs) * cfg.train_frac))
    n_va = max(1, int(len(dfs) * cfg.val_frac))
    parts = {"train": [dfs[i] for i in idx[:n_tr]],
             "val": [dfs[i] for i in idx[n_tr:n_tr + n_va]],
             "test": [dfs[i] for i in idx[n_tr + n_va:]]}

    avail = [f for f in feats if all(f in d.columns for d in parts["train"])]
    tr = pd.concat(parts["train"], ignore_index=True)
    trf = tr[avail].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0)
    fmean = trf.values.astype(np.float32).mean(0)
    fstd = trf.values.astype(np.float32).std(0) + 1e-8
    tmean = np.array([fmean[avail.index(t)] for t in targs], np.float32)
    tstd = np.array([fstd[avail.index(t)] for t in targs], np.float32)
    clip_cols = [i for i, c in enumerate(avail) if c not in targs]

    out = {}
    for name, frames in parts.items():
        F, Hh, Dd, Y, S, M = [], [], [], [], [], []
        TS, SID, EV = [], [], []
        off = 0
        for df in frames:
            ts = [c for c in df.columns if "timestamp" in c.lower()]
            if ts:
                df = df.sort_values(ts[0]).reset_index(drop=True)
            x = df.reindex(columns=avail).apply(pd.to_numeric, errors="coerce")
            x = x.ffill().bfill().fillna(0).values.astype(np.float32)
            x = (x - fmean) / fstd
            if y3:
                x[:, clip_cols] = np.clip(x[:, clip_cols], -5.0, 5.0)
            y = pd.DataFrame({t: pd.to_numeric(df[target_source(df, t)], errors="coerce")
                              for t in targs}).ffill().bfill().fillna(0)
            y = ((y.values.astype(np.float32)) - tmean) / tstd
            im = np.stack([df[t + "__interp"].values if t + "__interp" in df.columns
                           else np.zeros(len(df)) for t in targs], 1).astype(np.float32)
            n = len(df)
            nw = n - input_len - output_len + 1
            if nw <= 0:
                continue
            F.append(x); Y.append(y); M.append(im)
            # Analysis-only metadata (not model inputs): time stamp, session index, handover event
            tcol = ts[0] if ts else None
            TS.append(pd.to_datetime(df[tcol], errors="coerce").astype("int64").values / 1e9 if tcol else np.zeros(len(df)))
            SID.append(np.full(len(df), len(SID)))
            evc = EVENT_COL.get(provider)
            EV.append(pd.to_numeric(df[evc], errors="coerce").fillna(0).values.astype(np.float32) if evc in df.columns else np.zeros(len(df), np.float32))
            Hh.append(df["hour"].values.astype(np.float32))
            Dd.append(df["dow"].values.astype(np.float32))
            S.append(off + np.arange(nw))
            off += n
        out[name] = SplitData(np.concatenate(F), np.concatenate(Hh), np.concatenate(Dd),
                              np.concatenate(Y), np.concatenate(S).astype(np.int64), device)
        out[name].M = torch.from_numpy(np.concatenate(M)).to(device)   # 1 = interpolated target row
        out[name].ts, out[name].sid, out[name].ev = np.concatenate(TS), np.concatenate(SID), np.concatenate(EV)
    meta = dict(features=avail, targets=targs, tmean=tmean, tstd=tstd,
                fmean=fmean, fstd=fstd, clip_cols=clip_cols if y3 else [], clip_z=5.0 if y3 else None,
                variant=variant, causal_ping=causal_ping,
                n_sessions={k: len(v) for k, v in parts.items()},
                n_windows={k: len(v) for k, v in out.items()},
                target_idx=[avail.index(t) for t in targs])
    return out, meta


# ---------------------------------------------------------------------------
# Training (mirrors train.py: AdamW 1e-3/1e-3, clip 5, ReduceLROnPlateau 0.5/8,
# EMA(0.3) val-loss model selection, TF annealed 0.5 -> 0 over half the epochs)
# ---------------------------------------------------------------------------

def predict(model, split, L, H, bs=4096):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(split), bs):
            idx = torch.arange(i, min(i + bs, len(split)), device=split.starts.device)
            x, h, d, _ = split.batch(idx, L, H)
            p = model(x, h, d)
            outs.append((p[0] if isinstance(p, tuple) else p).float())
            if x.device.type == "mps":
                # The MPS allocator otherwise keeps ~2 GB per batch for the initial study
                # decoder, which ran the 5G jobs out of memory.
                torch.mps.synchronize(); torch.mps.empty_cache()
    return torch.cat(outs)


def gather_interp_mask(split, L, H, bs=8192):
    ms = []
    for i in range(0, len(split), bs):
        idx = torch.arange(i, min(i + bs, len(split)), device=split.starts.device)
        s = split.starts[idx]
        oo = s[:, None] + L + torch.arange(H, device=s.device)[None]
        ms.append(split.M[oo])
    return torch.cat(ms)


def gather_targets(split, L, H, bs=8192):
    ys = []
    for i in range(0, len(split), bs):
        idx = torch.arange(i, min(i + bs, len(split)), device=split.starts.device)
        x, _, _, y = split.batch(idx, L, H)
        ys.append(y)
    return torch.cat(ys)


def train_model(model, data, L, H, epochs, patience, loss_name, batch_size, seed, log):
    dev = data["train"].starts.device
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=8, min_lr=1e-6)
    if loss_name == "dist":
        crit = lambda o, y: seq2seq_dist.dist_loss(o, y, quantile_weight=QWEIGHT)
    else:
        crit = nn.MSELoss() if loss_name == "mse" else nn.HuberLoss(delta=1.0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    tr, va = data["train"], data["val"]
    best, best_state, ema, bad = float("inf"), None, None, 0
    yv = gather_targets(va, L, H)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tf = max(0.0, 0.5 * (1 - ep / max(1, epochs / 2)))
        perm = torch.randperm(len(tr), generator=g).to(dev)
        tot = torch.zeros((), device=dev)
        for i in range(0, len(tr), batch_size):
            idx = perm[i:i + batch_size]
            x, h, d, y = tr.batch(idx, L, H)
            opt.zero_grad(set_to_none=True)
            out = model(x, h, d, y, tf)
            p = out[0] if isinstance(out, tuple) else out
            loss = crit(p, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.detach() * idx.numel()
        pv = predict(model, va, L, H)
        vl = float(crit(pv, yv))  # selection on the training criterion, as in initial study
        ema = vl if ema is None else 0.3 * vl + 0.7 * ema
        sch.step(ema)
        if np.isfinite(ema) and ema < best:
            best, bad = ema, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"    ep {ep:3d} train={tot.item()/len(tr):.4f} val={vl:.4f} ema={ema:.4f} "
            f"lr={opt.param_groups[0]['lr']:.1e} [{time.time()-t0:.1f}s]")
        if bad >= patience:
            break
    if best_state is None:
        log("    [warn] validation loss never finite; keeping last state")
    else:
        model.load_state_dict(best_state)
    return ep


# ---------------------------------------------------------------------------
# Non-neural baselines
# ---------------------------------------------------------------------------

def persistence_pred(split, L, H, tidx, bs=8192):
    outs = []
    for i in range(0, len(split), bs):
        idx = torch.arange(i, min(i + bs, len(split)), device=split.starts.device)
        x, _, _, _ = split.batch(idx, L, H)
        outs.append(x[:, -1, tidx][:, None, :].expand(-1, H, -1))
    return torch.cat(outs)


def snaive_pred(split, L, H, tidx, period=5, bs=8192):
    """Seasonal naive: y(t+h) = y(t+h-period*ceil(h/period)). Period 5 rows = 1 s,
    the iperf report cycle (jitter/loss are 1 Hz values held over 5 rows)."""
    lag = [L - period + ((h - 1) % period) for h in range(1, H + 1)]
    outs = []
    for i in range(0, len(split), bs):
        idx = torch.arange(i, min(i + bs, len(split)), device=split.starts.device)
        x, _, _, _ = split.batch(idx, L, H)
        outs.append(x[:, lag][:, :, tidx])
    return torch.cat(outs)


def ridge_design(split, L, H, tidx, bs=8192):
    """Last input row (all features) + the full target history window."""
    rows = []
    for i in range(0, len(split), bs):
        idx = torch.arange(i, min(i + bs, len(split)), device=split.starts.device)
        x, _, _, _ = split.batch(idx, L, H)
        hist = x[:, :, tidx].reshape(x.size(0), -1)
        rows.append(torch.cat([x[:, -1, :], hist, torch.ones(x.size(0), 1, device=x.device)], 1))
    return torch.cat(rows)


def ridge_fit_predict(data, L, H, tidx, lams=(1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4)):
    """Direct multi-horizon ridge on the change from the last observed value.
    lambda is picked on the validation split; the intercept is not penalised."""
    Xtr = ridge_design(data["train"], L, H, tidx).cpu().double()
    Ytr = gather_targets(data["train"], L, H).reshape(len(data["train"]), -1).cpu().double()
    lastv = persistence_pred(data["train"], L, H, tidx).reshape(len(data["train"]), -1).cpu().double()
    G, B = Xtr.T @ Xtr, Xtr.T @ (Ytr - lastv)
    pen = torch.eye(Xtr.shape[1], dtype=torch.float64); pen[-1, -1] = 0.0
    sets = {}
    for k in ("val", "test"):
        Xs = ridge_design(data[k], L, H, tidx).cpu().double()
        lv = persistence_pred(data[k], L, H, tidx).reshape(len(data[k]), -1).cpu().double()
        sets[k] = (Xs, lv)
    yv = gather_targets(data["val"], L, H).reshape(len(data["val"]), -1).cpu().double()
    best = None
    for lam in lams:
        W = torch.linalg.solve(G + lam * pen, B)
        mse = float(((sets["val"][0] @ W + sets["val"][1] - yv) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, lam, W)
    _, lam, W = best
    preds = {k: (Xs @ W + lv).float().reshape(len(data[k]), H, len(tidx)) for k, (Xs, lv) in sets.items()}
    preds["lambda"] = lam
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(pred_te, y_te, pred_va, y_va, pers_te, meta, alpha=0.05, dist_te=None, dist_va=None):
    """All arrays denormalised, shape (N, H, T). dist_* (N, H, T, 6) are the
    distributional outputs (mean, q.025, q.25, q.5, q.75, q.975) when available."""
    rows = []
    T = y_te.shape[2]
    for ti in range(T):
        tname = SHORT.get(meta["targets"][ti], meta["targets"][ti])
        for h in range(y_te.shape[1]):
            yt, yp, pp = y_te[:, h, ti], pred_te[:, h, ti], pers_te[:, h, ti]
            r = yt - yp
            ss_tot = ((yt - yt.mean()) ** 2).sum()
            r2 = np.nan if ss_tot <= 1e-12 * len(yt) else 1 - (r ** 2).sum() / ss_tot
            # split-conformal half-width from validation residuals
            rv = np.abs(y_va[:, h, ti] - pred_va[:, h, ti])
            n = len(rv)
            q = np.quantile(rv, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
            lo, hi = yp - q, yp + q
            cov = ((yt >= lo) & (yt <= hi)).mean()
            mis = (hi - lo + (2 / alpha) * np.maximum(lo - yt, 0) + (2 / alpha) * np.maximum(yt - hi, 0)).mean()
            # initial study style (quantiles of the test residuals themselves)
            l2, h2 = yp + np.quantile(r, alpha / 2), yp + np.quantile(r, 1 - alpha / 2)
            cov2 = ((yt >= l2) & (yt <= h2)).mean()
            mis2 = (h2 - l2 + (2 / alpha) * np.maximum(l2 - yt, 0) + (2 / alpha) * np.maximum(yt - h2, 0)).mean()
            rmse_p = np.sqrt(((yt - pp) ** 2).mean())
            extra = {}
            if dist_te is not None:
                med = dist_te[:, h, ti, 3]
                extra["MAE_median"] = np.abs(yt - med).mean()
                extra["RMSE_median"] = np.sqrt(((yt - med) ** 2).mean())
                # conformalised quantile regression (Romano et al. 2019)
                lo_v, hi_v = dist_va[:, h, ti, 1], dist_va[:, h, ti, 5]
                sv = np.maximum(lo_v - y_va[:, h, ti], y_va[:, h, ti] - hi_v)
                c = np.quantile(sv, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
                l3, h3 = dist_te[:, h, ti, 1] - c, dist_te[:, h, ti, 5] + c
                extra["PICP_cqr"] = ((yt >= l3) & (yt <= h3)).mean()
                extra["MIS_cqr"] = (h3 - l3 + (2 / alpha) * np.maximum(l3 - yt, 0)
                                    + (2 / alpha) * np.maximum(yt - h3, 0)).mean()
                extra["width_cqr"] = (h3 - l3).mean()
                # raw (uncalibrated) model quantiles
                l4, h4 = dist_te[:, h, ti, 1], dist_te[:, h, ti, 5]
                extra["PICP_raw"] = ((yt >= l4) & (yt <= h4)).mean()
                # pinball loss averaged over the 5 quantiles
                taus = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
                e = yt[:, None] - dist_te[:, h, ti, 1:]
                extra["pinball"] = np.maximum(taus * e, (taus - 1) * e).mean()
            rows.append(dict(
                target=tname, h=h + 1,
                MAE=np.abs(r).mean(), RMSE=np.sqrt((r ** 2).mean()), R2=r2,
                MAE_persist=np.abs(yt - pp).mean(), RMSE_persist=rmse_p,
                PICP=cov, MIS=mis, PICP_test_quantile=cov2, MIS_test_quantile=mis2,
                halfwidth=q, **extra,
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

HIDDEN = 128
QWEIGHT = 1.0


def build_model(method, meta, L, H):
    F = len(meta["features"])
    T = len(meta["targets"])
    if method.startswith("y3_dist"):
        return seq2seq_dist.ThroughputSeq2SeqDist(num_features=F, embed_dim=16, hidden_size=HIDDEN,
                                                  dropout=0.3, input_len=L, output_len=H, num_targets=T,
                                                  target_feature_indices=meta["target_idx"])
    if method == "y2_seq2seq":
        return seq2seq_prev.ThroughputSeq2Seq(num_features=F, embed_dim=16, hidden_size=HIDDEN,
                                              dropout=0.3, input_len=L, output_len=H, num_targets=T)
    return seq2seq_y3.ThroughputSeq2Seq(num_features=F, embed_dim=16, hidden_size=HIDDEN, dropout=0.3,
                                        input_len=L, output_len=H, num_targets=T,
                                        target_feature_indices=meta["target_idx"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["persistence", "ridge", "y2_seq2seq", "y3_resid", "y3_full"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--input_len", type=int, default=30)
    ap.add_argument("--output_len", type=int, default=15)
    ap.add_argument("--data_root", default=str(ROOT / "data/lsn-data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results_final"))
    ap.add_argument("--save_models", action="store_true")
    ap.add_argument("--legacy_data", action="store_true",
                    help="keep parser-interpolated ping inputs (exact initial study data)")
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--qweight", type=float, default=1.0, help="weight of the pinball term in dist_loss")
    ap.add_argument("--train_stride", type=int, default=1,
                    help="keep every k-th training window (5G data is 7x larger)")
    a = ap.parse_args()
    globals()["HIDDEN"], globals()["QWEIGHT"] = a.hidden_size, a.qweight

    dev = get_device()
    L, H = a.input_len, a.output_len
    out_dir = Path(a.out_dir) / f"{a.provider}_{a.direction}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "log.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    cache = {}

    def get(variant):
        if variant not in cache:
            t0 = time.time()
            cache[variant] = load_splits(a.provider, a.direction, variant, L, H, dev, a.data_root,
                                         causal_ping=not a.legacy_data)
            d, m = cache[variant]
            if a.train_stride > 1:
                d["train"].starts = d["train"].starts[::a.train_stride].contiguous()
                m["n_windows"]["train"] = len(d["train"])
            log(f"[data {variant}] {len(m['features'])} features, sessions {m['n_sessions']}, "
                f"windows {m['n_windows']} ({time.time()-t0:.0f}s)")
        return cache[variant]

    def denorm(t, meta):
        return t.cpu().numpy() * meta["tstd"] + meta["tmean"]

    for method in a.methods:
        variant = "y3" if method.startswith(("y3_full", "y3_dist", "ridge_y3")) else "y2"
        data, meta = get(variant)
        tidx = meta["target_idx"]
        y_te = denorm(gather_targets(data["test"], L, H), meta)
        y_va = denorm(gather_targets(data["val"], L, H), meta)
        pers_te = denorm(persistence_pred(data["test"], L, H, tidx), meta)
        seeds = a.seeds if method not in ("persistence", "snaive", "ridge", "ridge_y3") else [0]
        for seed in seeds:
            csv = out_dir / f"{method}_s{seed}.csv"
            if csv.exists():
                log(f"[skip] {csv.name} exists")
                continue
            log(f"\n=== {a.provider} {a.direction} | {method} | seed {seed} ===")
            t0 = time.time()
            info = {}
            d_te = d_va = None
            if method == "persistence":
                p_te, p_va = pers_te, denorm(persistence_pred(data["val"], L, H, tidx), meta)
            elif method == "snaive":
                p_te = denorm(snaive_pred(data["test"], L, H, tidx), meta)
                p_va = denorm(snaive_pred(data["val"], L, H, tidx), meta)
            elif method in ("ridge", "ridge_y3"):
                pr = ridge_fit_predict(data, L, H, tidx)
                p_te, p_va = denorm(pr["test"], meta), denorm(pr["val"], meta)
                info["ridge_lambda"] = pr["lambda"]
            else:
                torch.manual_seed(seed); np.random.seed(seed)
                model = build_model(method, meta, L, H).to(dev)
                loss = ("dist" if method.startswith("y3_dist") else
                        "huber" if method.endswith("huber") else "mse")
                ep = train_model(model, data, L, H, a.epochs, a.patience, loss, a.batch_size, seed, log)
                info = {"epochs": ep, "params": sum(p.numel() for p in model.parameters())}
                o_te, o_va = predict(model, data["test"], L, H), predict(model, data["val"], L, H)
                if o_te.dim() == 4:   # distributional model: (N, H, T, 6)
                    sd = torch.tensor(meta["tstd"])[None, None, :, None]
                    mu = torch.tensor(meta["tmean"])[None, None, :, None]
                    d_te = (o_te.cpu() * sd + mu).numpy(); d_va = (o_va.cpu() * sd + mu).numpy()
                    p_te, p_va = d_te[..., 0], d_va[..., 0]
                else:
                    p_te, p_va = denorm(o_te, meta), denorm(o_va, meta)
                if a.save_models:
                    mdir = out_dir / "models"; mdir.mkdir(exist_ok=True)
                    torch.save({"state_dict": model.state_dict(), "method": method,
                                "features": meta["features"], "targets": meta["targets"],
                                "tmean": meta["tmean"].tolist(), "tstd": meta["tstd"].tolist(),
                                "fmean": meta["fmean"].tolist(), "fstd": meta["fstd"].tolist(),
                                "clip_cols": meta["clip_cols"], "clip_z": meta["clip_z"],
                                "variant": meta["variant"], "causal_ping": meta["causal_ping"],
                                "input_len": L, "output_len": H},
                               mdir / f"{method}_s{seed}.pt")
            # Predictions (val + test) for seed ensembles and figures.
            pdir = out_dir / "preds"; pdir.mkdir(exist_ok=True)
            np.savez_compressed(pdir / f"{method}_s{seed}.npz",
                                p_te=p_te.astype(np.float32), p_va=p_va.astype(np.float32),
                                **({} if d_te is None else {"d_te": d_te.astype(np.float16),
                                                            "d_va": d_va.astype(np.float16)}))
            if not (pdir / "truth.npz").exists():
                tmp = pdir / f"truth_tmp_{os.getpid()}.npz"
                np.savez_compressed(tmp, y_te=y_te.astype(np.float32),
                                    y_va=y_va.astype(np.float32), pers_te=pers_te.astype(np.float32),
                                    interp_te=gather_interp_mask(data["test"], L, H).cpu().numpy().astype(np.uint8),
                                    targets=np.array(meta["targets"]))
                os.replace(tmp, pdir / "truth.npz")
            df = metrics(p_te, y_te, p_va, y_va, pers_te, meta, dist_te=d_te, dist_va=d_va)
            df.insert(0, "seed", seed); df.insert(0, "method", method)
            df.insert(0, "dataset", f"{a.provider}_{a.direction}")
            df.to_csv(csv, index=False)
            info.update(settings=vars(a), causal_ping=meta["causal_ping"],
                        seconds=round(time.time() - t0, 1), n_test=len(y_te),
                        features=meta["features"], n_sessions=meta["n_sessions"])
            json.dump(info, open(out_dir / f"{method}_s{seed}.json", "w"), indent=1, default=str)
            agg = df.groupby("target")[["MAE", "RMSE", "R2", "RMSE_persist", "PICP"]].mean()
            log(agg.to_string(float_format=lambda v: f"{v:.4f}"))
            log(f"  AGG MAE={df.MAE.mean():.4f} RMSE={df.RMSE.mean():.4f} R2={df.R2.mean():.4f} "
                f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
