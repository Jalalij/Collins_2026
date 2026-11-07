"""
experiments/fr_attention.py - Which part of the 6 s input history does the forecaster use?

The DS2S decoder (models/seq2seq_dist.py) re-reads the encoder states of the
30 input rows at every one of its 15 forecast steps through Bahdanau
attention. This script runs the trained seed-0 checkpoints (no training) on
held-out TEST windows and measures where that attention goes. The previous
report's S2S model (prev_seq2seq, base features, scalar output) is analysed
the same way on the same windows for comparison.

  maps      mean attention over windows -> a 15 x 30 map per link
            (decoder step h = 1..15  x  input lag 0..29, lag 0 = newest row,
             one row = 200 ms, so lag 29 = 5.8 s ago)
  entropy   per-window entropy of the 30-way attention distribution,
            normalised by ln 30 (1 = uniform over the window, 0 = one row),
            and its exponential, the "effective number of rows" read
  recency   attention mass on the newest row, on the last second (lags 0-4),
            on rows older than 3 s (lags 15-29) and the attention-weighted
            mean lag in seconds (uniform attention: 2.9 s)
  input-dep total-variation distance between each window's attention and
            the mean map (0 = the same weights for every window)
  period    a phase-free test for a 1 s (5-row) comb: the power of each
            window's attention profile at 6 cycles per 30 rows, divided by
            the mean power of the two neighbouring frequencies (1 = no comb)
  phase     where the comb sits: iperf metrics are 1 Hz reports held for 5
            rows, and the engineered feature <target>__rsc counts the rows
            since the value last changed (0 = a new report arrived on this
            row). Attention on every (window, step, lag) cell is divided by
            the mean map at that (step, lag) and averaged by the rsc value of
            the row (1 = no preference). The mean map of the windows whose
            newest row is a fresh report (all such windows share one phase,
            so the comb survives the averaging) is saved too. The 'fr'
            inputs are z-clipped at 5, which merges the counts of targets
            that change on almost every row (RTT) or almost never (OneWeb
            jitter / loss); such targets are left out (rule in window_info).
  content   does attention follow salient rows or only their position?
            For test windows whose input contains an RTT spike (window max
            at or above the 95th percentile of VALIDATION window maxima),
            the attention on the spike row is divided by the mean attention
            that calm test windows put on the same (step, lag) cell
            (ratio of sums over windows; 1 = purely positional). The same
            lift is computed for the newest handover / satellite-change row,
            and for the history before and after that event.
  gradients a cross-check that does not rely on attention: the absolute
            gradient of the RTT mean forecast (step 1, step 15 and the sum
            over the 15 steps) with respect to every input value, summed over
            features, per lag, normalised to sum 1 per window and averaged
            over an evenly spaced subset of the test windows. DS2S adds its
            output to the last observed value, which is a direct path to the
            newest row, so for DS2S the profile of the learned change
            (forecast minus last value, "<out>_change") is reported as well.
            Attention only says which encoder states the decoder reads, and
            the GRU encoder already carries older inputs into the newer
            states, so attention and gradients can disagree.
            Total derivative: the S2S decoder feeds back its forecast with
            .detach() (a training device). Differentiating through that
            forward would drop every path through the 14 fed-back forecasts
            and make the step-15 profile look flat. S2S is therefore
            re-run with decode() below, which keeps the feedback (identical
            forward values, checked), so both models get the total gradient.
            Engineered inputs: the DS2S ('fr') rows also carry trailing
            statistics (__mean150/__mean300/__std300/__xmean summarise up to
            60 s and the session so far), so a gradient on the newest row is
            not only "recent" information. "<out>_raw" restricts the profile
            to the base input features (the S2S input set), and
            "<out>_eng_feature_share" is the share of |gradient| on the
            engineered features.
  ablation  does the decoder need its input-dependent attention? The test
            forecasts are recomputed with every window's attention replaced by
            (a) the mean attention map of VALIDATION windows (input
            independent) and (b) uniform weights 1/30, and the RTT MAE of the
            mean forecast is compared with the model's own attention
            (abl_* columns; ratio of MAEs, 1 = attention pattern irrelevant).

Calibration rule: the only quantities learned from data (the spike level and
the mean attention map of the ablation) come from validation windows;
everything reported is measured on test windows.

Sanity check: the mean forecasts of both models recomputed here on CPU are
compared with the forecasts saved by final_benchmark.py
(preds/<method>_s<seed>.npz, made on the GPU), so the checkpoints and the data
pipelines are the ones of the results.

Outputs (results_final/analysis/attention/):
  attn_maps.csv       link, model, step, lag, lag_s, weight (mean attention)
  attn_fresh_maps.csv the same for windows that end on a fresh 1 Hz report
  grad_profiles.csv   link, model, output, lag, lag_s, gradient share
                      (output h1/h15/sum, '_change', '_raw' as above)
  attn_phase.csv      link, model, target, rsc class, lift, cell share (all targets)
  attn_steps.csv      per (link, model, step) statistics
  attn_summary.csv    per (link, model) statistics averaged over steps
  attn_summary.json   the same plus per-step arrays, settings and sanity checks
  cache/<link>_<model>_s<seed>.npz   per-link aggregates (use --recompute)
Figure data (reports/final_report/data/), at most 451 rows per file:
  fr_attention_<link>.dat        DS2S map: lag, step, w, ratio = 30 w, lr = log2 ratio,
                                 lrc = lr clipped to the shared colour range
  fr_attention_fresh_<link>.dat  the same for windows ending on a fresh report
  fr_attention_prof_<link>.dat   lag profiles (mean over steps), x 30 (1 = uniform)
Figures (reports/final_report/figs/), written only when all six links are cached:
  fr_attention.tex       2 x 3 heatmaps of the DS2S attention (step x lag)
  fr_attention_prof.tex  2 x 3 lag profiles: DS2S and S2S attention, DS2S gradient,
                         with a zoom inset on the 1 s comb of Starlink DL
With --fr_method other than fr_dist (e.g. fr_gauss, fr_dist_w4) every output
name carries the method (attention_<method>/, fr_attention_<method>_*), so the
DS2S results are never overwritten; links without that checkpoint are skipped.

Usage (CPU only; on an idle machine ~1 min per satellite link, ~2 min per 5G link):
  LSN_DEVICE=cpu python experiments/fr_attention.py
  LSN_DEVICE=cpu python experiments/fr_attention.py --links starlink_dl --recompute
  LSN_DEVICE=cpu python experiments/fr_attention.py --links none     # tables from the cache only
"""

from __future__ import annotations

import os

os.environ.setdefault("LSN_DEVICE", "cpu")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from final_benchmark import load_splits  # noqa: E402
from models import seq2seq_dist, seq2seq_prev  # noqa: E402
from dataset import ENGINEERED_SUFFIXES, GLOBAL_ENGINEERED  # noqa: E402

LINKS = ["oneweb_dl", "oneweb_ul", "starlink_dl", "starlink_ul", "quectel_dl", "quectel_ul"]
SHORT = ["rtt", "jitter", "loss"]          # target order of every checkpoint
L, H = 30, 15
ROW_S = 0.2
COMB_K = L // 5                            # 6 cycles per 30 rows = a 5-row (1 s) period
PHASE_CLASSES = ["0", "1", "2", "3", "4", "5+"]
DT = np.float64


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, method: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feats, targs = ck["features"], ck["targets"]
    kw = dict(num_features=len(feats), embed_dim=16, hidden_size=128, dropout=0.3,
              input_len=ck.get("input_len", L), output_len=ck.get("output_len", H),
              num_targets=len(targs))
    if method == "prev_seq2seq":
        model = seq2seq_prev.ThroughputSeq2Seq(**kw)
    elif method.startswith("fr_gauss"):
        model = seq2seq_dist.ThroughputSeq2SeqGauss(
            **kw, target_feature_indices=[feats.index(t) for t in targs])
    else:
        model = seq2seq_dist.ThroughputSeq2SeqDist(
            **kw, target_feature_indices=[feats.index(t) for t in targs])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def forward_attention(model, split, idx, bs):
    """Run the model on test windows idx. Returns attn (n, H, L) with the last
    axis ordered by lag (0 = newest row) and mean (n, H, K), the normalised
    mean forecast."""
    attn, mean = [], []
    t_idx = torch.from_numpy(idx)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x, hr, dw, _ = split.batch(t_idx[i:i + bs], L, H)
            out, a = model(x, hr, dw)
            attn.append(a.flip(-1).float().numpy())
            mean.append((out[..., 0] if out.dim() == 4 else out).float().numpy())
    return np.concatenate(attn), np.concatenate(mean)


def is_anchored(model):
    """DS2S-type decoders add their output to the last observed value."""
    return hasattr(model, "target_feature_indices")


def decode(model, x, hr, dw, fixed=None):
    """Forward pass of both decoder types re-implemented step by step -> the
    normalised mean forecast (B, H, K). Used for (a) the S2S input gradient and
    (b) the attention ablation. fixed: (H, L) attention weights in chronological
    order applied to every window instead of the model's own attention (None =
    own attention, the exact operations of the model's forward()).
    Unlike seq2seq_prev.Decoder, the fed-back forecast is NOT detached. That
    changes no forward value (checked in analyse_link), but the input gradient
    then follows the 14 fed-back forecasts, i.e. it is the total derivative."""
    xin = x
    if model.time_embed is not None:
        xin = torch.cat([x, model.time_embed(hr.float(), dw.float())], dim=-1)
    enc_out, hidden = model.encoder(xin)
    dec = model.decoder
    anchored = is_anchored(model)
    if anchored:
        last = x[:, -1, model.target_feature_indices]
        keys = dec.attention.key_proj(enc_out)
    else:
        last = torch.zeros(x.size(0), dec.num_targets, dtype=x.dtype, device=x.device)
    means = []
    for t in range(model.output_len):
        if fixed is None:
            context, _ = (dec.attention(hidden[-1], enc_out, keys) if anchored
                          else dec.attention(hidden[-1], enc_out))
        else:
            context = (fixed[t].to(enc_out.dtype)[None, :, None] * enc_out).sum(1)
        gru_in = torch.cat([context, dec.output_embed(last)], dim=-1).unsqueeze(1)
        h, hidden = dec.gru(gru_in, hidden)
        top = h.squeeze(1)
        if anchored:
            delta = torch.stack([head(top) for head in dec.heads], dim=1)        # (B, K, 6)
            last = last + delta[..., 0]
        else:
            last = torch.cat([head(top) for head in dec.heads], dim=-1)          # (B, K)
        means.append(last)
    return torch.stack(means, dim=1)


def engineered_mask(features):
    """True for the engineered columns of the 'fr' inputs (trailing statistics, row_dt)."""
    return np.array([f in GLOBAL_ENGINEERED or any(f.endswith(s) for s in ENGINEERED_SUFFIXES)
                     for f in features])


GRAD_OUT = ("h1", "h15", "sum")   # RTT mean forecast at step 1, at step 15, summed over steps


def gradient_profile(model, split, idx, bs, rtt_k=0, anchor_col=None, eng=None):
    """Sum over windows idx of the per-window lag profile of |d out / d x[lag, :]|
    summed over features and normalised to sum 1 -> ({name: (L,)}, {name: share}), lag order.
    out = the RTT mean forecast at step 1, at step 15 and summed over steps, as the
    TOTAL derivative (S2S through decode(), which keeps the fed-back forecasts).
    With anchor_col (DS2S: the RTT input column the decoder adds its output to),
    also '<name>_change': the same for out minus the anchored last value, i.e.
    the gradient without the direct identity path (1 per step) to the newest row.
    With eng (boolean mask over features, some True), also '<name>_raw': the profile
    over the base (non-engineered) features only, and the second dict gives the mean
    share of |gradient| on the engineered features per output."""
    names = list(GRAD_OUT) + ([f"{n}_change" for n in GRAD_OUT] if anchor_col is not None else [])
    use_eng = eng is not None and bool(np.any(eng))
    gsum = {n: np.zeros(L, DT) for n in names + ([f"{n}_raw" for n in names] if use_eng else [])}
    esum = {n: 0.0 for n in (names if use_eng else [])}
    raw_t = torch.from_numpy(~eng) if use_eng else None

    def add(name, g):
        ga = g.abs()
        for nm, gg in ((name, ga),) + (((f"{name}_raw", ga[..., raw_t]),) if use_eng else ()):
            prof = gg.sum(-1).flip(-1).double()                                 # (B, L)
            prof = prof / prof.sum(-1, keepdim=True).clamp_min(1e-30)
            gsum[nm] += prof.sum(0).numpy()
        if use_eng:
            tot = ga.sum((1, 2)).double().clamp_min(1e-30)
            esum[name] += float((ga[..., ~raw_t].sum((1, 2)).double() / tot).sum())

    t_idx = torch.from_numpy(idx)
    for i in range(0, len(idx), bs):
        x, hr, dw, _ = split.batch(t_idx[i:i + bs], L, H)
        xg = x.clone().requires_grad_(True)
        with torch.enable_grad():
            if is_anchored(model):     # DS2S forward() has no detach: use it as is
                o, _ = model(xg, hr, dw)
                y = o[..., 0][:, :, rtt_k]                                       # (B, H)
            else:                      # S2S: keep the fed-back forecasts in the graph
                y = decode(model, xg, hr, dw)[:, :, rtt_k]
            outs = [("h1", y[:, 0].sum(), 1.0), ("h15", y[:, H - 1].sum(), 1.0), ("sum", y.sum(), float(H))]
            for j, (name, s, n_anchor) in enumerate(outs):
                # Windows do not interact in eval mode, so the gradient of the
                # batch sum gives every window's own gradient.
                g, = torch.autograd.grad(s, xg, retain_graph=j < len(outs) - 1)
                add(name, g)
                if anchor_col is not None:
                    gc = g.clone()
                    gc[:, -1, anchor_col] -= n_anchor
                    add(f"{name}_change", gc)
    return gsum, esum


def forecast_fixed_attention(model, split, idx, bs, fixed):
    """Normalised mean forecast (n, H, K) with the attention of every window
    replaced by fixed (H, L, chronological order)."""
    out = []
    t_idx = torch.from_numpy(idx)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x, hr, dw, _ = split.batch(t_idx[i:i + bs], L, H)
            out.append(decode(model, x, hr, dw, fixed).float().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Window information (computed once per link, shared by both models)
# ---------------------------------------------------------------------------

def window_rows(split, idx):
    s = split.starts[torch.from_numpy(idx)].numpy()
    return s[:, None] + np.arange(L)[None]                                      # chronological


def window_info(data, meta, idx, a):
    """Spike rows, event rows and the 1 Hz report phase of every analysed test window."""
    te, va = data["test"], data["val"]
    rtt_col = meta["target_idx"][0]
    tm, ts = float(meta["tmean"][0]), float(meta["tstd"][0])
    Xte, Xva = te.X.numpy(), va.X.numpy()

    # RTT spikes: level from VALIDATION windows only.
    rv = Xva[:, rtt_col][window_rows(va, np.arange(len(va)))] * ts + tm
    thr = float(np.quantile(rv.max(1), a.spike_q))
    rows = window_rows(te, idx)
    rt = (Xte[:, rtt_col][rows] * ts + tm)[:, ::-1]                             # lag order
    info = {"spike_thr": thr, "spike": rt.max(1) >= thr, "spike_lag": rt.argmax(1)}

    # Handover / satellite-change rows.
    ev = (te.ev[rows] > 0.5)[:, ::-1]
    info["event_lag"] = np.where(ev.any(1), ev.argmax(1), -1)
    info["event_row_rate"] = float(ev.mean())

    # Rows since the value last changed (engineered '__rsc' features, 'fr' inputs only),
    # for every target that has one. The 1 Hz report target is chosen in pick_phase_target.
    feats = meta["features"]
    fm, fs = np.asarray(meta["fmean"]), np.asarray(meta["fstd"])
    cz, clip_cols = meta.get("clip_z"), set(meta.get("clip_cols") or [])
    info["phase"], info["rsc_round_err"] = {}, {}
    for k, t in enumerate(meta["targets"]):
        c = f"{t}__rsc"
        if c in feats:
            j = feats.index(c)
            v = Xte[:, j][rows] * fs[j] + fm[j]
            # The counts are integers 0..10, but the 'fr' inputs are clipped at |z| <= clip_z,
            # which merges counts outside [fmean - clip_z fstd, fmean + clip_z fstd]. The
            # classes 0..4 and 5+ survive only if that range covers [0.5, 5]; otherwise
            # (e.g. RTT, which changes on almost every row) the target is left out.
            info["rsc_round_err"][SHORT[k]] = float(np.abs(v - np.rint(v)).max())
            if cz is not None and j in clip_cols and (fm[j] - cz * fs[j] >= 0.5 or fm[j] + cz * fs[j] < 5.0):
                continue
            info["phase"][SHORT[k]] = np.rint(v)[:, ::-1].astype(np.int16)
    return info


def pick_phase_target(phase_rows):
    """The first of jitter, loss, rtt whose rows-since-update value cycles like a
    1 Hz report held for 5 rows: every class 0..4 holds 10-30 % of the rows."""
    for t in ("jitter", "loss", "rtt"):
        rows = phase_rows.get(t)
        if rows and all(0.1 <= r["cell_share"] <= 0.3 for r in rows[:5]):
            return t
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def entropy(a):
    a = np.clip(a.astype(DT), 1e-12, 1.0)
    return -(a * np.log(a)).sum(-1)


def comb_power(a):
    """Mean power spectrum (over windows) of the de-meaned attention profiles, (H, L//2+1)."""
    f = np.fft.rfft(a.astype(DT) - a.mean(-1, keepdims=True), axis=-1)
    return (np.abs(f) ** 2).mean(0)


def attention_stats(attn, info, a):
    """Per-step statistics (dict name -> (H,) array), scalars, the mean map and
    the phase-aligned map."""
    A = attn.astype(DT)
    M = A.mean(0)                                                                # (H, L)
    ent = entropy(A)                                                             # (n, H)
    lags = np.arange(L) * ROW_S
    P = comb_power(A)
    st = {
        "entropy_norm": (ent / np.log(L)).mean(0),
        "eff_rows": np.exp(ent).mean(0),
        "entropy_norm_of_mean_map": entropy(M) / np.log(L),
        "mass_lag0": A[:, :, 0].mean(0),
        "mass_last1s": A[:, :, :5].sum(-1).mean(0),
        "mass_older3s": A[:, :, 15:].sum(-1).mean(0),
        "mean_lag_s": (A * lags).sum(-1).mean(0),
        "tv_from_mean": 0.5 * np.abs(A - M[None]).sum(-1).mean(0),
        "comb_ratio_1s": P[:, COMB_K] / (0.5 * (P[:, COMB_K - 1] + P[:, COMB_K + 1])),
        "comb_ratio_0p5s": P[:, 2 * COMB_K] / (0.5 * (P[:, 2 * COMB_K - 1] + P[:, 2 * COMB_K + 1])),
    }
    sc = {"spike_threshold_ms": info["spike_thr"], "event_row_rate": info["event_row_rate"],
          **{f"rsc_round_err_{t}": e for t, e in info["rsc_round_err"].items()},
          "rsc_class_targets": ",".join(info["phase"]) or "none"}

    # Spike rows versus the positional baseline of calm windows.
    spike, slag = info["spike"], info["spike_lag"]
    base = A[~spike].mean(0) if (~spike).sum() > 100 else M
    sel = spike & (slag >= a.min_lag)
    sc.update(n_spike=int(spike.sum()), n_spike_lift=int(sel.sum()))
    if sel.sum() >= 50:
        # Ratio of sums over windows: robust when attention is close to one-hot
        # (a geometric mean of per-window ratios collapses on near-zero weights).
        ii = np.where(sel)[0]
        st["spike_lift"] = A[ii, :, slag[ii]].sum(0) / base[:, slag[ii]].T.sum(0)

    # Event rows, and the history before / after the newest event.
    elag = info["event_lag"]
    ebase = A[elag < 0].mean(0) if (elag < 0).sum() > 100 else M
    sel = (elag >= a.min_lag) & (elag <= L - 2)
    sc["n_event_lift"] = int(sel.sum())
    if sel.sum() >= 50:
        ii = np.where(sel)[0]
        e = elag[ii]
        st["event_lift"] = A[ii, :, e].sum(0) / ebase[:, e].T.sum(0)
        # Mass on the history before / after the event, same ratio of sums.
        older = (np.arange(L)[None] > e[:, None])[:, None, :]                   # (m, 1, L)
        newer = (np.arange(L)[None] < e[:, None])[:, None, :]
        st["event_pre_lift"] = (A[ii] * older).sum((0, 2)) / (ebase[None] * older).sum((0, 2))
        st["event_post_lift"] = (A[ii] * newer).sum((0, 2)) / (ebase[None] * newer).sum((0, 2))

    # Rows since the value last changed, per target: attention lift by class and
    # the mean map of the windows whose newest row carries a new value.
    phase_rows, Mf = {}, {}
    if info["phase"]:
        R = (A / M[None]).transpose(0, 2, 1)                                     # (n, L, H)
        for t, ph in info["phase"].items():
            cls = np.minimum(ph, 5)                                              # (n, L)
            phase_rows[t] = []
            for c in range(6):
                m = cls == c
                phase_rows[t].append({"rsc": PHASE_CLASSES[c],
                                      "lift": float(R[m].mean()) if m.any() else float("nan"),
                                      "cell_share": float(m.mean())})
            fresh = ph[:, 0] == 0
            if fresh.sum() >= 100:
                Mf[t] = A[fresh].mean(0)
            sc[f"n_fresh_{t}"] = int(fresh.sum())
    return M, Mf, st, sc, phase_rows


# ---------------------------------------------------------------------------
# One link
# ---------------------------------------------------------------------------

def analyse_link(link, a, res_dir, cache_dir):
    provider, direction = link.split("_")
    methods = [a.fr_method, a.cmp_method]
    cache = {m: cache_dir / f"{link}_{m}_s{a.seed}.npz" for m in methods}
    pdir, mdir = res_dir / link / "preds", res_dir / link / "models"
    ckpts = {m: mdir / f"{m}_s{a.seed}.pt" for m in methods}
    todo = [m for m in methods if ckpts[m].exists() and (a.recompute or not cache[m].exists())]
    for m in methods:
        if not ckpts[m].exists():
            print(f"  [skip] {link} {m}: no checkpoint {ckpts[m].name}", flush=True)
    if not todo:
        return
    if not ckpts[a.fr_method].exists():
        return
    truth = np.load(pdir / "truth.npz")
    n_test = truth["y_te"].shape[0]

    data_cache, idx, info, starts_ref = {}, None, None, None
    # The DS2S ('fr') data come first: they hold the '__rsc' phase features.
    for m in [a.fr_method] + [x for x in methods if x != a.fr_method]:
        if not ckpts[m].exists():
            continue
        t0 = time.time()
        model, ck = load_model(ckpts[m], m)
        variant = ck["variant"]
        if variant not in data_cache:
            data_cache[variant] = load_splits(provider, direction, variant, L, H, torch.device("cpu"),
                                              a.data_root)
            print(f"  [data {link} {variant}] {time.time()-t0:.0f}s", flush=True)
        data, meta = data_cache[variant]
        assert meta["features"] == ck["features"], f"{link} {m}: feature list differs from checkpoint"
        te = data["test"]
        assert len(te) == n_test, f"{link}: {len(te)} test windows vs {n_test} saved"
        if idx is None:
            idx = np.unique(np.round(np.linspace(0, n_test - 1, min(a.max_windows, n_test))).astype(np.int64))
            starts_ref = te.starts.numpy().copy()
            info = window_info(data, meta, idx, a)
        else:
            assert np.array_equal(te.starts.numpy(), starts_ref), "window starts differ between variants"
        if m not in todo:
            continue

        attn, mean_n = forward_attention(model, te, idx, a.batch)
        mean = mean_n * meta["tstd"] + meta["tmean"]
        print(f"  [{link} {m}] forward on {len(idx)} windows {time.time()-t0:.0f}s", flush=True)

        # decode() (S2S gradient, ablation) must reproduce the model's own forward.
        x0, h0, d0, _ = te.batch(torch.from_numpy(idx[:256]), L, H)
        with torch.no_grad():
            dec_err = float(np.abs(decode(model, x0, h0, d0).float().numpy() - mean_n[:256]).max())
        assert dec_err < 1e-4, f"{link} {m}: decode() differs from forward() by {dec_err}"

        G, E, n_grad = {}, {}, 0
        if a.grad_windows > 0:
            gi = idx[np.unique(np.round(np.linspace(0, len(idx) - 1, min(a.grad_windows, len(idx))))
                               .astype(np.int64))]
            anchor = meta["target_idx"][0] if is_anchored(model) else None
            gs, es = gradient_profile(model, te, gi, a.batch, anchor_col=anchor,
                                      eng=engineered_mask(meta["features"]))
            G, n_grad = {k: v / len(gi) for k, v in gs.items()}, len(gi)
            E = {f"grad_{k}_eng_feature_share": v / len(gi) for k, v in es.items()}
            print(f"  [{link} {m}] gradients on {n_grad} windows {time.time()-t0:.0f}s", flush=True)

        check = {"decode_max_abs_err": dec_err}
        sp = pdir / f"{m}_s{a.seed}.npz"
        if sp.exists():
            ref = np.load(sp)["p_te"][idx]
            kk = [0] if link == "oneweb_ul" else [0, 1, 2]     # OneWeb UL jitter/loss are always 0
            d = np.abs(mean[..., kk] - ref[..., kk])
            check.update(max_abs_diff_vs_saved=float(d.max()), mean_abs_diff_vs_saved=float(d.mean()),
                         ref_mean_abs=float(np.abs(ref[..., kk]).mean()))

        # Ablation: the same test forecasts with input-independent attention. The
        # mean map comes from VALIDATION windows, uniform weights need no data.
        abl = {}
        if not a.no_ablation:
            va = data["val"]
            vidx = np.unique(np.round(np.linspace(0, len(va) - 1, min(a.max_windows, len(va)))).astype(np.int64))
            attn_va, _ = forward_attention(model, va, vidx, a.batch)
            Mva = attn_va.astype(DT).mean(0)                                     # (H, L), lag order
            del attn_va
            y = truth["y_te"][idx][..., 0].astype(DT)
            own = mean[..., 0].astype(DT)
            pers = truth["pers_te"][idx][..., 0].astype(DT)

            def errs(p):
                return float(np.abs(p - y).mean()), float(np.sqrt(((p - y) ** 2).mean())), \
                    float(np.abs(p[:, -1] - y[:, -1]).mean())

            abl["abl_mae_own"], abl["abl_rmse_own"], abl["abl_mae_own_h15"] = errs(own)
            abl["abl_mae_persistence"], abl["abl_rmse_persistence"], _ = errs(pers)
            abl["abl_move_ms"] = float(np.abs(own - pers).mean())     # size of the forecast's move
            abl["abl_tv_valmap_testmap"] = float(0.5 * np.abs(Mva - attn.astype(DT).mean(0)).sum(-1).mean())
            abl["abl_n_val_windows"] = len(vidx)
            for k, w in (("valmap", Mva[:, ::-1]), ("uniform", np.full((H, L), 1.0 / L))):
                p = forecast_fixed_attention(model, te, idx, a.batch,
                                             torch.from_numpy(np.ascontiguousarray(w)).float())
                p = p[..., 0].astype(DT) * float(meta["tstd"][0]) + float(meta["tmean"][0])
                abl[f"abl_mae_{k}"], abl[f"abl_rmse_{k}"], abl[f"abl_mae_{k}_h15"] = errs(p)
                abl[f"abl_mae_ratio_{k}"] = abl[f"abl_mae_{k}"] / abl["abl_mae_own"]
                abl[f"abl_rmse_ratio_{k}"] = abl[f"abl_rmse_{k}"] / abl["abl_rmse_own"]
                abl[f"abl_shift_{k}_ms"] = float(np.abs(p - own).mean())
            print(f"  [{link} {m}] ablation {time.time()-t0:.0f}s: MAE own {abl['abl_mae_own']:.3f} "
                  f"valmap x{abl['abl_mae_ratio_valmap']:.4f} uniform x{abl['abl_mae_ratio_uniform']:.4f}",
                  flush=True)

        M, Mf, st, sc, phase_rows = attention_stats(attn, info, a)
        lags = np.arange(L) * ROW_S
        for k, g in G.items():
            sc[f"grad_{k}_mass_lag0"] = float(g[0])
            sc[f"grad_{k}_mass_last1s"] = float(g[:5].sum())
            sc[f"grad_{k}_mass_older3s"] = float(g[15:].sum())
            sc[f"grad_{k}_mean_lag_s"] = float((g * lags).sum())
        sc.update(check, **E, **abl, n_windows=len(idx), n_grad=n_grad, n_test=n_test, variant=variant,
                  seconds=round(time.time() - t0, 1))
        np.savez(cache[m], M=M, Mf=np.array([Mf], dtype=object),
                 G=np.array([G], dtype=object), steps=np.array([st], dtype=object),
                 scalars=np.array([sc], dtype=object), phase=np.array([phase_rows], dtype=object),
                 P=comb_power(attn))
        pt = pick_phase_target(phase_rows)
        pl = [round(r["lift"], 2) for r in phase_rows[pt]] if pt else None
        print(f"  [{link} {m}] done {time.time()-t0:.0f}s: H_norm={st['entropy_norm'].mean():.3f} "
              f"eff_rows={st['eff_rows'].mean():.1f} lag0={st['mass_lag0'].mean():.3f} "
              f"last1s={st['mass_last1s'].mean():.3f} meanlag={st['mean_lag_s'].mean():.2f}s "
              f"comb={st['comb_ratio_1s'].mean():.2f} phase={pt}:{pl} {check}", flush=True)
        del model, attn


# ---------------------------------------------------------------------------
# Tables and figure data
# ---------------------------------------------------------------------------

def load_cache(f):
    """Cached aggregates of one (link, model). The phase results are reduced to
    the 1 Hz report target (if any): phase = its lift rows, Mf = its fresh-report map."""
    z = np.load(f, allow_pickle=True)
    r = {"M": z["M"], "G": z["G"][0], "steps": dict(z["steps"][0]), "scalars": dict(z["scalars"][0]),
         "P": z["P"], "phase_all": z["phase"][0], "Mf_all": z["Mf"][0]}
    pt = pick_phase_target(r["phase_all"])
    r["scalars"]["phase_target"] = pt
    r["phase"] = r["phase_all"].get(pt, []) if pt else []
    r["Mf"] = r["Mf_all"].get(pt) if pt else None
    for row in r["phase"]:
        r["scalars"][f"phase_lift_rsc{row['rsc'].replace('+', 'plus')}"] = row["lift"]
    if r["Mf"] is not None:
        on = np.arange(L) % 5 == 0
        r["steps"]["fresh_comb_contrast"] = r["Mf"][:, on].mean(-1) / r["Mf"][:, ~on].mean(-1)
    # Positional 1 s pattern of the mean map: the target of step h lies h + lag rows
    # after an input row, so (h + lag) mod 5 = c groups rows by their phase relative
    # to the target time. Local contrast = weight / mean of the two neighbouring lags,
    # geometric mean over steps and interior lags (equal values for all c = no stripes).
    M = r["M"]
    for c in range(5):
        v = [np.log(M[h, l] / (0.5 * (M[h, l - 1] + M[h, l + 1])))
             for h in range(H) for l in range(1, L - 1) if (h + 1 + l) % 5 == c]
        r["scalars"][f"target_phase_contrast_c{c}"] = float(np.exp(np.mean(v)))
    tc = [r["scalars"][f"target_phase_contrast_c{c}"] for c in range(5)]
    r["scalars"]["target_phase_modulation"] = max(tc) / min(tc)
    return r


def write_outputs(a, out_dir, fig_dir):
    """Tables (CSV + JSON) for every cached link and the figure data of the DS2S maps."""
    allres = {}
    for link in LINKS:
        rr = {m: load_cache(out_dir / "cache" / f"{link}_{m}_s{a.seed}.npz")
              for m in (a.fr_method, a.cmp_method)
              if (out_dir / "cache" / f"{link}_{m}_s{a.seed}.npz").exists()}
        if rr:
            allres[link] = rr
    if not allres:
        print("no cached results yet")
        return
    lag_s = np.round(np.arange(L) * ROW_S, 1)
    maps, fmaps, grads, phases, steps, summ = [], [], [], [], [], []
    js = {"settings": {k: v for k, v in vars(a).items() if k != "links"},
          "notes": {
              "lag": "lag 0 = newest input row, one row = 0.2 s, uniform weight = 1/30",
              "entropy_norm": "per-window entropy / ln 30, mean over windows (1 = uniform)",
              "eff_rows": "exp(entropy), mean over windows (30 = uniform)",
              "tv_from_mean": "mean total-variation distance between a window's attention and the mean map",
              "comb_ratio_1s": "per-window spectra: power at a 5-row period / mean power of the two "
                               "neighbouring frequencies (phase free, 1 = no comb)",
              "phase_lift_rsc<c>": "attention / mean map at the same (step, lag), averaged over rows whose "
                                   "rows-since-update value of the 1 Hz report target is c (1 = no "
                                   "preference); the target is the first of jitter, loss, rtt whose "
                                   "classes 0..4 each hold 10-30 % of the rows",
              "fresh_comb_contrast": "windows whose newest row is a fresh 1 Hz report: mean weight at "
                                     "lags 0,5,..,25 / mean weight at the other lags. Not meaningful "
                                     "where the oldest rows are an attention sink (5G, OneWeb): the "
                                     "sink at lags 26-29 falls in the 'other lags'",
              "rsc_round_err_<t>": "max |count - round(count)| of the denormalised rows-since-update "
                                   "input; large where z-clipping merged counts, such targets are "
                                   "excluded from the phase tables (rsc_class_targets lists the rest)",
              "spike_lift": "attention on the input RTT spike row / calm windows' mean attention on the "
                            "same (step, lag), ratio of sums over spike windows",
              "event_lift": "same for the newest handover / satellite-change row (baseline: windows "
                            "without an event)",
              "event_pre_lift": "attention mass on rows older than the event / baseline mass on those lags",
              "event_post_lift": "attention mass on rows newer than the event / baseline mass on those lags",
              "grad_<out>": "share of |d RTT mean forecast / d input| by lag, summed over features, "
                            "total derivative (S2S through its fed-back forecasts, which its decoder "
                            "detaches only as a training device); out = h1, h15 or sum over steps; "
                            "'_change' = forecast minus the anchored last value (DS2S only); "
                            "'_raw' = base input features only, i.e. without the engineered trailing "
                            "statistics (DS2S only, S2S has only base features)",
              "grad_<out>_eng_feature_share": "share of |gradient| (all lags) on the engineered "
                                              "features (trailing means up to 60 s, session mean, "
                                              "rows since update, row_dt), DS2S only",
              "abl_*": "test forecasts recomputed with every window's attention replaced by the mean "
                       "map of VALIDATION windows (valmap) or by uniform weights; RTT mean-forecast "
                       "MAE / RMSE over all steps (and MAE at step 15) on the analysed test windows; "
                       "ratio = ablated / own attention (1 = the attention pattern does not matter); "
                       "shift = mean |ablated - own forecast| in ms; move = mean |own forecast - "
                       "persistence| in ms; tv_valmap_testmap = total-variation distance between the "
                       "validation and test mean maps",
              "max_abs_diff_vs_saved": "CPU recomputation vs saved GPU test forecasts (ms, % for "
                                       "loss), both models"},
          "links": {}}
    for link, rr in allres.items():
        js["links"][link] = {}
        for m, r in rr.items():
            M, Mf, G, st, sc = r["M"], r["Mf"], r["G"], r["steps"], r["scalars"]
            for h in range(H):
                for l in range(L):
                    maps.append((link, m, h + 1, l, lag_s[l], M[h, l]))
                    if Mf is not None:
                        fmaps.append((link, m, h + 1, l, lag_s[l], Mf[h, l]))
            for k, g in G.items():
                for l in range(L):
                    grads.append((link, m, k, l, lag_s[l], g[l]))
            for t, prow in r["phase_all"].items():
                for p in prow:
                    phases.append({"link": link, "model": m, "target": t,
                                   "is_1hz_target": t == sc.get("phase_target"), **p})
            for h in range(H):
                steps.append({"link": link, "model": m, "step": h + 1, **{k: float(v[h]) for k, v in st.items()}})
            row = {"link": link, "model": m, **{k: float(np.mean(v)) for k, v in st.items()},
                   "mean_lag_s_step1": float(st["mean_lag_s"][0]),
                   "mean_lag_s_step15": float(st["mean_lag_s"][-1]),
                   "mass_last1s_step1": float(st["mass_last1s"][0]),
                   "mass_last1s_step15": float(st["mass_last1s"][-1]),
                   **sc}
            summ.append(row)
            js["links"][link][m] = {**row,
                                    "per_step": {k: [float(x) for x in v] for k, v in st.items()},
                                    "profile_over_steps_x30": [float(x) for x in M.mean(0) * L],
                                    "fresh_profile_over_steps_x30": None if Mf is None else
                                    [float(x) for x in Mf.mean(0) * L],
                                    "grad_profiles_x30": {k: [float(x) for x in g * L] for k, g in G.items()},
                                    "phase_lift_all_targets": r["phase_all"]}
    cols = ["link", "model", "step", "lag", "lag_s", "weight"]
    pd.DataFrame(maps, columns=cols).to_csv(out_dir / "attn_maps.csv", index=False, float_format="%.6g")
    if fmaps:
        pd.DataFrame(fmaps, columns=cols).to_csv(out_dir / "attn_fresh_maps.csv", index=False, float_format="%.6g")
    if grads:
        pd.DataFrame(grads, columns=["link", "model", "output", "lag", "lag_s", "share"]).to_csv(
            out_dir / "grad_profiles.csv", index=False, float_format="%.6g")
    if phases:
        pd.DataFrame(phases).to_csv(out_dir / "attn_phase.csv", index=False, float_format="%.6g")
    pd.DataFrame(steps).to_csv(out_dir / "attn_steps.csv", index=False, float_format="%.6g")
    sm = pd.DataFrame(summ)
    sm.to_csv(out_dir / "attn_summary.csv", index=False, float_format="%.6g")
    json.dump(js, open(out_dir / "attn_summary.json", "w"), indent=1, default=str)
    show = ["link", "model", "entropy_norm", "eff_rows", "mass_lag0", "mass_last1s", "mass_older3s",
            "mean_lag_s", "tv_from_mean", "comb_ratio_1s", "phase_target", "phase_lift_rsc0",
            "phase_lift_rsc5plus", "fresh_comb_contrast",
            "spike_lift", "event_lift", "event_pre_lift", "grad_h15_mass_last1s", "grad_h15_raw_mass_last1s",
            "grad_sum_change_mass_last1s", "grad_h15_eng_feature_share", "abl_mae_ratio_valmap",
            "abl_mae_ratio_uniform", "abl_rmse_ratio_uniform", "max_abs_diff_vs_saved"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(sm[[c for c in show if c in sm.columns]].to_string(float_format=lambda v: f"{v:.3f}"))

    # Figure data (450 rows per map file).
    for link, rr in allres.items():
        if a.fr_method not in rr:
            continue
        r = rr[a.fr_method]
        for M, name in ((r["M"], f"fr_attention{a.tag}_{link}.dat"), (r["Mf"], f"fr_attention{a.tag}_fresh_{link}.dat")):
            if M is None:
                continue
            rows = [(lag_s[l], h + 1, M[h, l], M[h, l] * L, np.log2(max(M[h, l] * L, 1e-6)))
                    for h in range(H) for l in range(L)]
            pd.DataFrame(rows, columns=["lag", "step", "w", "ratio", "lr"]).to_csv(
                fig_dir / name, sep=" ", index=False, float_format="%.5g")
        prof = {"lag": lag_s, "ds2s": r["M"].mean(0) * L}
        if r["Mf"] is not None:
            prof["ds2sfresh"] = r["Mf"].mean(0) * L
        for k in ("sum", "sum_change", "sum_change_raw"):
            if k in r["G"]:
                prof["grad" + k.replace("sum", "").replace("_", "")] = r["G"][k] * L
        if a.cmp_method in rr:
            prof["s2s"] = rr[a.cmp_method]["M"].mean(0) * L
            if "sum" in rr[a.cmp_method]["G"]:
                prof["s2sgrad"] = rr[a.cmp_method]["G"]["sum"] * L
        pd.DataFrame(prof).to_csv(fig_dir / f"fr_attention{a.tag}_prof_{link}.dat", sep=" ", index=False,
                                  float_format="%.5g")
    write_figures(allres, a)


# ---------------------------------------------------------------------------
# Figures (pgfplots; data paths are relative to reports/final_report/)
# ---------------------------------------------------------------------------

PANELS = ["oneweb_dl", "starlink_dl", "quectel_dl", "oneweb_ul", "starlink_ul", "quectel_ul"]
NICE = {"oneweb": "OneWeb", "starlink": "Starlink", "quectel": "5G"}


def nice(link):
    p, d = link.split("_")
    return f"{NICE[p]} {d.upper()}"


def write_figures(allres, a):
    tex_dir = Path(a.tex_dir)
    tex_dir.mkdir(parents=True, exist_ok=True)
    links = [l for l in PANELS if l in allres and a.fr_method in allres[l]]
    if len(links) < len(PANELS):
        print(f"figures: only {len(links)} of {len(PANELS)} links available, figures not written")
        return
    cl = a.color_lim
    # Heatmap data with the colour value clipped to the shared range.
    for link in links:
        f = Path(a.fig_dir) / f"fr_attention{a.tag}_{link}.dat"
        d = pd.read_csv(f, sep=" ")
        d["lrc"] = d["lr"].clip(-cl, cl)
        d.to_csv(f, sep=" ", index=False, float_format="%.5g")

    # 1) Heatmaps: DS2S mean attention, decoder step x input lag.
    ticks = list(range(-int(cl), int(cl) + 1))
    labels = []
    for t in ticks:
        v = 2.0 ** t
        s = f"{v:g}" if v >= 1 else f"1/{1 / v:g}"
        labels.append(("$\\le$" if t == ticks[0] else "$\\ge$" if t == ticks[-1] else "") + s)
    hs, vs, w, h = 0.32, 1.25, 4.70, 2.55     # 4.75 cm overflowed \textwidth by 0.5 pt
    out = ["% Generated by modeling/experiments/fr_attention.py (do not edit by hand).",
           f"% DS2S ({a.fr_method}, seed {a.seed}) mean decoder attention over test windows.",
           "% Colour = log2(weight x 30), 0 = uniform weight over the 30 input rows.",
           "\\begin{tikzpicture}",
           f"\\begin{{groupplot}}[group style={{group size=3 by 2, horizontal sep={hs}cm, vertical sep={vs}cm,",
           "    xticklabels at=edge bottom, yticklabels at=edge left, xlabels at=edge bottom, ylabels at=edge left},",
           f"  scale only axis, width={w}cm, height={h}cm, enlargelimits=false, axis on top,",
           "  xmin=-0.1, xmax=5.9, ymin=0.5, ymax=15.5, x dir=reverse,",
           "  xtick={0,1,2,3,4,5}, ytick={1,5,10,15}, yticklabels={0.2,1,2,3},",
           "  xlabel={input lag (s), 0 = newest row}, ylabel={horizon (s)},",
           "  tick label style={font=\\scriptsize}, label style={font=\\scriptsize},",
           "  title style={font=\\footnotesize, align=center, yshift=-0.5ex}, tick align=outside, major tick length=2pt,",
           f"  colormap/viridis, point meta min={-cl}, point meta max={cl}]"]
    for i, link in enumerate(links):
        st = allres[link][a.fr_method]["steps"]
        eff, last = float(np.mean(st["eff_rows"])), 100 * float(np.mean(st["mass_last1s"]))
        out.append(f"\\nextgroupplot[title={{({chr(97 + i)}) {nice(link)}\\\\[-1pt]{{\\scriptsize "
                   f"{eff:.0f} of 30 rows, {last:.0f}\\,\\% in last 1\\,s}}}}]")
        out.append(f"\\addplot[matrix plot*, mesh/cols={L}, point meta=explicit] "
                   f"table[x=lag, y=step, meta=lrc] {{data/fr_attention{a.tag}_{link}.dat}};")
        out.append("\\draw[white, densely dashed, thick] (axis cs:0.9,0.5) -- (axis cs:0.9,15.5);")
    out += ["\\end{groupplot}",
            f"\\begin{{axis}}[at={{($(group c3r1.north east)+(0.22cm,0)$)}}, anchor=north west, hide axis,",
            f"  scale only axis, width=1pt, height={2 * h + vs}cm, colormap/viridis,",
            f"  point meta min={-cl}, point meta max={cl}, colorbar, colorbar style={{width=0.2cm,",
            f"  ytick={{{','.join(map(str, ticks))}}}, yticklabels={{{','.join(labels)}}},",
            "  yticklabel style={font=\\scriptsize}, ylabel={attention weight / uniform weight},",
            "  ylabel style={font=\\scriptsize}}]",
            "\\addplot[draw=none] coordinates {(0,0)};",
            "\\end{axis}",
            "\\end{tikzpicture}"]
    (tex_dir / f"fr_attention{a.tag}.tex").write_text("\n".join(out) + "\n")

    # 2) Lag profiles (mean over decoder steps): DS2S attention, S2S attention and
    #    the DS2S gradient of the learned change, all relative to uniform. The panel
    #    with a 1 Hz report target also shows the windows that end on a new report,
    #    with an inset on the last 2.4 s where the two DS2S curves overlap.
    ph = 2.9
    out = ["% Generated by modeling/experiments/fr_attention.py (do not edit by hand).",
           "% Attention and gradient share per input lag, mean over the 15 decoder steps, x 30 (1 = uniform).",
           "\\begin{tikzpicture}",
           "\\begin{groupplot}[group style={group size=3 by 2, horizontal sep=0.32cm, vertical sep=1.1cm,",
           "    xticklabels at=edge bottom, yticklabels at=edge left, xlabels at=edge bottom, ylabels at=edge left},",
           f"  frbase, scale only axis, width=4.75cm, height={ph}cm, ymode=log, x dir=reverse,",
           "  xmin=0, xmax=5.8, ymin=0.01, ymax=30, xtick={0,1,2,3,4,5}, ytick={0.01,0.1,1,10},",
           "  yticklabels={0.01,0.1,1,10}, xlabel={input lag (s), 0 = newest row}, ylabel={weight / uniform},",
           "  title style={font=\\footnotesize, yshift=-0.5ex},",
           "  legend style={draw=black!50, font=\\scriptsize, /tikz/every even column/.append style={column sep=0.3cm}},",
           "  legend cell align={left}]"]
    zoom = None
    has_fresh = any(allres[l][a.fr_method]["Mf"] is not None for l in links)
    for i, link in enumerate(links):
        rr = allres[link]
        opts = f"title={{({chr(97 + i)}) {nice(link)}}}"
        if i == 0:
            opts += f", legend to name=legattn{a.tag.replace('_', '')}, legend columns={5 if not has_fresh else 3}"
        out.append(f"\\nextgroupplot[{opts}]")
        out.append("\\addplot[black!45, densely dotted, thick, domain=0:5.8, samples=2, forget plot] {1};")
        dat = f"data/fr_attention{a.tag}_prof_{link}.dat"
        out.append(f"\\addplot[mFR, mark repeat=3] table[x=lag, y=ds2s] {{{dat}}};")
        if i == 0:
            out.append("\\addlegendentry{DS2S attention}")
        if a.cmp_method in rr:
            out.append(f"\\addplot[mPrev, mark repeat=3] table[x=lag, y=s2s] {{{dat}}};")
            if i == 0:
                out.append("\\addlegendentry{S2S attention}")
        if "sum_change" in rr[a.fr_method]["G"]:
            out.append(f"\\addplot[black, dashed, thick, mark=triangle*, mark size=1.3pt, mark repeat=3] "
                       f"table[x=lag, y=gradchange] {{{dat}}};")
            if i == 0:
                out.append("\\addlegendentry{DS2S input gradient (learned change)}")
        if i == 0:
            out.append("\\addlegendimage{black!45, densely dotted, thick}\\addlegendentry{uniform weight}")
            if has_fresh:
                out.append("\\addlegendimage{cFR, thick, densely dotted, mark=*, mark size=0.9pt}"
                           "\\addlegendentry{DS2S attention, windows ending on a new 1\\,Hz report}")
        if rr[a.fr_method]["Mf"] is not None and zoom is None:
            # Windows ending on a fresh 1 Hz report share one phase, so the comb survives averaging.
            out.append(f"\\addplot[cFR, thick, densely dotted, mark=*, mark size=0.9pt] "
                       f"table[x=lag, y=ds2sfresh] {{{dat}}};")
            k = int(round(2.4 / ROW_S)) + 1                                   # lags 0 .. 2.4 s
            prof = np.r_[rr[a.fr_method]["Mf"].mean(0)[:k], rr[a.fr_method]["M"].mean(0)[:k]] * L
            lo, hi = float(prof.min()) * 0.95, float(prof.max()) * 1.05
            zoom = (i, link, lo, hi)
            out.append(f"\\draw[black!70] (axis cs:0,{lo:.3g}) rectangle (axis cs:2.4,{hi:.3g});")
            out.append(f"\\coordinate (zb) at (axis cs:1.2,{lo:.3g});")
    out.append("\\end{groupplot}")
    out.append("\\node at ($(group c2r1.north)+(0,1.0cm)$) {\\pgfplotslegendfromname{legattn" + a.tag.replace('_', '') + "}};")
    if zoom is not None:
        i, link, lo, hi = zoom
        c, r = i % 3 + 1, i // 3 + 1
        dat = f"data/fr_attention{a.tag}_prof_{link}.dat"
        out += [f"\\begin{{axis}}[at={{($(group c{c}r{r}.south)+(0,0.3cm)$)}}, anchor=south, "
                f"width=3.3cm, height=0.85cm, scale only axis, x dir=reverse, xmin=0, xmax=2.4, "
                f"ymin={lo:.3g}, ymax={hi:.3g}, xtick={{0,1,2}}, tick label style={{font=\\tiny}}, "
                f"ytick=\\empty, xtick pos=bottom, axis background/.style={{fill=white}}, grid=major, "
                f"grid style={{gray!25}}, name=ins]",
                f"\\addplot[mFR, mark size=0.9pt] table[x=lag, y=ds2s] {{{dat}}};",
                f"\\addplot[cFR, thick, densely dotted, mark=*, mark size=1.0pt] table[x=lag, y=ds2sfresh] {{{dat}}};",
                "\\end{axis}",
                "\\draw[->, black!70] (zb) -- (ins.north);"]
    out.append("\\end{tikzpicture}")
    (tex_dir / f"fr_attention{a.tag}_prof.tex").write_text("\n".join(out) + "\n")
    print(f"figures written to {tex_dir}/fr_attention{a.tag}.tex and fr_attention{a.tag}_prof.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Attention analysis of the DS2S and S2S forecasters")
    ap.add_argument("--fr_method", default="fr_dist")
    ap.add_argument("--cmp_method", default="prev_seq2seq")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--links", nargs="+", default=LINKS, help="links to (re)compute, 'none' = tables only")
    ap.add_argument("--max_windows", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--spike_q", type=float, default=0.95,
                    help="spike level = this quantile of VALIDATION window-max input RTT")
    ap.add_argument("--min_lag", type=int, default=3,
                    help="content tests use spike / event rows at least this many rows old")
    ap.add_argument("--grad_windows", type=int, default=2048,
                    help="evenly spaced test windows for the gradient cross-check, 0 = skip")
    ap.add_argument("--no_ablation", action="store_true",
                    help="skip the fixed-attention ablation (validation mean map / uniform weights)")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--res_dir", default=str(ROOT / "results_final/main"))
    ap.add_argument("--out_dir", default=None,
                    help="default results_final/analysis/attention (+ _<fr_method> unless fr_dist)")
    ap.add_argument("--fig_dir", default=str(ROOT.parent / "reports/final_report/data"))
    ap.add_argument("--tex_dir", default=str(ROOT.parent / "reports/final_report/figs"))
    ap.add_argument("--color_lim", type=float, default=2.0,
                    help="heatmap colour range in log2(weight x 30), shared by all panels")
    ap.add_argument("--data_root", default=str(ROOT / "data/lsn-data"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    np.random.seed(0)

    # Outputs of any --fr_method other than fr_dist carry its name, so the DS2S
    # figures and tables are never overwritten by a comparison run.
    a.tag = "" if a.fr_method == "fr_dist" else f"_{a.fr_method}"
    if a.out_dir is None:
        a.out_dir = str(ROOT / f"results_final/analysis/attention{a.tag}")
    res_dir, out_dir, fig_dir = Path(a.res_dir), Path(a.out_dir), Path(a.fig_dir)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for link in a.links:
        if link in LINKS:
            print(f"=== {link}", flush=True)
            analyse_link(link, a, res_dir, out_dir / "cache")
    write_outputs(a, out_dir, fig_dir)


if __name__ == "__main__":
    main()
