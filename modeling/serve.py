"""
serve.py - Sidecar inference server for the mpquic tunnel's FEC model.

Wire protocol (length-prefixed JSON over a Unix domain socket), matching
tunnel_app::tunnel::model_multipath::MultiProviderSidecarModel exactly:

    request  (Rust -> Python):
        [u32 BE length][JSON: {
            "provider": "starlink" | "oneweb" | "quectel",
            "path_id":  <u64>,
            "window":   [{"throughput_mbps": f32, "rtt_ms": f32, "loss_pct": f32}, ...]  (30 entries),
            "hour":     [f32, ...]  (30 entries, raw 0-23),
            "dow":      [f32, ...]  (30 entries, raw 0-6)
        }]

    response (Python -> Rust):
        [u32 BE length][JSON: {
            "throughput_mbps": [f32, ...]  (5 entries, real Mbps),
            "rtt_ms":          [f32, ...]  (5 entries, real ms),
            "loss_pct":        [f32, ...]  (5 entries, real percent)
        }]

Model loading
-------------
Each provider's model is loaded from its metadata.json (produced by
`python run.py train ...` — see models/seq2seq.py, which now persists
feature_stats/target_stats/actual_feature_cols/normalise there specifically
so this server doesn't have to guess or recompute them). Point CONFIG below
at each provider's metadata.json path.

CRITICAL naming convention: each model MUST have been trained with
`--targets <throughput> <rtt> <loss>` in EXACTLY that order (see training
commands in the project notes). This server maps output channel 0 -> 
throughput_mbps, 1 -> rtt_ms, 2 -> loss_pct positionally — it does not
inspect target column names. If you retrain with a different target order,
update TARGET_ROLE_INDEX below or fix the training command instead.

Phy-layer + weather data
-------------------------
This server is responsible for fetching physical-layer (rsrp/sinr/dish_snr/
etc.) and weather features itself, per the tunnel-side design (Rust only
sends throughput/rtt/loss + hour/dow — everything else the model needs is
filled in here). PhyWeatherClient below is a STUB that returns zeros for
every requested feature; replace fetch() per provider with real API calls.
Until you do, predictions will systematically ignore phy/weather signal
(equivalent to those columns always reading their training-set mean, since
zeros get z-score-normalised same as anything else) — the pipeline still
runs and returns plausible-shaped output, it just won't be as accurate as
once real phy/weather data is wired in.

Because only a *current snapshot* of phy/weather data is fetched per
request (not a historical time series), the same snapshot is broadcast
across all 30 window timesteps. This is a simplification — flagged here so
it's a conscious choice, not a hidden bug — revisit if per-timestep
historical phy/weather logging becomes available.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from models.seq2seq import ThroughputSeq2Seq

# ---------------------------------------------------------------------------
# Config — point these at your trained checkpoints
# ---------------------------------------------------------------------------

SOCK_PATH = "/tmp/fec_model.sock"

CONFIG = {
    "starlink": "results/starlink/seq2seq/metadata.json",
    "oneweb":   "results/oneweb/seq2seq/metadata.json",
    "quectel":  "results/quectel/seq2seq/metadata.json",
}

# Positional role -> output channel index. See module doc: training command
# order determines this, not column names.
ROLE_THROUGHPUT, ROLE_RTT, ROLE_LOSS = 0, 1, 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Phy-layer / weather client — STUB, replace per provider
# ---------------------------------------------------------------------------

class PhyWeatherClient:
    """
    Fetches the current snapshot of non-throughput/rtt/loss features for one
    provider (e.g. dish_snr, satellite_dtc, weather_humidity for Starlink;
    rsrp/rsrq/sinr/weather_* for Quectel; sinr_db/elevation_deg/... for
    OneWeb). Keys returned must match the provider's actual_feature_cols
    names from dataset.py exactly.

    STUB: returns {} (missing keys fall back to 0.0 — see build_feature_row).
    Replace `fetch()` with real API calls per provider before relying on
    phy/weather signal in production.
    """

    def __init__(self, provider: str):
        self.provider = provider

    async def fetch(self, path_id: int) -> dict[str, float]:
        # TODO: real per-provider API client. Examples:
        #   starlink: gRPC to local dish (100.64.0.1) for snr/obstruction,
        #             a weather API for humidity/pressure/rain_rate.
        #   quectel:  AT-command modem query for rsrp/rsrq/sinr,
        #             same weather API.
        #   oneweb:   provider-specific telemetry API for sinr_db/elevation/
        #             azimuth/handover counters.
        return {}


PHY_CLIENTS: dict[str, PhyWeatherClient] = {
    p: PhyWeatherClient(p) for p in CONFIG
}


# ---------------------------------------------------------------------------
# Loaded-model bundle
# ---------------------------------------------------------------------------

class LoadedModel:
    def __init__(self, meta_path: str):
        meta_path = Path(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)

        ckpt_path = meta.get("checkpoint", "")
        if not ckpt_path or not Path(ckpt_path).exists():
            siblings = list(meta_path.parent.rglob("best_model.pt"))
            if not siblings:
                raise FileNotFoundError(f"No checkpoint found near {meta_path}")
            ckpt_path = str(siblings[0])

        self.actual_feature_cols: list[str] = meta["actual_feature_cols"]
        self.feature_stats:       dict      = meta["feature_stats"]  # {"mean": [...], "std": [...]} or minmax
        self.target_stats:       dict       = meta["target_stats"]   # {col: {"mean":..,"std":..}, ...}
        self.target_cols:        list[str]  = meta["target_cols"]
        self.normalise:           str        = meta.get("normalise", "zscore")
        self.input_len:           int        = meta.get("input_len", 30)
        self.output_len:          int        = meta["output_len"]

        if len(self.target_cols) != 3:
            raise ValueError(
                f"{meta_path}: expected exactly 3 targets (throughput, rtt, loss), "
                f"got {self.target_cols}. Retrain with --targets <tput> <rtt> <loss>."
            )

        self.model = ThroughputSeq2Seq(
            num_features = meta["num_features"],
            embed_dim    = meta["embed_dim"],
            hidden_size  = meta["hidden_size"],
            input_len    = self.input_len,
            output_len   = self.output_len,
            num_targets  = len(self.target_cols),
        )
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        )
        self.model = self.model.to(DEVICE).eval()

        # Pre-convert feature_stats to numpy arrays once, not per-request.
        self._feat_mean = np.array(self.feature_stats.get("mean", []), dtype=np.float32) \
            if self.normalise == "zscore" else None
        self._feat_std = np.array(self.feature_stats.get("std", []), dtype=np.float32) \
            if self.normalise == "zscore" else None
        self._feat_min = np.array(self.feature_stats.get("min", []), dtype=np.float32) \
            if self.normalise == "minmax" else None
        self._feat_max = np.array(self.feature_stats.get("max", []), dtype=np.float32) \
            if self.normalise == "minmax" else None

        print(
            f"[serve] loaded {meta_path.parent.name}: "
            f"{len(self.actual_feature_cols)} features, targets={self.target_cols}, "
            f"checkpoint={ckpt_path}"
        )

    def normalise_features(self, raw: np.ndarray) -> np.ndarray:
        """raw: (input_len, num_features) -> normalised (input_len, num_features)."""
        if self.normalise == "zscore" and self._feat_mean is not None:
            return (raw - self._feat_mean) / self._feat_std
        if self.normalise == "minmax" and self._feat_min is not None:
            return (raw - self._feat_min) / (self._feat_max - self._feat_min + 1e-8)
        return raw

    def denormalise_target(self, arr: np.ndarray, col: str) -> np.ndarray:
        """arr: (output_len,) normalised -> real units."""
        stats = self.target_stats.get(col)
        if not stats or self.normalise == "none":
            return arr
        if self.normalise == "zscore":
            return arr * stats["std"] + stats["mean"]
        return arr * (stats["max"] - stats["min"]) + stats["min"]

    @torch.no_grad()
    def infer(self, feat_matrix: np.ndarray, hour: list[float], dow: list[float]) -> dict:
        """
        feat_matrix: (input_len, num_features) already normalised, in
        actual_feature_cols order.
        Returns {"throughput_mbps": [...], "rtt_ms": [...], "loss_pct": [...]}
        in real units, length output_len.
        """
        feat_t = torch.from_numpy(feat_matrix).float().unsqueeze(0).to(DEVICE)   # (1, T, F)
        hour_t = torch.tensor(hour, dtype=torch.float32).unsqueeze(0).to(DEVICE) # (1, T)
        dow_t  = torch.tensor(dow,  dtype=torch.float32).unsqueeze(0).to(DEVICE) # (1, T)

        preds, _ = self.model(feat_t, hour_t, dow_t)   # (1, output_len, 3), normalised
        preds = preds.squeeze(0).cpu().numpy()          # (output_len, 3)

        throughput = self.denormalise_target(preds[:, ROLE_THROUGHPUT], self.target_cols[ROLE_THROUGHPUT])
        rtt        = self.denormalise_target(preds[:, ROLE_RTT],        self.target_cols[ROLE_RTT])
        loss       = self.denormalise_target(preds[:, ROLE_LOSS],       self.target_cols[ROLE_LOSS])

        return {
            "throughput_mbps": throughput.tolist(),
            "rtt_ms":          rtt.tolist(),
            "loss_pct":        loss.tolist(),
        }


MODELS: dict[str, LoadedModel] = {}


def load_all_models():
    for provider, path in CONFIG.items():
        try:
            MODELS[provider] = LoadedModel(path)
        except Exception as e:
            print(f"[serve] WARNING: failed to load '{provider}' from {path}: {e}", file=sys.stderr)
    if not MODELS:
        raise RuntimeError("No models loaded — check CONFIG paths in serve.py")


# ---------------------------------------------------------------------------
# Feature-row construction: merge Rust's generic sample + phy/weather
# ---------------------------------------------------------------------------

# Maps the 3 canonical roles Rust always sends onto each provider's actual
# training-time column names. Positional target_cols[ROLE_*] already gives
# us this — reuse it so there's exactly one source of truth for the mapping.

def build_feature_matrix(
    lm: LoadedModel,
    window: list[dict],
    phy_snapshot: dict[str, float],
) -> np.ndarray:
    """
    window: 30 entries of {"throughput_mbps":.., "rtt_ms":.., "loss_pct":..}
    phy_snapshot: {feature_name: value} from PhyWeatherClient, broadcast
                  across every timestep (see module doc).

    Returns (input_len, num_features) raw (not yet normalised) matrix in
    lm.actual_feature_cols order. Missing columns fall back to 0.0, matching
    dataset.py's own ffill().bfill().fillna(0) convention for absent data.
    """
    throughput_col = lm.target_cols[ROLE_THROUGHPUT]
    rtt_col        = lm.target_cols[ROLE_RTT]
    loss_col       = lm.target_cols[ROLE_LOSS]

    T = len(window)
    F = len(lm.actual_feature_cols)
    mat = np.zeros((T, F), dtype=np.float32)

    col_index = {c: i for i, c in enumerate(lm.actual_feature_cols)}

    for t, sample in enumerate(window):
        row_values = dict(phy_snapshot)  # broadcast snapshot as the base row
        row_values[throughput_col] = sample["throughput_mbps"]
        row_values[rtt_col]        = sample["rtt_ms"]
        row_values[loss_col]       = sample["loss_pct"]

        for col, val in row_values.items():
            idx = col_index.get(col)
            if idx is not None:
                mat[t, idx] = val
            # Silently ignored if col isn't one of this provider's actual
            # trained-on features — e.g. phy_snapshot may return keys for
            # columns this particular experiment_type didn't include.

    return mat


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------

async def handle_request(req: dict) -> dict:
    provider = req["provider"]
    path_id  = req["path_id"]

    lm = MODELS.get(provider)
    if lm is None:
        raise ValueError(f"no model loaded for provider '{provider}'")

    phy_client = PHY_CLIENTS.get(provider)
    phy_snapshot = await phy_client.fetch(path_id) if phy_client else {}

    raw_matrix = build_feature_matrix(lm, req["window"], phy_snapshot)
    norm_matrix = lm.normalise_features(raw_matrix)

    return lm.infer(norm_matrix, req["hour"], req["dow"])


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername", "unix-client")
    try:
        while True:
            hdr = await reader.readexactly(4)
            (length,) = struct.unpack(">I", hdr)
            body = await reader.readexactly(length)
            req = json.loads(body)

            try:
                resp = await handle_request(req)
            except Exception as e:
                print(f"[serve] error handling request from {peer}: {e}", file=sys.stderr)
                # Close the connection on error — Rust's ensure_connected()
                # will reconnect and retry on its next tick; this avoids
                # trying to guess a partial/garbage response shape.
                break

            out = json.dumps(resp).encode()
            writer.write(struct.pack(">I", len(out)) + out)
            await writer.drain()

    except asyncio.IncompleteReadError:
        pass  # client closed the connection — normal on tunnel teardown
    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    load_all_models()

    sock_path = Path(SOCK_PATH)
    if sock_path.exists():
        sock_path.unlink()

    server = await asyncio.start_unix_server(handle_client, path=str(sock_path))
    print(f"[serve] listening on {SOCK_PATH}  (providers: {list(MODELS)})")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())