# Measurement data (ARA wireless living lab, 5 Hz)

Compressed copies of `modeling/data/lsn-data/` (the raw folder is git-ignored because of its size, about 820 MB uncompressed).

| File | Contents |
|---|---|
| `lsn-data_oneweb.tar.xz` | OneWeb DL/UL sessions (65 + 65 CSV files) |
| `lsn-data_starlink.tar.xz` | Starlink latency / latency_loss_throughput / throughput sessions |
| `lsn-data_quectel.tar.xz` | 5G (Quectel modem) sessions |
| `lsn-data_scripts.tar.xz` | Parsers (`parse_raw/`) and collector scripts |

Restore for the experiments:
```bash
cd modeling/data && for f in ../../data_release/lsn-data_*.tar.xz; do tar -xJf "$f"; done
```
Checksums: `SHA256SUMS`. The files are stored with Git LFS (`git lfs pull`).
