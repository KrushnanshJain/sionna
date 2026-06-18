# RA-LWLM-style localization dataset generator

`generate_dataset.py` produces a small synthetic CSI-fingerprint dataset with
Sionna RT, following the data-generation recipe of the RA-LWLM paper
(*Retrieval-Augmented In-Context Localization with Wireless Foundation Models*,
Sec. V-A).

## What it does

For each scene it randomizes:

- **Environment**: 2-4 rectangular concrete buildings (height 10 m, length
  `U(5,16)` m, width `U(5,10)` m) on a flat ground.
- **BS config `c_s`**: `N_ant ∈ {8,16,32}`, bandwidth `∈ {5,10,20}` MHz,
  height `U(15,20)` m, azimuth `U(25°,65°)`. Carrier frequency 3.5 GHz and
  128 subcarriers are fixed.

The BS is an uplink receiver with a horizontal ULA at the origin; the UE is a
single-antenna transmitter. Each random UE position is ray-traced and converted
to a channel frequency response `H` (the CSI matrix), with AWGN added.

## Usage

```bash
# from the repo root, with sionna installed in your environment
python localization/generate_dataset.py                 # tiny default set (CPU)

# scale toward the paper
python localization/generate_dataset.py \
    --seen-scenes 20 --unseen-scenes 10 \
    --db-size 4000 --query-size 1000
```

By default the CPU/LLVM backend is forced (works on any machine, and is required
for pre-Volta GPUs). Set `SIONNA_USE_GPU=1` if your GPU is Volta or newer.

## Output structure

One compressed `.npz` per scene under `dataset/`:

| Key | Shape | Meaning |
|---|---|---|
| `H_db` | `(db_size, N_ant, 128)` complex64 | database CSI fingerprints `D_s` |
| `pos_db` | `(db_size, 2)` float32 | their `(x, y)` labels |
| `H_query` | `(query_size, N_ant, 128)` complex64 | held-out query CSI |
| `pos_query` | `(query_size, 2)` float32 | query labels |
| `n_ant, bw_hz, z_bs, az_deg, fc_hz, n_subc` | scalars | shared config `c_s` |
| `footprints` | `(n_buildings, 4)` | building `(cx, cy, L, W)` |

Plus `dataset/meta.json` with all generation parameters.

Note: raw complex `H` is stored. Magnitude/phase preprocessing and the
angle-delay transform belong in the downstream ML code, not the dataset.
`N_ant` varies across scenes by design, hence the per-scene files.
