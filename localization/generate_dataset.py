#!/usr/bin/env python3
"""Generate a small RA-LWLM-style localization dataset with Sionna RT.

Mirrors the data-generation recipe of the RA-LWLM paper (Sec. V-A):
random concrete-building scenes, random BS configs, single-antenna UE,
uplink ULA at the BS, and per-scene (CSI, position, config) databases.
"""
import os
# Sionna auto-selects the CUDA backend if a GPU is visible. Pre-Volta GPUs
# (compute capability < 7.0) cannot run it -> force the CPU/LLVM backend.
# Set SIONNA_USE_GPU=1 only if your GPU is Volta or newer (e.g. T4, RTX 20xx+).
if os.environ.get("SIONNA_USE_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import argparse
import numpy as np
import mitsuba as mi
from sionna.rt import (load_scene, SceneObject, ITURadioMaterial, PlanarArray,
                       Transmitter, Receiver, PathSolver, subcarrier_frequencies)

CARRIER_FREQ = 3.5e9      # fixed across scenes (paper)
NUM_SUBC = 128            # fixed across scenes (paper)
BUILDING_HEIGHT = 10.0    # fixed (paper)
UE_HEIGHT = 1.5


def add_box(scene, name, size, center, material):
    """Add an axis-aligned concrete box (Mitsuba cube spans [-1,1]^3)."""
    obj = SceneObject(mi_mesh=mi.load_dict({"type": "cube"}),
                      name=name, radio_material=material)
    scene.edit(add=obj)  # must add before transforming
    obj.scaling = [float(size[0] / 2), float(size[1] / 2), float(size[2] / 2)]
    obj.position = [float(center[0]), float(center[1]), float(center[2])]
    return (center[0], center[1], size[0], size[1])  # (cx, cy, L, W) footprint


def build_random_scene(rng):
    """Random concrete-building environment + random BS config (paper Sec. V-A)."""
    cfg = {
        "n_ant": int(rng.choice([8, 16, 32])),
        "bw_hz": float(rng.choice([5e6, 10e6, 20e6])),
        "z_bs": float(rng.uniform(15.0, 20.0)),
        "az_deg": float(rng.uniform(25.0, 65.0)),
        "fc_hz": CARRIER_FREQ,
        "n_subc": NUM_SUBC,
    }
    scene = load_scene()              # empty scene
    scene.frequency = CARRIER_FREQ
    concrete = ITURadioMaterial(name="concrete", itu_type="concrete", thickness=0.3)

    add_box(scene, "ground", [200.0, 200.0, 0.2], [0.0, 0.0, 0.0], concrete)
    footprints = []
    for k in range(int(rng.integers(2, 5))):          # 2, 3, or 4 buildings
        length = rng.uniform(5.0, 16.0)
        width = rng.uniform(5.0, 10.0)
        cx, cy = rng.uniform(-40.0, 40.0), rng.uniform(-40.0, 40.0)
        footprints.append(
            add_box(scene, f"building_{k}", [length, width, BUILDING_HEIGHT],
                    [cx, cy, BUILDING_HEIGHT / 2], concrete))

    # BS = uplink receiver with a horizontal ULA, at the origin (paper)
    scene.rx_array = PlanarArray(num_rows=1, num_cols=cfg["n_ant"],
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")
    scene.add(Receiver("bs", position=[0.0, 0.0, cfg["z_bs"]],
                       orientation=[float(np.deg2rad(cfg["az_deg"])), 0.0, 0.0]))
    return scene, footprints, cfg


def _inside_building(x, y, footprints, margin=1.0):
    return any(abs(x - cx) <= L / 2 + margin and abs(y - cy) <= W / 2 + margin
               for (cx, cy, L, W) in footprints)


def sample_csi(scene, solver, freqs, footprints, n_samples, rng, area, snr_db, max_depth):
    """Trace n_samples random UE positions -> (H, position) pairs."""
    n_ant = scene.rx_array.num_ant
    H_out = np.empty((n_samples, n_ant, NUM_SUBC), dtype=np.complex64)
    pos_out = np.empty((n_samples, 2), dtype=np.float32)
    snr_lin = 10 ** (snr_db / 10.0)
    got = 0
    while got < n_samples:
        x = float(rng.uniform(-area, area))
        y = float(rng.uniform(-area, area))
        if _inside_building(x, y, footprints):
            continue
        if "ue" in scene.transmitters:
            scene.remove("ue")
        scene.add(Transmitter("ue", position=[x, y, UE_HEIGHT]))
        paths = solver(scene, max_depth=max_depth, refraction=True, seed=1)
        H = np.squeeze(paths.cfr(frequencies=freqs, normalize=True,
                                 normalize_delays=True, out_type="numpy"))
        # AWGN term n_{s,m} of Eq. (1)
        noise_p = np.mean(np.abs(H) ** 2) / snr_lin
        H = H + np.sqrt(noise_p / 2) * (rng.standard_normal(H.shape)
                                        + 1j * rng.standard_normal(H.shape))
        H_out[got] = H.astype(np.complex64)
        pos_out[got] = [x, y]
        got += 1
    return H_out, pos_out


def generate_scene(idx, kind, rng, args, out_dir):
    scene, footprints, cfg = build_random_scene(rng)
    solver = PathSolver()
    freqs = subcarrier_frequencies(NUM_SUBC, cfg["bw_hz"] / NUM_SUBC)
    H_db, p_db = sample_csi(scene, solver, freqs, footprints,
                            args.db_size, rng, args.area, args.snr_db, args.max_depth)
    H_q, p_q = sample_csi(scene, solver, freqs, footprints,
                          args.query_size, rng, args.area, args.snr_db, args.max_depth)
    path = os.path.join(out_dir, f"{kind}_scene_{idx:03d}.npz")
    np.savez_compressed(path, H_db=H_db, pos_db=p_db, H_query=H_q, pos_query=p_q,
                        **cfg, footprints=np.array(footprints, dtype=np.float32))
    print(f"  {kind} scene {idx:03d}: N_ant={cfg['n_ant']:2d}  "
          f"BW={cfg['bw_hz']/1e6:.0f}MHz  buildings={len(footprints)}  "
          f"db={H_db.shape}  query={H_q.shape}  -> {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser(description="RA-LWLM-style localization dataset generator")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--seen-scenes", type=int, default=2, help="SS scenes (paper: 20)")
    ap.add_argument("--unseen-scenes", type=int, default=1, help="US scenes (paper: 10)")
    ap.add_argument("--db-size", type=int, default=100, help="DB samples/scene (paper: 4000)")
    ap.add_argument("--query-size", type=int, default=20, help="query samples/scene (paper: 1000)")
    ap.add_argument("--area", type=float, default=50.0, help="UE sampling half-extent [m]")
    ap.add_argument("--snr-db", type=float, default=20.0)
    ap.add_argument("--max-depth", type=int, default=4, help="max ray bounces")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    print(f"Mitsuba variant: {mi.variant()}")
    print(f"Generating {args.seen_scenes} seen + {args.unseen_scenes} unseen scenes "
          f"({args.db_size} db + {args.query_size} query each)")

    for i in range(args.seen_scenes):
        generate_scene(i, "seen", rng, args, args.out)
    for i in range(args.unseen_scenes):
        generate_scene(i, "unseen", rng, args, args.out)

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({**vars(args),
                   "carrier_freq_hz": CARRIER_FREQ, "num_subcarriers": NUM_SUBC,
                   "building_height_m": BUILDING_HEIGHT, "ue_height_m": UE_HEIGHT}, f, indent=2)
    print(f"Done. Dataset written to '{args.out}/'")


if __name__ == "__main__":
    main()
