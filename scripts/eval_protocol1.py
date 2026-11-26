"""Official 8-benchmark Protocol 1 evaluation runner for DP-SB + DER.

Wraps the untouched official evaluation scripts:
  - src/full_length_eval.py      (PreVAD, XD-Violence, UCF-Crime, MSAD, UBnormal, TAD, LAD)
  - src/full_length_eval_dota.py (DoTA with per-video min-max normalization)

Supports:
  1. Single-checkpoint zero-leakage evaluation across all 8 benchmarks (--ckpt).
  2. Multi-epoch evaluation over saved checkpoints (--run-dir --epochs).
  3. Parameter-free DER (c0=0, tau_der=1.0, rho0=5.0, ctr+gate) and Table IV(a)
     controlled single-scale ablations (--fixed-wl, --backbone-only, --disable-gate).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_DATASETS = [
    ("prevad", "AP"),
    ("xd", "AP"),
    ("ucf", "AUC"),
    ("msad", "AUC"),
    ("ubn", "AUC"),
    ("dota", "AUC"),
    ("tad", "AUC"),
    ("lad", "AUC"),
]


def build_eval_config(base_cfg_path: Path, args) -> dict:
    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    mc = cfg["model"]["init_args"]["model_config"]["init_args"]
    mc["verbalizer_type"] = None

    if args.backbone_only:
        mc["beam_entropy_router"] = False
        mc["beam_global_scale_override"] = 0.0
        mc["beam_local_scale_override"] = 0.0
    elif args.fixed_wl is not None:
        wL = float(args.fixed_wl)
        mc["beam_entropy_router"] = False
        mc["beam_center_local"] = not args.no_center_local
        mc["beam_disable_cfar_gate"] = bool(args.disable_gate)
        mc["beam_global_scale_override"] = round(1.0 - wL, 6)
        mc["beam_local_scale_override"] = round(float(args.rho0) * wL, 6)
    elif args.der:
        mc["beam_entropy_router"] = True
        mc["beam_router_temp"] = float(args.tau_der)
        mc["beam_center_local"] = not args.no_center_local
        mc["beam_disable_cfar_gate"] = bool(args.disable_gate)
        mc["beam_global_scale_override"] = 1.0
        mc["beam_local_scale_override"] = float(args.rho0)
    return cfg


def run_one_eval(config_path: Path, ckpt_path: Path, dataset: str, data_root: Path, log_path: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{REPO_ROOT / 'src'}"
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    script = "src/full_length_eval_dota.py" if dataset == "dota" else "src/full_length_eval.py"
    cmd = [
        sys.executable,
        str(REPO_ROOT / script),
        "--config", str(config_path),
        "--ckpt", str(ckpt_path),
        "--dataset", dataset,
        "--data_root", str(data_root),
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    dt = time.time() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"Eval failed for {dataset} ({ckpt_path.name}), rc={proc.returncode}. See {log_path}")
    if "Load may failed" in proc.stdout:
        raise RuntimeError(f"Checkpoint load mismatch detected for {dataset} ({ckpt_path.name}). See {log_path}")

    auc_m = re.findall(r"^AUC:\s*([0-9.]+)", proc.stdout, flags=re.MULTILINE)
    ap_m = re.findall(r"^AP:\s*([0-9.]+)", proc.stdout, flags=re.MULTILINE)
    if not auc_m or not ap_m:
        raise RuntimeError(f"Failed to parse AUC/AP from {log_path}")

    return {
        "AUC": round(float(auc_m[-1]) * 100.0, 2),
        "AP": round(float(ap_m[-1]) * 100.0, 2),
        "seconds": round(dt, 2),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Official 8-benchmark Protocol 1 evaluation for DP-SB + DER.")
    parser.add_argument("--config", type=str, required=True, help="Path to model YAML config.")
    parser.add_argument("--ckpt", type=str, default=None, help="Single checkpoint path (zero-leakage evaluation).")
    parser.add_argument("--run-dir", type=str, default=None, help="Training run dir containing ckpts/epoch_*.ckpt.")
    parser.add_argument("--epochs", type=str, default=None, help="Comma-separated epoch indices when using --run-dir.")
    parser.add_argument("--data-root", type=str, default="data/PreVAD", help="Root directory containing PreVAD and other_datasets.")
    parser.add_argument("--datasets", type=str, default="prevad,xd,ucf,msad,ubn,dota,tad,lad", help="Comma-separated dataset list.")
    parser.add_argument("--out-dir", type=str, default="outputs/protocol1", help="Directory for logs and JSON summary.")

    # DER & Table IV(a) controlled routing switches
    parser.add_argument("--der", action="store_true", help="Enable parameter-free DER routing (c0=0, tau_der=1.0, rho0=5.0).")
    parser.add_argument("--rho0", type=float, default=5.0, help="PreVAD-train variance equalization constant (default: 5.0).")
    parser.add_argument("--tau-der", type=float, default=1.0, help="DER temperature tau_der (default: 1.0).")
    parser.add_argument("--no-center-local", action="store_true", help="Disable valid-mask chunk centering (raw ablation).")
    parser.add_argument("--disable-gate", action="store_true", help="Disable video confidence gate g_vid (no-gate ablation).")
    parser.add_argument("--fixed-wl", type=float, default=None, help="Fixed constant w_L in [0, 1] instead of DER (Table IV(a)).")
    parser.add_argument("--backbone-only", action="store_true", help="Set Delta_b = 0 to isolate base backbone on same checkpoints.")
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.ckpt) == bool(args.run_dir):
        raise ValueError("Specify exactly one of --ckpt or --run-dir.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    active_ds = [(d, m) for d, m in ALL_DATASETS if d in set(args.datasets.split(","))]

    eval_cfg = build_eval_config(Path(args.config), args)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        yaml.safe_dump(eval_cfg, tf, sort_keys=False)
        tmp_cfg_path = Path(tf.name)

    try:
        if args.ckpt:
            ckpt_list = [("single", Path(args.ckpt))]
        else:
            ckpt_dir = Path(args.run_dir) / "ckpts"
            if args.epochs:
                eps = [int(x.strip()) for x in args.epochs.split(",")]
            else:
                eps = sorted(int(p.stem.split("_")[1]) for p in ckpt_dir.glob("epoch_*.ckpt"))
            ckpt_list = [(f"ep{ep:02d}", ckpt_dir / f"epoch_{ep:02d}.ckpt") for ep in eps]

        results = {"datasets": [d for d, _ in active_ds], "per_ckpt": {}}
        for tag, ckpt_path in ckpt_list:
            rec = {}
            print(f"\n=== Evaluating {tag}: {ckpt_path.name} ===", flush=True)
            for ds, mname in active_ds:
                log_path = out_dir / "logs" / f"{tag}_{ds}.log"
                r = run_one_eval(tmp_cfg_path, ckpt_path, ds, data_root, log_path)
                rec[ds] = r
                print(f"  {ds:6s} | {mname}={r[mname]:6.2f}% (AUC={r['AUC']:.2f}, AP={r['AP']:.2f}, {r['seconds']:.1f}s)", flush=True)
            ds_names = [d for d, _ in active_ds]
            main_vals = [rec[d][m] for d, m in active_ds]
            zs_vals = [rec[d][m] for d, m in active_ds if d != "prevad"]
            rec["8DS_Mean"] = round(float(np.mean(main_vals)), 2) if len(ds_names) == 8 else None
            rec["7ZS_Mean"] = round(float(np.mean(zs_vals)), 2) if len(zs_vals) == 7 else None
            if rec["7ZS_Mean"] is not None:
                print(f"  --> 7-ZS Mean: {rec['7ZS_Mean']:.2f}% | 8-DS Mean: {rec['8DS_Mean']:.2f}%", flush=True)
            results["per_ckpt"][tag] = rec

        out_json = out_dir / "protocol1_results.json"
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved Protocol 1 results to {out_json}", flush=True)
    finally:
        tmp_cfg_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
