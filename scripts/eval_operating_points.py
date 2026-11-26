"""Operating-point FPR/FNR audit (Table III) and latency/VRAM profiler (Table V).

Evaluates:
  1. ROC operating points: FPR@80%TPR, FPR@90%TPR, FNR@10%FPR.
  2. Fixed threshold (tau = 0.5) FPR and FNR across normal-containing benchmarks.
  3. Zero-shot threshold transfer: calibrate tau_PreVAD@10%FPR on PreVAD and transfer
     directly across the six unseen normal-containing benchmarks (XD, UCF, MSAD, UBN, TAD, LAD).
  4. Optional inference latency (ms/clip) and peak GPU memory (MiB) profiling at B=1 and B=16.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import src.full_length_eval as fle

MAX_LEN = 512
NORMAL_DS7 = ["prevad", "xd", "ucf", "msad", "ubn", "tad", "lad"]
UNSEEN_DS6 = ["xd", "ucf", "msad", "ubn", "tad", "lad"]


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def load_model(cfg_path: Path, ckpt_path: Path, der: bool, rho0: float, center_local: bool, disable_gate: bool):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    mc = cfg["model"]["init_args"]["model_config"]["init_args"]
    mc["verbalizer_type"] = None
    if mc.get("use_semantic_beamformer", False) and der:
        mc["beam_entropy_router"] = True
        mc["beam_router_temp"] = 1.0
        mc["beam_center_local"] = bool(center_local)
        mc["beam_disable_cfar_gate"] = bool(disable_gate)
        mc["beam_global_scale_override"] = 1.0
        mc["beam_local_scale_override"] = float(rho0)
    model = fle.build_model(cfg, str(ckpt_path))
    model.eval()
    return model


@torch.no_grad()
def collect_scores(model, ds_name: str, data_root: Path):
    fle.seed_everything(7)
    dataset, verbalizer, classes = fle.build_dataset(ds_name, str(data_root))
    verbalizer.set_dataset(dataset.abbr)
    gts, preds = [], []
    for i in range(len(dataset)):
        data = dataset.get_full_item(i)
        T = int(data["v_feat_l"])
        gt = np.zeros(T, dtype=np.uint8)
        for sp in data["temp_span"]:
            if sp[0] != sp[1]:
                gt[int(sp[0]):int(sp[1])] = 1
        chunks = []
        for j in range(math.ceil(T / MAX_LEN)):
            v = data["v_feat"][j * MAX_LEN:(j + 1) * MAX_LEN].unsqueeze(0).to(fle.device)
            L = torch.tensor([v.shape[1]], device=fle.device)
            out = model({"v_feat": v, "v_feat_l": L}, class_names=verbalizer(classes))
            chunks.append(out["cls_bin_logits"][0].float().cpu().numpy())
        gts.append(gt)
        preds.append(sigmoid_np(np.concatenate(chunks)[:T]))
    return np.concatenate(gts), np.concatenate(preds)


def compute_op_metrics(gt: np.ndarray, prob: np.ndarray, tau_transfer: float | None = None) -> dict:
    fpr, tpr, thresh = roc_curve(gt, prob)
    auc = float(roc_auc_score(gt, prob) * 100.0)
    fpr_at_80 = float(fpr[np.searchsorted(tpr, 0.80, side="left")] * 100.0)
    fpr_at_90 = float(fpr[np.searchsorted(tpr, 0.90, side="left")] * 100.0)
    idx_f10 = max(int(np.searchsorted(fpr, 0.10, side="right")) - 1, 0)
    fnr_at_10 = float((1.0 - tpr[idx_f10]) * 100.0)
    tau_10 = float(thresh[idx_f10])

    def rate_at(tau: float):
        pred = prob >= tau
        neg = gt == 0
        pos = gt == 1
        return float(pred[neg].mean() * 100.0), float((~pred[pos]).mean() * 100.0)

    fpr_05, fnr_05 = rate_at(0.5)
    out = {
        "AUC": round(auc, 2),
        "FPR@80": round(fpr_at_80, 2),
        "FPR@90": round(fpr_at_90, 2),
        "FNR@10": round(fnr_at_10, 2),
        "tau@10%FPR": round(tau_10, 6),
        "FPR_tau05": round(fpr_05, 2),
        "FNR_tau05": round(fnr_05, 2),
    }
    if tau_transfer is not None:
        tfpr, tfnr = rate_at(tau_transfer)
        out["FPR_transfer"] = round(tfpr, 2)
        out["FNR_transfer"] = round(tfnr, 2)
    return out


@torch.no_grad()
def benchmark_latency_vram(model, T: int = 512, C: int = 14, warmup: int = 10, iters: int = 50) -> dict:
    device = fle.device
    prompts = ["Normal"] + [f"Anomaly class {i}" for i in range(1, C)]
    out = {
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "beamformer_params": int(sum(p.numel() for p in model.beamformer.parameters())) if model.beamformer is not None else 0,
    }
    for B in (1, 16):
        v = torch.randn(B, T, 512, device=device)
        L = torch.full((B,), T, device=device, dtype=torch.long)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        for _ in range(warmup):
            model({"v_feat": v, "v_feat_l": L}, class_names=prompts)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model({"v_feat": v, "v_feat_l": L}, class_names=prompts)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ms_per_clip = ((time.perf_counter() - t0) * 1000.0) / (iters * B)
        vram_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        out[f"lat_ms_B{B}"] = round(ms_per_clip, 3)
        out[f"vram_mib_B{B}"] = round(vram_mib, 1)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Table III operating-point FPR/FNR & Table V latency/VRAM audit.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/PreVAD")
    parser.add_argument("--der", action="store_true")
    parser.add_argument("--rho0", type=float, default=5.0)
    parser.add_argument("--no-center-local", action="store_true")
    parser.add_argument("--disable-gate", action="store_true")
    parser.add_argument("--benchmark-latency", action="store_true", help="Also run Table V latency/VRAM benchmark.")
    parser.add_argument("--out-json", type=str, default="outputs/operating_points.json")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    model = load_model(
        Path(args.config), Path(args.ckpt),
        der=args.der, rho0=args.rho0,
        center_local=not args.no_center_local,
        disable_gate=args.disable_gate,
    )

    per_ds = {}
    gt_pre, pr_pre = collect_scores(model, "prevad", data_root)
    per_ds["prevad"] = compute_op_metrics(gt_pre, pr_pre)
    tau_prevad = per_ds["prevad"]["tau@10%FPR"]
    print(f"  PREVAD | tau@10%FPR={tau_prevad:.4f} | FPR_05={per_ds['prevad']['FPR_tau05']:.2f}% | FNR_05={per_ds['prevad']['FNR_tau05']:.2f}%", flush=True)

    for ds in UNSEEN_DS6:
        gt, pr = collect_scores(model, ds, data_root)
        per_ds[ds] = compute_op_metrics(gt, pr, tau_transfer=tau_prevad)
        r = per_ds[ds]
        print(
            f"  {ds.upper():6s} | AUC={r['AUC']:.2f}% | FPR@80={r['FPR@80']:.2f}% | FPR@90={r['FPR@90']:.2f}% | "
            f"FNR@10={r['FNR@10']:.2f}% | tau0.5(FPR/FNR)={r['FPR_tau05']:.2f}/{r['FNR_tau05']:.2f}%",
            flush=True,
        )

    summary = {
        "7DS_FPR@80": round(float(np.mean([per_ds[d]["FPR@80"] for d in NORMAL_DS7])), 2),
        "7DS_FPR@90": round(float(np.mean([per_ds[d]["FPR@90"] for d in NORMAL_DS7])), 2),
        "7DS_FNR@10": round(float(np.mean([per_ds[d]["FNR@10"] for d in NORMAL_DS7])), 2),
        "7DS_Fixed05_FPR": round(float(np.mean([per_ds[d]["FPR_tau05"] for d in NORMAL_DS7])), 2),
        "7DS_Fixed05_FNR": round(float(np.mean([per_ds[d]["FNR_tau05"] for d in NORMAL_DS7])), 2),
        "6Unseen_Transfer_FPR_mean": round(float(np.mean([per_ds[d]["FPR_transfer"] for d in UNSEEN_DS6])), 2),
        "6Unseen_Transfer_FPR_std": round(float(np.std([per_ds[d]["FPR_transfer"] for d in UNSEEN_DS6])), 2),
        "6Unseen_Transfer_FNR_mean": round(float(np.mean([per_ds[d]["FNR_transfer"] for d in UNSEEN_DS6])), 2),
        "per_dataset": per_ds,
    }
    if args.benchmark_latency:
        summary["efficiency"] = benchmark_latency_vram(model)
        print(f"  [Efficiency] {summary['efficiency']}", flush=True)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved operating-point audit to {out_path}", flush=True)


if __name__ == "__main__":
    main()
