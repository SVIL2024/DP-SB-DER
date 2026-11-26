"""Protocol 2 (drift@5) open-world definition drift evaluation for DP-SB + DER.

Supports both benchmarks in Table II of the paper:
  1. --mode table_d   : Exact published 5-trial anomaly subsets in Appendix Table D
                        of LaGoVAD (ICLR 2026) on XD-Violence and MSAD (Table II(a)).
  2. --mode paired_5ds: Five-dataset balanced 50% subsets (XD, UCF, MSAD, LAD, PreVAD;
                        2,120 paired abnormal clips) auditing 5-DS AUC/AP, normal-frame-free
                        Abn-Only AUC, and same-video paired exclusion score drop Delta_flip
                        with two-sided Wilcoxon signed-rank tests (Table II(b)).
"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import src.full_length_eval as fle

MAX_LEN = 512

# Exact subsets published in Appendix Table D of LaGoVAD (Liu et al., ICLR 2026)
TABLE_D_SUBSETS = {
    "xd": [
        ["Fighting", "Shooting", "Riot"],
        ["Abuse", "Car accident", "Explosion"],
        ["Fighting", "Abuse", "Explosion"],
        ["Shooting", "Car accident", "Riot"],
        ["Riot", "Abuse", "Car accident"],
    ],
    "msad": [
        ["Assault", "Fighting", "Robbery", "Shooting", "Vandalism", "Explosion"],
        ["People_falling", "Traffic_accident", "Fire", "Object_falling", "Water_incident", "Explosion"],
        ["Assault", "People_falling", "Traffic_accident", "Vandalism", "Fire", "Object_falling"],
        ["Fighting", "Robbery", "Shooting", "Traffic_accident", "Water_incident", "Fire"],
        ["People_falling", "Robbery", "Object_falling", "Water_incident", "Explosion", "Assault"],
    ],
}

XD_CODE_MAP = {
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "Car accident",
    "G": "Explosion",
}


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def parse_xd_all_classes(stem: str, primary_cls: str) -> set[str]:
    if primary_cls == "Normal" or "_label_A" in stem:
        return set()
    m = re.search(r"_label_([A-Z0-9\-]+)$", stem)
    if not m:
        return {primary_cls}
    codes = m.group(1).split("-")
    out = {XD_CODE_MAP[c] for c in codes if c in XD_CODE_MAP}
    return out if out else {primary_cls}


def make_balanced_5_subsets(classes: list[str], seed: int = 2024) -> list[list[str]]:
    """Generate 5 balanced ~50% active anomaly subsets so each class appears in both active and excluded trials."""
    abn_cls = [c for c in classes if c != "Normal"]
    rng = random.Random(seed)
    k = max(1, len(abn_cls) // 2)
    subsets = []
    for t in range(5):
        pool = list(abn_cls)
        rng.shuffle(pool)
        subsets.append(sorted(pool[:k]))
    return subsets


def preload_multiclass_dataset(ds_name: str, data_root: Path):
    dataset, verbalizer, classes = fle.build_dataset(ds_name, str(data_root))
    verbalizer.set_dataset(dataset.abbr)
    items = []
    for i in range(len(dataset)):
        data = dataset.get_full_item(i)
        cls_name = dataset.gts[i]["class_name"]
        T = int(data["v_feat_l"])
        raw_gt = np.zeros(T, dtype=np.uint8)
        for span in data["temp_span"]:
            if span[0] != span[1]:
                raw_gt[int(span[0]):int(span[1])] = 1
        stem = Path(data["video_path"]).stem
        all_cls = parse_xd_all_classes(stem, cls_name) if ds_name == "xd" else ({cls_name} if cls_name != "Normal" else set())
        items.append({
            "name": stem,
            "v_feat": data["v_feat"],
            "T": T,
            "class_name": cls_name,
            "all_classes": all_cls,
            "raw_gt": raw_gt,
        })
    return verbalizer, classes, items


def load_model(cfg_path: Path, ckpt_path: Path, der: bool, rho0: float, branch_g_only: bool):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    mc = cfg["model"]["init_args"]["model_config"]["init_args"]
    mc["verbalizer_type"] = None
    if mc.get("use_semantic_beamformer", False):
        if branch_g_only:
            mc["beam_entropy_router"] = False
            mc["beam_global_scale_override"] = 1.0
            mc["beam_local_scale_override"] = 0.0
        elif der:
            mc["beam_entropy_router"] = True
            mc["beam_router_temp"] = 1.0
            mc["beam_center_local"] = True
            mc["beam_disable_cfar_gate"] = False
            mc["beam_global_scale_override"] = 1.0
            mc["beam_local_scale_override"] = float(rho0)
    model = fle.build_model(cfg, str(ckpt_path))
    model.eval()
    return model


@torch.no_grad()
def evaluate_drift5(model, verbalizer, items, subsets: list[list[str]]) -> dict:
    device = fle.device
    N = len(items)
    offsets = np.zeros(N + 1, dtype=np.int64)
    for i, it in enumerate(items):
        offsets[i + 1] = offsets[i] + it["T"]
    total_T = int(offsets[-1])

    # Cache visual temporal encodings once across all 5 definition trials
    vis_chunks = []
    for it in items:
        chs = []
        for j in range(math.ceil(it["T"] / MAX_LEN)):
            v = it["v_feat"][j * MAX_LEN:(j + 1) * MAX_LEN].unsqueeze(0).to(device)
            L = torch.tensor([v.shape[1]], device=device)
            vf, _ = model._temporal_encoding(v, L)
            chs.append(vf.cpu())
        vis_chunks.append(chs)

    trial_aucs, trial_aps = [], []
    abn_only_aucs = []
    prob_matrix = np.zeros((5, total_T), dtype=np.float32)
    active_flags = np.zeros((5, N), dtype=bool)

    for t_idx, active_cls in enumerate(subsets):
        active_set = set(active_cls)
        fle.seed_everything(7 + t_idx)
        prompts = verbalizer(["Normal"] + list(active_cls))
        ti = model.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77 - model.clip_text_model.num_soft_prompts,
            return_tensors="pt",
        ).to(device)
        tf = model.clip_text_model(input_ids=ti.input_ids, attention_mask=ti.attention_mask).pooler_output

        gt_t = np.zeros(total_T, dtype=np.uint8)
        abn_mask_t = np.zeros(total_T, dtype=bool)
        for i, it in enumerate(items):
            s, e = int(offsets[i]), int(offsets[i + 1])
            T = it["T"]
            is_abn = len(it["all_classes"]) > 0
            is_act = is_abn and bool(it["all_classes"] & active_set)
            active_flags[t_idx, i] = is_act
            if is_act:
                gt_t[s:e] = it["raw_gt"]
            if is_abn:
                abn_mask_t[s:e] = (it["raw_gt"] == 1)

            chunks_l = []
            for vf_c in vis_chunks[i]:
                vf = vf_c.to(device)
                L = torch.tensor([vf.shape[1]], device=device)
                fl, _ = model._fuse_and_head(vf, tf, L, update_prior=False)
                chunks_l.append(fl[0].float().cpu().numpy())
            prob_matrix[t_idx, s:e] = sigmoid_np(np.concatenate(chunks_l)[:T])

        trial_aucs.append(float(roc_auc_score(gt_t, prob_matrix[t_idx]) * 100.0))
        trial_aps.append(float(average_precision_score(gt_t, prob_matrix[t_idx]) * 100.0))
        gt_abn = gt_t[abn_mask_t]
        pr_abn = prob_matrix[t_idx, abn_mask_t]
        if gt_abn.min() != gt_abn.max():
            abn_only_aucs.append(float(roc_auc_score(gt_abn, pr_abn) * 100.0))

    # Same-video paired definition-exclusion flip (Delta_flip = mean_active - mean_excluded on GT event frames)
    paired_flips = []
    for i, it in enumerate(items):
        if len(it["all_classes"]) == 0 or it["raw_gt"].sum() == 0:
            continue
        inc_trials = [t for t in range(5) if active_flags[t, i]]
        exc_trials = [t for t in range(5) if not active_flags[t, i]]
        if not inc_trials or not exc_trials:
            continue
        s, e = int(offsets[i]), int(offsets[i + 1])
        ev = it["raw_gt"] == 1
        inc_score = float(np.mean([prob_matrix[t, s:e][ev].mean() for t in inc_trials]))
        exc_score = float(np.mean([prob_matrix[t, s:e][ev].mean() for t in exc_trials]))
        paired_flips.append(inc_score - exc_score)

    paired_arr = np.array(paired_flips, dtype=np.float64)
    pval = float(stats.wilcoxon(paired_arr).pvalue) if len(paired_arr) > 0 and np.any(paired_arr != 0) else 1.0
    return {
        "AUC": round(float(np.mean(trial_aucs)), 2),
        "AP": round(float(np.mean(trial_aps)), 2),
        "Abn_Only_AUC": round(float(np.mean(abn_only_aucs)), 2) if abn_only_aucs else None,
        "Paired_N": int(len(paired_arr)),
        "Paired_Delta_Flip": round(float(paired_arr.mean()), 4) if len(paired_arr) > 0 else 0.0,
        "Wilcoxon_p_vs_0": pval,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Protocol 2 (drift@5) open-world definition drift evaluation.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/PreVAD")
    parser.add_argument("--mode", choices=["table_d", "paired_5ds"], default="table_d")
    parser.add_argument("--der", action="store_true", help="Enable parameter-free DER routing.")
    parser.add_argument("--rho0", type=float, default=5.0)
    parser.add_argument("--branch-g-only", action="store_true", help="Isolate semantic Branch G (w_L = 0).")
    parser.add_argument("--out-json", type=str, default="outputs/protocol2_drift5.json")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    model = load_model(Path(args.config), Path(args.ckpt), args.der, args.rho0, args.branch_g_only)

    datasets = ["xd", "msad"] if args.mode == "table_d" else ["xd", "ucf", "msad", "lad", "prevad"]
    results = {"mode": args.mode, "per_dataset": {}}
    for ds in datasets:
        verbalizer, classes, items = preload_multiclass_dataset(ds, data_root)
        subsets = TABLE_D_SUBSETS[ds] if args.mode == "table_d" else make_balanced_5_subsets(classes, seed=2024)
        res = evaluate_drift5(model, verbalizer, items, subsets)
        results["per_dataset"][ds] = res
        print(
            f"  [{args.mode}] {ds.upper():6s} | AUC={res['AUC']:.2f}% | AP={res['AP']:.2f}% | "
            f"Abn-Only AUC={res['Abn_Only_AUC']} | Delta_flip={res['Paired_Delta_Flip']:+.4f} (N={res['Paired_N']})",
            flush=True,
        )

    results["Mean_AUC"] = round(float(np.mean([results["per_dataset"][d]["AUC"] for d in datasets])), 2)
    results["Mean_AP"] = round(float(np.mean([results["per_dataset"][d]["AP"] for d in datasets])), 2)
    print(f"\n==> {args.mode} Mean AUC={results['Mean_AUC']:.2f}% | Mean AP={results['Mean_AP']:.2f}%", flush=True)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved Protocol 2 results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
