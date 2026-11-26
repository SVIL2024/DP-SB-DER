"""Deterministic from-scratch training runner for DP-SB + DER.

Trains strictly from scratch with fixed seed (default 2024), saves weights-only
checkpoints (frozen CLIP weights stripped), exports an evaluation-compatible
config.yaml, and verifies strict checkpoint reload integrity.
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import Callback, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from datasets.datamodules import PreVADDataModule
from models.LaGoVAD.configs import LaGoVADModelConfig, LaGoVADTrainingConfig
from models.LaGoVAD.lagovad import LaGoVADLightModel

torch.set_float32_matmul_precision("high")


class PeriodicWeightsOnlySaver(Callback):
    """Save weights-only checkpoints (frozen CLIP stripped) at specified 1-indexed epochs."""

    def __init__(self, save_dir: Path, save_epochs: list[int]):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_epochs = {int(e) for e in save_epochs}
        self.saved_paths: dict[int, str] = {}

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch_1idx = trainer.current_epoch + 1
        if epoch_1idx not in self.save_epochs:
            return
        state_dict = {k: v.detach().cpu() for k, v in pl_module.state_dict().items()}
        payload = {
            "state_dict": state_dict,
            "epoch": epoch_1idx,
            "global_step": trainer.global_step,
        }
        pl_module.on_save_checkpoint(payload)
        ckpt_path = self.save_dir / f"epoch_{epoch_1idx:02d}.ckpt"
        torch.save(payload, ckpt_path)
        size_mb = ckpt_path.stat().st_size / (1024 ** 2)
        self.saved_paths[epoch_1idx] = str(ckpt_path)
        print(f"[Saver] Saved {ckpt_path.name} ({size_mb:.2f} MB, step={trainer.global_step})", flush=True)


class EpochTimerLogger(Callback):
    """Record per-epoch wall-clock time, peak GPU memory, and training metrics."""

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = Path(log_path)
        self.t0 = None
        self.epoch_records = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        dt = time.time() - self.t0
        ep = trainer.current_epoch + 1
        metrics = {
            k: float(v.item()) if torch.is_tensor(v) else float(v)
            for k, v in trainer.callback_metrics.items()
        }
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        rec = {
            "epoch": ep,
            "seconds": round(dt, 2),
            "global_step": trainer.global_step,
            "peak_mem_gb": round(peak_gb, 3),
            "metrics": metrics,
        }
        self.epoch_records.append(rec)
        self.log_path.write_text(json.dumps(self.epoch_records, indent=2), encoding="utf-8")
        loss_s = metrics.get("loss", float("nan"))
        extra = ""
        if "beam_gamma" in metrics:
            extra = (
                f" | gamma={metrics['beam_gamma']:.3f}"
                f" alpha_n={metrics.get('beam_alpha_norm', 0.0):.3f}"
                f" alpha_a={metrics.get('beam_alpha_ano', 0.0):.3f}"
                f" |db|={metrics.get('beam_bin_delta_abs', 0.0):.3f}"
            )
        print(f"[Epoch {ep:02d}] {dt:.1f}s | step={trainer.global_step} | loss={loss_s:.4f} | peak={peak_gb:.2f}GB{extra}", flush=True)


def verify_checkpoint_reload(model_cfg: LaGoVADModelConfig, train_cfg: LaGoVADTrainingConfig, ckpt_path: Path):
    """Verify strict reload: zero unexpected keys, missing keys strictly limited to frozen CLIP."""
    fresh = LaGoVADLightModel(model_cfg, train_cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in {ckpt_path}: {unexpected}")
    bad_missing = [
        k for k in missing
        if not (k.startswith("clip_text_model.model.") or k.startswith("clip_text_model.text_projection."))
    ]
    if bad_missing:
        raise RuntimeError(f"Unexpected missing keys in {ckpt_path}: {bad_missing}")
    print(f"[Verify] Reload OK for {ckpt_path.name}: {len(sd)} loaded, {len(missing)} frozen-CLIP missing, 0 unexpected.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DP-SB or LaGoVAD baseline from scratch on PreVAD.")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config (e.g. configs/dpsb_plugin_train.yaml).")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for checkpoints and logs.")
    parser.add_argument("--data-root", type=str, default=None, help="Override PreVAD data root directory.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--limit-train-batches", type=float, default=None, help="Limit batches per epoch (for smoke testing).")
    parser.add_argument("--no-preload", action="store_true", help="Disable RAM preloading of PreVAD features.")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed_everything", 2024))
    L.seed_everything(seed, workers=True)

    model_cfg_args = dict(cfg["model"]["init_args"]["model_config"]["init_args"])
    train_cfg_args = dict(cfg["model"]["init_args"]["training_config"]["init_args"])
    data_args = dict(cfg["data"]["init_args"])

    assert model_cfg_args.get("pretrained_weight_path") is None, "Training must start from scratch (pretrained_weight_path=None)."
    assert train_cfg_args.get("pretrain_weights") is None, "Training must start from scratch (pretrain_weights=None)."

    if args.data_root is not None:
        data_args["train_data_root"] = args.data_root
        data_args["val_data_root"] = args.data_root
    if args.no_preload:
        data_args["preload_features"] = False

    max_epochs = int(args.max_epochs if args.max_epochs is not None else cfg["trainer"]["max_epochs"])
    save_epochs = cfg.get("save_epochs", list(range(1, max_epochs + 1)))
    if args.max_epochs is not None and max_epochs not in save_epochs:
        save_epochs = [max_epochs]

    eval_cfg = {
        "seed_everything": seed,
        "model": {
            "class_path": "models.LaGoVAD.LaGoVADLightModel",
            "init_args": {
                "model_config": {
                    "class_path": "models.LaGoVAD.LaGoVADModelConfig",
                    "init_args": model_cfg_args,
                },
                "training_config": {
                    "class_path": "models.LaGoVAD.LaGoVADTrainingConfig",
                    "init_args": train_cfg_args,
                },
            },
        },
        "data": {
            "class_path": "datasets.PreVADDataModule",
            "init_args": data_args,
        },
    }
    (out_dir / "config.yaml").write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8")

    model_cfg = LaGoVADModelConfig(**model_cfg_args)
    train_cfg = LaGoVADTrainingConfig(**train_cfg_args)
    model = LaGoVADLightModel(model_cfg, train_cfg)

    if cfg["trainer"].get("gradient_checkpointing", False):
        if hasattr(model.temporal_encoder, "gradient_checkpointing"):
            model.temporal_encoder.gradient_checkpointing = True
            model.temporal_encoder._gradient_checkpointing_func = functools.partial(
                torch.utils.checkpoint.checkpoint, use_reentrant=False
            )
        if hasattr(model.fusion, "gradient_checkpointing"):
            model.fusion.gradient_checkpointing = True
        if hasattr(model.clip_text_model, "model") and hasattr(model.clip_text_model.model, "gradient_checkpointing_enable"):
            model.clip_text_model.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        print("[Memory] Gradient checkpointing enabled (FP32, use_reentrant=False).", flush=True)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {n_total:,} | Trainable params: {n_train:,}", flush=True)

    dm = PreVADDataModule(**data_args)
    saver = PeriodicWeightsOnlySaver(ckpt_dir, save_epochs)
    timer_cb = EpochTimerLogger(out_dir / "epoch_log.json")
    logger = CSVLogger(save_dir=str(out_dir), name="csv_logs", version="run")

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg["trainer"].get("precision", 32),
        max_epochs=max_epochs,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=int(cfg["trainer"].get("log_every_n_steps", 20)),
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval="epoch"), saver, timer_cb],
    )

    t_start = time.time()
    trainer.fit(model, datamodule=dm)
    total_sec = time.time() - t_start
    print(f"[Done] Finished {max_epochs} epochs in {total_sec / 60.0:.2f} min.", flush=True)

    if saver.saved_paths:
        last_ep = max(saver.saved_paths.keys())
        verify_checkpoint_reload(model_cfg, train_cfg, Path(saver.saved_paths[last_ep]))

    meta = {
        "config": str(args.config),
        "seed": seed,
        "max_epochs": max_epochs,
        "total_seconds": round(total_sec, 2),
        "total_params": n_total,
        "trainable_params": n_train,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
