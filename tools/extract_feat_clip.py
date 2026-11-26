"""Extract frozen CLIP ViT-B/16 snippet features from raw videos (following LaGoVAD).

Example:
    python tools/extract_feat_clip.py \
        --video-dir data/PreVAD/other_datasets/ucf_videos \
        --output-dir data/PreVAD/other_datasets/ucf_features \
        --interval 8 \
        --device cuda:0
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import decord
import numpy as np
import torch
from rich.progress import Progress

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import src.models.clip as clip
from src.utils.video_loader import VideoLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CLIP ViT-B/16 frame features for VAD benchmarks.")
    parser.add_argument("--video-dir", type=str, nargs="+", required=True, help="One or more input video directories.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save extracted .npy feature arrays.")
    parser.add_argument("--ext", type=str, default="mp4", help="Video file extension (default: mp4).")
    parser.add_argument("--interval", type=int, default=8, help="Frame sampling stride (default: 8).")
    parser.add_argument("--batch-size", type=int, default=256, help="Frames per forward batch (default: 256).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (default: cuda:0).")
    parser.add_argument("--workers", type=int, default=2, help="Background threads for writing .npy files.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vid_paths = []
    for vd in args.video_dir:
        vid_paths.extend(sorted(Path(vd).rglob(f"*.{args.ext}")))
    print(f"Found {len(vid_paths)} videos.", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = clip.load("ViT-B/16", jit=True, device=device)
    model.eval()

    save_executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = []
    try:
        with Progress() as progress:
            vid_task = progress.add_task("Extracting CLIP ViT-B/16", total=len(vid_paths))
            for vid_path in vid_paths:
                output_path = output_dir / f"{vid_path.stem}.npy"
                if output_path.exists():
                    progress.update(vid_task, advance=1)
                    continue

                loader = VideoLoader(
                    str(vid_path),
                    batch_size=args.batch_size,
                    interval=args.interval,
                    augmentation="no_center_crop",
                )
                feats = []
                try:
                    for frame_batch in loader:
                        with torch.no_grad():
                            frame_batch = frame_batch.to(device).squeeze(0)
                            feat = model.encode_image(frame_batch)
                            feats.append(feat.cpu().numpy())
                except decord.DECORDError as exc:
                    print(f"[Skip] Failed to decode {vid_path.name}: {exc}", flush=True)
                    progress.update(vid_task, advance=1)
                    continue

                if not feats:
                    print(f"[Skip] Empty frames for {vid_path.name}", flush=True)
                    progress.update(vid_task, advance=1)
                    continue

                arr = np.concatenate(feats, axis=0)
                futures.append(save_executor.submit(np.save, str(output_path), arr))
                progress.update(vid_task, advance=1)
    finally:
        save_executor.shutdown(wait=True)
        wait(futures)


if __name__ == "__main__":
    main()
