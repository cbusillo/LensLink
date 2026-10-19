"""Object masks from SAM 2's video mode run on every --every-th frame
only, forward only, each result applied --every frames late and held
until the next: what a CPU background thread at ~130-150 ms a step can
deliver live. Prompted by the setup clicks on frame 0.

    uv run python sam_sparse.py TAKE NAME:X,Y[+X,Y...] [...] [--every 8]

Writes <take>/samsp-<NAME>.bin (960x540, original orientation).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from lab import CACHE, AlphaWriter, device, frame_count


sys.path.insert(0, str(CACHE / "src" / "sam2"))
from sam2.build_sam import build_sam2_video_predictor  # type: ignore  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("take", type=Path)
    p.add_argument("objects", nargs="+")
    p.add_argument("--every", type=int, default=8)
    a = p.parse_args()
    jpegs = a.take / "jpeg540"
    if not jpegs.exists():
        sys.exit("run sam_tracks.py once first: it writes jpeg540/")
    count = frame_count(a.take)
    sparse = a.take / f"jpeg540-every{a.every}"
    sparse.mkdir(exist_ok=True)
    picks = list(range(0, count, a.every))
    for k, index in enumerate(picks):
        link = sparse / f"{k:05d}.jpg"
        if not link.exists():
            link.symlink_to(jpegs / f"{index:05d}.jpg")

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(CACHE / "weights" / "sam2.1_hiera_small.pt"), device=device())
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(sparse), offload_video_to_cpu=True)
        for spec in a.objects:
            name, points = spec.split(":")
            xy = np.array([((3839 - int(x)) / 4, (2159 - int(y)) / 4)
                           for x, y in (q.split(",") for q in points.split("+"))], dtype=np.float32)
            predictor.reset_state(state)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=xy,
                                            labels=np.ones(len(xy), dtype=np.int32))
            results = {}
            for k, _, logits in predictor.propagate_in_video(state, start_frame_idx=0):
                results[k] = (logits[0, 0] > 0).cpu().numpy()
            writer = AlphaWriter(a.take / f"samsp-{name}.bin")
            for index in range(count):
                # Frame 0's setup result is waited for; later steps land
                # one interval after the frame they were computed from.
                k = 0 if index < a.every else (index - a.every) // a.every
                writer.write(np.where(results.get(k, results[0]), 255, 0).astype(np.uint8))
            writer.close()
            print(f"samsp-{name}: {writer.count} frames from {len(results)} sparse steps")


if __name__ == "__main__":
    main()
