"""Object tracks for a take with SAM 2: one click point per object.

    uv run python sam_tracks.py TAKE NAME:FRAME:X,Y[+X,Y...] [...]

X and Y are in upright 3840x2160 coordinates (as the review sheets show
the picture); FRAME is where the points are placed. Several points per
object cover thin or partly hidden parts (the boom, the chair base). Each object's mask is
tracked forward and backward through the whole take and written as
<take>/sam-<NAME>.bin (lab.py alpha format, original orientation,
960x540). This is a lab stand-in for automatic object detection: it
answers whether explicit object masks fix the keep/drop rules, not how
to find the objects live.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lab import CACHE, AlphaWriter, device, frames

sys.path.insert(0, str(CACHE / "src" / "sam2"))
from sam2.build_sam import build_sam2_video_predictor  # type: ignore  # noqa: E402

W, H = 960, 540


def main() -> None:
    take = Path(sys.argv[1])
    prompts = []
    for spec in sys.argv[2:]:
        name, frame, points = spec.split(":")
        # Upright 4K -> original orientation at 960x540.
        xy = [((3839 - int(x)) / 4, (2159 - int(y)) / 4)
              for x, y in (p.split(",") for p in points.split("+"))]
        prompts.append((name, int(frame), xy))

    jpegs = take / "jpeg540"
    if not jpegs.exists():
        jpegs.mkdir()
        for index, rgb in frames(take):
            Image.fromarray(rgb).resize((W, H), Image.BILINEAR).save(jpegs / f"{index:05d}.jpg", quality=92)

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(CACHE / "weights" / "sam2.1_hiera_small.pt"),
        device=device(),
    )
    names = [p[0] for p in prompts]
    count = len(list(jpegs.glob("*.jpg")))
    masks = {n: np.zeros((count, H, W), dtype=np.uint8) for n in names}
    # One pass per object: on MPS, propagating objects prompted on
    # different frames in one state trips an MPS matmul dtype assertion.
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(jpegs), offload_video_to_cpu=True)
        for name, frame, xy in prompts:
            predictor.reset_state(state)
            predictor.add_new_points_or_box(
                state, frame_idx=frame, obj_id=1,
                points=np.array(xy, dtype=np.float32),
                labels=np.ones(len(xy), dtype=np.int32))
            for reverse in (False, True):
                for index, _, logits in predictor.propagate_in_video(
                        state, start_frame_idx=frame, reverse=reverse):
                    masks[name][index] = np.where((logits[0, 0] > 0).cpu().numpy(), 255, 0)
    for name in names:
        writer = AlphaWriter(take / f"sam-{name}.bin")
        for m in masks[name]:
            writer.write(m)
        writer.close()
        covered = (masks[name] > 0).mean(axis=(1, 2))
        print(f"{take.name} sam-{name}: {count} frames, present in "
              f"{(covered > 0.0005).sum()} of them")


if __name__ == "__main__":
    main()
