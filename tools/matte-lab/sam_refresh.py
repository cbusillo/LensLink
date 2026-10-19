"""Object masks the way a live session can afford them: SAM 2 in image
mode (no video memory), re-run every --every frames from a box around the
object's last mask, the result held until the next refresh. Frame 0 uses
the setup clicks. Each refresh is applied --every frames late, as a CPU
background thread taking ~130 ms would deliver it.

    uv run python sam_refresh.py TAKE NAME:X,Y[+X,Y...] [...] [--every 8]

X and Y are upright 3840x2160 setup clicks on frame 0 (as for
sam_tracks.py). Writes <take>/samref-<NAME>.bin (960x540, original
orientation).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lab import CACHE, AlphaWriter, device, frames

sys.path.insert(0, str(CACHE / "src" / "sam2"))
from sam2.build_sam import build_sam2  # type: ignore  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore  # noqa: E402

W, H = 960, 540


def box_of(mask: np.ndarray, pad: float) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = (x1 - x0) * pad + 4, (y1 - y0) * pad + 4
    return np.array([max(0.0, float(x0 - px)), max(0.0, float(y0 - py)),
                     min(W - 1.0, float(x1 + px)), min(H - 1.0, float(y1 + py))], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("take", type=Path)
    p.add_argument("objects", nargs="+")
    p.add_argument("--every", type=int, default=8)
    p.add_argument("--pad", type=float, default=0.15)
    a = p.parse_args()
    clicks = {}
    for spec in a.objects:
        name, points = spec.split(":")
        clicks[name] = np.array([((3839 - int(x)) / 4, (2159 - int(y)) / 4)
                                 for x, y in (q.split(",") for q in points.split("+"))], dtype=np.float32)

    model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml",
                       str(CACHE / "weights" / "sam2.1_hiera_small.pt"), device=device())
    predictor = SAM2ImagePredictor(model)
    shown: dict[str, np.ndarray] = {n: np.zeros((H, W), dtype=bool) for n in clicks}  # what the frame path uses
    pending: dict[str, np.ndarray | None] = {n: None for n in clicks}  # refresh in flight
    writers = {n: AlphaWriter(a.take / f"samref-{n}.bin") for n in clicks}
    with torch.inference_mode():
        for index, rgb in frames(a.take):
            if index % a.every == 0:
                small = np.asarray(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
                predictor.set_image(small)
                for name in clicks:
                    if index == 0:
                        masks, _, _ = predictor.predict(point_coords=clicks[name],
                                                        point_labels=np.ones(len(clicks[name])),
                                                        multimask_output=False)
                        shown[name] = masks[0] > 0  # setup is waited for
                    else:
                        # The last result lands now; start the next one.
                        if pending[name] is not None:
                            shown[name] = pending[name]
                        box = box_of(shown[name], a.pad)
                        if box is None:
                            pending[name] = None
                            continue
                        masks, _, _ = predictor.predict(box=box, multimask_output=False)
                        pending[name] = masks[0] > 0
            for name in clicks:
                writers[name].write(np.where(shown[name], 255, 0).astype(np.uint8))
    for name, w in writers.items():
        w.close()
        print(f"samref-{name}: {w.count} frames")


if __name__ == "__main__":
    main()
