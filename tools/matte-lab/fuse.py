"""Edge fusion: a base matte decides inside and outside; a detail matte
supplies the edge band.

    uv run python fuse.py TAKE BASE DETAIL [--band 16] [--keep sam-mic] [--tag NAME]

Writes <take>/fused-<BASE>-<tag>.bin at 1920x1080. In a band of --band
pixels around the base's edge (where the base is neither clearly 0 nor
clearly 1, grown by --band), the detail matte is used; elsewhere the
base. Objects named by --keep (the mic) keep the base everywhere, since a
person-only detail model drops them.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from lab import AlphaWriter, read_alpha

W, H = 1920, 1080


def fit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.shape != (H, W):
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    return a.astype(np.float32) / 255


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("take", type=Path)
    p.add_argument("base")
    p.add_argument("detail")
    p.add_argument("--band", type=int, default=16)
    p.add_argument("--keep", default="")
    p.add_argument("--tag", default="fused")
    args = p.parse_args()

    base = read_alpha(args.take / f"{args.base}.bin")
    detail = read_alpha(args.take / f"{args.detail}.bin")
    keeps = [read_alpha(args.take / f"{k}.bin") for k in args.keep.split(",") if k]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.band + 1, 2 * args.band + 1))
    writer = AlphaWriter(args.take / f"fused-{args.base}-{args.tag}.bin")
    for i in range(min(len(base), len(detail))):
        b = fit(base[i])
        d = fit(detail[i])
        edge = ((b > 0.02) & (b < 0.98)).astype(np.uint8)
        band = cv2.dilate(edge, kernel) > 0
        for k in keeps:
            band &= cv2.dilate((fit(k[i]) > 0.5).astype(np.uint8), kernel) == 0
        out = np.where(band, d, b)
        writer.write((np.clip(out, 0, 1) * 255).astype(np.uint8))
    writer.close()
    print(f"fused-{args.base}-{args.tag}: {writer.count} frames")


if __name__ == "__main__":
    main()
