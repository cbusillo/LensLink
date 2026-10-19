"""Solid core: alpha forced to 1 more than --inset pixels (at 1920 wide)
inside the matte's outline, so only an outer band stays soft and hair is
never see-through inside.

    uv run python core.py TAKE MATTE [--inset 10]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from fuse import fit
from lab import AlphaWriter, read_alpha

p = argparse.ArgumentParser()
p.add_argument("take", type=Path)
p.add_argument("matte")
p.add_argument("--inset", type=int, default=10)
a = p.parse_args()
matte = read_alpha(a.take / f"{a.matte}.bin")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.inset + 1, 2 * a.inset + 1))
writer = AlphaWriter(a.take / f"{a.matte}-core{a.inset}.bin")
for i in range(len(matte)):
    m = fit(matte[i])
    inside = cv2.erode((m > 0.5).astype(np.uint8), kernel) > 0
    # Feather the core's own edge over the inset so the step never shows.
    core = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), a.inset / 2)
    writer.write((np.maximum(m, core) * 255).clip(0, 255).astype(np.uint8))
writer.close()
print(f"{a.matte}-core{a.inset}: {writer.count} frames")
