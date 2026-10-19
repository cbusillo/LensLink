"""Guided upsampling: the matte, stretched to 1920x1080, is snapped to
edges in the full-resolution color frame with a color guided filter (He
et al.), the job MPSImageGuidedFilter does on the GPU live.

    uv run python guided.py TAKE MATTE [--radius 8] [--eps 1e-4]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from fuse import fit
from lab import AlphaWriter, frames, read_alpha

p = argparse.ArgumentParser()
p.add_argument("take", type=Path)
p.add_argument("matte")
p.add_argument("--radius", type=int, default=8)
p.add_argument("--eps", type=float, default=1e-4)
a = p.parse_args()
matte = read_alpha(a.take / f"{a.matte}.bin")
tag = f"gf{a.radius}"
writer = AlphaWriter(a.take / f"{a.matte}-{tag}.bin")
for i, rgb in frames(a.take, len(matte)):
    guide = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    m = fit(matte[i])
    out = cv2.ximgproc.guidedFilter(guide, m, a.radius, a.eps)
    writer.write((np.clip(out, 0, 1) * 255).astype(np.uint8))
writer.close()
print(f"{a.matte}-{tag}: {writer.count} frames")
