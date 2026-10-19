"""Applies the keep/drop rules to any matte, every frame, with no memory
correction: alpha = max(matte * allowed, forced), where allowed is the
person mask (grown by --grow pixels at 1920 wide) plus the objects plus
the chair while seated, and forced is the objects plus the chair while
seated.

    uv run python gate.py TAKE MATTE PERSON --objects sam-mic --chair sam-chair --seated SPEC [--grow 48]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from fuse import fit
from lab import AlphaWriter, read_alpha, seated_frames

p = argparse.ArgumentParser()
p.add_argument("take", type=Path)
p.add_argument("matte")
p.add_argument("person")
p.add_argument("--objects", default="")
p.add_argument("--chair", default="")
p.add_argument("--seated", default="none")
p.add_argument("--grow", type=int, default=48)
p.add_argument("--tag", default="gated")
a = p.parse_args()
matte = read_alpha(a.take / f"{a.matte}.bin")
person = read_alpha(a.take / f"{a.person}.bin")
objects = [read_alpha(a.take / f"{o}.bin") for o in a.objects.split(",") if o]
chair = read_alpha(a.take / f"{a.chair}.bin") if a.chair else None
sitting = seated_frames(a.seated, len(matte))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.grow + 1, 2 * a.grow + 1))
writer = AlphaWriter(a.take / f"{a.matte}-{a.tag}.bin")
for i in range(len(matte)):
    forced = np.zeros((1080, 1920), dtype=bool)
    for o in objects:
        forced |= fit(o[i]) > 0.5
    if chair is not None and sitting[i]:
        forced |= fit(chair[i]) > 0.5
    allowed = cv2.dilate(((fit(person[i]) > 0.5) | forced).astype(np.uint8), kernel) > 0
    out = np.maximum(fit(matte[i]) * allowed, forced)
    writer.write((out * 255).astype(np.uint8))
writer.close()
print(f"{a.matte}-{a.tag}: {writer.count} frames")
