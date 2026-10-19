"""1:1 crops of a hair edge for each method, stacked, at one frame.

    uv run python crops.py TAKE FRAME METHOD [METHOD ...]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lab import composite, frames, read_alpha

take, index, methods = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
rgb = next(f for i, f in frames(take, index + 1) if i == index)[::-1, ::-1]
h, w = rgb.shape[:2]
ref = np.asarray(read_alpha(take / f"{methods[-1]}.bin")[index])[::-1, ::-1]
ref = np.array(Image.fromarray(ref).resize((w, h)))
rows = np.where(ref.max(axis=1) > 127)[0]
top = rows[0]
# Both sides of the head, a third of the way down from the top of the hair.
band = ref[top + 300]
cols = np.where(band > 127)[0]
tiles = []
for side, x in (("left", cols[0]), ("right", cols[-1])):
    x0 = int(np.clip(x - 240, 0, w - 480)); y0 = int(np.clip(top + 60, 0, h - 480))
    for m in methods:
        comp = composite(rgb, np.asarray(read_alpha(take / f"{m}.bin")[index])[::-1, ::-1])
        tile = Image.fromarray(comp[y0:y0 + 480, x0:x0 + 480])
        ImageDraw.Draw(tile).text((4, 4), f"{m} {side}", fill="white")
        tiles.append(tile)
cols_n = len(methods)
page = Image.new("RGB", (480 * cols_n, 960), "white")
for i, t in enumerate(tiles):
    page.paste(t, (480 * (i % cols_n), 480 * (i // cols_n)))
out = take.parent.parent / "sheets" / f"{take.name}-f{index:04d}-hair.png"
page.save(out)
print(out)
