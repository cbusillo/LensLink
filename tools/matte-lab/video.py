"""Side-by-side review video of methods over a plain backdrop.

    uv run python video.py TAKE OUT.mp4 METHOD [METHOD ...]

A METHOD ending in "+fg" is composited with blur-fusion foreground
colors instead of the camera's pixels.
"""
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from blurfusion import estimate
from lab import frames, read_alpha

take, out, methods = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
alphas = [read_alpha(take / f"{m.removesuffix('+fg')}.bin") for m in methods]
tw, th = 960, 540
backdrop = np.zeros((th, tw, 3), dtype=np.float32)
backdrop[:] = np.linspace([40, 70, 110], [15, 25, 45], th)[:, None, :]
container = av.open(str(out), "w")
stream = container.add_stream("h264", rate=30)
stream.width, stream.height = tw * len(methods), th + 24
stream.pix_fmt = "yuv420p"
stream.options = {"crf": "20"}
for index, rgb in frames(take, min(len(a) for a in alphas)):
    small = np.asarray(Image.fromarray(rgb[::-1, ::-1]).resize((tw, th), Image.BILINEAR), dtype=np.float32)
    canvas = Image.new("RGB", (tw * len(methods), th + 24), "black")
    for k, a in enumerate(alphas):
        al = np.asarray(Image.fromarray(np.asarray(a[index])[::-1, ::-1]).resize((tw, th), Image.BILINEAR), dtype=np.float32)[..., None] / 255
        color = small
        if methods[k].endswith("+fg"):
            color = estimate(small / 255, al[..., 0], tw / 1920) * 255
        canvas.paste(Image.fromarray((color * al + backdrop * (1 - al)).astype(np.uint8)), (tw * k, 24))
        ImageDraw.Draw(canvas).text((tw * k + 8, 6), f"{methods[k]}   frame {index}", fill="white")
    for packet in stream.encode(av.VideoFrame.from_image(canvas)):
        container.mux(packet)
for packet in stream.encode():
    container.mux(packet)
container.close()
print(out)
