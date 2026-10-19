"""Foreground color estimation by blur fusion (Forte, Pitie and
Photoroom's fast-foreground-estimation), so edge pixels carry the
person's color instead of the old background's. Compares composites
with and without it at one frame, 1:1 hair crops.

    uv run python blurfusion.py TAKE MATTE FRAME
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fuse import fit
from lab import frames, read_alpha


def blur_fusion(image: np.ndarray, alpha: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """One pass of F, B estimation from image I and alpha (both float)."""
    def blur(x):
        return cv2.blur(x, (r, r))
    a = alpha[..., None]
    blurred_a = blur(alpha)[..., None]
    blurred_fa = blur(image * a)
    blurred_f = blurred_fa / (blurred_a + 1e-5)
    blurred_b1a = blur(image * (1 - a))
    blurred_b = blurred_b1a / ((1 - blurred_a) + 1e-5)
    f = blurred_f + a * (image - a * blurred_f - (1 - a) * blurred_b)
    return np.clip(f, 0, 1), blurred_b


def estimate(image, alpha, scale: float = 1.0):
    """Blur sizes are for 1920 wide; scale them for other sizes."""
    f, b = blur_fusion(image, alpha, max(3, int(91 * scale) | 1))
    # Second, finer pass on the first estimate (as in the reference).
    a = alpha[..., None]
    def blur(x, r=max(3, int(7 * scale) | 1)):
        return cv2.blur(x, (r, r))
    blurred_a = blur(alpha)[..., None]
    bf = blur(f * a) / (blurred_a + 1e-5)
    bb = blur(b * (1 - a)) / ((1 - blurred_a) + 1e-5)
    return np.clip(bf + a * (image - a * bf - (1 - a) * bb), 0, 1)


def main() -> None:
    take, matte, index = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    rgb = next(f for i, f in frames(take, index + 1) if i == index)
    image = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    alpha = fit(read_alpha(take / f"{matte}.bin")[index])
    fg = estimate(image, alpha)
    # A dark blue backdrop, closer to a real replacement than magenta.
    backdrop = np.zeros_like(image)
    backdrop[:] = (0.10, 0.16, 0.30)
    a = alpha[..., None]
    plain = image * a + backdrop * (1 - a)
    fixed = fg * a + backdrop * (1 - a)
    tiles = []
    up = lambda x: np.ascontiguousarray((x[::-1, ::-1] * 255).astype(np.uint8))
    pl, fx = up(plain), up(fixed)
    ref = alpha[::-1, ::-1]
    top = np.where(ref.max(axis=1) > 0.5)[0][0]
    cols = np.where(ref[top + 150] > 0.5)[0]
    for side, x in (("left", cols[0]), ("right", cols[-1])):
        x0 = int(np.clip(x - 120, 0, 1920 - 240)); y0 = int(np.clip(top + 30, 0, 1080 - 240))
        for name, img in (("original pixels", pl), ("blur fusion", fx)):
            t = Image.fromarray(img[y0:y0 + 240, x0:x0 + 240]).resize((480, 480), Image.NEAREST)
            ImageDraw.Draw(t).text((4, 4), f"{name} {side}", fill="white")
            tiles.append(t)
    page = Image.new("RGB", (960, 960))
    for i, t in enumerate(tiles):
        page.paste(t, (480 * (i % 2), 480 * (i // 2)))
    out = take.parent.parent / "sheets" / f"{take.name}-f{index:04d}-{matte}-blurfusion.png"
    page.save(out)
    print(out)


if __name__ == "__main__":
    main()
