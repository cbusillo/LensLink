"""The keep/drop rules with only what a live session has: MatAnyone2 plus
a cheap per-frame person mask, a seated flag, and object masks from ONE
setup frame (the owner's clicks). No per-frame object tracker: the mic is
followed through MatAnyone2's own output.

    uv run python live_rules.py TAKE --size 1024x576 --person vision-balanced \
        --setup-mic sam-mic --setup-chair sam-chair --seated SPEC [--reseed 15]

Per frame:
  you      = the person mask, grown by a margin (hair beyond it survives)
  mic      = last frame's mic, grown a little, where the output still shows
             something (so it follows a moved arm)
  touching = output regions touching you (the chair while you sit in it)
  allowed  = you + mic + (touching, only while seated)
  output   = MatAnyone2's matte * allowed
Every --reseed frames that output is written back into memory, so the
tracker forgets what the rules dropped (a set-down mug, a vacated chair).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lab import CACHE, AlphaWriter, device, frames, read_alpha, seated_frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("take", type=Path)
    p.add_argument("--size", default="1024x576")
    p.add_argument("--seed", default="vision-accurate", help="person mask for the setup frame")
    p.add_argument("--person", default="vision-balanced", help="cheap per-frame person mask")
    p.add_argument("--setup-mic", default="sam-mic")
    p.add_argument("--setup-chair", default="")
    p.add_argument("--seated", default="none")
    p.add_argument("--reseed", type=int, default=15)
    p.add_argument("--tag", default="live")
    a = p.parse_args()

    sys.path.insert(0, str(CACHE / "src" / "matanyone2"))
    from matanyone2.inference.inference_core import InferenceCore  # type: ignore
    from matanyone2.utils.get_default_model import get_matanyone2_model  # type: ignore

    w, h = (int(v) for v in a.size.split("x"))
    dev = device()
    net = get_matanyone2_model(str(CACHE / "weights" / "matanyone2.pth"), dev)
    core = InferenceCore(net, cfg=net.cfg)
    person = read_alpha(a.take / f"{a.person}.bin")
    sitting = seated_frames(a.seated, len(person))

    def fit(x) -> np.ndarray:
        return cv2.resize(np.asarray(x), (w, h), interpolation=cv2.INTER_LINEAR) > 127

    def disk(r: int) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    grow_you, grow_mic = disk(w // 40), disk(max(2, w // 128))
    setup = fit(read_alpha(a.take / f"{a.seed}.bin")[0])
    mic = fit(read_alpha(a.take / f"{a.setup_mic}.bin")[0])
    setup |= mic
    if a.setup_chair:
        setup |= fit(read_alpha(a.take / f"{a.setup_chair}.bin")[0])
    seed_mask = torch.from_numpy(np.where(setup, 255.0, 0.0).astype(np.float32)).to(dev)

    writer = AlphaWriter(a.take / f"matanyone-{w}x{h}-{a.tag}.bin")
    spent = []
    with torch.no_grad():
        for index, rgb in frames(a.take):
            small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            image = torch.from_numpy(small).permute(2, 0, 1).float().to(dev) / 255
            start = time.perf_counter()
            if index == 0:
                core.step(image, seed_mask, objects=[1])
                for _ in range(11):
                    prob = core.step(image, first_frame_pred=True)
            else:
                prob = core.step(image)
            alpha = core.output_prob_to_mask(prob).clamp(0, 1).cpu().numpy()

            solid = (alpha > 0.5).astype(np.uint8)
            you = cv2.dilate(fit(person[index]).astype(np.uint8), grow_you) > 0
            mic = (cv2.dilate(mic.astype(np.uint8), grow_mic) > 0) & (solid > 0)
            allowed = you | mic
            if sitting[index]:
                _, labels = cv2.connectedComponents(solid)
                near = cv2.dilate(you.astype(np.uint8), disk(2)) > 0
                touching = np.isin(labels, np.unique(labels[near & (solid > 0)]))
                allowed |= touching & (labels > 0)
            out = alpha * allowed

            if a.reseed and index and index % a.reseed == 0:
                core.step(image, torch.from_numpy(out * 255).float().to(dev), objects=[1])
            spent.append(time.perf_counter() - start)
            writer.write((out * 255).astype(np.uint8))
    writer.close()
    print(f"matanyone-{w}x{h}-{a.tag}: {writer.count} frames, mean {np.mean(spent[1:]) * 1000:.1f} ms (MPS)")


if __name__ == "__main__":
    main()
