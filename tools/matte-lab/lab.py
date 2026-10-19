"""Offline replay of recorded LensLink takes through cutout methods.

A take folder (recorded by the app's DatasetRecorder) holds video.mov,
frames.bin, depth.bin, motion.bin and calibration.json. Each method writes
an alpha stream beside it as <method>.bin: width u32, height u32, frame
count u32 (little-endian), then one 8-bit alpha per frame.

    uv run python lab.py rvm TAKE [--backbone mobilenetv3|resnet50] [--ratio 0.125]
    uv run python lab.py matanyone TAKE --seed vision-accurate [--size 512x288]
    uv run python lab.py sheet TAKE METHOD [METHOD ...] [--frames N]

`sheet` renders, for evenly spaced frames, each method's cutout over
magenta: the whole frame and a full-resolution crop around the head.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

CACHE = Path("/Volumes/Developer-Artifacts/host-caches/matte-lab")
MAGENTA = np.array([255, 0, 255], dtype=np.float32)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def frames(take: Path, limit: int | None = None):
    """Yields (index, HxWx3 uint8 RGB) for each video frame."""
    with av.open(str(take / "video.mov")) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if limit is not None and index >= limit:
                return
            yield index, frame.to_ndarray(format="rgb24")


def frame_count(take: Path) -> int:
    return (take / "frames.bin").stat().st_size // 8


class AlphaWriter:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("wb")
        self.count = 0
        self.shape: tuple[int, int] | None = None

    def write(self, alpha: np.ndarray) -> None:
        if self.shape is None:
            self.shape = alpha.shape
            self.file.write(struct.pack("<III", alpha.shape[1], alpha.shape[0], 0))
        self.file.write(np.ascontiguousarray(alpha, dtype=np.uint8).tobytes())
        self.count += 1

    def close(self) -> None:
        self.file.seek(8)
        self.file.write(struct.pack("<I", self.count))
        self.file.close()


def read_alpha(path: Path) -> np.ndarray:
    """All frames of an alpha stream as (N, H, W) uint8 (memory-mapped)."""
    w, h, n = struct.unpack("<III", path.read_bytes()[:12])
    return np.memmap(path, dtype=np.uint8, mode="r", offset=12, shape=(n, h, w))


# --- Robust Video Matting -------------------------------------------------


def run_rvm(take: Path, backbone: str, ratio: float, limit: int | None) -> None:
    sys.path.insert(0, str(CACHE / "src" / "rvm"))
    from model import MattingNetwork  # type: ignore

    dev = device()
    model = MattingNetwork(backbone).eval().to(dev)
    model.load_state_dict(torch.load(CACHE / "weights" / f"rvm_{backbone}.pth", map_location="cpu"))
    writer = AlphaWriter(take / f"rvm-{backbone}.bin")
    rec = [None] * 4
    spent = []
    with torch.no_grad():
        for _, rgb in frames(take, limit):
            src = torch.from_numpy(rgb).to(dev).permute(2, 0, 1)[None].float() / 255
            start = time.perf_counter()
            _, pha, *rec = model(src, *rec, downsample_ratio=ratio)
            alpha = (pha[0, 0].clamp(0, 1) * 255).byte().cpu().numpy()
            spent.append(time.perf_counter() - start)
            # Kept at half size: plenty to judge edges, a quarter of the disk.
            writer.write(alpha[::2, ::2])
    writer.close()
    print(f"rvm-{backbone}: {writer.count} frames, mean {np.mean(spent) * 1000:.1f} ms "
          f"(MPS, incl. readback; not a deployment figure)")


# --- MatAnyone2 -----------------------------------------------------------


def seated_frames(spec: str, count: int) -> np.ndarray:
    """'all', 'none', or ranges like '120-480,560-600' (inclusive)."""
    out = np.zeros(count, dtype=bool)
    if spec == "all":
        out[:] = True
    elif spec != "none":
        for part in spec.split(","):
            a, b = (int(v) for v in part.split("-"))
            out[a:b + 1] = True
    return out


def run_matanyone(take: Path, seed: str, size: tuple[int, int], limit: int | None,
                  reseed: int = 0, objects: list[str] | None = None,
                  chair: str = "", seated: str = "none", tag: str = "",
                  gate: str = "") -> None:
    """Mirrors the reference inference_matanyone2.py loop: the seed mask
    (0-255) dilated then eroded by 10, and the first frame repeated 10
    times as warm-up before the take plays.

    With `reseed` N > 0, every N frames the tracker is told the target
    again: the person seed for that frame, plus each object track in
    `objects` (kept always, e.g. the mic), plus the `chair` track on
    frames marked seated. Everything outside is taught as background, so
    things the tracker absorbed (a set-down mug, a vacated chair) can
    leave."""
    sys.path.insert(0, str(CACHE / "src" / "matanyone2"))
    from matanyone2.inference.inference_core import InferenceCore  # type: ignore
    from matanyone2.utils.get_default_model import get_matanyone2_model  # type: ignore
    from matanyone2.utils.inference_utils import gen_dilate, gen_erosion  # type: ignore

    dev = device()
    net = get_matanyone2_model(str(CACHE / "weights" / "matanyone2.pth"), dev)
    core = InferenceCore(net, cfg=net.cfg)
    w, h = size
    seeds = read_alpha(take / f"{seed}.bin")
    tracks = [read_alpha(take / f"{o}.bin") for o in (objects or [])]
    chair_track = read_alpha(take / f"{chair}.bin") if chair else None
    sitting = seated_frames(seated, len(seeds))

    gate_seeds = read_alpha(take / f"{gate}.bin") if gate else None

    def target(index: int, person=None) -> torch.Tensor:
        def fit(a: np.ndarray) -> np.ndarray:
            return np.array(Image.fromarray(np.asarray(a)).resize((w, h), Image.BILINEAR)) > 127

        keep = fit((seeds if person is None else person)[index])
        for t in tracks:
            keep |= fit(t[index])
        if chair_track is not None and sitting[index]:
            keep |= fit(chair_track[index])
        mask = np.where(keep, 255, 0).astype(np.uint8)
        mask = gen_erosion(gen_dilate(mask, 10, 10), 10, 10)
        return torch.from_numpy(mask).float().to(dev)

    def fit_track(a: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(np.asarray(a)).resize((w, h), Image.BILINEAR)) > 127

    def allowed_region(index: int, person=None) -> torch.Tensor:
        """Where the matte may be non-zero: the target grown by a margin
        (~2.5% of the width) so hair beyond Vision's mask survives."""
        grown = gen_dilate(target(index, person).cpu().numpy().astype(np.uint8), w // 40, w // 40)
        return torch.from_numpy(grown).float().to(dev)

    def forced_region(index: int) -> torch.Tensor:
        """Always kept: the object tracks, and the chair while seated."""
        keep = np.zeros((h, w), dtype=bool)
        for t in tracks:
            keep |= fit_track(t[index])
        if chair_track is not None and sitting[index]:
            keep |= fit_track(chair_track[index])
        return torch.from_numpy(keep.astype(np.float32)).to(dev)

    name = f"matanyone-{w}x{h}-{seed}" + (f"-{tag}" if tag else "")
    writer = AlphaWriter(take / f"{name}.bin")
    spent = []
    warmup = 10

    def to_input(rgb: np.ndarray) -> torch.Tensor:
        small = np.array(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
        return torch.from_numpy(small).permute(2, 0, 1).float().to(dev) / 255

    with torch.no_grad():
        for index, rgb in frames(take, limit):
            image = to_input(rgb)
            start = time.perf_counter()
            if index == 0:
                core.step(image, target(0), objects=[1])
                prob = core.step(image, first_frame_pred=True)
                for _ in range(warmup):
                    prob = core.step(image, first_frame_pred=True)
            elif reseed and index % reseed == 0:
                # Correct, don't replace: keep the prediction's soft edges,
                # zero it well outside what the rules allow, force the
                # always-kept objects on, and store that as memory.
                predicted = core.output_prob_to_mask(core.step(image)).clamp(0, 1)
                allowed = (allowed_region(index) > 0).float()
                forced = (forced_region(index) > 0).float()
                corrected = torch.maximum(predicted * allowed, forced)
                core.step(image, corrected * 255, objects=[1])
                prob = None
                alpha = corrected
            else:
                prob = core.step(image)
            if prob is not None:
                alpha = core.output_prob_to_mask(prob)
            if gate_seeds is not None:
                # Every frame: the rules applied to the output itself, from
                # a cheap per-frame person mask, so memory can never keep
                # what the rules drop between corrections.
                allowed = (allowed_region(index, gate_seeds) > 0).float()
                alpha = torch.maximum(alpha.clamp(0, 1) * allowed,
                                      (forced_region(index) > 0).float())
            out = (alpha.clamp(0, 1) * 255).byte().cpu().numpy()
            spent.append(time.perf_counter() - start)
            writer.write(out)
    writer.close()
    print(f"{name}: {writer.count} frames, "
          f"mean {np.mean(spent[1:]) * 1000:.1f} ms (MPS reference code)")


# --- Review sheets ----------------------------------------------------------


def composite(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if alpha.shape != rgb.shape[:2]:
        alpha = np.array(Image.fromarray(alpha).resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))
    a = alpha.astype(np.float32)[..., None] / 255
    return (rgb.astype(np.float32) * a + MAGENTA * (1 - a)).astype(np.uint8)


def head_box(alpha: np.ndarray, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """A 1280x720 full-resolution box around the top of the subject."""
    rows = np.where(alpha.max(axis=1) > 127)[0]
    cols = np.where(alpha.max(axis=0) > 127)[0]
    sy, sx = frame_h / alpha.shape[0], frame_w / alpha.shape[1]
    top = int(rows[0] * sy) if len(rows) else 0
    cx = int(np.median(cols) * sx) if len(cols) else frame_w // 2
    x0 = min(max(cx - 640, 0), frame_w - 1280)
    y0 = min(max(top - 80, 0), frame_h - 720)
    return x0, y0, x0 + 1280, y0 + 720


def sheet(take: Path, methods: list[str], count: int, out_dir: Path, at: str = "") -> None:
    alphas = {m: read_alpha(take / f"{m}.bin") for m in methods}
    total = min(len(a) for a in alphas.values())
    picks = ({int(v) for v in at.split(",")} if at else set(np.linspace(0, total - 1, count).astype(int).tolist()))
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, rgb in frames(take, total):
        if index not in picks:
            continue
        h, w = rgb.shape[:2]
        box = head_box(np.asarray(alphas[methods[0]][index])[::-1, ::-1], w, h)
        tiles = []
        for m in methods:
            # Upright: the phone streams sensor-landscape, upside down.
            comp = np.ascontiguousarray(composite(rgb, np.asarray(alphas[m][index]))[::-1, ::-1])
            whole = Image.fromarray(comp).resize((640, 360), Image.BILINEAR)
            crop = Image.fromarray(comp[box[1]:box[3], box[0]:box[2]]).resize((640, 360), Image.LANCZOS)
            tile = Image.new("RGB", (1280, 380), "white")
            tile.paste(whole, (0, 20))
            tile.paste(crop, (640, 20))
            ImageDraw.Draw(tile).text((6, 4), m, fill="black")
            tiles.append(tile)
        page = Image.new("RGB", (1280, 380 * len(tiles)), "white")
        for i, tile in enumerate(tiles):
            page.paste(tile, (0, 380 * i))
        page.save(out_dir / f"{take.name}-f{index:04d}.png")
    print(f"sheets for {sorted(picks)} in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rvm")
    p.add_argument("take", type=Path)
    p.add_argument("--backbone", default="mobilenetv3")
    p.add_argument("--ratio", type=float, default=0.125)
    p.add_argument("--limit", type=int)
    p = sub.add_parser("matanyone")
    p.add_argument("take", type=Path)
    p.add_argument("--seed", default="vision-accurate")
    p.add_argument("--size", default="512x288")
    p.add_argument("--limit", type=int)
    p.add_argument("--reseed", type=int, default=0)
    p.add_argument("--objects", default="")
    p.add_argument("--chair", default="")
    p.add_argument("--seated", default="none")
    p.add_argument("--tag", default="")
    p.add_argument("--gate", default="", help="person mask stream applied to every frame")
    p = sub.add_parser("sheet")
    p.add_argument("take", type=Path)
    p.add_argument("methods", nargs="+")
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--at", default="", help="exact frame numbers, comma-separated")
    p.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cmd == "rvm":
        run_rvm(args.take, args.backbone, args.ratio, args.limit)
    elif args.cmd == "matanyone":
        w, h = (int(v) for v in args.size.split("x"))
        run_matanyone(args.take, args.seed, (w, h), args.limit, args.reseed,
                      [o for o in args.objects.split(",") if o], args.chair,
                      args.seated, args.tag, args.gate)
    else:
        sheet(args.take, args.methods, args.frames, args.out or args.take.parent.parent / "sheets", args.at)


if __name__ == "__main__":
    main()
