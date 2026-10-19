# matte-lab

Offline replay of takes recorded by the app's `record_dataset` command
(`ios-app/Sources/DatasetRecorder.swift`) through cutout methods, to compare
them on the same frames before building anything live. This fork only.

- `vision-masks.swift`: Apple Vision person segmentation at fast, balanced
  or accurate, with per-frame timing. Build with
  `swiftc -O vision-masks.swift -o vision-masks`.
- `lab.py rvm|matanyone|sheet`: Robust Video Matting, MatAnyone2 (reference
  PyTorch on MPS, seeded by a Vision mask) and side-by-side review sheets.
- `crops.py`: 1:1 crops of both hair edges for each method at one frame.
- `sam_tracks.py`: SAM 2 object tracks (mic, chair) from click points; a lab
  stand-in for automatic detection.
- `video.py`: side-by-side review video of methods.
- `lab.py matanyone --reseed --objects --chair --seated --gate`: the keep/drop
  rules: memory corrected every N frames and the output gated every frame.

Model source and weights are not vendored; `lab.py` expects them under
`/Volumes/Developer-Artifacts/host-caches/matte-lab` (`src/rvm`,
`src/matanyone2` at `0079197`, `weights/*.pth`). Timings from this harness
are MPS reference-code figures, not deployment numbers.

Findings that shaped the live design (October 19, 2026, on this M4 Max):

- MatAnyone2 Core ML per frame: 512x288 14.5 ms, 768x432 29.9 ms on the
  Neural Engine; 1024x576 54.5 ms there, 49.4 ms on the GPU, so 1024 needs
  the encoder and decoder on different units, overlapped across frames
  (`bench_ma.py`).
- Apple's SAM 2.1 Core ML image encoder: 36-43 ms GPU, 105-120 ms Neural
  Engine, 130-150 ms CPU only (`bench_coreml.py ... CPU_ONLY`). Objects are
  tracked with SAM 2's video mode on every 8th frame on the CPU
  (`sam_sparse.py`); image-mode box refresh (`sam_refresh.py`) loses the
  mic when a hand grabs it.
- Blur-fusion foreground colors (`blurfusion.py`) remove the old-wall fringe
  at hair edges; edge tricks on low-resolution mattes (`fuse.py`, `core.py`,
  `guided.py`) did not reach the 1024 look.

