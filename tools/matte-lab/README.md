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
