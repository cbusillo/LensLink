"""Times each compiled MatAnyone2 Core ML piece per compute unit, from
the export manifest's input shapes (zeros as input; speed only).

    uv run python bench_ma.py MODELS_DIR [RUNS]
"""
import json
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np

root, runs = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 40
manifest = json.loads((root / "manifest.json").read_text())
print(f"working {manifest['working_w']}x{manifest['working_h']}")
totals = {}
for unit in ("CPU_AND_NE", "CPU_AND_GPU", "ALL"):
    row = []
    for m in manifest["models"]:
        model = ct.models.CompiledMLModel(str(root / f"{m['name']}.mlmodelc"),
                                          compute_units=getattr(ct.ComputeUnit, unit))
        feeds = {i["name"]: np.zeros(i["shape"], dtype=np.float32) for i in m["inputs"]}
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model.predict(feeds)
            times.append(time.perf_counter() - start)
        med = float(np.median(times[5:]) * 1000)
        row.append((m["name"], med))
    per_frame = sum(t for n, t in row if n in ("encoder", "uncert", "readout", "decoder"))
    memory = sum(t for n, t in row if n in ("maskencoder", "objsummary"))
    print(f"{unit:12s} " + "  ".join(f"{n} {t:.1f}" for n, t in row)
          + f"  | every frame {per_frame:.1f} ms, memory frames +{memory:.1f} ms")
