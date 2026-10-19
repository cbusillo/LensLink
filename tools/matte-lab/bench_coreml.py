"""Times a Core ML model on this Mac per compute unit, feeding recurrent
outputs back as inputs (RVM's r1..r4) when names match.

    uv run python bench_coreml.py MODEL.mlmodel [RUNS]
"""
import sys
import time

import coremltools as ct
import numpy as np
from PIL import Image

path, runs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
for unit in (sys.argv[3].split(",") if len(sys.argv) > 3 else ("CPU_AND_NE", "CPU_AND_GPU", "ALL")):
    model = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, unit))
    spec = model.get_spec()
    feeds = {}
    for inp in spec.description.input:
        t = inp.type.WhichOneof("Type")
        if t == "imageType":
            feeds[inp.name] = Image.new("RGB", (inp.type.imageType.width, inp.type.imageType.height), (90, 90, 90))
        else:
            shape = list(inp.type.multiArrayType.shape)
            feeds[inp.name] = np.zeros(shape, dtype=np.float32)
    outs = [o.name for o in spec.description.output]
    times = []
    for i in range(runs):
        start = time.perf_counter()
        result = model.predict(feeds)
        times.append(time.perf_counter() - start)
        for name in feeds:  # r1i <- r1o
            if name.endswith("i") and name[:-1] + "o" in result:
                feeds[name] = result[name[:-1] + "o"]
    t = np.array(times[10:]) * 1000
    print(f"{unit:12s} median {np.median(t):.1f} ms  p90 {np.percentile(t, 90):.1f} ms  (inputs {[i.name for i in spec.description.input]}, outputs {outs})")
