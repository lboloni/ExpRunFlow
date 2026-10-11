"""The metrics of a run (metrics.json): plain data from which replications are aggregated without loading
project objects. Needs no optional dependency, so that every stage can write its metrics.
See docs/DESIGN-Replication.md, Section 3."""

import json
import os
import pathlib
import platform
import subprocess
import sys

METRICS_FILE = "metrics.json"


def host_info():
    """The machine and the Python a run executed on, for reporting how long experiments took"""
    processor = platform.processor()
    if sys.platform == "darwin":
        try:
            processor = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                       text=True, timeout=5).stdout.strip() or processor
        except Exception:
            pass
    return {"platform": platform.platform(), "processor": processor, "cpus": os.cpu_count(),
            "python": platform.python_version()}


def save_metrics(exp, scalars, series=None, entities=None, label_keys=None):
    """Write the metrics.json of a run into its data directory. The identification is filled in from the
    exp/run: a replication's block gives its base run, factor values and labels; a plain run is its own
    base run, has no factor values, and its labels are read from label_keys.
    scalars: name -> number; series: name -> list, the first key being the shared index (e.g. timestep);
    entities: entity -> {"group": ..., name -> number}"""
    block = exp.get("replication")
    if block:
        identification = {"base-run": block["base-run"], "factors": dict(block["factors"]),
                          "labels": dict(block.get("labels", {}))}
    else:
        identification = {"base-run": exp.run(), "factors": {},
                          "labels": {key: exp.get(key) for key in (label_keys or [])}}
    metrics = {"run": exp.run(), **identification, "scalars": scalars, "entities": entities or {},
               "series": series or {}, "host": host_info()}
    path = pathlib.Path(exp.data_dir(), METRICS_FILE)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=1)
    return path


def load_metrics(data_dir):
    with open(pathlib.Path(data_dir, METRICS_FILE)) as f:
        return json.load(f)
