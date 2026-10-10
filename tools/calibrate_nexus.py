#!/usr/bin/env python3
"""Measure Nexus rigid-body buffer demand for every scene the suites run.

Benchmarks run Nexus with fixed buffers (collision pairs, multibody contact slots, solver
colors). This runs each (scenario, size) of every suite with Nexus, untimed, under its
adaptive policies, and records per-environment peaks over the longest horizon any suite uses
for that scenario. The adapter then applies them with headroom; sizes between calibrated ones
use the next larger entry. Rerun after changing the Nexus pin, solver settings or scenes,
then rebuild the Nexus adapter (the table is compiled in).

    python3 tools/calibrate_nexus.py [--repeats 3] [--scenarios brick_ring,pyramids]
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import tomllib

from capabilities import candidates, detect

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "adapters/nexus/capacities.json"
PEAKS = ["collision_pairs", "collision_pairs_capacity", "mb_contact_slots", "mb_contact_slots_capacity", "solver_colors"]


def suite_scenes():
    """(scenario, size) -> (dt, horizon) over every suite that can run Nexus rigid bodies."""
    scenes, horizons = {}, defaultdict(int)
    for path in sorted((ROOT / "suites").glob("*.toml")):
        suite = tomllib.loads(path.read_text())
        for scenario in suite["scenarios"]:
            if scenario.startswith("mpm_"):
                continue
            horizons[scenario] = max(horizons[scenario], suite["steps"], suite["warmup"])
            for size in suite["sizes"]:
                previous = scenes.setdefault((scenario, size), suite["dt"])
                if previous != suite["dt"]:
                    raise SystemExit(f"{scenario} size {size} uses several timesteps; calibrate each")
    return {key: (dt, horizons[key[0]]) for key, dt in sorted(scenes.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--scenarios", help="comma-separated subset; other entries are kept")
    p.add_argument("--backend", help="Nexus backend (default: the full round's choice for this machine)")
    args = p.parse_args()
    backend = args.backend or candidates(detect())[0]["nexus"]
    controller = os.environ.get("PHYSMARK_BIN") or str(ROOT / "target/release/physmark")
    adapter = str(ROOT / "adapters/nexus/target/release/physmark-nexus")
    wanted = set(args.scenarios.split(",")) if args.scenarios else None
    table = json.loads(TABLE.read_text()) if TABLE.exists() else {}
    scenes = {k: v for k, v in suite_scenes().items() if wanted is None or k[0] in wanted}
    measured = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="physmark-calibrate-") as tmp:
        tmp = Path(tmp)
        for index, ((scenario, size), (dt, horizon)) in enumerate(scenes.items(), 1):
            scene_path = tmp / "scene.json"
            subprocess.run([controller, "--root", str(ROOT), "scene", scenario, "--size", str(size), "--dt", repr(dt),
                            "--output", str(scene_path)], check=True, capture_output=True)
            job = dict(schema_version=1, scene=json.loads(scene_path.read_text()), engine="nexus", backend=backend,
                       mode="timing", worlds=1, steps=horizon, warmup=horizon, capture_stride=1, threads=8)
            (tmp / "job.json").write_text(json.dumps(job))
            peaks = {}
            start = time.monotonic()
            for _ in range(args.repeats):
                subprocess.run([adapter, "--calibrate", str(tmp / "job.json"), str(tmp / "peaks.json")], check=True)
                result = json.loads((tmp / "peaks.json").read_text())
                for key in PEAKS:
                    peaks[key] = max(peaks.get(key, 0), result[key])
            # Keyed by the scene's own size: robot scenes count links, not robots.
            measured[scenario].append(dict(size=job["scene"]["size"], steps=horizon, **peaks))
            print(f"[{index}/{len(scenes)}] {scenario} {size}: {peaks} ({time.monotonic() - start:.0f} s)", flush=True)
    for scenario, entries in measured.items():
        table[scenario] = sorted(entries, key=lambda e: e["size"])
    nexus = tomllib.loads((ROOT / "engines.lock").read_text())["engines"]["nexus"]["commit"]
    table["_meta"] = dict(nexus_commit=nexus, backend=backend, repeats=args.repeats, worlds=1,
                          method="adaptive policies from generous buffers; per-environment peaks over the horizon")
    TABLE.write_text(json.dumps(dict(sorted(table.items())), indent=1) + "\n")
    print(f"wrote {TABLE}; rebuild the Nexus adapter to apply it", flush=True)


if __name__ == "__main__":
    main()
