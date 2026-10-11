#!/usr/bin/env python3
"""Shared adapter harness: scene translation helpers and the timed loop.

Each engine lives in its own `adapters/<engine>` directory and calls `main` with a factory
that builds its simulation. The Rust controller never participates in the timed step loop.
"""

import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET


def numbers(v):
    return " ".join(format(float(x), ".17g") for x in v)


def arena_cap_bytes():
    """MuJoCo arena for models whose automatic size cannot be allocated: min(16 GiB, RAM / 2).

    The automatic size grows far faster than use (23 GiB at 4,000 bodies, while 10,000 rigid
    bodies peaked at 383 MiB); Linux refuses single allocations beyond RAM + swap, so compiling
    fails from about 8,000 bodies. Untouched arena pages are never resident."""
    try:
        ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        ram = 2**35
    return min(16 * 2**30, ram // 2)


def is_arena_allocation_failure(error):
    return "Could not allocate memory" in str(error)


def compile_mjcf(xml):
    """Compiles with MuJoCo's automatic arena, falling back to `arena_cap_bytes` only when that
    arena cannot be allocated (the arena size never changes results unless it runs out)."""
    import mujoco

    try:
        return mujoco.MjModel.from_xml_string(xml)
    except ValueError as error:
        if not is_arena_allocation_failure(error):
            raise
        capped = xml.replace("<worldbody>", f'<size memory="{arena_cap_bytes()}"/><worldbody>', 1)
        return mujoco.MjModel.from_xml_string(capped)


def mjcf(scene):
    if scene.get("robot"):
        return scene["robot"]["xml"]
    root = ET.Element("mujoco", model="physmark")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    option = ET.SubElement(
        root,
        "option",
        timestep=str(scene["dt"]),
        gravity=numbers(scene["gravity"]),
        integrator="Euler",
        solver="Newton",
        iterations="50",
        tolerance="1e-8",
    )
    ET.SubElement(option, "flag", sleep="disable")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0", armature="0", frictionloss="0")
    ET.SubElement(
        default,
        "geom",
        condim="3",
        friction="0.5 0 0",
        margin="0",
        gap="0",
        solref="0.02 1",
    )
    world = ET.SubElement(root, "worldbody")
    hinges = {j["body"]: j for j in scene["hinges"]}
    for b in scene["bodies"]:
        q = b["rotation"]
        body = ET.SubElement(
            world,
            "body",
            name=f"body_{b['id']}",
            pos=numbers(b["position"]),
            quat=numbers([q[3], *q[:3]]),
        )
        if b["dynamic"]:
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass=str(b["mass"]),
                diaginertia=numbers(b["inertia"]),
            )
            if b["id"] in hinges:
                j = hinges[b["id"]]
                ET.SubElement(
                    body,
                    "joint",
                    name=f"joint_{b['id']}",
                    type="hinge",
                    pos=numbers(j["local_anchor"]),
                    axis=numbers(j["axis"]),
                    limited="false",
                )
            else:
                ET.SubElement(body, "freejoint", name=f"joint_{b['id']}")
        shape = b["shape"]
        size = (
            b["size"]
            if shape == "box"
            else ([b["size"][0]] if shape == "sphere" else [b["size"][0], b["size"][2]])
        )
        ET.SubElement(
            body,
            "geom",
            name=f"geom_{b['id']}",
            type=shape,
            size=numbers(size),
            friction=numbers([b["friction"], 0, 0]),
        )
    return ET.tostring(root, encoding="unicode")


def initial_velocities(scene):
    """(body, linear world velocity, angular world velocity) for bodies that start moving."""
    return [
        (b, b["velocity"], b.get("angular_velocity", [0, 0, 0]))
        for b in scene["bodies"]
        if b["dynamic"] and (any(b["velocity"]) or any(b.get("angular_velocity", [0, 0, 0])))
    ]


def to_body_frame(q, v):
    """Rotates world vector `v` by the inverse of XYZW quaternion `q`."""
    import numpy as np

    u, w = -np.asarray(q[:3]), q[3]
    v = np.asarray(v, dtype=float)
    c = np.cross(u, v)
    return (v + 2 * (w * c + np.cross(u, c))).tolist()


def robot_settings(job):
    r = job["scene"]["robot"]
    return {k: r[k] for k in ("model", "source_commit", "source_path", "initial_keyframe", "target_keyframe", "ctrl", "self_contacts")}


def run(job, make_sim, evidence_dir=None):
    import numpy as np

    assert job["schema_version"] == 1
    gpu_timing = None
    if job["mode"] == "gpu_timing":
        from gpu_timing import preflight, profile

        gpu_timing = preflight(job)
        if gpu_timing is not None:
            # A capability result contains no simulation sample and is never ranked.
            return dict(
                schema_version=1,
                engine=job["engine"],
                backend=job["backend"],
                setup_seconds=0,
                warmup_seconds=0,
                elapsed_seconds=0,
                final_poses=[],
                awake_count=0,
                settings={},
                frames=[],
                gpu_timing=gpu_timing,
            )
    memory = None
    if job["mode"] == "memory":
        from memory_usage import MemorySampler

        memory = MemorySampler(job["backend"])
    start = time.perf_counter()
    if job["scene"].get("mpm") and job["worlds"] != 1:
        raise ValueError("MPM benchmarks require exactly one environment")
    sim = make_sim(job)
    if job["scene"].get("robot"):
        sim.settings["robot"] = robot_settings(job)
        initial_poses = np.asarray(sim.poses()).reshape(job["worlds"], -1, 7)
        expected = np.asarray([b["position"] for b in job["scene"]["bodies"] if b["dynamic"]])
        if not np.allclose(initial_poses[:, :, :3], expected, atol=1e-4):
            raise RuntimeError("MJCF importer initial link poses differ from canonical crouch")
    sim.sync()
    setup = time.perf_counter() - start
    if memory:
        memory.sample("setup")
    sim.settings["warmup_steps_effective"] = max(job["warmup"], job["steps"])
    start = time.perf_counter()
    sim.advance(max(job["warmup"], job["steps"]))
    sim.sync()
    warmup = time.perf_counter() - start
    if memory:
        memory.sample("warmup")
    sim.reset()
    sim.sync()
    if memory:
        memory.sample("reset")
    frames = []
    nb = sum(b["dynamic"] for b in job["scene"]["bodies"])
    if job["mode"] == "capture":
        frames.append(
            {"step": 0, "poses": np.asarray(sim.poses()[:nb], dtype=float).tolist()}
        )
        start = time.perf_counter()
        for step in range(0, job["steps"], job["capture_stride"]):
            n = min(job["capture_stride"], job["steps"] - step)
            sim.advance(n)
            sim.sync()
            frames.append(
                {
                    "step": step + n,
                    "poses": np.asarray(sim.poses()[:nb], dtype=float).tolist(),
                }
            )
        elapsed = time.perf_counter() - start
    elif job["mode"] == "gpu_timing":
        start = time.perf_counter()
        gpu_timing = profile(sim, job, evidence_dir)
        elapsed = time.perf_counter() - start
    else:
        start = time.perf_counter()
        sim.advance(job["steps"])
        sim.sync()
        elapsed = time.perf_counter() - start
    memory_result = memory.finish(job["steps"]) if memory else None
    return {
        "schema_version": 1,
        "engine": job["engine"],
        "backend": job["backend"],
        "setup_seconds": setup,
        "warmup_seconds": warmup,
        "elapsed_seconds": elapsed,
        "final_poses": np.asarray(sim.poses(), dtype=float).tolist(),
        "awake_count": sim.awake(),
        "settings": sim.settings,
        "frames": frames,
        "gpu_timing": gpu_timing,
        "memory": memory_result,
    }


def main(make_sim):
    """Adapter entry point: `adapter JOB.json RESULT.json`."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: adapter JOB.json RESULT.json")
    job = json.loads(Path(sys.argv[1]).read_text())
    path = Path(sys.argv[2])
    sims = []
    result = run(job, lambda job: sims.append(make_sim(job)) or sims[-1], path.parent)
    import math

    if not all(math.isfinite(v) for pose in result["final_poses"] for v in pose):
        # Otherwise the JSON writer fails on NaN and hides the cause.
        raise RuntimeError("simulation diverged: final poses contain NaN or infinite values")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False, indent=2))
    temporary.replace(path)
    # Some runtimes (Isaac Sim's Kit) must be shut down explicitly, after the result is safe.
    for sim in sims:
        if hasattr(sim, "close"):
            sim.close()
