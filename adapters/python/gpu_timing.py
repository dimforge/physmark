"""Device profilers used only in a separate gpu_timing subprocess.

Chrome trace timestamps/durations are microseconds (displayTimeUnit is cosmetic).
Never use host events, host profiler totals, or wall time as GPU measurements.
"""

import gzip
import json
import math
from pathlib import Path
import re


def unavailable(reason, status="unavailable"):
    return dict(
        status=status,
        seconds=None,
        method=None,
        scope=None,
        measured_steps=0,
        sample_count=0,
        reason=reason,
        details={},
    )


def preflight(job):
    if job["backend"] == "cpu":
        return unavailable("CPU backend", "not_applicable")
    if job["engine"] == "genesis" and job["backend"] == "metal":
        return unavailable(
            "Quadrants 1.3.3 Metal does not wire a hardware profiler into GfxRuntime. "
            "Its generic kernel profiler is a host-clock profiler. PyTorch MPS uses "
            "a separate command queue, so MPS events cannot time Genesis kernels."
        )
    if job["backend"] != "cuda":
        return unavailable("No verified device profiler for this engine/backend")
    return None


def available(seconds, method, scope, steps, count, **details):
    if not math.isfinite(seconds) or seconds <= 0 or count <= 0:
        raise ValueError("GPU profiler returned invalid or empty durations")
    return dict(
        status="available",
        seconds=seconds,
        method=method,
        scope=scope,
        measured_steps=steps,
        sample_count=count,
        reason=None,
        details=details,
    )


def trace_device_time(trace, kind, device=0):
    """Union of actual GPU activity intervals; reject host and aggregate lanes.

    Kineto identifies kernels/copies/memsets by category and device argument.
    XLA identifies devices and streams through Chrome metadata events.
    Union avoids double counting simultaneous streams and nested/duplicate events.
    """
    events = trace.get("traceEvents", [])
    processes, threads = {}, {}
    for event in events:
        if event.get("ph") != "M":
            continue
        name = event.get("args", {}).get("name", "")
        if event.get("name") == "process_name":
            processes[event.get("pid")] = str(name)
        elif event.get("name") == "thread_name":
            threads[event.get("pid"), event.get("tid")] = str(name)
    intervals = []
    for event in events:
        if event.get("ph") != "X":
            continue
        if kind == "kineto":
            if event.get("cat") not in {"kernel", "gpu_memcpy", "gpu_memset"}:
                continue
            if str(event.get("args", {}).get("device")) != str(device):
                continue
        elif kind == "xla":
            process = processes.get(event.get("pid"), "")
            match = re.fullmatch(r"/device:GPU:(\d+)(?:\s.*)?", process)
            thread = threads.get((event.get("pid"), event.get("tid")), "")
            if not match or int(match[1]) != device:
                continue
            if not re.match(r"stream[ #:_-]*\d", thread, re.IGNORECASE):
                continue
        else:
            raise ValueError(f"unknown trace format: {kind}")
        start, duration = float(event["ts"]), float(event["dur"])
        if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid GPU activity timestamp")
        if duration > 0:
            intervals.append((start, start + duration))
    if not intervals:
        return None
    intervals.sort()
    start, end = intervals[0]
    total = 0.0
    for left, right in intervals[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    total += end - start
    return total / 1_000_000, len(intervals)


def profile(sim, job, directory):
    steps = job["steps"]
    directory = Path(directory)
    if job["engine"] == "newton":
        import warp as wp

        with wp.ScopedTimer(
            "physmark GPU replay",
            print=False,
            cuda_filter=wp.TIMING_ALL,
            synchronize=True,
        ) as timer:
            sim.advance(steps)
            sim.sync()
        samples = [
            dict(name=r.name, milliseconds=r.elapsed, activity=r.filter)
            for r in timer.timing_results
            if r.device == sim.device
        ]
        (directory / "gpu-samples.json").write_text(
            json.dumps(samples, allow_nan=False)
        )
        if not samples:
            return unavailable("Warp returned no device activity timestamps")
        if any(
            not math.isfinite(r["milliseconds"]) or r["milliseconds"] < 0
            for r in samples
        ):
            raise ValueError("invalid Warp CUDA event duration")
        # ScopedTimer.elapsed is a HOST time; only TimingResult.elapsed uses CUDA events.
        return available(
            sum(r["milliseconds"] for r in samples) / 1000,
            "warp_cuda_events",
            "sum_warp_activity_durations",
            steps,
            len(samples),
            evidence="gpu-samples.json",
            replay="separate process",
            includes="Warp kernels, builtins, copies and memsets; sums may overlap across streams",
        )
    if job["engine"] == "genesis":
        import torch

        if (
            torch.profiler.ProfilerActivity.CUDA
            not in torch.profiler.supported_activities()
        ):
            return unavailable(
                "PyTorch profiler has no CUDA activity collector (CUPTI required)"
            )
        path = directory / "gpu-trace.json"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            sim.advance(steps)
            sim.sync()
        profiler.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
        kind, method = "kineto", "pytorch_cupti_activity"
    elif job["engine"] == "mjx":
        import jax

        trace_dir = directory / "gpu-trace"
        with jax.profiler.trace(str(trace_dir)):
            sim.advance(steps)
            sim.sync()
        paths = list(trace_dir.rglob("*.trace.json.gz"))
        if len(paths) != 1:
            return unavailable(f"Expected one XLA device trace, found {len(paths)}")
        path = paths[0]
        with gzip.open(path, "rt") as f:
            trace = json.load(f)
        kind, method = "xla", "jax_cupti_activity"
    else:
        raise ValueError("no GPU profiler implementation")
    result = trace_device_time(trace, kind)
    if result is None:
        return unavailable(
            "Trace contains no GPU device activity; CUPTI support/permissions are required. Host events were ignored."
        )
    seconds, count = result
    return available(
        seconds,
        method,
        "device_activity_union",
        steps,
        count,
        evidence=str(path.relative_to(directory)),
        replay="separate process",
        includes="union of GPU kernel/copy/memset intervals on device 0; queue gaps excluded",
    )
