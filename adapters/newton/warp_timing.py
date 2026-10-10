"""Warp CUDA event timings for Newton's rigid and MPM solvers."""
import json
import math

from gpu_timing import available, unavailable


def warp_profile(sim, job, directory):
    import warp as wp

    with wp.ScopedTimer(
        "physmark GPU replay",
        print=False,
        cuda_filter=wp.TIMING_ALL,
        synchronize=True,
    ) as timer:
        sim.advance(job["steps"])
        sim.sync()
    samples = [
        dict(name=r.name, milliseconds=r.elapsed, activity=r.filter)
        for r in timer.timing_results
        if r.device == sim.device
    ]
    (directory / "gpu-samples.json").write_text(json.dumps(samples, allow_nan=False))
    if not samples:
        return unavailable("Warp returned no device activity timestamps")
    if any(not math.isfinite(r["milliseconds"]) or r["milliseconds"] < 0 for r in samples):
        raise ValueError("invalid Warp CUDA event duration")
    # ScopedTimer.elapsed is a HOST time; only TimingResult.elapsed uses CUDA events.
    return available(
        sum(r["milliseconds"] for r in samples) / 1000,
        "warp_cuda_events",
        "sum_warp_activity_durations",
        job["steps"],
        len(samples),
        evidence="gpu-samples.json",
        replay="separate process",
        includes="Warp kernels, builtins, copies and memsets; sums may overlap across streams",
    )
