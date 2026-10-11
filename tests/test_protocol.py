import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adapters/common"))
sys.path.insert(0, str(ROOT / "adapters/mujoco"))
sys.path.insert(0, str(ROOT / "tools"))
from physmark_adapter import mjcf
from mujoco_adapter import Mujoco
from report import summarize, case_table, timing_value


def scene(name="mixed_pile", size=6):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "scene.json"
        subprocess.run(
            [
                str(ROOT / "target/release/physmark"),
                "--root",
                str(ROOT),
                "scene",
                name,
                "--size",
                str(size),
                "--output",
                str(out),
            ],
            check=True,
        )
        return json.loads(out.read_text())


def test_mjcf_preserves_mass_shapes_and_hinge_anchor():
    import mujoco
    import numpy as np

    for name in ["mixed_pile", "pendulums"]:
        s = scene(name)
        xml = mjcf(s)
        model = mujoco.MjModel.from_xml_string(xml)
        assert not model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        for b in s["bodies"]:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"body_{b['id']}")
            if b["dynamic"]:
                assert model.body_mass[idx] == b["mass"]
                np.testing.assert_allclose(model.body_inertia[idx], b["inertia"])
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        for j in s["hinges"]:
            idx = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{j['body']}"
            )
            np.testing.assert_allclose(d.xanchor[idx], j["anchor"], atol=1e-12)


def test_reset_replays_from_initial_state():
    import numpy as np

    s = scene("box_pile", 4)
    job = dict(scene=s, backend="cpu", worlds=2)
    sim = Mujoco(job, mjcf(s))
    initial = np.array(sim.poses())
    sim.advance(80)
    final = np.array(sim.poses())
    assert not np.allclose(final, initial)
    sim.reset()
    np.testing.assert_allclose(sim.poses(), initial)
    sim.advance(80)
    np.testing.assert_allclose(sim.poses(), final)


def test_failed_trials_cannot_improve_rank():
    records = []
    for engine in ["mujoco", "rapier"]:
        for size in [4, 8]:
            for repeat in [0, 1]:
                records.append(
                    dict(
                        engine=engine,
                        backend="cpu",
                        scenario="box_pile",
                        size=size,
                        worlds=1,
                        status="passed",
                        ms_per_step=2 if engine == "mujoco" else 1,
                    )
                )
    table = case_table(records, ["mujoco", "rapier"], timing_value)
    assert table["engines"] == ["rapier", "mujoco"]
    assert table["rows"][0]["cells"]["rapier"]["median"] == 1
    assert table["rows"][0]["cells"]["rapier"]["best"] and not table["rows"][0]["cells"]["mujoco"]["best"]
    for r in records:
        if r["engine"] == "rapier":
            r.update(status="skipped", outcome_label="skipped (likely timeout)")
    table = case_table(records, ["mujoco", "rapier"], timing_value)
    assert table["engines"] == ["rapier", "mujoco"]
    cell = table["rows"][1]["cells"]["rapier"]
    assert cell["median"] is None and not cell["best"] and cell["failure"] == "timeout?"
    assert table["rows"][1]["cells"]["mujoco"]["best"]
    records[-1].update(status="passed")
    records[-2].update(status="error", error="adapter out of memory")
    cell = case_table(records, ["mujoco", "rapier"], timing_value)["rows"][1]["cells"]["rapier"]
    assert (cell["passed"], cell["total"], cell["failure"], cell["best"]) == (1, 2, "OOM", True)


def test_missing_and_duplicate_trials_are_ineligible():
    suite = dict(
        engines=["mujoco", "rapier"],
        scenarios=["box_pile"],
        sizes=[4, 8],
        worlds=[1],
        seeds=[42],
        repeats=2,
    )
    records = [
        dict(
            engine="mujoco",
            backend="cpu",
            scenario="box_pile",
            size=4,
            worlds=1,
            status="passed",
            ms_per_step=1,
            seed=42,
            repeat=0,
        )
    ] * 2
    rows = summarize(records, suite)
    assert len(rows) == 4
    assert not any(r["eligible"] for r in rows)


def test_gpu_statistics_require_complete_consistent_coverage():
    import copy

    records = [
        dict(
            engine="nexus",
            backend="metal",
            scenario="box_pile",
            size=4,
            worlds=1,
            seed=42,
            repeat=i,
            status="passed",
            ms_per_step=10 + i * 2,
            gpu_ms_per_step=1 + i * 2,
            gpu_timing=dict(status="available", method="metal", scope="passes"),
        )
        for i in range(2)
    ]
    row = summarize(records)[0]
    assert row["median_ms"] == 11 and row["gpu_median_ms"] == 2
    assert row["eligible"] and row["gpu_eligible"]
    partial = copy.deepcopy(records)
    partial[1]["gpu_timing"] = dict(status="unavailable")
    partial[1].pop("gpu_ms_per_step")
    row = summarize(partial)[0]
    assert row["eligible"] and not row["gpu_eligible"]
    assert row["gpu_available"] == 1
    records[1]["gpu_timing"]["scope"] = "different"
    assert not summarize(records)[0]["gpu_eligible"]
    records[1]["gpu_ms_per_step"] = float("nan")
    assert not summarize(records)[0]["gpu_eligible"]
    legacy = [
        dict(
            engine="mujoco",
            backend="cpu",
            scenario="box_pile",
            size=4,
            worlds=1,
            status="passed",
            ms_per_step=1,
        )
    ]
    assert summarize(legacy)[0]["gpu_median_ms"] is None


def test_trace_times_exclude_host_and_union_overlapping_gpu_intervals():
    from gpu_timing import trace_device_time

    def event(cat, start, duration, device=0):
        return dict(ph="X", cat=cat, ts=start, dur=duration, args=dict(device=device))

    trace = dict(
        displayTimeUnit="ns",
        traceEvents=[
            event("cpu_op", 0, 10_000),
            event("cuda_runtime", 0, 50_000),
            event("kernel", 10, 10),
            event("gpu_memcpy", 15, 10),
            event("gpu_memset", 30, 5),
            event("kernel", 0, 1000, device=1),
        ],
    )
    # Union is [10,25] + [30,35] = 20 us, not 25 us, regardless of displayTimeUnit.
    seconds, count = trace_device_time(trace, "kineto")
    assert seconds == 20 / 1_000_000 and count == 3
    trace["traceEvents"] = [event("cpu_op", 0, 100)]
    assert trace_device_time(trace, "kineto") is None


def test_xla_trace_requires_gpu_device_and_stream_metadata():
    from gpu_timing import trace_device_time

    def metadata(name, pid, tid, label):
        return dict(ph="M", name=name, pid=pid, tid=tid, args=dict(name=label))

    trace = dict(
        traceEvents=[
            metadata("process_name", 10, 0, "/device:GPU:0"),
            metadata("process_name", 20, 0, "/host:CPU"),
            metadata("thread_name", 10, 1, "Stream #7 (Kernel)"),
            metadata("thread_name", 10, 2, "XLA Modules"),
            metadata("thread_name", 20, 1, "Stream #7"),
            dict(ph="X", pid=10, tid=1, ts=100, dur=25),
            dict(ph="X", pid=10, tid=2, ts=0, dur=10000),
            dict(ph="X", pid=20, tid=1, ts=0, dur=10000),
        ]
    )
    assert trace_device_time(trace, "xla") == (25 / 1_000_000, 1)
    assert trace_device_time(trace, "xla", device=1) is None


def test_gpu_preflight_never_labels_cpu_or_genesis_metal_as_device_time():
    from gpu_timing import preflight

    for engine, backend in [("mjx", "cpu"), ("genesis", "metal")]:
        result = preflight(dict(engine=engine, backend=backend))
        assert result["seconds"] is None and result["sample_count"] == 0
        assert result["reason"]
    assert preflight(dict(engine="mjx", backend="cuda")) is None


def test_memory_statistics_require_complete_matching_replays():
    import copy

    records = [
        dict(
            engine="nexus",
            backend="metal",
            scenario="stacks",
            size=8,
            worlds=1,
            status="passed",
            ms_per_step=1,
            steps=4,
            memory=dict(
                status="complete",
                completed_steps=4,
                metrics={
                    key: dict(
                        status="available",
                        method=key,
                        scope="process",
                        peak_kind="sampled" if key == "gpu" else "os_high_water",
                        peak_bytes=1024 * (i + 1),
                    )
                    for key in ["ram", "gpu"]
                },
            ),
        )
        for i in range(2)
    ]
    row = summarize(records)[0]
    assert row["memory"]["ram"]["eligible"]
    assert row["memory"]["gpu"]["median_bytes"] == 1536
    for mutate in [
        lambda r: r.pop("memory"),
        lambda r: r["memory"].update(status="error"),
        lambda r: r["memory"].update(completed_steps=3),
        lambda r: r["memory"]["metrics"]["gpu"].update(
            status="unavailable", peak_bytes=None
        ),
        lambda r: r["memory"]["metrics"]["gpu"].update(peak_bytes=0),
        lambda r: r["memory"]["metrics"]["gpu"].update(scope="another_counter"),
        lambda r: r["memory"]["metrics"]["gpu"].update(peak_kind="os_high_water"),
    ]:
        partial = copy.deepcopy(records)
        mutate(partial[1])
        row = summarize(partial)[0]
        assert row["eligible"]  # memory failures cannot rewrite primary timings
        assert not row["memory"]["gpu"]["eligible"]
    legacy = [{k: v for k, v in record.items() if k != "memory"} for record in records]
    assert summarize(legacy)[0]["memory"]["ram"]["median_bytes"] is None


def test_shared_native_memory_sampler_observes_real_metal_allocation():
    import ctypes as ct
    import pytest
    from memory_usage import MemorySampler

    if sys.platform != "darwin":
        pytest.skip("Metal allocation test requires macOS")
    metal = ct.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    objc = ct.CDLL("/usr/lib/libobjc.A.dylib")
    metal.MTLCreateSystemDefaultDevice.restype = ct.c_void_p
    objc.sel_registerName.argtypes = [ct.c_char_p]
    objc.sel_registerName.restype = ct.c_void_p
    allocate = ct.CFUNCTYPE(
        ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint64, ct.c_uint64
    )(("objc_msgSend", objc))
    release = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_void_p)(("objc_msgSend", objc))
    sampler = MemorySampler("metal")
    device = metal.MTLCreateSystemDefaultDevice()
    buffer = allocate(
        device, objc.sel_registerName(b"newBufferWithLength:options:"), 64 * 2**20, 0
    )
    assert buffer
    try:
        sampler.sample("setup")
        sampler.sample("warmup")
        sampler.sample("reset")
        result = sampler.finish(1)
        baseline, setup = result["snapshots"][:2]
        assert setup["bytes"]["gpu"] - baseline["bytes"]["gpu"] >= 64 * 2**20
        assert result["metrics"]["gpu"]["peak_bytes"] >= setup["bytes"]["gpu"]
        assert result["metrics"]["ram"]["peak_bytes"] > 0
        assert result["metrics"]["footprint"]["peak_bytes"] > 0
        assert result["unified_memory"] is True
    finally:
        release(buffer, objc.sel_registerName(b"release"))
        release(device, objc.sel_registerName(b"release"))
