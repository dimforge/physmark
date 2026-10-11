"""MPM reports must use the chosen baseline and particle dimensions."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from report import case_table, memory_tables, startup_value, timing_value, summarize, curves, count_unit


def records():
    return [dict(engine=e, backend="metal", scenario="mpm_elastic_drop", size=n,
                 worlds=1, status="passed", ms_per_step=ms, seed=42, repeat=0)
            for n in (512, 4096) for e, ms in (("genesis", 2.), ("nexus", 1.))]


def test_tables_use_fixed_order_and_mark_the_best_engine_per_device_group():
    rs = records() + [dict(r, engine="rapier", backend="cpu", ms_per_step=5.) for r in records()[::2]]
    table = case_table(rs, ["genesis", "rapier", "nexus"], timing_value)
    assert table["engines"] == ["rapier", "nexus", "genesis"]
    assert not table["batching"]
    for row in table["rows"]:
        assert row["cells"]["rapier"]["best"] and row["cells"]["nexus"]["best"] and not row["cells"]["genesis"]["best"]
    for r in rs:
        r["memory"] = dict(status="complete", metrics=dict(gpu=dict(status="available", peak_bytes=2**20 * (1 if r["engine"] == "genesis" else 2))))
    tables = memory_tables(rs, ["genesis", "rapier", "nexus"])
    assert list(tables) == ["gpu"] and tables["gpu"]["engines"] == ["nexus", "genesis"]
    assert all(row["cells"]["genesis"]["best"] for row in tables["gpu"]["rows"])


def test_mpm_curves_label_particles_and_generate_throughput(tmp_path):
    (tmp_path / "curves").mkdir()
    files = curves(summarize(records()), tmp_path, [])
    assert len(files) == 2
    assert len(list((tmp_path / "curves").glob("*.svg"))) == 4
    svg = "\n".join(p.read_text() for p in (tmp_path / "curves").glob("*.svg"))
    assert "Particles per environment" in svg
    assert "Particle-steps / second" in svg
    assert "higher is better" in svg and "lower is better" in svg
    assert count_unit("mpm_sand_collapse") == "particles"
    assert count_unit("box_pile") == "bodies"


def test_startup_table_sums_setup_and_warmup():
    rs = [dict(r, setup_seconds=1., warmup_seconds=2. if r["engine"] == "nexus" else 5.) for r in records()]
    rs[0]["warmup_seconds"] = None
    table = case_table(rs, ["genesis", "nexus"], startup_value)
    first = table["rows"][0]["cells"]
    assert first["genesis"]["median"] is None and first["genesis"]["failure"] == "not measured"
    assert first["nexus"]["median"] == 3. and first["nexus"]["best"]
