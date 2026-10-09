"""MPM reports must use the chosen baseline and particle dimensions."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from report import leaderboard, summarize, curves, count_unit


def records():
    return [dict(engine=e, backend="metal", scenario="mpm_elastic_drop", size=n,
                 worlds=1, status="passed", ms_per_step=ms, seed=42, repeat=0)
            for n in (512, 4096) for e, ms in (("genesis", 2.), ("nexus", 1.))]


def test_mpm_ranks_against_explicit_baseline_and_requires_complete_coverage():
    rows = summarize(records())
    scores = {r["engine"]: r["score"] for r in leaderboard(rows, ["genesis", "nexus"], "genesis")}
    assert scores == pytest.approx({"genesis": 1., "nexus": 2.})
    assert all(r["score"] is None for r in leaderboard(rows, ["genesis", "nexus"]))
    broken = records(); broken[0]["status"] = "error"
    assert all(r["score"] is None for r in leaderboard(summarize(broken), ["genesis", "nexus"], "genesis"))


def test_mpm_curves_label_particles_and_generate_throughput(tmp_path):
    (tmp_path / "curves").mkdir()
    files = curves(summarize(records()), tmp_path, [])
    assert len(files) == 2
    svg = "\n".join(p.read_text() for p in (tmp_path / "curves").glob("*.svg"))
    assert "Particles per environment" in svg
    assert "Particle-steps / second" in svg
    assert "higher is better" in svg and "lower is better" in svg
    assert count_unit("mpm_sand_collapse") == "particles"
    assert count_unit("box_pile") == "bodies"
