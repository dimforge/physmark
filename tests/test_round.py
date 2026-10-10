"""Backend selection of the full round for each kind of machine."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from capabilities import candidates, nexus_feature  # noqa: E402


def machine(**overrides):
    caps = dict(system="Linux", machine="x86_64", metal=False, cuda=False, nvidia_gpus=[],
                isaac_wheels=True, isaac_eula_accepted=False)
    return {**caps, **overrides}


def test_apple_silicon_uses_metal_and_skips_cuda_engines():
    selected, skipped = candidates(machine(system="Darwin", machine="arm64", metal=True, isaac_wheels=False))
    assert selected == dict(rapier="cpu", mujoco="cpu", nexus="metal", genesis="metal")
    assert set(skipped) == {"newton", "mjx", "isaac"}


def test_nvidia_linux_runs_every_engine_on_cuda_once_the_eula_is_accepted():
    gpu = dict(cuda=True, nvidia_gpus=["NVIDIA RTX 6000, 580.1"])
    selected, skipped = candidates(machine(**gpu, isaac_eula_accepted=True))
    assert selected == dict(rapier="cpu", mujoco="cpu", nexus="cuda", genesis="cuda", newton="cuda", mjx="cuda", isaac="cuda")
    assert not skipped
    selected, skipped = candidates(machine(**gpu))
    assert "isaac" not in selected and "EULA" in skipped["isaac"]


def test_machine_without_cuda_or_metal_skips_cuda_runs():
    selected, skipped = candidates(machine())
    assert selected == dict(rapier="cpu", mujoco="cpu", nexus="webgpu", genesis="cpu")
    assert all("CUDA" in reason for reason in skipped.values())
    assert nexus_feature("webgpu") == "webgpu"


def test_isaac_needs_its_wheel_platforms_even_with_cuda():
    selected, skipped = candidates(machine(machine="ppc64le", cuda=True, isaac_wheels=False, isaac_eula_accepted=True))
    assert selected["newton"] == "cuda" and "isaac" not in selected
    assert "no wheels" in skipped["isaac"]
