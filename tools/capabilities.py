"""Hardware detection and per-engine backend selection, shared by the build and the full round.

Standard library only: this runs before the benchmark Python environment exists.
"""
import os
import platform
import shutil
import subprocess


def _output(*command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def nvidia_gpus():
    """Names of the NVIDIA GPUs the driver exposes; empty without a working driver."""
    if not shutil.which("nvidia-smi"):
        return []
    lines = _output("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
    return [line.strip() for line in lines.splitlines() if line.strip()]


def isaac_wheels():
    """NVIDIA publishes Isaac Sim wheels for Linux and Windows on x86-64 and aarch64 only."""
    return platform.system() in ("Linux", "Windows") and platform.machine().lower() in (
        "x86_64", "amd64", "aarch64", "arm64")


def memory_bytes():
    if platform.system() == "Darwin":
        value = _output("sysctl", "-n", "hw.memsize").strip()
        return int(value) if value.isdigit() else None
    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return None


def detect():
    system = platform.system()
    cpu = platform.processor() or platform.machine()
    if system == "Darwin":
        cpu = _output("sysctl", "-n", "machdep.cpu.brand_string").strip() or cpu
    gpus = nvidia_gpus()
    return dict(
        system=system,
        machine=platform.machine(),
        cpu=cpu,
        cpu_count=os.cpu_count() or 1,
        memory_bytes=memory_bytes(),
        metal=system == "Darwin",
        cuda=bool(gpus),
        nvidia_gpus=gpus,
        isaac_wheels=isaac_wheels(),
        isaac_eula_accepted=os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() in ("YES", "Y", "1", "TRUE"),
    )


def candidates(caps):
    """Backend per engine for this machine, plus the reason each excluded engine is skipped."""
    selected, skipped = {}, {}
    selected["rapier"] = "cpu"
    selected["mujoco"] = "cpu"
    if caps["metal"]:
        selected["nexus"] = selected["genesis"] = "metal"
    elif caps["cuda"]:
        selected["nexus"] = selected["genesis"] = "cuda"
    else:
        # Vulkan, DirectX 12 or Metal through wgpu; the probe drops Nexus if no adapter exists.
        selected["nexus"] = "webgpu"
        selected["genesis"] = "cpu"
    # Newton (MuJoCo Warp) and MJX are GPU engines here; their CPU paths are too slow for a full round.
    for engine in ("newton", "mjx"):
        if caps["cuda"]:
            selected[engine] = "cuda"
        else:
            skipped[engine] = "no NVIDIA CUDA GPU detected"
    if not caps["cuda"]:
        skipped["isaac"] = "no NVIDIA CUDA GPU detected"
    elif not caps["isaac_wheels"]:
        skipped["isaac"] = f"Isaac Sim has no wheels for {caps['system']} {caps['machine']}"
    elif not caps["isaac_eula_accepted"]:
        skipped["isaac"] = "Isaac Sim EULA not accepted (export OMNI_KIT_ACCEPT_EULA=YES after reading it)"
    else:
        selected["isaac"] = "cuda"
    return selected, skipped


def nexus_feature(backend):
    """Cargo feature of the Nexus adapter for a backend name."""
    return {"metal": "metal", "cuda": "cuda", "webgpu": "webgpu", "cpu": "cpu"}[backend]
