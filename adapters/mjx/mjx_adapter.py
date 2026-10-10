#!/usr/bin/env python3
"""MuJoCo MJX (JAX implementation), JIT-compiled and vmapped across worlds."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mujoco"))
from physmark_adapter import main, mjcf  # noqa: E402
from mujoco_adapter import Mujoco  # noqa: E402


class MJX(Mujoco):
    def __init__(self, job, xml):
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        cpu_job = {**job, "backend": "cpu"}
        super().__init__(cpu_job, xml)
        self.job = job
        self.jax, self.jnp, self.mjx = jax, jnp, mjx
        platform = {"cpu": "cpu", "cuda": "gpu", "metal": "METAL"}.get(job["backend"])
        if platform is None:
            raise ValueError("unsupported MJX backend")
        devices = jax.devices(platform)
        self.device = devices[0]
        with jax.default_device(self.device):
            self.mx = mjx.put_model(self.model, impl="jax")
            initial = mjx.put_data(self.model, self.data[0], impl="jax")
            self.initial = jax.tree.map(
                lambda x: jnp.broadcast_to(x, (job["worlds"],) + x.shape), initial
            )
            self.d = self.initial
            self.step_batch = jax.vmap(lambda d: mjx.step(self.mx, d))
            self.functions = {}
            # Explicitly compile every block length before timing or capture.
            for n in {
                job["steps"],
                job["warmup"],
                job["capture_stride"],
                job["steps"] % job["capture_stride"],
                1,
            }:
                if n:
                    fn = jax.jit(
                        lambda d, n=n: jax.lax.fori_loop(
                            0, n, lambda _, state: self.step_batch(state), d
                        )
                    )
                    self.functions[n] = fn.lower(self.initial).compile()
            self.forward = jax.jit(jax.vmap(lambda d: mjx.kinematics(self.mx, d)))
        self.settings.update(
            precision="f32",
            implementation="mjx-jax",
            jax_version=jax.__version__,
            device=str(self.device),
            world_execution="vmap",
            time_execution="compiled_fori_loop",
        )

    def reset(self):
        if hasattr(self, "initial"):
            self.d = self.initial
        else:
            super().reset()

    def advance(self, steps):
        if steps:
            self.d = self.functions[steps](self.d)

    def sync(self):
        self.jax.block_until_ready(self.d)

    def poses(self):
        d = self.jax.device_get(self.forward(self.d))
        return [
            list(d.xpos[e, i]) + list(d.xquat[e, i, 1:]) + [float(d.xquat[e, i, 0])]
            for e in range(self.job["worlds"])
            for i in self.ids
        ]

    def gpu_profile(self, job, directory):
        """Union of device activity in the XLA trace (CUPTI) around the measured steps."""
        import gzip
        import json

        import jax
        from gpu_timing import trace_profile, unavailable

        trace_dir = directory / "gpu-trace"
        with jax.profiler.trace(str(trace_dir)):
            self.advance(job["steps"])
            self.sync()
        paths = list(trace_dir.rglob("*.trace.json.gz"))
        if len(paths) != 1:
            return unavailable(f"Expected one XLA device trace, found {len(paths)}")
        with gzip.open(paths[0], "rt") as f:
            trace = json.load(f)
        return trace_profile(job, directory, trace, "xla", "jax_cupti_activity", paths[0])


if __name__ == "__main__":
    main(lambda job: MJX(job, mjcf(job["scene"])))
