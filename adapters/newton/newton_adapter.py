#!/usr/bin/env python3
"""Newton with SolverMuJoCo (MuJoCo Warp) for rigid bodies and SolverImplicitMPM for MPM."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from physmark_adapter import arena_cap_bytes, initial_velocities, is_arena_allocation_failure, main, mjcf  # noqa: E402


class Newton:
    def __init__(self, job, xml):
        import newton
        import numpy as np
        import warp as wp

        self.n, self.wp, self.job = newton, wp, job
        wp.init()
        wp.config.quiet = True
        device = "cpu" if job["backend"] == "cpu" else "cuda:0"
        if job["backend"] not in ("cpu", "cuda"):
            raise ValueError("Newton supports CPU or CUDA, not Metal")
        self.device = wp.get_device(device)
        robot = job["scene"].get("robot")
        with wp.ScopedDevice(self.device):
            single = newton.ModelBuilder(up_axis=newton.Axis.Z)
            if robot:
                # Robots keep the Menagerie model: its <option> (integrator, cone, impratio,
                # iterations) becomes per-world solver settings, and its position actuators run
                # natively from control.mujoco.ctrl, as in MuJoCo.
                newton.solvers.SolverMuJoCo.register_custom_attributes(single)
            single.add_mjcf(
                xml, ignore_inertial_definitions=False, collapse_fixed_joints=False,
                **({"ctrl_direct": True} if robot else {}),
            )
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            for i in range(job["worlds"]):
                builder.add_world(single, label_prefix=f"env{i}")
            self.model = builder.finalize(device=self.device)
            # SolverMuJoCo compiles its own MjSpec with MuJoCo's automatic arena, which cannot be
            # allocated on Linux from about 8,000 bodies; retry those with the shared cap.
            import mujoco

            compile_spec = mujoco.MjSpec.compile

            def compile_with_arena(spec, *args, **kwargs):
                try:
                    return compile_spec(spec, *args, **kwargs)
                except ValueError as error:
                    if spec.memory >= 0 or not is_arena_allocation_failure(error):
                        raise
                    spec.memory = arena_cap_bytes()
                    return compile_spec(spec, *args, **kwargs)

            mujoco.MjSpec.compile = compile_with_arena
            if robot:
                # MuJoCo Warp's automatic estimates (njmax and nconmax 64) overflow on the Go1 and
                # drop constraints. Standing up peaks at 60 (Go1) and 103 (T1) constraint rows in
                # MuJoCo; the narrowphase counts candidate contacts before filtering, which the
                # Go1's leg capsules exceed at 64. Keep ample headroom per world.
                options = dict(nconmax=512, njmax=256)
            else:
                options = dict(
                    integrator="euler",
                    solver="newton",
                    iterations=50,
                    nconmax=max(64, job["scene"]["size"] * 32),
                    njmax=max(128, job["scene"]["size"] * 128),
                )
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_cpu=False,
                use_mujoco_contacts=True,
                enable_sleeping=False,
                **options,
            )
            if self.solver.use_mujoco_cpu:
                raise RuntimeError("Newton unexpectedly selected native MuJoCo")
            if robot:
                # MuJoCo qpos stores the free joint's quaternion as wxyz; Newton uses xyzw.
                q = list(robot["qpos"])
                q[3:7] = [q[4], q[5], q[6], q[3]]
                per_world = self.model.joint_coord_count // job["worlds"]
                if per_world != len(q):
                    raise RuntimeError("Newton robot coordinates differ from the MuJoCo qpos")
                self.model.joint_q.assign(np.tile(np.asarray(q, dtype=np.float32), job["worlds"]))
            self.state0, self.state1 = self.model.state(), self.model.state()
            newton.eval_fk(
                self.model, self.model.joint_q, self.model.joint_qd, self.state0
            )
        names = self.model.body_label
        self.ids = []
        for env in range(job["worlds"]):
            for body in job["scene"]["bodies"]:
                if body["dynamic"]:
                    link = robot["body_names"][body["id"] - 1] if robot else f"body_{body['id']}"
                    wanted = f"env{env}/{link}"
                    matches = [
                        i
                        for i, name in enumerate(names)
                        if name.startswith(f"env{env}/")
                        and name.endswith(f"/{link}")
                    ]
                    if len(matches) != 1:
                        raise ValueError(f"could not map Newton body {wanted}: {names}")
                    self.ids.append(matches[0])
        with wp.ScopedDevice(self.device):
            moving = initial_velocities(job["scene"])
            if moving:
                # Body twists are (COM velocity, angular velocity), both in world frame.
                index = {b["id"]: k for k, b in enumerate(b for b in job["scene"]["bodies"] if b["dynamic"])}
                per_world = len(index)
                qd = self.state0.body_qd.numpy()
                for env in range(job["worlds"]):
                    for b, v, w in moving:
                        qd[self.ids[env * per_world + index[b["id"]]]] = [*v, *w]
                self.state0.body_qd.assign(qd)
                newton.eval_ik(self.model, self.state0, self.state0.joint_q, self.state0.joint_qd)
                newton.eval_fk(self.model, self.state0.joint_q, self.state0.joint_qd, self.state0)
            self.initial_q = self.state0.body_q.numpy().copy()
            self.initial_qd = self.state0.body_qd.numpy().copy()
            self.initial_jq = self.state0.joint_q.numpy().copy()
            self.initial_jqd = self.state0.joint_qd.numpy().copy()
            self.control = self.model.control()
            if robot:
                self.control.mujoco.ctrl.assign(np.tile(np.asarray(robot["ctrl"], dtype=np.float32), job["worlds"]))
        self.settings = {
            "solver": "SolverMuJoCo",
            "use_mujoco_cpu": False,
            "collision_pipeline": "mujoco",
            "iterations": 50,
            "integrator": "Euler",
            "sleeping": False,
            "precision": "f32",
            "world_execution": "native_batch",
            "graph_capture": False,
            "newton_version": newton.__version__,
            "warp_version": wp.__version__,
            "device": str(self.device),
        }
        if robot:
            self.settings.update(
                iterations="MJCF <option>", integrator="MJCF <option>",
                actuators="MJCF position actuators (control.mujoco.ctrl)",
            )

    def reset(self):
        self.solver.reset(self.state0)
        for s in (self.state0, self.state1):
            s.body_q.assign(self.initial_q)
            s.body_qd.assign(self.initial_qd)
            s.joint_q.assign(self.initial_jq)
            s.joint_qd.assign(self.initial_jqd)

    def advance(self, steps):
        with self.wp.ScopedDevice(self.device):
            for _ in range(steps):
                self.state0.clear_forces()
                self.solver.step(
                    self.state0,
                    self.state1,
                    self.control,
                    None,
                    self.job["scene"]["dt"],
                )
                self.state0, self.state1 = self.state1, self.state0

    def sync(self):
        self.wp.synchronize_device(self.device)

    def poses(self):
        data = self.state0.body_q.numpy()
        return data[self.ids].tolist()

    def awake(self):
        if self.solver.enable_sleeping:
            raise RuntimeError("Newton sleeping unexpectedly enabled")
        return len(self.ids)

    def gpu_profile(self, job, directory):
        from warp_timing import warp_profile

        return warp_profile(self, job, directory)


def make(job):
    if job["scene"].get("mpm"):
        from newton_mpm import NewtonMPM

        return NewtonMPM(job)
    return Newton(job, mjcf(job["scene"]))


if __name__ == "__main__":
    main(make)
