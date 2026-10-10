#!/usr/bin/env python3
"""Genesis rigid and MPM solvers on its Quadrants backend."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from physmark_adapter import initial_velocities, main, mjcf, to_body_frame  # noqa: E402


class Genesis:
    def __init__(self, job, xml):
        import genesis as gs

        self.gs, self.job = gs, job
        backend = {"cpu": gs.cpu, "metal": gs.metal, "cuda": gs.cuda}.get(
            job["backend"]
        )
        if backend is None:
            raise ValueError("unsupported Genesis backend")
        gs.init(
            backend=backend,
            precision="32",
            logging_level="warning",
            seed=job["scene"]["seed"] % (2**31),
            performance_mode=True,
        )
        if gs.backend != backend:
            raise RuntimeError(f"Genesis selected {gs.backend} instead of {backend}")
        self.tmp = tempfile.TemporaryDirectory(prefix="physmark-genesis-")
        path = Path(self.tmp.name) / "scene.xml"
        path.write_text(xml)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=job["scene"]["dt"],
                substeps=1,
                gravity=tuple(job["scene"]["gravity"]),
            ),
            rigid_options=gs.options.RigidOptions(use_hibernation=False, iterations=50,
                **(dict(friction_cone=gs.friction_cone.elliptic,
                        enable_torsional_friction=True, enable_rolling_friction=True)
                   if job["scene"].get("robot") else
                   # The default (150 pairs) overflows on brick walls; resting bricks touch about four others.
                   dict(max_collision_pairs=max(150, 4 * job["scene"]["size"])))),
            show_viewer=False,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        )
        self.entity = self.scene.add_entity(
            gs.morphs.MJCF(file=str(path)), material=gs.materials.Rigid()
        )
        self.scene.build(n_envs=job["worlds"])
        if job["scene"].get("robot"):
            import numpy as np
            config = job["scene"]["robot"]
            self.entity.set_qpos(np.tile(config["qpos"], (job["worlds"], 1)))
            self.robot_dofs = [self.entity.get_joint(n).dofs_idx_local[0] for n in config["joint_names"]]
            self.entity.control_dofs_position(np.tile(config["ctrl"], (job["worlds"], 1)), self.robot_dofs)
        moving = initial_velocities(job["scene"])
        if moving:
            import numpy as np
            dofs, velocity = [], []
            for b, v, w in moving:
                # Free joint DoFs: world-frame linear, body-frame angular.
                joint = self.entity.get_joint(f"joint_{b['id']}")
                if len(joint.dofs_idx_local) != 6:
                    raise ValueError("initial velocities require a free body")
                dofs += list(joint.dofs_idx_local)
                velocity += [*v, *to_body_frame(b["rotation"], w)]
            self.entity.set_dofs_velocity(np.tile(velocity, (job["worlds"], 1)), dofs)
        self.initial = self.scene.get_state()
        self.ids = [
            self.entity.get_link(job["scene"]["robot"]["body_names"][b["id"]-1] if job["scene"].get("robot") else f"body_{b['id']}").idx_local
            for b in job["scene"]["bodies"]
            if b["dynamic"]
        ]
        self.settings = {
            "solver": "Genesis rigid",
            "iterations": 50,
            "sleeping": False,
            "use_hibernation": False,
            "precision": "f32",
            "world_execution": "native_batch",
            "performance_mode": True,
            "genesis_version": gs.__version__,
            "device": str(gs.device),
            "actual_backend": str(gs.backend),
            "rigid_options": self.scene.options.rigid.model_dump(mode="json"),
        }

    def reset(self):
        self.scene.reset(self.initial)
        if self.job["scene"].get("robot"):
            import numpy as np
            self.entity.control_dofs_position(np.tile(self.job["scene"]["robot"]["ctrl"], (self.job["worlds"], 1)), self.robot_dofs)

    def advance(self, steps):
        for _ in range(steps):
            self.scene.step()

    def sync(self):
        import quadrants as qd

        qd.sync()

    def poses(self):
        p = self.entity.get_links_pos(self.ids).cpu().numpy()
        q = self.entity.get_links_quat(self.ids).cpu().numpy()
        if p.ndim == 2:
            p, q = p[None, ...], q[None, ...]
        return [
            list(p[e, i]) + list(q[e, i, 1:]) + [float(q[e, i, 0])]
            for e in range(self.job["worlds"])
            for i in range(len(self.ids))
        ]

    def awake(self):
        if self.scene.options.rigid.use_hibernation:
            raise ValueError("Genesis hibernation unexpectedly enabled")
        return len(self.ids) * self.job["worlds"]

    def gpu_profile(self, job, directory):
        from gpu_timing import torch_cupti_profile

        return torch_cupti_profile(self, job, directory)


def make(job):
    if job["scene"].get("mpm"):
        from genesis_mpm import GenesisMPM

        return GenesisMPM(job)
    return Genesis(job, mjcf(job["scene"]))


if __name__ == "__main__":
    main(make)
