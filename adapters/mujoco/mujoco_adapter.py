#!/usr/bin/env python3
"""Native MuJoCo, built from the pinned source; worlds run serially on the CPU."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from physmark_adapter import compile_mjcf, main, mjcf, robot_settings, to_body_frame, initial_velocities  # noqa: E402


class Mujoco:
    def __init__(self, job, xml):
        import mujoco
        import numpy as np

        self.mj, self.np, self.job = mujoco, np, job
        if job["backend"] != "cpu":
            raise ValueError("native MuJoCo physics only supports backend=cpu")
        self.model = compile_mjcf(xml)
        if self.model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
            raise RuntimeError("MuJoCo sleeping must be disabled")
        self.ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                job["scene"]["robot"]["body_names"][b["id"]-1] if job["scene"].get("robot") else f"body_{b['id']}")
            for b in job["scene"]["bodies"]
            if b["dynamic"]
        ]
        self.data = [mujoco.MjData(self.model) for _ in range(job["worlds"])]
        self.settings = {
            "solver": "Newton",
            "iterations": 50,
            "integrator": "Euler",
            "precision": "f64",
            "sleeping": False,
            "world_execution": "serial",
            "mujoco_version": mujoco.__version__,
            "contact_solref": [0.02, 1],
            "friction_cone": "pyramidal",
        }
        if job["scene"].get("robot"):
            self.settings.update(robot=robot_settings(job), iterations=int(self.model.opt.iterations),
                friction_cone=str(self.model.opt.cone), contact_solref="Menagerie defaults")
        self.reset()

    def reset(self):
        for d in self.data:
            self.mj.mj_resetData(self.model, d)
            if self.job["scene"].get("robot"):
                d.qpos[:] = self.job["scene"]["robot"]["qpos"]
                d.ctrl[:] = self.job["scene"]["robot"]["ctrl"]
            for b, v, w in initial_velocities(self.job["scene"]):
                # Free joint velocities: world-frame linear, body-frame angular.
                joint = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, f"joint_{b['id']}")
                if self.model.jnt_type[joint] != self.mj.mjtJoint.mjJNT_FREE:
                    raise ValueError("initial velocities require a free body")
                adr = self.model.jnt_dofadr[joint]
                d.qvel[adr:adr + 3] = v
                d.qvel[adr + 3:adr + 6] = to_body_frame(b["rotation"], w)
            self.mj.mj_forward(self.model, d)

    def advance(self, steps):
        for d in self.data:
            self.mj.mj_step(self.model, d, nstep=steps)

    def sync(self):
        pass

    def poses(self):
        # mj_step's position-dependent fields describe the beginning of its last step.
        # Refresh kinematics outside timing, without advancing simulated time.
        for d in self.data:
            self.mj.mj_kinematics(self.model, d)
        return [
            list(d.xpos[i]) + list(d.xquat[i][1:]) + [float(d.xquat[i][0])]
            for d in self.data
            for i in self.ids
        ]

    def awake(self):
        return len(self.ids) * self.job["worlds"]


if __name__ == "__main__":
    main(lambda job: Mujoco(job, mjcf(job["scene"])))
