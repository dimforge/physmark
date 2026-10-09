#!/usr/bin/env python3
"""Engine adapters. The Rust controller never participates in the timed step loop."""

import json
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


def numbers(v):
    return " ".join(format(float(x), ".17g") for x in v)


def mjcf(scene):
    if scene.get("robot"):
        return scene["robot"]["xml"]
    root = ET.Element("mujoco", model="physmark")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    option = ET.SubElement(
        root,
        "option",
        timestep=str(scene["dt"]),
        gravity=numbers(scene["gravity"]),
        integrator="Euler",
        solver="Newton",
        iterations="50",
        tolerance="1e-8",
    )
    ET.SubElement(option, "flag", sleep="disable")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0", armature="0", frictionloss="0")
    ET.SubElement(
        default,
        "geom",
        condim="3",
        friction="0.5 0 0",
        margin="0",
        gap="0",
        solref="0.02 1",
    )
    world = ET.SubElement(root, "worldbody")
    hinges = {j["body"]: j for j in scene["hinges"]}
    for b in scene["bodies"]:
        q = b["rotation"]
        body = ET.SubElement(
            world,
            "body",
            name=f"body_{b['id']}",
            pos=numbers(b["position"]),
            quat=numbers([q[3], *q[:3]]),
        )
        if b["dynamic"]:
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass=str(b["mass"]),
                diaginertia=numbers(b["inertia"]),
            )
            if b["id"] in hinges:
                j = hinges[b["id"]]
                ET.SubElement(
                    body,
                    "joint",
                    name=f"joint_{b['id']}",
                    type="hinge",
                    pos=numbers(j["local_anchor"]),
                    axis=numbers(j["axis"]),
                    limited="false",
                )
            else:
                ET.SubElement(body, "freejoint", name=f"joint_{b['id']}")
        shape = b["shape"]
        size = (
            b["size"]
            if shape == "box"
            else ([b["size"][0]] if shape == "sphere" else [b["size"][0], b["size"][2]])
        )
        ET.SubElement(
            body,
            "geom",
            name=f"geom_{b['id']}",
            type=shape,
            size=numbers(size),
            friction=numbers([b["friction"], 0, 0]),
        )
    return ET.tostring(root, encoding="unicode")


def robot_settings(job):
    r = job["scene"]["robot"]
    return {k: r[k] for k in ("model", "source_commit", "source_path", "initial_keyframe", "target_keyframe", "ctrl", "self_contacts")}


class Mujoco:
    def __init__(self, job, xml):
        import mujoco
        import numpy as np

        self.mj, self.np, self.job = mujoco, np, job
        if job["backend"] != "cpu":
            raise ValueError("native MuJoCo physics only supports backend=cpu")
        self.model = mujoco.MjModel.from_xml_string(xml)
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


class Newton:
    def __init__(self, job, xml):
        import newton
        import warp as wp

        self.n, self.wp, self.job = newton, wp, job
        wp.init()
        wp.config.quiet = True
        device = "cpu" if job["backend"] == "cpu" else "cuda:0"
        if job["backend"] not in ("cpu", "cuda"):
            raise ValueError("Newton supports CPU or CUDA, not Metal")
        self.device = wp.get_device(device)
        with wp.ScopedDevice(self.device):
            single = newton.ModelBuilder(up_axis=newton.Axis.Z)
            single.add_mjcf(
                xml, ignore_inertial_definitions=False, collapse_fixed_joints=False
            )
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            for i in range(job["worlds"]):
                builder.add_world(single, label_prefix=f"env{i}")
            self.model = builder.finalize(device=self.device)
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_cpu=False,
                use_mujoco_contacts=True,
                enable_sleeping=False,
                integrator="euler",
                solver="newton",
                iterations=50,
                nconmax=max(64, job["scene"]["size"] * 32),
                njmax=max(128, job["scene"]["size"] * 128),
            )
            if self.solver.use_mujoco_cpu:
                raise RuntimeError("Newton unexpectedly selected native MuJoCo")
            self.state0, self.state1 = self.model.state(), self.model.state()
            newton.eval_fk(
                self.model, self.model.joint_q, self.model.joint_qd, self.state0
            )
            self.initial_q = self.state0.body_q.numpy().copy()
            self.initial_qd = self.state0.body_qd.numpy().copy()
            self.initial_jq = self.state0.joint_q.numpy().copy()
            self.initial_jqd = self.state0.joint_qd.numpy().copy()
            self.control = self.model.control()
        names = self.model.body_label
        self.ids = []
        for env in range(job["worlds"]):
            for body in job["scene"]["bodies"]:
                if body["dynamic"]:
                    wanted = f"env{env}/body_{body['id']}"
                    matches = [
                        i
                        for i, name in enumerate(names)
                        if name.startswith(f"env{env}/")
                        and name.endswith(f"/body_{body['id']}")
                    ]
                    if len(matches) != 1:
                        raise ValueError(f"could not map Newton body {wanted}: {names}")
                    self.ids.append(matches[0])
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
                   if job["scene"].get("robot") else {})),
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


def run(job, evidence_dir=None):
    import numpy as np

    assert job["schema_version"] == 1
    gpu_timing = None
    if job["mode"] == "gpu_timing":
        from gpu_timing import preflight, profile

        gpu_timing = preflight(job)
        if gpu_timing is not None:
            # A capability result contains no simulation sample and is never ranked.
            return dict(
                schema_version=1,
                engine=job["engine"],
                backend=job["backend"],
                setup_seconds=0,
                warmup_seconds=0,
                elapsed_seconds=0,
                final_poses=[],
                awake_count=0,
                settings={},
                frames=[],
                gpu_timing=gpu_timing,
            )
    memory = None
    if job["mode"] == "memory":
        from memory_usage import MemorySampler

        memory = MemorySampler(job["backend"])
    start = time.perf_counter()
    if job["scene"].get("mpm"):
        from mpm import GenesisMPM, NewtonMPM
        if job["worlds"] != 1:
            raise ValueError("MPM benchmarks require exactly one environment")
        sim = {"genesis": GenesisMPM, "newton": NewtonMPM}[job["engine"]](job)
    else:
        xml = mjcf(job["scene"])
        sim = {"mujoco": Mujoco, "mjx": MJX, "newton": Newton, "genesis": Genesis}[
            job["engine"]
        ](job, xml)
    if job["scene"].get("robot"):
        sim.settings["robot"] = robot_settings(job)
        initial_poses = np.asarray(sim.poses()).reshape(job["worlds"], -1, 7)
        expected = np.asarray([b["position"] for b in job["scene"]["bodies"] if b["dynamic"]])
        if not np.allclose(initial_poses[:, :, :3], expected, atol=1e-4):
            raise RuntimeError("MJCF importer initial link poses differ from canonical crouch")
    sim.sync()
    setup = time.perf_counter() - start
    if memory:
        memory.sample("setup")
    sim.settings["warmup_steps_effective"] = max(job["warmup"], job["steps"])
    start = time.perf_counter()
    sim.advance(max(job["warmup"], job["steps"]))
    sim.sync()
    warmup = time.perf_counter() - start
    if memory:
        memory.sample("warmup")
    sim.reset()
    sim.sync()
    if memory:
        memory.sample("reset")
    frames = []
    nb = sum(b["dynamic"] for b in job["scene"]["bodies"])
    if job["mode"] == "capture":
        frames.append(
            {"step": 0, "poses": np.asarray(sim.poses()[:nb], dtype=float).tolist()}
        )
        start = time.perf_counter()
        for step in range(0, job["steps"], job["capture_stride"]):
            n = min(job["capture_stride"], job["steps"] - step)
            sim.advance(n)
            sim.sync()
            frames.append(
                {
                    "step": step + n,
                    "poses": np.asarray(sim.poses()[:nb], dtype=float).tolist(),
                }
            )
        elapsed = time.perf_counter() - start
    elif job["mode"] == "gpu_timing":
        start = time.perf_counter()
        gpu_timing = profile(sim, job, evidence_dir)
        elapsed = time.perf_counter() - start
    else:
        start = time.perf_counter()
        sim.advance(job["steps"])
        sim.sync()
        elapsed = time.perf_counter() - start
    memory_result = memory.finish(job["steps"]) if memory else None
    return {
        "schema_version": 1,
        "engine": job["engine"],
        "backend": job["backend"],
        "setup_seconds": setup,
        "warmup_seconds": warmup,
        "elapsed_seconds": elapsed,
        "final_poses": np.asarray(sim.poses(), dtype=float).tolist(),
        "awake_count": sim.awake(),
        "settings": sim.settings,
        "frames": frames,
        "gpu_timing": gpu_timing,
        "memory": memory_result,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: adapter.py JOB.json RESULT.json")
    job = json.loads(Path(sys.argv[1]).read_text())
    path = Path(sys.argv[2])
    result = run(job, path.parent)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False, indent=2))
    temporary.replace(path)
