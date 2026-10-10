#!/usr/bin/env python3
"""Isaac Sim adapter: PhysX 5 rigid bodies built directly as a USD stage, read through the
PhysX tensor API. Isaac Sim publishes Linux and Windows wheels only (NVIDIA GPU required).

Rigid primitive scenes and world-anchored hinges only; robots and MPM are not wired.
Isaac Sim requires its EULA to be accepted by the user (OMNI_KIT_ACCEPT_EULA=YES).
"""
from importlib import metadata
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from physmark_adapter import initial_velocities, main  # noqa: E402

POSITION_ITERATIONS = 4
VELOCITY_ITERATIONS = 1


def quat_mul(a, b):
    """XYZW quaternion product a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def x_to(axis):
    """XYZW quaternion rotating +X onto the unit vector `axis` (USD joints act along X)."""
    n = math.sqrt(sum(a * a for a in axis))
    a = [v / n for v in axis]
    dot = a[0]
    if dot < -1 + 1e-12:
        return [0.0, 0.0, 1.0, 0.0]
    cross = [0.0, -a[2], a[1]]
    q = [*cross, 1 + dot]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


class Isaac:
    def __init__(self, job):
        scene = job["scene"]
        if scene.get("robot") or scene.get("mpm"):
            raise ValueError("the Isaac adapter supports rigid primitive scenes only")
        if job["backend"] not in ("cuda", "cpu"):
            raise ValueError("Isaac Sim supports backend=cuda or backend=cpu")
        if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in ("YES", "Y", "1", "TRUE"):
            # Kit would otherwise wait for an interactive answer until the trial times out.
            raise RuntimeError("accept the NVIDIA Isaac Sim EULA yourself first: export OMNI_KIT_ACCEPT_EULA=YES")
        self.job, self.gpu = job, job["backend"] == "cuda"
        from isaacsim import SimulationApp

        self.app = SimulationApp({"headless": True})
        import carb
        import omni.physx
        import omni.physics.tensors as tensors
        import omni.usd
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

        settings = carb.settings.get_settings()
        settings.set_int("/physics/numThreads", job["threads"])
        # Keep state on the device and out of USD: only the tensor API reads poses.
        settings.set_bool("/physics/updateToUsd", False)
        settings.set_bool("/physics/updateVelocitiesToUsd", False)
        settings.set_bool("/physics/suppressReadback", self.gpu)
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

        gravity = scene["gravity"]
        magnitude = math.sqrt(sum(g * g for g in gravity))
        physics = UsdPhysics.Scene.Define(stage, "/physicsScene")
        physics.CreateGravityDirectionAttr(Gf.Vec3f(*[g / magnitude for g in gravity]))
        physics.CreateGravityMagnitudeAttr(magnitude)
        physx = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
        physx.CreateEnableGPUDynamicsAttr(self.gpu)
        physx.CreateBroadphaseTypeAttr("GPU" if self.gpu else "MBP")
        physx.CreateSolverTypeAttr("TGS")
        physx.CreateTimeStepsPerSecondAttr(1 / scene["dt"])
        physx.CreateEnableStabilizationAttr(False)
        physx.CreateEnableCCDAttr(False)
        # GPU buffers are fixed at start; size them for the whole batch.
        total = job["worlds"] * len(scene["bodies"])
        for name, per_body, default in [
            ("GpuMaxRigidContactCount", 16, 2**19),
            ("GpuMaxRigidPatchCount", 8, 2**18),
            ("GpuFoundLostPairsCapacity", 8, 2**18),
            ("GpuFoundLostAggregatePairsCapacity", 8, 2**18),
            ("GpuTotalAggregatePairsCapacity", 8, 2**18),
        ]:
            getattr(physx, f"Create{name}Attr")(max(default, per_body * total))

        materials = {}

        def material(friction):
            if friction not in materials:
                path = f"/World/materials/friction_{len(materials)}"
                shade = UsdShade.Material.Define(stage, path)
                api = UsdPhysics.MaterialAPI.Apply(shade.GetPrim())
                api.CreateStaticFrictionAttr(friction)
                api.CreateDynamicFrictionAttr(friction)
                api.CreateRestitutionAttr(0.0)
                materials[friction] = shade
            return materials[friction]

        # Independent environments sit side by side on a grid; spacing keeps their grounds apart.
        extent = max(
            max(abs(b["position"][0]), abs(b["position"][1])) + max(b["size"])
            for b in scene["bodies"]
        )
        spacing = 2 * extent + 1
        columns = math.ceil(math.sqrt(job["worlds"]))
        self.offsets = [
            [(e % columns) * spacing, (e // columns) * spacing, 0.0] for e in range(job["worlds"])
        ]
        hinges = {h["body"]: h for h in scene["hinges"]}
        paths = []
        for e, offset in enumerate(self.offsets):
            for b in scene["bodies"]:
                path = f"/World/envs/env_{e}/body_{b['id']}"
                xform = UsdGeom.Xform.Define(stage, path)
                q = b["rotation"]
                xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Vec3d(*[p + o for p, o in zip(b["position"], offset)])
                )
                xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Quatd(q[3], q[0], q[1], q[2])
                )
                s = b["size"]
                if b["shape"] == "sphere":
                    geom = UsdGeom.Sphere.Define(stage, path + "/collider")
                    geom.CreateRadiusAttr(s[0])
                elif b["shape"] == "capsule":
                    geom = UsdGeom.Capsule.Define(stage, path + "/collider")
                    geom.CreateRadiusAttr(s[0])
                    geom.CreateHeightAttr(2 * s[2])
                    geom.CreateAxisAttr("Z")
                else:
                    geom = UsdGeom.Cube.Define(stage, path + "/collider")
                    geom.CreateSizeAttr(2.0)
                    geom.AddScaleOp().Set(Gf.Vec3f(*s))
                UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
                UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(
                    material(b["friction"]), UsdShade.Tokens.weakerThanDescendants, "physics"
                )
                if not b["dynamic"]:
                    continue
                prim = xform.GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(prim)
                mass = UsdPhysics.MassAPI.Apply(prim)
                mass.CreateMassAttr(b["mass"])
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*b["inertia"]))
                mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
                mass.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
                body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                # A zero threshold never puts bodies to sleep.
                body.CreateSleepThresholdAttr(0.0)
                body.CreateStabilizationThresholdAttr(0.0)
                body.CreateLinearDampingAttr(0.0)
                body.CreateAngularDampingAttr(0.0)
                body.CreateMaxAngularVelocityAttr(1e7)
                body.CreateSolverPositionIterationCountAttr(POSITION_ITERATIONS)
                body.CreateSolverVelocityIterationCountAttr(VELOCITY_ITERATIONS)
                paths.append(path)
                if b["id"] in hinges:
                    h = hinges[b["id"]]
                    joint = UsdPhysics.RevoluteJoint.Define(stage, path + "/hinge")
                    frame = x_to(h["axis"])
                    inverse = [-q[0], -q[1], -q[2], q[3]]
                    local = quat_mul(inverse, frame)
                    joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
                    joint.CreateAxisAttr("X")
                    joint.CreateLocalPos0Attr(Gf.Vec3f(*[a + o for a, o in zip(h["anchor"], offset)]))
                    joint.CreateLocalRot0Attr(Gf.Quatf(frame[3], *frame[:3]))
                    joint.CreateLocalPos1Attr(Gf.Vec3f(*h["local_anchor"]))
                    joint.CreateLocalRot1Attr(Gf.Quatf(local[3], *local[:3]))
        self.paths = paths

        self.physx = omni.physx.get_physx_interface()
        self.stepper = omni.physx.get_physx_simulation_interface()
        self.physx.force_load_physics_from_usd()
        self.physx.start_simulation()
        # One untimed step creates the GPU state the tensor views attach to.
        self.stepper.simulate(scene["dt"], 0.0)
        self.stepper.fetch_results()
        import torch

        self.torch = torch
        self.device = "cuda:0" if self.gpu else "cpu"
        self.view = tensors.create_simulation_view("torch")
        self.view.set_subspace_roots("/")
        self.bodies = self.view.create_rigid_body_view(paths)
        order = {p: i for i, p in enumerate(self.bodies.prim_paths)}
        if sorted(order) != sorted(paths):
            raise RuntimeError("Isaac rigid body view does not cover the scene")
        # Tensor rows follow the view; map them back to scene order.
        self.rows = torch.tensor([order[p] for p in paths], device=self.device)
        self.indices = torch.arange(len(paths), dtype=torch.int32, device=self.device)
        self.offset_rows = torch.tensor(
            [o for o in self.offsets for b in scene["bodies"] if b["dynamic"]],
            dtype=torch.float32,
            device=self.device,
        )
        self.initial_poses, self.initial_velocities = self.canonical_state()
        self.reset()
        self.time = 0.0
        self.settings = {
            "solver": "PhysX TGS",
            "position_iterations": POSITION_ITERATIONS,
            "velocity_iterations": VELOCITY_ITERATIONS,
            "precision": "f32",
            "sleeping": False,
            "sleep_threshold": 0.0,
            "stabilization": False,
            "ccd": False,
            "broadphase": "GPU" if self.gpu else "MBP",
            "gpu_dynamics": self.gpu,
            "world_execution": "native_batch (side-by-side environments on a grid)",
            "environment_spacing_m": spacing,
            "isaacsim_version": metadata.version("isaacsim"),
            "device": self.device,
            "threads": job["threads"],
        }

    def canonical_state(self):
        """Scene-order initial poses (with environment offsets) and twists."""
        scene = self.job["scene"]
        poses, twists = [], []
        moving = {b["id"]: (v, w) for b, v, w in initial_velocities(scene)}
        for offset in self.offsets:
            for b in scene["bodies"]:
                if b["dynamic"]:
                    poses.append([*[p + o for p, o in zip(b["position"], offset)], *b["rotation"]])
                    v, w = moving.get(b["id"], ([0, 0, 0], [0, 0, 0]))
                    twists.append([*v, *w])
        as_tensor = lambda x: self.torch.tensor(x, dtype=self.torch.float32, device=self.device)
        return as_tensor(poses), as_tensor(twists)

    def reset(self):
        # Rows are written in view order; the scene-order tensors are scattered into it.
        poses = self.torch.empty_like(self.initial_poses)
        twists = self.torch.empty_like(self.initial_velocities)
        poses[self.rows] = self.initial_poses
        twists[self.rows] = self.initial_velocities
        self.bodies.set_transforms(poses, self.indices)
        self.bodies.set_velocities(twists, self.indices)

    def advance(self, steps):
        dt = self.job["scene"]["dt"]
        for _ in range(steps):
            self.stepper.simulate(dt, self.time)
            self.stepper.fetch_results()
            self.time += dt

    def sync(self):
        if self.gpu:
            self.torch.cuda.synchronize()

    def poses(self):
        p = self.bodies.get_transforms()[self.rows].clone()
        p[:, :3] -= self.offset_rows
        return p.cpu().double().tolist()

    def awake(self):
        # Sleeping is disabled through a zero sleep threshold on every body.
        return len(self.paths)

    def close(self):
        self.app.close()

    def gpu_profile(self, job, directory):
        from gpu_timing import torch_cupti_profile

        return torch_cupti_profile(self, job, directory)


if __name__ == "__main__":
    main(Isaac)
