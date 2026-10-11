"""Isaac Sim robots: the job's MJCF converted to a PhysX articulation, one per environment.

Isaac Sim's MJCF importer (mujoco-usd-converter) writes the links, collision primitives,
joints, limits and actuator force ranges, but leaves the joint drives without gains. The drives
take MuJoCo's position-actuator law from the compiled model instead: stiffness kp, damping
kv plus the joint's own damping, the actuator force range as the drive's maximum force, the
joint armature, and the actuator targets clamped to their control range.
"""
from importlib import metadata
import math
from pathlib import Path
import tempfile

# Primitive scenes use 4. With 4 or 8, the T1 slowly tips over on its force-limited ankles
# (16° after 1 s, 8° after 2 s); 16 holds it upright (2° after 2 s), so robots use 16.
POSITION_ITERATIONS = 16
VELOCITY_ITERATIONS = 1
# Wide enough that neighbouring robots never touch, however they fall.
SPACING = 3.0


class IsaacRobot:
    def __init__(self, job):
        scene, robot = job["scene"], job["scene"]["robot"]
        self.job, self.gpu = job, job["backend"] == "cuda"
        from isaacsim import SimulationApp

        # Primitive scenes keep their lighter experience; robots also need the MJCF importer.
        self.app = SimulationApp({"headless": True}, experience=str(Path(__file__).with_name("physmark-robot.kit")))
        import carb
        import mujoco
        import numpy as np
        import omni.physics.tensors as tensors
        import omni.physx
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

        settings = carb.settings.get_settings()
        settings.set_int("/physics/numThreads", job["threads"])
        settings.set_bool("/physics/updateToUsd", False)
        settings.set_bool("/physics/updateVelocitiesToUsd", False)
        settings.set_bool("/physics/suppressReadback", self.gpu)

        self.tmp = tempfile.TemporaryDirectory(prefix="physmark-isaac-robot-")
        xml_path = Path(self.tmp.name) / "robot.xml"
        xml_path.write_text(robot["xml"])
        asset = MJCFImporter(MJCFImporterConfig(
            mjcf_path=str(xml_path), usd_path=self.tmp.name, allow_self_collision=False)).import_mjcf()

        stage = Usd.Stage.CreateInMemory()
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
        total = job["worlds"] * scene["size"]
        for name, per_body, default in [
            ("GpuMaxRigidContactCount", 40, 2**19),
            ("GpuMaxRigidPatchCount", 20, 2**18),
            ("GpuFoundLostPairsCapacity", 16, 2**18),
            ("GpuFoundLostAggregatePairsCapacity", 16, 2**18),
            ("GpuTotalAggregatePairsCapacity", 16, 2**18),
        ]:
            getattr(physx, f"Create{name}Attr")(max(default, per_body * total))

        # One shared ground replaces each robot's own floor plane. MuJoCo combines a contact
        # pair's friction with max(); PhysX's "max" combine mode does the same.
        floor = UsdGeom.Plane.Define(stage, "/World/ground")
        floor.CreateAxisAttr("Z")
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        material = UsdShade.Material.Define(stage, "/World/ground_material")
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(0.8)
        api.CreateDynamicFrictionAttr(0.8)
        api.CreateRestitutionAttr(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("max")
        UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")

        columns = math.ceil(math.sqrt(job["worlds"]))
        self.offsets = [[(e % columns) * SPACING, (e // columns) * SPACING, 0.0] for e in range(job["worlds"])]
        roots = []
        for e, offset in enumerate(self.offsets):
            env = UsdGeom.Xform.Define(stage, f"/World/envs/block_{e // 64}/env_{e}")
            env.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*offset))
            prim = stage.DefinePrim(f"{env.GetPath()}/robot", "Xform")
            prim.GetReferences().AddReference(asset)
            prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
            stage.GetPrimAtPath(f"{prim.GetPath()}/Geometry/physmark_floor").SetActive(False)
            root = next(p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.ArticulationRootAPI))
            articulation = PhysxSchema.PhysxArticulationAPI.Apply(root)
            articulation.CreateSolverPositionIterationCountAttr(POSITION_ITERATIONS)
            articulation.CreateSolverVelocityIterationCountAttr(VELOCITY_ITERATIONS)
            articulation.CreateSleepThresholdAttr(0.0)
            articulation.CreateStabilizationThresholdAttr(0.0)
            articulation.CreateEnabledSelfCollisionsAttr(False)
            roots.append(str(root.GetPath()))

        self.stage = stage
        self.stepper = omni.physx.get_physx_simulation_interface()
        stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        if not self.stepper.attach_stage(stage_id):
            raise RuntimeError("PhysX could not attach the Isaac robot stage")
        self.stepper.simulate(scene["dt"], 0.0)
        self.stepper.fetch_results()
        import torch

        self.torch, self.np = torch, np
        self.device = "cuda:0" if self.gpu else "cpu"
        self.view = tensors.create_simulation_view("torch", stage_id)
        self.view.set_subspace_roots("/")
        root_name = roots[0].rsplit("/", 1)[1]
        self.art = self.view.create_articulation_view(f"/World/envs/*/*/robot/Geometry/{root_name}")
        if self.art.count != job["worlds"]:
            raise RuntimeError("Isaac articulation view does not cover every environment")
        order = {p: i for i, p in enumerate(self.art.prim_paths)}
        # View rows -> scene environment order.
        self.env_rows = torch.tensor([order[p] for p in roots], device=self.device)
        self.indices = torch.arange(job["worlds"], dtype=torch.int32, device=self.device)
        dof_names = list(self.art.shared_metatype.dof_names)
        link_names = list(self.art.shared_metatype.link_names)

        # MuJoCo's position actuators, read from the compiled model.
        model = mujoco.MjModel.from_xml_string(robot["xml"])
        n = len(dof_names)
        stiffness, damping, max_force, armature = (np.zeros(n) for _ in range(4))
        target, position = np.zeros(n), np.zeros(n)
        qpos, ctrl = np.asarray(robot["qpos"]), np.asarray(robot["ctrl"])
        frictionloss = 0.0
        for a in range(model.nu):
            joint = model.actuator_trnid[a, 0]
            dof = dof_names.index(model.joint(joint).name)
            adr = model.jnt_dofadr[joint]
            stiffness[dof] = model.actuator_gainprm[a, 0]
            damping[dof] = -model.actuator_biasprm[a, 2] + model.dof_damping[adr]
            max_force[dof] = (max(abs(model.actuator_forcerange[a]))
                              if model.actuator_forcelimited[a] else 1e30)
            armature[dof] = model.dof_armature[adr]
            frictionloss = max(frictionloss, model.dof_frictionloss[adr])
            lo, hi = model.actuator_ctrlrange[a]
            target[dof] = min(max(ctrl[a], lo), hi) if model.actuator_ctrllimited[a] else ctrl[a]
            position[dof] = qpos[model.jnt_qposadr[joint]]
        rows = lambda v: torch.tensor(np.tile(v, (job["worlds"], 1)), dtype=torch.float32, device=self.device)
        self.art.set_dof_stiffnesses(rows(stiffness).cpu(), self.indices.cpu())
        self.art.set_dof_dampings(rows(damping).cpu(), self.indices.cpu())
        self.art.set_dof_max_forces(rows(max_force).cpu(), self.indices.cpu())
        self.art.set_dof_armatures(rows(armature).cpu(), self.indices.cpu())
        self.targets, self.positions = rows(target), rows(position)
        root_pose = [*qpos[:3], qpos[4], qpos[5], qpos[6], qpos[3]]  # MuJoCo wxyz -> xyzw
        self.root_poses = torch.tensor(
            [[root_pose[0] + o[0], root_pose[1] + o[1], root_pose[2] + o[2], *root_pose[3:]] for o in self.offsets],
            dtype=torch.float32, device=self.device)
        self.link_rows = torch.tensor(
            [link_names.index(robot["body_names"][b["id"] - 1]) for b in scene["bodies"] if b["dynamic"]],
            device=self.device)
        self.offset_rows = torch.tensor(self.offsets, dtype=torch.float32, device=self.device)
        self.reset()
        self.time = 0.0
        self.settings = {
            "solver": "PhysX TGS articulations",
            "position_iterations": POSITION_ITERATIONS,
            "velocity_iterations": VELOCITY_ITERATIONS,
            "precision": "f32",
            "sleeping": False,
            "self_collisions": False,
            "importer": "isaacsim.asset.importer.mjcf (mujoco-usd-converter), PhysX variant",
            "drives": "MuJoCo position actuators: stiffness kp, damping kv + joint damping, max force from "
                      "forcerange, joint armature, targets clamped to ctrlrange",
            "ignored_joint_frictionloss": float(frictionloss),
            "ground": "one shared plane, friction 0.8 with max combine",
            "gpu_dynamics": self.gpu,
            "world_execution": "native_batch (side-by-side environments on a grid)",
            "environment_spacing_m": SPACING,
            "isaacsim_version": metadata.version("isaacsim"),
            "device": self.device,
            "threads": job["threads"],
        }

    def reset(self):
        torch = self.torch
        poses = torch.empty_like(self.root_poses)
        poses[self.env_rows] = self.root_poses
        zeros = torch.zeros((self.job["worlds"], 6), dtype=torch.float32, device=self.device)
        self.art.set_root_transforms(poses, self.indices)
        self.art.set_root_velocities(zeros, self.indices)
        self.art.set_dof_positions(self.positions, self.indices)
        self.art.set_dof_velocities(torch.zeros_like(self.positions), self.indices)
        self.art.set_dof_position_targets(self.targets, self.indices)
        self.view.update_articulations_kinematic()

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
        links = self.art.get_link_transforms()[self.env_rows][:, self.link_rows].clone()
        links[:, :, :3] -= self.offset_rows[:, None, :]
        return links.reshape(-1, 7).cpu().double().tolist()

    def awake(self):
        return self.job["worlds"] * self.job["scene"]["size"]

    def close(self):
        self.app.close()

    def gpu_profile(self, job, directory):
        from gpu_timing import torch_cupti_profile

        return torch_cupti_profile(self, job, directory)
