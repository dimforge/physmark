"""Single-environment MPM adapters; exact material points come from the Rust scene."""
import math
import numpy as np


def particle_poses(positions):
    positions = np.asarray(positions).reshape(-1, 3)
    out = np.zeros((len(positions), 7))
    out[:, :3] = positions
    out[:, 6] = 1  # Protocol padding: MPM particles have no rigid orientation.
    return out.tolist()


def common_settings(job):
    return dict(mpm=job["scene"]["mpm"], material=job["scene"]["mpm"]["material"],
                precision="f32", sleeping=False, world_execution="single", graph_capture=False,
                comparison="common initial material parameters, different constitutive/discretization implementations")


class GenesisMPM:
    def __init__(self, job):
        import genesis as gs
        self.gs, self.job = gs, job
        cfg = job["scene"]["mpm"]
        backend = {"cpu": gs.cpu, "metal": gs.metal, "cuda": gs.cuda}[job["backend"]]
        gs.init(backend=backend, precision="32", logging_level="warning",
                seed=job["scene"]["seed"] % 2**31, performance_mode=True)
        if gs.backend != backend:
            raise RuntimeError("Genesis selected a different backend")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=job["scene"]["dt"], substeps=cfg["substeps"],
                substeps_local=1, requires_grad=False, gravity=tuple(job["scene"]["gravity"])),
            mpm_options=gs.options.MPMOptions(particle_size=cfg["particle_spacing"],
                grid_density=1 / cfg["cell_width"], lower_bound=tuple(cfg["domain_lower"]),
                upper_bound=tuple(cfg["domain_upper"]), enable_CPIC=False),
            rigid_options=gs.options.RigidOptions(use_hibernation=False),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False), show_viewer=False)
        for b in job["scene"]["bodies"]:
            if not b["dynamic"]:
                self.scene.add_entity(gs.morphs.Box(pos=tuple(b["position"]),
                    size=tuple(2 * x for x in b["size"]), fixed=True),
                    material=gs.materials.Rigid(friction=cfg["boundary_friction"]),
                    surface=gs.surfaces.Default())
        params = dict(E=cfg["young_modulus"], nu=cfg["poisson_ratio"], rho=cfg["density"], sampler="regular")
        if cfg["material"] == "elastic":
            material = gs.materials.MPM.Elastic(**params, model="neohooken")
            law = "Neo-Hookean"
        elif cfg["material"] == "sand":
            material = gs.materials.MPM.Sand(**params, friction_angle=cfg["friction_angle_degrees"])
            law = "logarithmic strain / Drucker-Prager, fixed friction angle"
        else:
            # Genesis liquid pressure is lambda * J * (J-1); match its tangent at J=1.
            nu = cfg["poisson_ratio"]
            params["E"] = cfg["bulk_modulus"] * (1 + nu) * (1 - 2 * nu) / nu
            material = gs.materials.MPM.Liquid(**params, viscous=False)
            law = "lambda J(J-1) pressure, zero shear"
        self.entity = self.scene.add_entity(gs.morphs.Nowhere(n_particles=job["scene"]["size"]),
            material=material, surface=gs.surfaces.Default())
        self.scene.build()  # One actual environment, no batched axis.
        self.ensure_cpic_disabled()
        points = [b for b in job["scene"]["bodies"] if b["dynamic"]]
        self.entity.set_particles_pos(np.array([b["position"] for b in points], dtype=np.float32))
        self.entity.set_particles_vel(np.array([b["velocity"] for b in points], dtype=np.float32))
        self.entity.set_particles_active(np.ones(len(points), dtype=bool))
        self.sync()
        self.initial = self.scene.get_state()
        if not np.allclose(np.array(self.poses())[:, :3], [b["position"] for b in points], atol=1e-6):
            raise ValueError("Genesis did not preserve the canonical particle positions")
        if self.awake() != len(points):
            raise ValueError("Genesis did not activate every material point")
        # Per particle: the f32 total from get_mass() drifts beyond 1e-5 near 200K particles.
        solver = self.scene.sim.mpm_solver
        start = self.entity._particle_start
        masses = solver.particles_info.mass.to_numpy()[start:start + len(points)] / solver._particle_volume_scale
        if not np.allclose(masses, [b["mass"] for b in points], rtol=1e-5):
            raise ValueError("Genesis particle mass does not match the shared scene")
        self.settings = {**common_settings(job), "solver": "Genesis MLS-MPM", "constitutive_law": law,
                        "grid": "bounded dense with active-node tracking", "transfer": "APIC", "cpic": False,
                        "genesis_version": gs.__version__, "actual_backend": str(gs.backend),
                        "mpm_options": self.scene.options.mpm.model_dump(mode="json"),
                        "material_options": material.model_dump(mode="json"),
                        "particle_initialization": "Nowhere reservation, explicit positions/velocity/activation"}

    def ensure_cpic_disabled(self):
        # CPIC is a different rigid-coupling scheme than the analytic boundaries of the other engines.
        if self.scene.sim.mpm_solver.enable_CPIC:
            raise RuntimeError("Genesis MPM CPIC must stay disabled")

    def reset(self):
        self.scene.reset(self.initial)

    def advance(self, steps):
        for _ in range(steps):
            self.scene.step()

    def sync(self):
        import quadrants as qd
        qd.sync()

    def poses(self):
        return particle_poses(self.entity.get_particles_pos().cpu().numpy())

    def awake(self):
        return int(self.entity.get_particles_active().cpu().numpy().sum())


class NewtonMPM:
    def __init__(self, job):
        import newton
        import warp as wp
        self.n, self.wp, self.job = newton, wp, job
        cfg = job["scene"]["mpm"]
        if job["backend"] not in ("cpu", "cuda"):
            raise ValueError("Newton MPM supports CPU/CUDA, not Metal")
        wp.init()
        wp.config.quiet = True
        self.device = wp.get_device("cpu" if job["backend"] == "cpu" else "cuda:0")
        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            newton.solvers.SolverImplicitMPM.register_custom_attributes(builder)
            for b in job["scene"]["bodies"]:
                if b["dynamic"]:
                    builder.add_particle(pos=wp.vec3(*b["position"]), vel=wp.vec3(*b["velocity"]),
                        mass=b["mass"], radius=cfg["particle_spacing"] / 2.)
                else:
                    builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*b["position"]), wp.quat_identity()),
                        hx=b["size"][0], hy=b["size"][1], hz=b["size"][2],
                        cfg=newton.ModelBuilder.ShapeConfig(mu=cfg["boundary_friction"], density=0.))
            self.model = builder.finalize(device=self.device)
            self.model.set_gravity(tuple(job["scene"]["gravity"]))
            props = dict(young_modulus=cfg["young_modulus"], poisson_ratio=cfg["poisson_ratio"],
                         damping=0., hardening=0., viscosity=0., dilatancy=0., yield_pressure=1e15)
            if cfg["material"] == "elastic":
                props.update(friction=0., yield_stress=1e15, tensile_yield_ratio=1.)
            elif cfg["material"] == "sand":
                props.update(friction=math.tan(math.radians(cfg["friction_angle_degrees"])),
                             yield_stress=0., tensile_yield_ratio=0.)
            else:
                props.update(young_modulus=3 * cfg["bulk_modulus"] * (1 - 2 * cfg["poisson_ratio"]),
                             friction=0., yield_stress=0., tensile_yield_ratio=0.)
            for name, value in props.items():
                getattr(self.model.mpm, name).fill_(value)
            self.config = newton.solvers.SolverImplicitMPM.Config()
            self.config.voxel_size = cfg["cell_width"]
            # NanoVDB sparse grids require CUDA; dense grid is an explicit CPU choice.
            self.config.grid_type = "dense" if self.device.is_cpu else "sparse"
            self.config.transfer_scheme = "apic"
            # Use the basis combination in Newton's reference dam-break example.
            self.config.velocity_basis = "Q1"
            self.config.strain_basis = "P0"
            self.config.collider_basis = "pic"
            self.config.max_iterations = 100
            self.config.tolerance = 1e-4
            self.config.air_drag = 0.
            self.reset()
            if not np.allclose(self.model.particle_mass.numpy(), [b["mass"] for b in job["scene"]["bodies"] if b["dynamic"]], rtol=1e-6):
                raise ValueError("Newton particle masses do not match the shared scene")
        self.settings = {**common_settings(job), "solver": "SolverImplicitMPM", "grid": self.config.grid_type,
                        "transfer": "APIC", "grid_padding": self.config.grid_padding, "constitutive_law": "implicit corotated elasticity / yield-surface rheology",
                        "velocity_basis": self.config.velocity_basis, "strain_basis": self.config.strain_basis,
                        "collider_basis": self.config.collider_basis,
                        "iterations": 100, "tolerance": 1e-4, "material_options": props,
                        "newton_version": newton.__version__, "warp_version": wp.__version__,
                        "device": str(self.device), "collision_projection": "solver.project_outside"}

    def reset(self):
        with self.wp.ScopedDevice(self.device):
            if hasattr(self, "solver"):
                del self.solver
            self.solver = self.n.solvers.SolverImplicitMPM(self.model, self.config)
            self.state0, self.state1 = self.model.state(), self.model.state()

    def advance(self, steps):
        cfg = self.job["scene"]["mpm"]
        with self.wp.ScopedDevice(self.device):
            for _ in range(steps):
                for _ in range(cfg["substeps"]):
                    self.state0.clear_forces()
                    dt = self.job["scene"]["dt"] / cfg["substeps"]
                    self.solver.step(self.state0, self.state1, None, None, dt)
                    self.solver.project_outside(self.state1, self.state1, dt)
                    self.state0, self.state1 = self.state1, self.state0

    def sync(self):
        self.wp.synchronize_device(self.device)

    def poses(self):
        return particle_poses(self.state0.particle_q.numpy())

    def awake(self):
        return int(np.count_nonzero(self.model.particle_flags.numpy() & int(self.n.ParticleFlags.ACTIVE)))
