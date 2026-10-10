"""Newton SolverImplicitMPM adapter; exact material points come from the Rust scene."""
import math

import numpy as np

from mpm_common import common_settings, particle_poses


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

    def gpu_profile(self, job, directory):
        from warp_timing import warp_profile

        return warp_profile(self, job, directory)

    def awake(self):
        return int(np.count_nonzero(self.model.particle_flags.numpy() & int(self.n.ParticleFlags.ACTIVE)))
