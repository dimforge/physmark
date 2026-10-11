"""Genesis MPM adapter; exact material points come from the Rust scene."""
import numpy as np

from mpm_common import common_settings, particle_poses


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
        else:
            material = gs.materials.MPM.Sand(**params, friction_angle=cfg["friction_angle_degrees"])
            law = "logarithmic strain / Drucker-Prager, fixed friction angle"
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

    def gpu_profile(self, job, directory):
        from gpu_timing import torch_cupti_profile

        return torch_cupti_profile(self, job, directory)
