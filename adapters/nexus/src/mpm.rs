use super::*;
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use physmark_core::mpm::MpmConfig;

pub async fn state(job: &Job, backend: &GpuBackend) -> Result<NexusState> {
    let cfg = job.scene.mpm.as_ref().unwrap();
    let mut state = NexusState::default();
    state.compute_graphs = false;
    let mut model = match cfg.material.as_str() {
        "elastic" => {
            ParticleModel::elastic_neo_hookean(cfg.young_modulus as f32, cfg.poisson_ratio as f32)
        }
        "sand" => {
            ParticleModel::sand_neo_hookean(cfg.young_modulus as f32, cfg.poisson_ratio as f32)
        }
        _ => bail!("unsupported MPM material"),
    };
    if let ParticleModel::SandNeoHookean(ref mut sand) = model {
        sand.plastic.ha = cfg.friction_angle_degrees.to_radians() as f32;
        sand.plastic.hb = 0.;
        sand.plastic.hc = 0.;
        sand.plastic.hd = 0.;
    }
    let mut particles = vec![];
    for body in &job.scene.bodies {
        if body.dynamic {
            let mut p = Particle::new(
                Vector::from_array(body.position.map(|x| x as f32)),
                (cfg.particle_spacing / 2.) as f32,
                cfg.density as f32,
                model,
            );
            p.dynamics.velocity = Vector::from_array(body.velocity.map(|x| x as f32));
            particles.push(p);
        } else {
            let rb = RigidBodyBuilder::fixed()
                .translation(Vector::from_array(body.position.map(|x| x as f32)))
                .build();
            let shape = ColliderBuilder::cuboid(
                body.size[0] as f32,
                body.size[1] as f32,
                body.size[2] as f32,
            )
            .build();
            state.insert_rigid_body(
                rb,
                shape,
                RbdCoupling::MpmOneWay(BoundaryCondition::separate(cfg.boundary_friction as f32)),
            );
        }
    }
    state.set_mpm_params(
        backend,
        SimulationParams {
            gravity: Vector::from_array(job.scene.gravity.map(|x| x as f32)),
            dt: job.scene.dt as f32,
        },
        cfg.cell_width as f32,
    )?;
    state.set_mpm_substeps(cfg.substeps as u32);
    state.add_particles(backend, particles)?;
    state.finalize(backend).await?;
    // Building the rigid coupling in finalize enables CPIC in this engine pin.
    // Select analytic grid collisions only after the coupling has been built.
    state.set_mpm_use_cpic(false);
    ensure_cpic_disabled(&state)?;
    Ok(state)
}
/// Fails if CPIC rigid coupling is enabled; it is a different collision scheme than the
/// analytic grid boundaries used by every other engine.
pub fn ensure_cpic_disabled(state: &NexusState) -> Result<()> {
    anyhow::ensure!(!state.mpm_use_cpic(), "Nexus MPM CPIC must stay disabled");
    Ok(())
}
pub fn settings(cfg: &MpmConfig) -> serde_json::Value {
    serde_json::json!({"solver":"Nexus MLS-MPM", "material":cfg.material, "mpm":cfg,
        "constitutive_law":if cfg.material == "elastic" { "Neo-Hookean" } else { "Neo-Hookean / Drucker-Prager, fixed friction angle" },
        "grid":"sparse hashed", "transfer":"APIC", "precision":"f32", "sleeping":false,
        "world_execution":"single", "graph_capture":false, "cpic":false,
        "comparison":"common initial material parameters, different constitutive/discretization implementations"})
}
