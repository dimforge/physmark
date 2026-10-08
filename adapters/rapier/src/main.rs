use rapier3d as engine;
include!("../../common_rust.rs");
use anyhow::{Result, ensure};
use physmark_core::{AdapterResult, Frame, read_job};
use std::time::Instant;
fn main() -> Result<()> {
    let (job, path) = read_job()?;
    ensure!(job.backend == "cpu", "Rapier only supports cpu");
    rayon::ThreadPoolBuilder::new()
        .num_threads(job.threads)
        .build_global()?;
    let start = Instant::now();
    let make = || {
        (0..job.worlds)
            .map(|_| build_world(&job.scene))
            .collect::<Vec<_>>()
    };
    let mut worlds = make();
    let setup = start.elapsed().as_secs_f64();
    let start = Instant::now();
    for _ in 0..job.warmup.max(job.steps) {
        for (w, _) in &mut worlds {
            w.step();
        }
    }
    let warmup = start.elapsed().as_secs_f64();
    worlds = make();
    let poses = |worlds: &Vec<(PhysicsWorld, Vec<RigidBodyHandle>)>| -> Vec<[f64; 7]> {
        worlds
            .iter()
            .flat_map(|(w, h)| {
                job.scene
                    .bodies
                    .iter()
                    .filter(|b| b.dynamic)
                    .map(move |b| pose_array(w.bodies[h[b.id]].position()))
            })
            .collect()
    };
    let n = job.scene.size;
    let mut frames = vec![];
    if job.mode == "capture" {
        frames.push(Frame {
            step: 0,
            poses: poses(&worlds)[..n].to_vec(),
        });
    }
    let start = Instant::now();
    for step in 1..=job.steps {
        for (w, _) in &mut worlds {
            w.step();
        }
        if job.mode == "capture" && (step % job.capture_stride == 0 || step == job.steps) {
            frames.push(Frame {
                step,
                poses: poses(&worlds)[..n].to_vec(),
            });
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let awake = worlds
        .iter()
        .map(|(w, _)| {
            w.bodies
                .iter()
                .filter(|(_, b)| b.is_dynamic() && !b.is_sleeping())
                .count()
        })
        .sum();
    let result = AdapterResult {
        schema_version: 1,
        engine: job.engine,
        backend: job.backend,
        setup_seconds: setup,
        warmup_seconds: warmup,
        elapsed_seconds: elapsed,
        final_poses: poses(&worlds),
        awake_count: awake,
        settings: serde_json::json!({"solver":"Rapier default TGS", "solver_iterations":worlds[0].0.integration_parameters.num_solver_iterations,"precision":"f32","sleeping":false,"warmup_steps_effective":job.warmup.max(job.steps),"threads":job.threads,"world_execution":"serial","parallel_feature":true}),
        frames,
    };
    std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
