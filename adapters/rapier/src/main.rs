use rapier3d as engine;
include!("../../common/common_rust.rs");
use anyhow::{Result, ensure};
use physmark_core::memory::MemorySampler;
use physmark_core::{AdapterResult, Frame, read_job};
use std::time::Instant;
// Robot stand-up settings; other scenes keep Rapier's defaults (4 substeps, 1 PGS iteration).
const ROBOT_SUBSTEPS: usize = 1;
const ROBOT_PGS_ITERATIONS: usize = 4;
/// Robot environments step as one rayon task each, every world single-threaded. Rapier picks
/// its internal parallelism from the calling rayon pool, so that needs a build without its
/// `parallel` feature: the default build hands robot jobs to that one.
#[cfg(feature = "parallel")]
fn run_serial_build() -> Result<()> {
    let exe = std::env::current_exe()?;
    let serial = exe
        .parent()
        .and_then(|d| d.parent())
        .map(|target| target.join("serial/release").join(exe.file_name().unwrap()))
        .filter(|p| p.is_file())
        .ok_or_else(|| anyhow::anyhow!("serial Rapier build missing; run `cargo run --release -- build`"))?;
    let mut command = std::process::Command::new(serial);
    command.args(std::env::args_os().skip(1));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        return Err(command.exec().into());
    }
    #[cfg(not(unix))]
    std::process::exit(command.status()?.code().unwrap_or(1));
}
fn step_all(worlds: &mut [(PhysicsWorld, Vec<RigidBodyHandle>)]) {
    #[cfg(not(feature = "parallel"))]
    {
        use rayon::prelude::*;
        worlds.par_iter_mut().for_each(|(w, _)| w.step());
    }
    #[cfg(feature = "parallel")]
    for (w, _) in worlds {
        w.step();
    }
}
fn main() -> Result<()> {
    let (job, path) = read_job()?;
    #[cfg(feature = "parallel")]
    if job.scene.robot.is_some() {
        return run_serial_build();
    }
    let memory = (job.mode == "memory").then(|| MemorySampler::new(&job.backend));
    ensure!(job.backend == "cpu", "Rapier only supports cpu");
    rayon::ThreadPoolBuilder::new()
        .num_threads(job.threads)
        .build_global()?;
    let start = Instant::now();
    let make = || {
        (0..job.worlds)
            .map(|_| {
                let (mut w, h) = build_world(&job.scene)?;
                if job.scene.robot.is_some() {
                    w.integration_parameters.num_solver_iterations = ROBOT_SUBSTEPS;
                    w.integration_parameters.num_internal_pgs_iterations = ROBOT_PGS_ITERATIONS;
                }
                Ok((w, h))
            })
            .collect::<Result<Vec<_>>>()
    };
    let mut worlds = make()?;
    let setup = start.elapsed().as_secs_f64();
    if let Some(memory) = &memory {
        memory.sample("setup");
    }
    let start = Instant::now();
    for _ in 0..job.warmup.max(job.steps) {
        step_all(&mut worlds);
    }
    let warmup = start.elapsed().as_secs_f64();
    if let Some(memory) = &memory {
        memory.sample("warmup");
    }
    worlds = make()?;
    if let Some(memory) = &memory {
        memory.sample("reset");
    }
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
        step_all(&mut worlds);
        if job.mode == "capture" && (step % job.capture_stride == 0 || step == job.steps) {
            frames.push(Frame {
                step,
                poses: poses(&worlds)[..n].to_vec(),
            });
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let memory = memory.map(|sampler| sampler.finish(job.steps));
    let awake = worlds
        .iter()
        .map(|(w, _)| {
            w.bodies
                .iter()
                .filter(|(_, b)| b.is_dynamic() && !b.is_sleeping())
                .count()
        })
        .sum();
    let mut result = AdapterResult {
        schema_version: 1,
        engine: job.engine.clone(),
        backend: job.backend.clone(),
        setup_seconds: setup,
        warmup_seconds: warmup,
        elapsed_seconds: elapsed,
        final_poses: poses(&worlds),
        awake_count: awake,
        settings: serde_json::json!({"solver":"Rapier default TGS", "solver_iterations":worlds[0].0.integration_parameters.num_solver_iterations,"precision":"f32","sleeping":false,"warmup_steps_effective":job.warmup.max(job.steps),"threads":job.threads,"world_execution":if cfg!(feature = "parallel") {"serial"} else {"one rayon task per environment, single-threaded worlds"},"parallel_feature":cfg!(feature = "parallel")}),
        frames,
        gpu_timing: None,
        memory,
    };
    if let Some(robot) = &job.scene.robot {
        let params = &worlds[0].0.integration_parameters;
        result.settings["robot"] = serde_json::json!({"model":robot.model,"source_commit":robot.source_commit,"initial_keyframe":robot.initial_keyframe,"target_keyframe":robot.target_keyframe,"ctrl":robot.ctrl,"self_contacts":false,
            "substeps":params.num_solver_iterations,"pgs_iterations":params.num_internal_pgs_iterations,
            "coriolis":"semi-implicit, explicit fallback on divergence (Rapier has no switch)"});
    }
    std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
