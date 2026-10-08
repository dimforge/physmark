use nexus3d::rapier as engine;
include!("../../common_rust.rs");
use anyhow::{Result, bail};
use khal::backend::{Backend, GpuBackend};
use nexus3d::prelude::*;
use physmark_core::{AdapterResult, Frame, Job, read_job};
use std::time::Instant;
async fn state(job: &Job, b: &GpuBackend) -> Result<(NexusState, Vec<Vec<RigidBodyHandle>>)> {
    let mut s = NexusState::default();
    let mut h = vec![];
    for e in 0..job.worlds {
        if e > 0 {
            s.add_environment();
        }
        let (world, handles) = build_world(&job.scene);
        *s.rbd_world_mut(e) = world;
        h.push(handles);
    }
    s.set_rbd_timestep(job.scene.dt as f32, 4);
    s.set_rbd_collisions_capacity((job.scene.size * 32).max(128) as u32);
    s.finalize(b).await?;
    s.set_rbd_gravity(b, job.scene.gravity.map(|x| x as f32));
    Ok((s, h))
}
async fn poses(
    job: &Job,
    b: &GpuBackend,
    s: &NexusState,
    h: &[Vec<RigidBodyHandle>],
) -> Result<Vec<[f64; 7]>> {
    let p = s.read_rigid_body_poses(b).await;
    let mut out = vec![];
    for (e, handles) in h.iter().enumerate() {
        for body in &job.scene.bodies {
            if body.dynamic {
                let index = s
                    .rigid_body_gpu_index(e, handles[body.id])
                    .ok_or_else(|| anyhow::anyhow!("missing body mapping"))?
                    as usize;
                let p = p
                    .get(index)
                    .ok_or_else(|| anyhow::anyhow!("missing GPU pose"))?;
                let t = p.translation;
                let q = p.rotation;
                out.push([
                    t.x as f64, t.y as f64, t.z as f64, q.x as f64, q.y as f64, q.z as f64,
                    q.w as f64,
                ]);
            }
        }
    }
    Ok(out)
}
fn main() -> Result<()> {
    pollster::block_on(run())
}
async fn run() -> Result<()> {
    let (job, path) = read_job()?;
    let b = match job.backend.as_str() {
        #[cfg(feature = "metal")]
        "metal" => GpuBackend::Metal(khal::backend::metal::Metal::new()?),
        #[cfg(feature = "webgpu")]
        "webgpu" => GpuBackend::WebGpu(khal::backend::WebGpu::default().await?),
        #[cfg(feature = "cuda")]
        "cuda" => GpuBackend::Cuda(khal::backend::Cuda::new(0)?),
        #[cfg(feature = "cpu")]
        "cpu" => GpuBackend::Cpu,
        _ => bail!(
            "this Nexus build does not include requested backend {}",
            job.backend
        ),
    };
    let start = Instant::now();
    let mut pipeline = NexusPipeline::default();
    let (mut s, _) = state(&job, &b).await?;
    pipeline.preload_pipelines(&b, NexusPipelineMask::RBD)?;
    b.synchronize()?;
    let setup = start.elapsed().as_secs_f64();
    let start = Instant::now();
    for _ in 0..job.warmup.max(job.steps) {
        pipeline.simulate(&b, &mut s, None).await?;
    }
    b.synchronize()?;
    let warmup = start.elapsed().as_secs_f64();
    let (mut s, h) = state(&job, &b).await?;
    b.synchronize()?;
    let mut frames = vec![];
    if job.mode == "capture" {
        frames.push(Frame {
            step: 0,
            poses: poses(&job, &b, &s, &h).await?[..job.scene.size].to_vec(),
        });
    }
    b.synchronize()?;
    let start = Instant::now();
    for step in 1..=job.steps {
        pipeline.simulate(&b, &mut s, None).await?;
        if job.mode == "capture" && (step % job.capture_stride == 0 || step == job.steps) {
            b.synchronize()?;
            frames.push(Frame {
                step,
                poses: poses(&job, &b, &s, &h).await?[..job.scene.size].to_vec(),
            });
        }
    }
    b.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();
    let final_poses = poses(&job, &b, &s, &h).await?;
    let result = AdapterResult {
        schema_version: 1,
        engine: job.engine,
        backend: job.backend,
        setup_seconds: setup,
        warmup_seconds: warmup,
        elapsed_seconds: elapsed,
        awake_count: job.scene.size * job.worlds,
        final_poses,
        settings: serde_json::json!({"solver":"TGS soft", "solver_iterations":4,"precision":"f32","sleeping":false,"warmup_steps_effective":job.warmup.max(job.steps),"sleeping_implementation":"no sleeping in GPU rigid pipeline", "world_execution":"native_batch","graph_capture":false,"collision_capacity":(job.scene.size*32).max(128)}),
        frames,
    };
    std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
