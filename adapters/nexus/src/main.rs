use nexus3d::rapier as engine;
include!("../../common_rust.rs");
use anyhow::{Result, bail};
use khal::backend::{Backend, GpuBackend, GpuTimestamps};
use nexus3d::prelude::*;
use physmark_core::{AdapterResult, Frame, GpuTiming, Job, read_job};
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
    // Bounded batches avoid a timestamp readback barrier on every physics step.
    // Each step gets its own manager because simulate starts an asynchronous readback.
    const PROFILE_BATCH: usize = 8;
    let profiling = job.mode == "gpu_timing" && job.backend != "cpu";
    let mut timers = vec![];
    let mut query_capacity = 0;
    if profiling {
        // Apple M4 Max limits the sample buffer to 32 KiB = 2048 pairs.
        // Allocation failure and unsupported queries both produce Noop in Khal.
        for capacity in [2048, 1024, 512, 256, 128, 64] {
            let timer = GpuTimestamps::new(&b, capacity);
            if timer.is_enabled() {
                timers.push(timer);
                query_capacity = capacity;
                break;
            }
        }
        if query_capacity > 0 {
            for _ in 1..PROFILE_BATCH {
                let timer = GpuTimestamps::new(&b, query_capacity);
                anyhow::ensure!(timer.is_enabled(), "GPU timestamp batch allocation failed");
                timers.push(timer);
            }
        }
    }
    let enabled = timers.first().is_some_and(GpuTimestamps::is_enabled);
    let mut step_seconds = Vec::new();
    let mut pass_totals = std::collections::BTreeMap::<String, f64>::new();
    let mut sample_count = 0;
    let start = Instant::now();
    for step in 1..=job.steps {
        let slot = (step - 1) % PROFILE_BATCH;
        if enabled {
            timers[slot].reset();
            anyhow::ensure!(timers[slot].is_idle(), "timestamp manager still pending");
            pipeline
                .simulate(&b, &mut s, Some(&mut timers[slot]))
                .await?;
        } else {
            pipeline.simulate(&b, &mut s, None).await?;
        }
        if enabled && (step % PROFILE_BATCH == 0 || step == job.steps) {
            b.synchronize()?;
            for timer in &mut timers[..=slot] {
                let deadline = Instant::now() + std::time::Duration::from_secs(5);
                let samples = loop {
                    if let Some(samples) = timer.try_take(&b) {
                        break samples;
                    }
                    anyhow::ensure!(
                        Instant::now() < deadline,
                        "GPU timestamp readback timed out"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(1));
                };
                anyhow::ensure!(
                    !samples.is_empty()
                        && (job.backend == "cuda" || samples.len() < query_capacity as usize),
                    "empty or possibly truncated GPU timestamp samples"
                );
                let mut seconds = 0.;
                for sample in &samples {
                    anyhow::ensure!(
                        sample.duration_ms.is_finite() && sample.duration_ms >= 0.,
                        "invalid GPU timestamp"
                    );
                    seconds += sample.duration_ms / 1000.;
                    *pass_totals.entry(sample.label.clone()).or_default() +=
                        sample.duration_ms / 1000.;
                }
                anyhow::ensure!(
                    seconds > 0.,
                    "GPU timestamp queries returned no positive duration"
                );
                sample_count += samples.len();
                step_seconds.push(seconds);
            }
        }
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
    let gpu_timing = if enabled {
        let evidence = serde_json::json!({"step_seconds":step_seconds,"pass_seconds":pass_totals});
        std::fs::write(
            path.with_file_name("gpu-samples.json"),
            serde_json::to_vec_pretty(&evidence)?,
        )?;
        Some(GpuTiming {
            status: "available".into(),
            seconds: Some(step_seconds.iter().sum()),
            method: Some(format!("khal_{}_timestamps", job.backend)),
            scope: Some("sum_compute_pass_durations".into()),
            measured_steps: step_seconds.len(),
            sample_count,
            reason: None,
            details: serde_json::json!({"batch_steps":PROFILE_BATCH,"query_capacity_per_step":if job.backend == "cuda" { None } else { Some(query_capacity) },"evidence":"gpu-samples.json","excludes":"host work, transfers outside compute passes, queue gaps","replay":"separate process"}),
        })
    } else if job.mode == "gpu_timing" {
        Some(GpuTiming::unavailable(
            "GPU timestamp queries are unsupported or all query-buffer allocations failed",
        ))
    } else {
        None
    };
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
        gpu_timing,
    };
    std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
