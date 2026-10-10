//! Fixed GPU buffer budgets for the rigid-body pipeline, from `capacities.json`.
//!
//! Benchmarks use fixed buffers only: growing them from lagging readbacks drops contacts
//! until the resize lands. The budgets are measured by `--calibrate` (adaptive policies,
//! tiny initial buffers, peaks recorded over the full horizon) through
//! `tools/calibrate_nexus.py`, then applied with headroom. Every trial checks the
//! collision-pair and multibody-contact demand against them.
use super::*;
use nexus3d::rbd::pipeline::{RbdCapacities, RbdResizePolicy};
use serde_json::Value;

/// Margin over the calibrated peaks, for run-to-run GPU nondeterminism.
const HEADROOM: f64 = 1.25;
/// Extra colors over the calibrated peak budget.
const COLOR_MARGIN: u32 = 4;
const CALIBRATION: &str = include_str!("../capacities.json");
/// Starting color budget of a calibration run (Nexus's default). The adaptive policy never
/// shrinks below it and grows it whenever the coloring fails to converge.
const CALIBRATION_COLORS: u32 = 8;
/// Steps before colors are measured: until its first readback, Nexus floors the budget of
/// large scenes at 32 or 64 colors regardless of need, then shrinks to the colors in use + 4.
/// A real need keeps the budget up: it only shrinks with 8 unused colors.
const COLOR_SETTLE_STEPS: usize = 16;

/// Per-environment budgets and the allocation policy they are used with.
#[derive(Copy, Clone, Debug)]
pub struct Budget {
    pub collision_pairs: u32,
    pub mb_contact_slots: u32,
    pub solver_colors: u32,
    pub policy: RbdResizePolicy,
}

impl Budget {
    /// Adaptive buffers for a calibration run: generous pairs and contact slots so nothing is
    /// dropped while the demand is measured; colors start low so their growth is the measure.
    pub fn calibration(job: &Job) -> Self {
        Self {
            collision_pairs: (job.scene.size as u32 * 4).max(64),
            mb_contact_slots: 256,
            solver_colors: CALIBRATION_COLORS,
            policy: RbdResizePolicy::Grow,
        }
    }

    /// The fixed budget for `job`, from the calibrated entry of its scenario with the
    /// smallest size at least as large as the job's (never extrapolated).
    pub fn fixed(job: &Job) -> Result<Self> {
        let table: Value = serde_json::from_str(CALIBRATION)?;
        let entries = table[&job.scene.scenario].as_array().ok_or_else(|| {
            anyhow::anyhow!(
                "{} has no Nexus capacity calibration; run tools/calibrate_nexus.py",
                job.scene.scenario
            )
        })?;
        let entry = entries
            .iter()
            .filter(|e| e["size"].as_u64().is_some_and(|s| s as usize >= job.scene.size))
            .min_by_key(|e| e["size"].as_u64())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} size {} exceeds the calibrated Nexus capacities; run tools/calibrate_nexus.py",
                    job.scene.scenario,
                    job.scene.size
                )
            })?;
        let peak = |key: &str| entry[key].as_f64().unwrap_or(0.);
        // Measured demand (the GPU counters count past capacity), with Nexus's own 1.5× resize
        // slack and our headroom. PFM entries share the pair buffer, at most one per pair.
        let per_env = |demand: &str| (peak(demand) * 1.5 * HEADROOM).ceil() as u32;
        Ok(Self {
            collision_pairs: per_env("collision_pairs").max(64),
            mb_contact_slots: per_env("mb_contact_slots").max(1),
            solver_colors: peak("solver_colors") as u32 + COLOR_MARGIN,
            policy: RbdResizePolicy::Fixed,
        })
    }

    pub fn capacities(&self, worlds: usize) -> NexusCapacities {
        NexusCapacities {
            rbd: RbdCapacities {
                batches: worlds as u32,
                collisions_capacity: self.collision_pairs,
                mb_contact_constraints_capacity: self.mb_contact_slots,
                collisions_resize_policy: self.policy,
                solver_colors: self.solver_colors,
                solver_colors_resize_policy: self.policy,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub fn settings(&self) -> serde_json::Value {
        serde_json::json!({
            "capacity_policy": format!("{:?}", self.policy).to_lowercase(),
            "collision_pairs_per_environment": self.collision_pairs,
            "mb_contact_slots_per_environment": self.mb_contact_slots,
            "solver_colors": self.solver_colors,
            "capacity_source": "adapters/nexus/capacities.json (calibrated peaks with headroom)",
        })
    }
}

/// Collision pairs and multibody contact slots the last step needed, over all environments.
pub async fn demand(b: &GpuBackend, s: &NexusState) -> Result<(u32, u32)> {
    let rbd = s
        .rbd
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no rigid-body state"))?;
    let pairs: Vec<u32> = b
        .slow_read_vec(rbd.collision_pairs_len_gpu().buffer())
        .await?;
    let slots: Vec<u32> = b
        .slow_read_vec(rbd.multibodies().mb_cons_demand().buffer())
        .await?;
    Ok((pairs[0], slots[0]))
}

/// Fails if the last step needed more than the fixed buffers hold (contacts were dropped).
pub async fn ensure_fits(b: &GpuBackend, s: &NexusState) -> Result<(u32, u32)> {
    let rbd = s
        .rbd
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no rigid-body state"))?;
    let (pairs, slots) = demand(b, s).await?;
    anyhow::ensure!(
        pairs <= rbd.collision_pairs_capacity(),
        "collision-pair demand {pairs} exceeds the fixed capacity {}; recalibrate Nexus capacities",
        rbd.collision_pairs_capacity()
    );
    anyhow::ensure!(
        slots <= rbd.mb_contact_constraints_capacity(),
        "multibody contact demand {slots} exceeds the fixed capacity {}; recalibrate Nexus capacities",
        rbd.mb_contact_constraints_capacity()
    );
    Ok((pairs, slots))
}

/// Steps `job` under adaptive policies and records per-environment peaks of the demand and of
/// the budgets Nexus grew to. Untimed; reads the GPU counters after every step.
pub async fn calibrate(job: &Job, b: &GpuBackend) -> Result<serde_json::Value> {
    anyhow::ensure!(
        job.scene.mpm.is_none(),
        "MPM scenes have no rigid-body buffers to calibrate"
    );
    let mut pipeline = NexusPipeline::default();
    let (mut s, _) = state(job, b, Budget::calibration(job)).await?;
    let worlds = job.worlds as f64;
    let mut peaks = [0f64; 4];
    let mut solver_colors = 0;
    for step in 0..job.warmup.max(job.steps) {
        pipeline.simulate(b, &mut s, None).await?;
        b.synchronize()?;
        let (pairs, slots) = demand(b, &s).await?;
        let rbd = s.rbd.as_ref().unwrap();
        let values = [
            pairs as f64 / worlds,
            rbd.collision_pairs_capacity() as f64 / worlds,
            slots as f64 / worlds,
            rbd.mb_contact_constraints_capacity() as f64 / worlds,
        ];
        for (peak, value) in peaks.iter_mut().zip(values) {
            *peak = peak.max(value);
        }
        if step >= COLOR_SETTLE_STEPS {
            solver_colors = solver_colors.max(rbd.max_colors());
        }
    }
    Ok(serde_json::json!({
        "size": job.scene.size,
        "worlds": job.worlds,
        "steps": job.warmup.max(job.steps),
        "collision_pairs": peaks[0],
        "collision_pairs_capacity": peaks[1],
        "mb_contact_slots": peaks[2],
        "mb_contact_slots_capacity": peaks[3],
        "solver_colors": solver_colors,
    }))
}
