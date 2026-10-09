//! Shared material points and SI parameters. Engine constitutive laws still differ.
use super::{Body, Scene, make_body};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const ENGINES: &[&str] = &["nexus", "genesis", "newton"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MpmConfig {
    pub material: String,
    pub particle_spacing: f64,
    pub cell_width: f64,
    pub density: f64,
    pub young_modulus: f64,
    pub poisson_ratio: f64,
    pub bulk_modulus: f64,
    pub friction_angle_degrees: f64,
    pub boundary_friction: f64,
    pub substeps: usize,
    pub domain_lower: [f64; 3],
    pub domain_upper: [f64; 3],
}
impl MpmConfig {
    pub fn validate(&self, scene: &Scene) -> Result<()> {
        ensure!(
            ["elastic", "sand", "liquid"].contains(&self.material.as_str()),
            "unknown MPM material"
        );
        ensure!(
            self.substeps > 0 && self.substeps <= 1024,
            "invalid MPM substeps"
        );
        ensure!(
            [
                self.particle_spacing,
                self.cell_width,
                self.density,
                self.young_modulus,
                self.bulk_modulus
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0.),
            "invalid MPM physical parameters"
        );
        ensure!(
            self.poisson_ratio.is_finite() && (0.0..0.5).contains(&self.poisson_ratio),
            "invalid Poisson ratio"
        );
        ensure!(
            self.boundary_friction.is_finite()
                && self.boundary_friction >= 0.
                && self.friction_angle_degrees.is_finite()
                && (0.0..90.0).contains(&self.friction_angle_degrees),
            "invalid MPM friction"
        );
        ensure!(
            (0..3).all(|k| self.domain_lower[k].is_finite()
                && self.domain_upper[k].is_finite()
                && self.domain_lower[k] < self.domain_upper[k]),
            "invalid MPM domain"
        );
        ensure!(scene.hinges.is_empty(), "MPM scenes cannot contain hinges");
        let points: Vec<_> = scene.bodies.iter().filter(|b| b.dynamic).collect();
        ensure!(points.len() == scene.size, "MPM particle count mismatch");
        let mass = self.density * self.particle_spacing.powi(3);
        ensure!(
            points.iter().all(|b| b.shape == "sphere"
                && (b.mass - mass).abs() <= mass * 1.0e-6
                && (0..3).all(|k| b.position[k] > self.domain_lower[k]
                    && b.position[k] < self.domain_upper[k])),
            "invalid MPM particle mass/position"
        );
        // Conservative explicit acoustic CFL guard, applied to all engines equally.
        let stiffness = self.young_modulus / (1. - 2. * self.poisson_ratio);
        let speed = (stiffness.max(self.bulk_modulus * 7.) / self.density).sqrt();
        ensure!(
            scene.dt / self.substeps as f64 <= 0.25 * self.cell_width / speed,
            "MPM timestep exceeds shared acoustic CFL bound"
        );
        Ok(())
    }
}

pub fn scene(name: &str, size: usize, seed: u64, dt: f64) -> Result<Scene> {
    let n = (size as f64).cbrt().round() as usize;
    ensure!(
        n >= 4 && n.is_multiple_of(2) && n.pow(3) == size,
        "MPM size must be an even cube (64, 512, 4096, 32768, ...)"
    );
    let spacing = 0.4 / n as f64;
    let material = match name {
        "mpm_elastic_drop" => "elastic",
        "mpm_sand_collapse" => "sand",
        "mpm_dam_break" => "liquid",
        _ => anyhow::bail!("unknown MPM scenario"),
    };
    let cfg = MpmConfig {
        material: material.into(),
        particle_spacing: spacing,
        cell_width: 2. * spacing,
        density: 1000.,
        young_modulus: 10_000.,
        poisson_ratio: 0.2,
        bulk_modulus: 10_000.,
        friction_angle_degrees: 30.,
        // Genesis's rigid coupling requires friction >= 0.01; keep it equal in every engine.
        boundary_friction: if material == "liquid" { 0.01 } else { 0.5 },
        substeps: 20,
        // Genesis pads its grid boundary inward. Keep that numerical boundary
        // outside the shared colliders, even in the coarsest smoke scene.
        domain_lower: [-0.6 - 8. * spacing, -0.6 - 8. * spacing, -8. * spacing],
        domain_upper: [0.6 + 8. * spacing, 0.6 + 8. * spacing, 1.2 + 8. * spacing],
    };
    let mut bodies: Vec<Body> = vec![];
    // A common open tank, floor at z=0 and inner side walls at +/-0.6 m.
    for (pos, half) in [
        ([0., 0., -0.1], [0.8, 0.8, 0.1]),
        ([-0.65, 0., 0.6], [0.05, 0.7, 0.6]),
        ([0.65, 0., 0.6], [0.05, 0.7, 0.6]),
        ([0., -0.65, 0.6], [0.6, 0.05, 0.6]),
        ([0., 0.65, 0.6], [0.6, 0.05, 0.6]),
    ] {
        let mut b = make_body(bodies.len(), false, "box", half, pos);
        b.friction = cfg.boundary_friction;
        bodies.push(b);
    }
    let (nx, ny, bottom, left) = if material == "elastic" {
        (n, n, 0.22, -0.2)
    } else {
        (n / 2, n, 0.02, -0.5)
    };
    for i in 0..size {
        let position = [
            left + (i % nx) as f64 * spacing + spacing / 2.,
            -0.2 + ((i / nx) % ny) as f64 * spacing + spacing / 2.,
            bottom + (i / (nx * ny)) as f64 * spacing + spacing / 2.,
        ];
        let mut b = make_body(bodies.len(), true, "sphere", [spacing / 2.; 3], position);
        b.mass = cfg.density * spacing.powi(3);
        b.inertia = [0.4 * b.mass * (spacing / 2.).powi(2); 3];
        b.friction = cfg.boundary_friction;
        bodies.push(b);
    }
    let scene = Scene {
        schema_version: 1,
        scenario: name.into(),
        size,
        seed,
        gravity: [0., 0., -9.81],
        dt,
        bodies,
        hinges: vec![],
        robot: None,
        mpm: Some(cfg),
    };
    scene.mpm.as_ref().unwrap().validate(&scene)?;
    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdapterResult, Job, hash, validate_result};

    fn job() -> Job {
        Job {
            schema_version: 1,
            scene: scene("mpm_elastic_drop", 512, 42, 1. / 240.).unwrap(),
            engine: "nexus".into(),
            backend: "metal".into(),
            mode: "timing".into(),
            worlds: 1,
            steps: 64,
            warmup: 64,
            capture_stride: 4,
            threads: 8,
        }
    }
    #[test]
    fn canonical_particles_preserve_volume_mass_and_resolution() {
        for name in ["mpm_elastic_drop", "mpm_sand_collapse", "mpm_dam_break"] {
            for size in [64, 512, 4096, 32768] {
                let s = scene(name, size, 42, 1. / 240.).unwrap();
                assert_eq!(hash(&s), hash(&scene(name, size, 42, 1. / 240.).unwrap()));
                let points: Vec<_> = s.bodies.iter().filter(|b| b.dynamic).collect();
                assert_eq!(points.len(), size);
                assert!((points.iter().map(|b| b.mass).sum::<f64>() - 64.).abs() < 1e-8);
                let cfg = s.mpm.as_ref().unwrap();
                assert_eq!(cfg.cell_width, 2. * cfg.particle_spacing);
                assert!(points.iter().all(|p| p.position[0].abs() < 0.6
                    && p.position[1].abs() < 0.6
                    && p.position[2] > 0.));
            }
        }
        assert!(scene("mpm_sand_collapse", 100, 42, 1. / 240.).is_err());
        assert!(scene("mpm_sand_collapse", 512, 42, 1.).is_err());
    }
    #[test]
    fn mpm_rejects_unsupported_engines_multiple_worlds_and_bad_materials() {
        let mut j = job();
        j.validate().unwrap();
        j.engine = "rapier".into();
        assert!(j.validate().is_err());
        j.engine = "genesis".into();
        j.worlds = 2;
        assert!(j.validate().is_err());
        j.worlds = 1;
        j.scene.mpm.as_mut().unwrap().density = f64::NAN;
        assert!(j.validate().is_err());
        let mut j = job();
        j.scene.mpm = None;
        assert!(j.validate().is_err());
    }
    #[test]
    fn mpm_validation_rejects_loss_inactivation_escape_and_floor_tunneling() {
        let j = job();
        let mut r = AdapterResult {
            schema_version: 1,
            engine: j.engine.clone(),
            backend: j.backend.clone(),
            setup_seconds: 0.,
            warmup_seconds: 0.,
            elapsed_seconds: 1.,
            awake_count: j.scene.size,
            final_poses: j
                .scene
                .bodies
                .iter()
                .filter(|b| b.dynamic)
                .map(|b| [b.position[0], b.position[1], b.position[2], 0., 0., 0., 1.])
                .collect(),
            settings: serde_json::json!({}),
            frames: vec![],
            gpu_timing: None,
            memory: None,
        };
        assert!(validate_result(&j, &r).passed);
        r.awake_count -= 1;
        assert!(!validate_result(&j, &r).passed);
        r.awake_count += 1;
        let p = r.final_poses[0];
        r.final_poses[0][2] = -2. * j.scene.mpm.as_ref().unwrap().cell_width;
        assert!(!validate_result(&j, &r).passed);
        r.final_poses[0] = p;
        r.final_poses[0][0] = 100.;
        assert!(!validate_result(&j, &r).passed);
        r.final_poses.pop();
        assert!(!validate_result(&j, &r).passed);
    }
}
