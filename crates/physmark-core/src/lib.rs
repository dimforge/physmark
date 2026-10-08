//! Versioned, engine-neutral scene and subprocess protocol. Coordinates are metres, kilograms,
//! seconds, Z-up; quaternions are XYZW and body origins coincide with centres of mass.
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::f64::consts::PI;

pub const SCENARIOS: &[&str] = &[
    "free_fall",
    "box_pile",
    "sphere_pile",
    "mixed_pile",
    "stacks",
    "dominoes",
    "pendulums",
    "friction_ramp",
];
pub const ENGINES: &[&str] = &["rapier", "nexus", "genesis", "newton", "mujoco", "mjx"];
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Body {
    pub id: usize,
    pub dynamic: bool,
    pub shape: String,
    /// Box half extents, sphere radius in x, or capsule (radius, unused, cylinder half length).
    pub size: [f64; 3],
    pub position: [f64; 3],
    pub rotation: [f64; 4],
    pub velocity: [f64; 3],
    pub mass: f64,
    pub inertia: [f64; 3],
    pub friction: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hinge {
    pub body: usize,
    pub anchor: [f64; 3],
    pub local_anchor: [f64; 3],
    pub axis: [f64; 3],
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    pub schema_version: u32,
    pub scenario: String,
    pub size: usize,
    pub seed: u64,
    pub gravity: [f64; 3],
    pub dt: f64,
    pub bodies: Vec<Body>,
    pub hinges: Vec<Hinge>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub schema_version: u32,
    pub scene: Scene,
    pub engine: String,
    pub backend: String,
    pub mode: String,
    pub worlds: usize,
    pub steps: usize,
    pub warmup: usize,
    pub capture_stride: usize,
    pub threads: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub step: usize,
    pub poses: Vec<[f64; 7]>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterResult {
    pub schema_version: u32,
    pub engine: String,
    pub backend: String,
    pub setup_seconds: f64,
    pub warmup_seconds: f64,
    pub elapsed_seconds: f64,
    /// Environment-major, dynamic bodies in scene order, positions followed by XYZW quaternion.
    pub final_poses: Vec<[f64; 7]>,
    pub awake_count: usize,
    pub settings: serde_json::Value,
    pub frames: Vec<Frame>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Validation {
    pub passed: bool,
    pub reasons: Vec<String>,
    pub max_floor_penetration: f64,
    pub max_free_fall_error: Option<f64>,
    pub max_joint_error: Option<f64>,
}
impl Job {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1 && self.scene.schema_version == 1,
            "unsupported protocol version"
        );
        ensure!(ENGINES.contains(&self.engine.as_str()), "unknown engine");
        ensure!(
            self.worlds > 0 && self.steps > 0 && self.capture_stride > 0 && self.threads > 0,
            "counts must be positive"
        );
        ensure!(
            self.scene.dt.is_finite() && self.scene.dt > 0.,
            "invalid timestep"
        );
        ensure!(
            self.mode == "timing" || self.mode == "capture",
            "invalid run mode"
        );
        ensure!(
            self.scene.bodies.iter().enumerate().all(|(i, b)| i == b.id),
            "body IDs must be contiguous"
        );
        for b in &self.scene.bodies {
            ensure!(
                ["box", "sphere", "capsule"].contains(&b.shape.as_str()),
                "unsupported shape"
            );
            ensure!(
                b.position
                    .iter()
                    .chain(&b.rotation)
                    .chain(&b.size)
                    .chain(&b.inertia)
                    .all(|x| x.is_finite()),
                "non-finite scene"
            );
            ensure!(
                !b.dynamic || (b.mass > 0. && b.inertia.iter().all(|x| *x > 0.)),
                "invalid mass properties"
            );
            ensure!(b.size[0] > 0., "invalid geometry");
        }
        Ok(())
    }
}
pub fn hash<T: Serialize>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}
pub fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[0], q[1], q[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let c = cross(u, v);
    let d = cross(u, c);
    std::array::from_fn(|i| v[i] + 2. * (q[3] * c[i] + d[i]))
}
fn make_body(id: usize, dynamic: bool, shape: &str, size: [f64; 3], position: [f64; 3]) -> Body {
    let mass = if dynamic { 1. } else { 0. };
    let inertia = match shape {
        "sphere" => [0.4 * mass * size[0] * size[0]; 3],
        "capsule" => {
            // Exact solid capsule: cylinder and two hemispheres, parallel-axis theorem.
            let r = size[0];
            let h = 2. * size[2];
            let vc = PI * r * r * h;
            let vs = 4. * PI * r * r * r / 3.;
            let mc = mass * vc / (vc + vs);
            let ms = mass - mc;
            let iz = 0.5 * mc * r * r + 0.4 * ms * r * r;
            let ix = mc * (3. * r * r + h * h) / 12.
                + ms * (83. * r * r / 320. + (h / 2. + 3. * r / 8.).powi(2));
            [ix, ix, iz]
        }
        _ => [
            mass * (size[1] * size[1] + size[2] * size[2]) / 3.,
            mass * (size[0] * size[0] + size[2] * size[2]) / 3.,
            mass * (size[0] * size[0] + size[1] * size[1]) / 3.,
        ],
    };
    Body {
        id,
        dynamic,
        shape: shape.into(),
        size,
        position,
        rotation: [0., 0., 0., 1.],
        velocity: [0.; 3],
        mass,
        inertia,
        friction: 0.5,
    }
}
pub fn scene(name: &str, size: usize, seed: u64, dt: f64) -> Result<Scene> {
    ensure!(
        size > 0 && size <= 1_000_000,
        "scene size must be in 1..=1000000"
    );
    ensure!(SCENARIOS.contains(&name), "unknown scenario: {name}");
    let mut out = Scene {
        schema_version: 1,
        scenario: name.into(),
        size,
        seed,
        gravity: [0., 0., -9.81],
        dt,
        bodies: vec![],
        hinges: vec![],
    };
    let width = (size as f64).cbrt().ceil() as usize;
    if name != "free_fall" {
        out.bodies.push(make_body(
            0,
            false,
            "box",
            [
                (size as f64).sqrt().max(20.) * 2.,
                (size as f64).sqrt().max(20.) * 2.,
                0.5,
            ],
            [0., 0., -0.5],
        ));
    }
    let mut rng = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut random = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / ((1u64 << 53) as f64)
    };
    if name == "friction_ramp" {
        let mut b = make_body(
            out.bodies.len(),
            false,
            "box",
            [(size as f64).sqrt().max(3.), 3., 0.2],
            [0., 0., 2.],
        );
        b.rotation = [0., (0.18f64).sin(), 0., (0.18f64).cos()];
        out.bodies.push(b);
    }
    for i in 0..size {
        let x = ((i % width) as f64 - width as f64 / 2.) * 1.1;
        let y = (((i / width) % width) as f64 - width as f64 / 2.) * 1.1;
        let z = 0.6 + (i / (width * width)) as f64 * 1.1;
        let shape = match name {
            "sphere_pile" => "sphere",
            "mixed_pile" => ["box", "sphere", "capsule"][i % 3],
            _ => "box",
        };
        let mut b = make_body(
            out.bodies.len(),
            true,
            shape,
            if shape == "capsule" {
                [0.18, 0.18, 0.22]
            } else {
                [0.35; 3]
            },
            [x, y, z],
        );
        match name {
            "free_fall" => {
                b.position = [x, y, 10. + z];
            }
            "stacks" => {
                let n = (size.div_ceil(8) as f64).sqrt().ceil() as usize;
                b.position = [
                    (i / 8 % n) as f64 * 1.5,
                    (i / 8 / n) as f64 * 1.5,
                    0.351 + (i % 8) as f64 * 0.702,
                ];
            }
            "dominoes" => {
                b = make_body(
                    b.id,
                    true,
                    "box",
                    [0.08, 0.35, 0.65],
                    [(i % 40) as f64 * 0.42, (i / 40) as f64 * 2., 0.65],
                );
                if i % 40 == 0 {
                    b.rotation = [0., 0.12f64.sin(), 0., 0.12f64.cos()];
                }
            }
            "pendulums" => {
                b = make_body(b.id, true, "box", [0.12, 0.12, 0.6], [0.; 3]);
                let a = 0.25 + 0.15 * random();
                b.rotation = [0., a.sin(), 0., a.cos()];
                let anchor = [(i % 16) as f64 * 2., (i / 16) as f64 * 2., 2.];
                let local = [0., 0., 0.6];
                let offset = rotate(b.rotation, local);
                b.position = std::array::from_fn(|k| anchor[k] - offset[k]);
                out.hinges.push(Hinge {
                    body: b.id,
                    anchor,
                    local_anchor: local,
                    axis: [0., 1., 0.],
                });
            }
            "friction_ramp" => {
                b.position = [x, y, 3. + z];
            }
            _ => {
                b.position[0] += (random() - 0.5) * 0.1;
                b.position[1] += (random() - 0.5) * 0.1;
                let a = (random() - 0.5) * 0.2;
                b.rotation = [0., 0., a.sin(), a.cos()];
            }
        }
        out.bodies.push(b);
    }
    // Keep the ground under elongated layouts at every scale (notably domino rows).
    if name != "free_fall" {
        let x = out
            .bodies
            .iter()
            .filter(|b| b.dynamic)
            .map(|b| b.position[0].abs() + b.size[0] + 10.)
            .fold(20., f64::max);
        let y = out
            .bodies
            .iter()
            .filter(|b| b.dynamic)
            .map(|b| b.position[1].abs() + b.size[1] + 10.)
            .fold(20., f64::max);
        out.bodies[0].size[0] = x;
        out.bodies[0].size[1] = y;
    }
    Ok(out)
}
pub fn validate_result(job: &Job, r: &AdapterResult) -> Validation {
    let bodies: Vec<_> = job.scene.bodies.iter().filter(|b| b.dynamic).collect();
    let mut v = Validation {
        passed: true,
        reasons: vec![],
        max_floor_penetration: 0.,
        max_free_fall_error: None,
        max_joint_error: None,
    };
    let mut bad = |s: &str| {
        v.passed = false;
        v.reasons.push(s.into());
    };
    if r.schema_version != 1 || r.engine != job.engine || r.backend != job.backend {
        bad("adapter identity/protocol mismatch");
    }
    if r.final_poses.len() != bodies.len() * job.worlds {
        bad("missing or extra dynamic body poses");
        return v;
    }
    if !r.elapsed_seconds.is_finite() || r.elapsed_seconds <= 0. {
        bad("invalid elapsed time");
    }
    if r.awake_count != bodies.len() * job.worlds {
        bad("sleeping or missing dynamic bodies");
    }
    if r.final_poses.iter().flatten().any(|x| !x.is_finite()) {
        bad("non-finite poses");
        return v;
    }
    for (i, p) in r.final_poses.iter().enumerate() {
        let b = bodies[i % bodies.len()];
        let q = [p[3], p[4], p[5], p[6]];
        let qnorm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        if (qnorm - 1.).abs() > 0.01 {
            v.reasons.push("non-unit quaternion".into());
        }
        if job.scene.scenario == "free_fall" {
            let t = job.steps as f64 * job.scene.dt;
            let expected = b.position[2] + b.velocity[2] * t + 0.5 * job.scene.gravity[2] * t * t;
            let e = ((p[0] - b.position[0]).powi(2)
                + (p[1] - b.position[1]).powi(2)
                + (p[2] - expected).powi(2))
            .sqrt();
            v.max_free_fall_error = Some(v.max_free_fall_error.unwrap_or(0.).max(e));
        } else {
            let extent = match b.shape.as_str() {
                "sphere" => b.size[0],
                "capsule" => b.size[0] + b.size[2] * rotate(q, [0., 0., 1.])[2].abs(),
                _ => (0..3)
                    .map(|k| {
                        let mut ax = [0.; 3];
                        ax[k] = 1.;
                        b.size[k] * rotate(q, ax)[2].abs()
                    })
                    .sum(),
            };
            v.max_floor_penetration = v.max_floor_penetration.max((extent - p[2]).max(0.));
        }
        if let Some(j) = job.scene.hinges.iter().find(|j| j.body == b.id) {
            let a = rotate(q, j.local_anchor);
            let e = (0..3)
                .map(|k| (p[k] + a[k] - j.anchor[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            v.max_joint_error = Some(v.max_joint_error.unwrap_or(0.).max(e));
        }
    }
    if v.max_floor_penetration > 0.15 {
        v.reasons
            .push("floor penetration exceeds 0.15 m baseline gate".into());
    }
    if v.max_joint_error.is_some_and(|x| x > 0.08) {
        v.reasons
            .push("hinge anchor error exceeds 0.08 m baseline gate".into());
    }
    let fall_tolerance = 0.02 + 9.81 * job.steps as f64 * job.scene.dt * job.scene.dt;
    if v.max_free_fall_error.is_some_and(|x| x > fall_tolerance) {
        v.reasons
            .push("free-fall error exceeds timestep-scaled tolerance".into());
    }
    v.reasons.sort();
    v.reasons.dedup();
    v.passed = v.reasons.is_empty();
    v
}
pub fn read_job() -> Result<(Job, std::path::PathBuf)> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        bail!("usage: adapter JOB.json RESULT.json");
    }
    let j: Job = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    j.validate()?;
    Ok((j, args[2].clone().into()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scenes_reproduce_and_validate() {
        for n in SCENARIOS {
            let s = scene(n, 19, 42, 1. / 120.).unwrap();
            assert_eq!(hash(&s), hash(&scene(n, 19, 42, 1. / 120.).unwrap()));
            assert_eq!(s.bodies.iter().filter(|b| b.dynamic).count(), 19);
        }
    }
    #[test]
    fn validation_catches_missing_and_sleeping_bodies() {
        let j = Job {
            schema_version: 1,
            scene: scene("free_fall", 1, 42, 0.01).unwrap(),
            engine: "rapier".into(),
            backend: "cpu".into(),
            mode: "timing".into(),
            worlds: 1,
            steps: 10,
            warmup: 1,
            capture_stride: 1,
            threads: 1,
        };
        let mut r = AdapterResult {
            schema_version: 1,
            engine: "rapier".into(),
            backend: "cpu".into(),
            setup_seconds: 0.,
            warmup_seconds: 0.,
            elapsed_seconds: 1.,
            final_poses: vec![],
            awake_count: 0,
            settings: serde_json::json!({}),
            frames: vec![],
        };
        assert!(!validate_result(&j, &r).passed);
        let b = &j.scene.bodies[0];
        r.final_poses = vec![[
            b.position[0],
            b.position[1],
            b.position[2] - 0.04905,
            0.,
            0.,
            0.,
            1.,
        ]];
        assert!(!validate_result(&j, &r).passed);
        r.awake_count = 1;
        assert!(validate_result(&j, &r).passed);
        r.final_poses[0][0] = f64::NAN;
        assert!(!validate_result(&j, &r).passed);
    }
    #[test]
    fn hinge_initial_anchors_match() {
        let s = scene("pendulums", 8, 42, 0.01).unwrap();
        for j in &s.hinges {
            let b = &s.bodies[j.body];
            let a = rotate(b.rotation, j.local_anchor);
            for (k, offset) in a.iter().enumerate() {
                assert!((b.position[k] + offset - j.anchor[k]).abs() < 1e-12);
            }
        }
    }
}
