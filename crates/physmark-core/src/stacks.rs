//! Large brick structures smashed by rolling spheres, scaled from the Nexus stress scenes
//! (`brick_ring_110k` and `pyramids_500k` on its `rbd-performance` branch).
//!
//! Every brick is a 1 × 0.5 × 0.5 m box of density 1; sizes count every dynamic body,
//! spheres included. The last course (ring) or wall (pyramids) is left partial so the
//! count is exact. The ground top is at z = 0 instead of the source's z = 0.5.
use super::{Body, Scene, make_body};
use anyhow::{Result, ensure};
use std::f64::consts::PI;

pub const SCENARIOS: &[&str] = &["brick_ring", "pyramids"];

/// Brick half extents (long axis along local x).
const BRICK: [f64; 3] = [0.5, 0.25, 0.25];
const BRICK_DENSITY: f64 = 1.;
const FRICTION: f64 = 0.5;
const BALL_SPEED: f64 = 30.;

fn solid_box(id: usize, half: [f64; 3], density: f64, position: [f64; 3], angle: f64) -> Body {
    let mut b = make_body(id, true, "box", half, position);
    let mass = density * 8. * half[0] * half[1] * half[2];
    b.inertia = b.inertia.map(|i| i * mass);
    b.mass = mass;
    b.friction = FRICTION;
    b.rotation = [0., 0., (angle / 2.).sin(), (angle / 2.).cos()];
    b
}

/// A sphere resting on the ground at (x, y), rolling along +y without slipping.
fn rolling_ball(id: usize, r: f64, density: f64, x: f64, y: f64) -> Body {
    let mut b = make_body(id, true, "sphere", [r; 3], [x, y, r]);
    let mass = density * 4. / 3. * PI * r.powi(3);
    b.inertia = b.inertia.map(|i| i * mass);
    b.mass = mass;
    b.friction = FRICTION;
    b.velocity = [0., BALL_SPEED, 0.];
    b.angular_velocity = [-BALL_SPEED / r, 0., 0.];
    b
}

fn ground(extent: f64) -> Body {
    let half = (2. * extent).max(100.);
    let mut b = make_body(0, false, "box", [half, half, 0.5], [0., 0., -0.5]);
    b.friction = FRICTION;
    b
}

struct Ring {
    radius: f64,
    courses: usize,
    rows: usize,
}

impl Ring {
    /// Bricks of `course`, `rows` deep at the bottom and tapering linearly to one at the top.
    fn depth(&self, course: usize) -> usize {
        self.rows - course * self.rows / self.courses
    }

    fn ring_radius(&self, j: usize) -> f64 {
        self.radius + 0.25 + 0.52 * j as f64
    }

    fn ring_bricks(&self, j: usize) -> usize {
        (2. * PI * (self.ring_radius(j) - 0.25) / 1.02).floor() as usize
    }

    fn bricks(&self) -> usize {
        (0..self.courses)
            .flat_map(|c| 0..self.depth(c))
            .map(|j| self.ring_bricks(j))
            .sum()
    }
}

/// A running-bond ring wall; 110,000 bodies reproduce the Nexus 110k ring proportions
/// (radius 80, 40 courses, 10 bricks deep).
fn brick_ring(size: usize) -> Result<Vec<Body>> {
    ensure!(size >= 32, "brick_ring needs at least 32 bodies");
    let bricks = size - 1;
    let scale = (size as f64 / 110_000.).sqrt();
    let courses = ((40. * scale).round() as usize).max(2);
    let rows = courses.min(10);
    // The smallest radius holding every brick: the top course is then the only partial one.
    let (mut lo, mut hi) = (0.5, 1.0e5);
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        let ring = Ring {
            radius: mid,
            courses,
            rows,
        };
        if ring.bricks() >= bricks {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let ring = Ring {
        radius: hi,
        courses,
        rows,
    };
    let mut bodies = vec![ground(ring.radius + rows as f64)];
    'courses: for c in 0..courses {
        for j in 0..ring.depth(c) {
            let r = ring.ring_radius(j);
            let n = ring.ring_bricks(j);
            for i in 0..n {
                if bodies.len() > bricks {
                    break 'courses;
                }
                let a = (i as f64 + (c % 2) as f64 * 0.5) / n as f64 * 2. * PI;
                let position = [r * a.cos(), r * a.sin(), 0.25 + 0.5 * c as f64];
                bodies.push(solid_box(
                    bodies.len(),
                    BRICK,
                    BRICK_DENSITY,
                    position,
                    a + PI / 2.,
                ));
            }
        }
    }
    let r = courses as f64 / 5.;
    let y = -(ring.radius + 0.52 * rows as f64 + r + 2.);
    bodies.push(rolling_ball(bodies.len(), r, 10., 0., y));
    Ok(bodies)
}

/// Rows of triangular walls, 30 bricks wide at the base, laid out 16:68 like the Nexus 500k
/// scene (16 × 68 walls and two spheres at 505,922 bodies). Below one wall, the base shrinks.
fn pyramids(size: usize) -> Result<Vec<Body>> {
    ensure!(size >= 4, "pyramids needs at least 4 bodies");
    let triangle = |b: usize| b * (b + 1) / 2;
    let mut base = 30;
    let walls_for = |bricks: usize, base: usize| bricks.div_ceil(triangle(base));
    // Two spheres when there are at least two wall columns.
    let mut balls = 2;
    let mut walls = walls_for(size - balls, base);
    let columns_for = |walls: usize| ((walls as f64 * 16. / 68.).sqrt().round() as usize).max(1);
    if columns_for(walls) < 2 {
        balls = 1;
        walls = walls_for(size - balls, base);
    }
    let bricks = size - balls;
    if walls == 1 {
        base = (1..=30).find(|&b| triangle(b) >= bricks).unwrap();
    }
    let columns = columns_for(walls);
    let rows = walls.div_ceil(columns);
    let (pitch_x, pitch_y) = ((base + 2) as f64, 5.);
    let extent = (columns as f64 * pitch_x).max(rows as f64 * pitch_y) / 2.;
    let mut bodies = vec![ground(extent)];
    'walls: for cx in 0..columns {
        for cy in 0..rows {
            let x0 = (cx as f64 - (columns - 1) as f64 / 2.) * pitch_x;
            let y = (cy as f64 - (rows - 1) as f64 / 2.) * pitch_y;
            for k in 0..base {
                let m = base - k;
                for i in 0..m {
                    if bodies.len() > bricks {
                        break 'walls;
                    }
                    let x = x0 + (i as f64 - (m - 1) as f64 / 2.) * 1.01;
                    let position = [x, y, 0.25 + 0.5 * k as f64];
                    bodies.push(solid_box(bodies.len(), BRICK, BRICK_DENSITY, position, 0.));
                }
            }
        }
    }
    let r = base as f64 / 7.;
    let ball_columns: Vec<usize> = if balls == 2 {
        let last = (columns - 1) as f64;
        let a = (last * 6. / 15.).round() as usize;
        let b = ((last * 9. / 15.).round() as usize).max(a + 1);
        vec![a, b]
    } else {
        vec![(columns - 1) / 2]
    };
    for cx in ball_columns {
        let x = (cx as f64 - (columns - 1) as f64 / 2.) * pitch_x;
        let y = -((rows - 1) as f64 / 2.) * pitch_y - r - 12.;
        bodies.push(rolling_ball(bodies.len(), r, 50., x, y));
    }
    Ok(bodies)
}

pub fn scene(name: &str, size: usize, seed: u64, dt: f64) -> Result<Scene> {
    let bodies = match name {
        "brick_ring" => brick_ring(size)?,
        "pyramids" => pyramids(size)?,
        _ => anyhow::bail!("unknown stack scenario {name}"),
    };
    ensure!(
        bodies.iter().filter(|b| b.dynamic).count() == size,
        "stack body count mismatch"
    );
    Ok(Scene {
        schema_version: 1,
        scenario: name.into(),
        size,
        seed,
        gravity: [0., 0., -9.81],
        dt,
        bodies,
        hinges: vec![],
        mpm: None,
        robot: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_counts_at_every_scale() {
        for name in SCENARIOS {
            for size in [32, 100, 467, 1000, 4096, 28_000, 110_000] {
                let s = scene(name, size, 42, 1. / 120.).unwrap();
                assert_eq!(s.bodies.iter().filter(|b| b.dynamic).count(), size);
                assert!(
                    s.bodies
                        .iter()
                        .all(|b| b.position.iter().all(|x| x.is_finite()))
                );
            }
        }
    }

    #[test]
    fn reference_scales_match_nexus() {
        // 16 × 68 walls of 465 bricks plus two spheres.
        let s = scene("pyramids", 505_922, 42, 1. / 120.).unwrap();
        let spheres: Vec<_> = s.bodies.iter().filter(|b| b.shape == "sphere").collect();
        assert_eq!(spheres.len(), 2);
        assert!((spheres[0].size[0] - 30. / 7.).abs() < 1e-12);
        let ring = scene("brick_ring", 110_000, 42, 1. / 120.).unwrap();
        let ball = ring.bodies.last().unwrap();
        assert_eq!(ball.shape, "sphere");
        assert!((ball.size[0] - 8.).abs() < 1e-12);
        // Radius within a few percent of the source's 80 m.
        let r = ring.bodies[1].position[0].hypot(ring.bodies[1].position[1]);
        assert!((r - 80.25).abs() < 2., "{r}");
    }

    #[test]
    fn bricks_rest_on_the_ground_without_overlap() {
        for name in SCENARIOS {
            let s = scene(name, 5_000, 42, 1. / 120.).unwrap();
            let ground = &s.bodies[0];
            for b in s.bodies.iter().filter(|b| b.dynamic) {
                assert!(b.position[2] >= b.size[2] - 1e-9 || b.shape == "sphere");
                assert!(b.position[0].abs() + 10. < ground.size[0]);
                assert!(b.position[1].abs() + 10. < ground.size[1]);
            }
            // Bricks in one course never overlap their neighbours.
            let bricks: Vec<_> = s
                .bodies
                .iter()
                .filter(|b| b.shape == "box" && b.dynamic)
                .collect();
            for (k, a) in bricks.iter().enumerate().take(400) {
                for b in &bricks[k + 1..k + 40.min(bricks.len() - k - 1) + 1] {
                    if (a.position[2] - b.position[2]).abs() < 1e-9 {
                        let d =
                            (a.position[0] - b.position[0]).hypot(a.position[1] - b.position[1]);
                        assert!(d > 0.5, "{name}: bricks {} and {} overlap", a.id, b.id);
                    }
                }
            }
        }
    }
}
