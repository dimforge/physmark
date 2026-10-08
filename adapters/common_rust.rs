use engine::prelude::*;
use physmark_core::Scene;
fn vector(v: [f64; 3]) -> Vector {
    Vector::new(v[0] as f32, v[1] as f32, v[2] as f32)
}
fn build_world(scene: &Scene) -> (PhysicsWorld, Vec<RigidBodyHandle>) {
    let mut world = PhysicsWorld::new();
    world.gravity = vector(scene.gravity);
    world.integration_parameters.dt = scene.dt as f32;
    let mut handles = vec![];
    for b in &scene.bodies {
        let rb = if b.dynamic {
            RigidBodyBuilder::dynamic()
        } else {
            RigidBodyBuilder::fixed()
        };
        let rb = rb
            .pose(Pose::from_parts(
                vector(b.position),
                Rotation::from_xyzw(
                    b.rotation[0] as f32,
                    b.rotation[1] as f32,
                    b.rotation[2] as f32,
                    b.rotation[3] as f32,
                ),
            ))
            .linvel(vector(b.velocity))
            .can_sleep(false)
            .linear_damping(0.)
            .angular_damping(0.);
        let rb = if b.dynamic {
            rb.additional_mass_properties(MassProperties::new(
                Vector::ZERO,
                b.mass as f32,
                vector(b.inertia),
            ))
        } else {
            rb
        };
        let s = b.size.map(|x| x as f32);
        let collider = match b.shape.as_str() {
            "sphere" => ColliderBuilder::ball(s[0]),
            "capsule" => ColliderBuilder::capsule_z(s[2], s[0]),
            _ => ColliderBuilder::cuboid(s[0], s[1], s[2]),
        }
        .density(0.)
        .friction(b.friction as f32)
        .restitution(0.);
        let (handle, _) = world.insert(rb, collider);
        handles.push(handle);
    }
    for j in &scene.hinges {
        // The canonical ground is a real collider-backed static body in both engines.
        // Reusing it avoids a second, engine-specific representation of world anchors.
        let ground = scene
            .bodies
            .iter()
            .find(|b| !b.dynamic)
            .expect("hinges require static ground");
        let q = ground.rotation;
        let inv = [-q[0], -q[1], -q[2], q[3]];
        let local = physmark_core::rotate(
            inv,
            std::array::from_fn(|k| j.anchor[k] - ground.position[k]),
        );
        let joint = RevoluteJointBuilder::new(vector(j.axis))
            .local_anchor1(vector(local))
            .local_anchor2(vector(j.local_anchor));
        world
            .impulse_joints
            .insert(handles[ground.id], handles[j.body], joint, true);
    }
    world.wake_up_all(true);
    (world, handles)
}
#[allow(dead_code)]
fn pose_array(p: &Pose) -> [f64; 7] {
    let t = p.translation;
    let q = p.rotation;
    [
        t.x as f64, t.y as f64, t.z as f64, q.x as f64, q.y as f64, q.z as f64, q.w as f64,
    ]
}
