# PhysMark

A local, source-pinned physics benchmark suite with a Rust controller, six real engine adapters, shared scenarios, physical sanity checks, and an offline report containing curves, a leaderboard, screenshots and MP4 videos.

| Engine | Default on this Mac | Implementation |
| --- | --- | --- |
| Rapier | CPU | Native Rust, parallel feature enabled; explicit thread budget |
| Nexus | Metal | Native Rust GPU pipeline |
| Genesis | Metal | Genesis rigid solver through Python |
| Newton | CPU | `SolverMuJoCo(use_mujoco_cpu=False)` — MuJoCo Warp even on CPU |
| MuJoCo | CPU | Native library and Python extension built from the pinned source commit |
| MJX | CPU | MuJoCo MJX **JAX** implementation, JIT + `vmap` |

MuJoCo Warp is a pinned Newton dependency, not a separately benchmarked engine. No engine is replaced by a mock or by another solver when its backend is unavailable. An unavailable backend produces a failed trial.

## Quick start

Prerequisites: Git, Rust (the toolchain file selects the tested version), `uv`, CMake, Ninja, and FFmpeg for MP4 generation. Python 3.12 is managed by `uv`. On macOS, Xcode command-line tools and the Metal toolchain are required by Nexus.

```sh
cd ~/work/physmark
cargo run --release -- fetch
cargo run --release -- build
cargo run --release -- doctor
cargo run --release -- run --suite suites/smoke.toml
```

`build` installs into `.physmark/envs/python`, builds the pinned upstream sources in `.physmark/sources`, and compiles the adapters in release mode. It never installs into your system Python. All Python engines share a version-locked environment in this first release; each trial runs in a fresh process. Native adapters have independent Cargo lockfiles.

Open the printed `results/<run>/index.html` directly in a browser. The report works offline and links to all jobs, raw timings, logs, trajectories, settings and source/machine metadata. GPU trials also collect device timestamps in a separate profiling replay when supported: solid curves show synchronized time and dashed curves show GPU time. The coverage table explains unavailable measurements; the leaderboard uses synchronized time. See [timing methodology](docs/methodology.md#gpu-timestamps). Results and build caches are ignored by Git. No data is uploaded.

```sh
# A scale demonstration with repeated trials and common-renderer videos:
cargo run --release -- run --suite suites/demo.toml

# All eight features, each with two independent worlds:
cargo run --release -- run --suite suites/coverage.toml

# CPU-only comparison:
cargo run --release -- run --engines rapier,mujoco,mjx,newton,genesis --backend genesis=cpu --no-capture

# Native batched execution versus serial independent worlds:
cargo run --release -- run --suite suites/batch.toml

# Larger single-world workloads (intentionally demanding):
cargo run --release -- run --suite suites/scale.toml --no-capture

# Rebuild the report without rerunning physics:
cargo run --release -- report results/<run>

# Export an exact scenario:
cargo run --release -- scene mixed_pile --size 32 --seed 42 --output scene.json
```

A run returns a nonzero exit code if any trial fails. The report still includes those trials. An existing output directory is never overwritten by `run`; select another with `--output`.

## Updating engines

`engines.lock` contains full GitHub commit SHAs. MJX deliberately shares MuJoCo's commit. To select a new upstream tag, branch or commit:

```sh
cargo run --release -- pin genesis <tag-or-commit>
cargo run --release -- pin mujoco <tag-or-commit>  # updates MJX as well
cargo run --release -- fetch
cargo run --release -- build
cargo run --release -- run --suite suites/smoke.toml
```

Review and commit the lockfile changes after validation. An upstream API change may require an adapter change; the smoke suite is the compatibility gate. `requirements.lock` pins Python dependency versions. The adapter Cargo lockfiles pin Rust dependencies, including Nexus's Khal and Vortx revisions. Keep native dependency revisions coherent with an updated Nexus pin. Newton, MuJoCo and its Warp dependency may also require coordinated pin updates (`pin mujoco-warp <ref>`). If dependency requirements change, resolve and review the Python lock and Cargo locks explicitly; the build does not silently select a newer engine.

The runner checks source checkouts against the pins and verifies a build fingerprint. Changes to adapters, shared code, build tools or dependency locks require `build` again. Upstream checkout modifications are rejected. Build products are kept outside the upstream trees.

## Scenarios and coverage

All scenes use metres, kilograms, seconds, Z-up, identical explicit mass/inertia, shared initial transforms, gravity and timestep. A scenario is generated once by Rust and serialized for every engine.

| Scenario | Common features exercised |
| --- | --- |
| `free_fall` | Integration, gravity, independent bodies, analytic sanity check |
| `box_pile` | Box collision detection, contact generation and friction |
| `sphere_pile` | Sphere contacts and dense piles |
| `mixed_pile` | Boxes, spheres and capsules with explicit mass properties |
| `stacks` | Persistent contact, stacking and stability |
| `dominoes` | Rotational dynamics, contact propagation and slender bodies |
| `pendulums` | Fixed-world revolute constraints and anchor drift |
| `friction_ramp` | Inclined static geometry and frictional contact |

Sizes and world counts are independent suite dimensions. Native MuJoCo and Rapier execute worlds serially; MJX uses `vmap`; Nexus, Genesis and Newton use their native batching. The report exposes this distinction. Sleeping/hibernation is disabled everywhere.

This first version covers rigid primitives and world-anchored hinges. It does **not** claim coverage of every common feature: articulated chains, motors, limits, meshes, CCD, ray queries, deformables and differentiability are future extensions. See [methodology](docs/methodology.md) for what the measurements establish and what they do not.

## NVIDIA machines

The Python adapters accept `--backend genesis=cuda,newton=cuda,mjx=cuda`; Nexus accepts CUDA when built with `PHYSMARK_NEXUS_FEATURES=cuda`. These paths are wired but require validation on NVIDIA hardware. The checked dependency lock was resolved on macOS arm64; a Linux CUDA environment requires compatible CUDA-enabled Torch/Warp/JAX packages and a reviewed platform lock. The build helper accepts `PHYSMARK_NEXUS_FEATURES=webgpu` as well. This Mac cannot validate CUDA execution.

## Development

```sh
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
.physmark/envs/python/bin/python -m pytest tests
```

The subprocess protocol and deterministic scene generator live in `crates/physmark-core`; orchestration is in `crates/physmark`; the small Python layer calls the Python-first engines and renders results. Python does not orchestrate benchmark scheduling or sit between the controller and a native engine's timed step loop.

