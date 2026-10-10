# Measurement contract

## What is measured

Each timing trial has its own process. Trials run serially in a deterministic shuffled order. Setup and JIT compilation occur before a full warmup trajectory (`max(warmup, steps)`) that exercises the same horizon and contact paths as the measured rollout. The scene then resets to the canonical initial state, including solver buffers where the engine exposes them. A synchronized monotonic wall-clock interval covers the fixed number of physics steps across all worlds. Final state readback, physical checks, report generation and rendering happen after the interval. Setup and warmup durations are retained separately.

The elapsed interval includes each adapter's actual stepping API overhead. MuJoCo uses its native `nstep` loop, Rapier a Rust loop, MJX a compiled JAX `fori_loop`, and Genesis/Newton their Python stepping APIs. GPU command submission, required synchronization and internal housekeeping are included. This is an end-to-end warm rollout comparison, not a kernel-only timer. CUDA graph capture is not enabled in the initial adapters.

`ms_per_step = elapsed_seconds × 1000 / steps` covers **all** worlds. `world_steps_per_second = steps × worlds / elapsed_seconds`. It does not multiply throughput by body count. Curves show per-case medians with interquartile bands across seeds/repeats. A one-repeat smoke run is a functionality check, not a statistically useful ranking.

The leaderboard uses a geometric mean of per-case speed ratios relative to native MuJoCo. Every case receives equal weight. Every requested trial for an engine must pass; missing/failed cases never improve its score. A score is withheld when the MuJoCo baseline is incomplete. CPU and GPU backends appear explicitly; do not interpret mixed-device rankings as an intrinsic algorithm ranking.

## Physical comparability

All engines receive the same scene JSON. Static/dynamic status, primitive dimensions, mass, diagonal inertia, gravity, initial transforms, friction, hinge anchors and timestep are explicit. Restitution and damping are zero. Sleeping is disabled; Rapier checks actual awake bodies, other adapters verify the disabled setting and report the number of simulated bodies (rather than inferring activity from nonzero velocity).

Solver algorithms, precision, iteration meaning, contact regularization and friction models differ. Rapier/Nexus use four solver substeps; MuJoCo/MJX/Newton use Euler and Newton solver iterations capped at 50; Genesis uses its rigid solver with 50 iterations. MuJoCo-based contact regularization uses `solref="0.02 1"`. Exact effective settings are attached to results. MuJoCo is double precision; other current adapters are single precision. Equal iteration numbers do not imply equal accuracy or cost.

The validation gates reject missing/nonfinite poses, nonunit quaternions, sleeping bodies, excessive floor penetration, hinge-anchor drift and timestep-scaled free-fall error. These gates are intentionally broad sanity checks (0.15 m floor penetration, 0.08 m anchor drift); passing is **not** a certificate of matched physical accuracy. A serious performance study should introduce workload-specific fidelity tiers, sweep solver settings, add energy/contact residual metrics and rank only configurations satisfying the same targets.

Only final physics state is gated in timing trials. Capture replays provide trajectories for visual diagnosis. Collision geometry is engine neutral, but analytic primitive collision algorithms can still differ.

## GPU timestamps

Every passing GPU wall-time trial is followed by a **separate profiling process** with the same scene, seed, worlds, timestep, warmup and measured horizon. It warms up and resets independently. Profiling, timestamp readback and trace export never run in the primary synchronized interval. Both replays must pass the physics gates before a GPU sample is accepted. CPU backends have no GPU sample. A capability-only response may return without running physics; it contains no duration and cannot enter a curve.

| Engine/backend | Device timing method | Scope |
| --- | --- | --- |
| Nexus Metal | Khal Metal counter samples | Sum of timed compute-pass durations |
| Nexus CUDA | Khal CUDA events | Sum of timed compute-pass durations |
| Nexus WebGPU | Khal timestamp queries, if supported by adapter/device | Sum of timed compute-pass durations |
| Newton CUDA (MuJoCo Warp solver) | Warp `ScopedTimer` activity results backed by CUDA events | Sum of Warp kernel, builtin, copy and memset durations |
| Genesis CUDA | PyTorch/Kineto CUPTI activity trace, including non-Torch CUDA kernels | Union of GPU kernel, copy and memset intervals on device 0 |
| MJX CUDA | JAX/XLA CUPTI activity trace | Union of GPU stream activity intervals on device 0 |
| Genesis Metal | Unavailable with pinned Quadrants 1.3.3 | Its Metal runtime does not connect a device profiler; the generic kernel profiler is a host-clock profiler |

Nexus collects each step exactly once, in batches of eight independent query managers. It synchronizes and harvests every batch so pending readbacks cannot silently skip steps. The adapter rejects empty, invalid or potentially exhausted query buffers. Raw per-step durations and totals per pass label are saved in `gpu-samples.json`. Profiling pass boundaries and batch barriers can change GPU scheduling; these samples describe the instrumented replay, not a subtraction from the uninstrumented wall time. Per-pass counter sampling can have significant overhead: GPU sample totals can exceed the uninstrumented synchronized duration. There is no assertion that the GPU curve must be lower. Query-buffer allocation adapts to supported sizes (up to 2048 pairs, the 32 KiB sample-buffer limit of some Metal devices), and records the selected capacity.

Warp's outer `ScopedTimer.elapsed` is a **host** duration and is not used. Individual `timing_results[].elapsed` are device-event durations. Their sum can double-count overlapping streams and excludes work outside Warp's instrumentation. Genesis and MJX instead use a union of intervals to count overlapping device work once. Host trace lanes, CUDA API calls, aggregate XLA module lanes and other devices are excluded. These scopes are intentionally recorded rather than presented as equivalent kernel-only measurements. GPU clocks, contention and profiling overhead can differ between the two processes.

CUPTI must be available and permitted on the NVIDIA machine. If a profiler produces only host events, GPU timing is **unavailable**, never zero and never replaced with synchronized time. Unsupported APIs and devices remain visible in the coverage table. Profiler process errors/timeouts and malformed data have a separate **error** status, preserve a passing wall result, and make the overall command fail so automation notices. Failed physics replays cannot supply a device sample.

Reports draw solid synchronized-time curves and dashed GPU-timestamp curves with independent complete-coverage requirements. The GPU curve requires the same method and scope across repeats/seeds. Both use the median and interquartile range of per-rollout average milliseconds per step (across all worlds). The leaderboard uses synchronized time only. Old result files load unchanged and show GPU timing as not collected. Every GPU attempt retains `gpu-job.json`, `gpu.log`, and, on success, `gpu-result.json`; available profilers also retain their raw samples/traces.

Source references: [Khal pinned backends](https://github.com/dimforge/khal/tree/58b02f2379f3542dbf5269804c3f2dfcdfc4e0b4/crates/khal/src/backend), [Genesis profiling guide](https://genesis-world.readthedocs.io/en/latest/user_guide/developers/profiling.html), [JAX profiling guide](https://docs.jax.dev/en/latest/profiling.html), [Quadrants 1.3.3 Metal runtime](https://github.com/Genesis-Embodied-AI/quadrants/blob/00423caa93b620f6f3ebbb5d3f26e74cf56ecf5b/quadrants/runtime/program_impls/metal/metal_program.cpp), [Quadrants profiler selection](https://github.com/Genesis-Embodied-AI/quadrants/blob/00423caa93b620f6f3ebbb5d3f26e74cf56ecf5b/quadrants/program/kernel_profiler.cpp).

## RAM and GPU memory

Each passing timing trial has a **separate fresh-process memory replay**, using the same initial scene, world count, warmup and full rollout. The stepping API and batching match the unprofiled trial. A native sampling thread polls at a target interval of 10 ms; synchronized snapshots are also taken before setup, after setup, after warmup, after reset and after the rollout. Collection stops before final pose readback, validation, serialization or rendering. The profiler’s elapsed time never supplies a performance sample. Its final physics state must pass the same gates before its memory measurements are accepted.

| Metric | Counter | Meaning |
| --- | --- | --- |
| RAM, macOS | `proc_pid_rusage` resident size and `getrusage(RUSAGE_SELF).ru_maxrss` | Current process RSS and OS resident-set high-water mark, in bytes |
| Physical footprint, macOS | `proc_pid_rusage` V4 `ri_phys_footprint` / `ri_lifetime_max_phys_footprint` | Current physical footprint and OS lifetime maximum, including accounted/compressed memory |
| RAM, Linux | `/proc/self/status` `VmRSS` / `VmHWM` | Current process RSS and kernel RSS high-water mark, converted from KiB to bytes |
| GPU, Metal | `MTLDevice.currentAllocatedSize` on the process’s default device | Actual allocated Metal resources, covering both Nexus and Genesis/Quadrants; not limited to Torch’s allocator |
| GPU, Linux CUDA | NVML v2 compute-process `usedGpuMemory` for the adapter PID, summed over physical devices | Actual driver-reported device memory, including context and retained pool overhead; not global device usage or free-memory deltas |

**RAM and GPU are different views of memory, not additive totals.** Apple Silicon has unified memory, and GPU allocations can overlap the process’s RAM accounting. RSS includes resident shared pages; it is not exclusive/private ownership. Metal allocation size is not a measurement of physical residency. NVML and Metal have different scopes and should not be interpreted as interchangeable allocator counters. CPU backends show GPU memory as not applicable. WebGPU has no verified portable allocation counter here and reports unavailable. CUDA/NVML is wired and source-checked but has not been validated on NVIDIA hardware. NVML permission failures, unsupported drivers, inaccessible devices, and missing per-PID allocations do not produce a fabricated zero.

RAM peaks are kernel-maintained high-water marks from process launch through the end of physics. They include module imports, setup, JIT, warmup, reset, the sampler itself and any retained caches. GPU peaks are **sampled maxima**, lower bounds on the true transient peak: a 10 ms target interval is not a scheduling guarantee. The phase snapshots also retain current byte counts. Persistent allocations and process overhead are included; these are absolute footprints, not estimated per-body costs or baseline-subtracted deltas. Child-process allocations are outside the scope of these per-PID counters. A counter read failure makes that metric unavailable for the replay, while preserving other valid counters and the primary timing result.

`memory-result.json` stores the metrics, counter methods/scopes, sample counts, five phase snapshots, unified-memory flag and completed step count. `memory-job.json` and `memory.log` retain replay evidence. Reports show median peak bytes and interquartile bands across repeats/seeds, with independent complete-coverage and consistent-counter requirements. The existing speed leaderboard remains based on synchronized times. A failed memory replay is recorded as an error and makes the overall command fail; counter unavailability is an explicit capability outcome. Rebuilding old reports does not backfill memory or rerun physics.

Rust adapters and Python engines use the same native sampler in `physmark-core`. `build` produces its shared library as well as the adapters. The 1 Hz macOS controller memory cap is a separate process-safety mechanism, not the memory measurement source.

Counter references: [Apple Metal allocation counter](https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize), [Apple rusage structure](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/resource.h), [Linux proc status](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html), [NVIDIA NVML process queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html).

## Engines and environments

| Engine | Implementation |
| --- | --- |
| Rapier | Native Rust, `parallel` feature, explicit thread budget; worlds run serially |
| Nexus | Native Rust GPU pipeline (Metal, CUDA, WebGPU or CPU feature); native batching |
| Genesis | Genesis rigid and MPM solvers through Python; native batching |
| Newton | `SolverMuJoCo(use_mujoco_cpu=False)`, i.e. MuJoCo Warp, for rigid bodies; `SolverImplicitMPM` for MPM |
| MuJoCo | Native library and Python extension built from the pinned source commit; worlds run serially |
| MJX | MuJoCo MJX JAX implementation, JIT-compiled and `vmap`ped across worlds |
| Isaac Sim | PhysX 5 through Isaac Sim's wheels, USD stage built directly (see below) |

MuJoCo Warp is a pinned Newton dependency, not a separately benchmarked engine. MJX shares MuJoCo's pin; Newton, MuJoCo and MuJoCo Warp may need coordinated pin updates (`pin mujoco-warp <ref>`). No engine is replaced by a mock or by another solver when its backend is unavailable: an unavailable backend produces a failed trial, and the full round skips the engine instead.

`build` installs the Python engines into `.physmark/envs/python` (Isaac Sim into `.physmark/envs/isaac`), builds pinned upstream sources from `.physmark/sources`, and never installs into the system Python. `requirements.lock` was resolved on macOS arm64; every locked package also has Linux distributions, and Torch pulls its CUDA libraries as dependencies there. On NVIDIA machines the build adds the JAX CUDA 13 plugin matching the locked JAX; this CUDA path has not yet been validated. Linux memory collection uses NVML per-process counters when the driver exposes them; Windows has no RAM/GPU counters yet and reports them unavailable.

Coverage includes rigid primitives, world-anchored hinges, the MJCF robots and single-environment MPM. General mesh collision, CCD, ray queries, other deformable solvers and differentiability are not benchmarked.

## Runs, resumption and skips

A run returns a nonzero exit code if any trial fails or is skipped; the report still includes those trials. A fresh run never overwrites an existing output directory. `--output <dir> --resume` continues an interrupted run: completed results are kept, interrupted attempts are archived, and resumed provenance is saved in `resume-history.json`. Resumption requires the same suite, engine pins, backends, requested threads, Python package versions, Rust version and machine metadata.

`--skip-case 'engine:scenario:size:worlds=reason'` excludes a known resource-failing case; skipped trials stay visible and prevent a complete-suite score. Suites with `skip_after_resource_limit` record the remaining identical/larger environment batches of an engine/scene/size as skipped after a timeout or memory-cap failure. Environment counts run in ascending order, with deterministic shuffled trials within each count. On macOS, `PHYSMARK_MEMORY_LIMIT_GIB` terminates an adapter whose physical footprint exceeds the limit (sampled once per second and recorded in provenance).

In reports, solid curves show synchronized wall time and dashed curves device timestamps from the separate profiling replay; the leaderboard uses synchronized time. On unified-memory GPUs (Apple), RAM and GPU allocations overlap and must not be added.

## Repeatability and scope

Record full source pins, Python/Rust locks, adapter build fingerprint, scene/job hashes, harness revision/dirty status, machine/OS/GPU description, settings and raw samples. The harness fingerprint includes uncommitted source content; publish results only with the corresponding source revision and lockfiles. Timings are hardware-, OS-, power-, temperature- and driver-dependent. Running unrelated heavy work affects them. Thread limits are requested through Rayon and common runtime environment variables; those are not a promise of identical framework worker counts.

The report renderer consumes a separate capture process; it uses common geometry, camera rules, colours and video timing for all engines. It never calls an engine renderer. Capture files store only world 0 to bound output size. Rendering costs are excluded and renderer failures remain visible in `render-status.json`. Captures above 2,000 bodies use the pinned VTK dependency for instanced, depth-buffered offscreen rendering of every body; smaller captures use Matplotlib. Both paths use the same canonical primitive meshes, engine colors, camera direction and simulated-time playback. The renderer and rendered body count are recorded with large-scene media. Report rendering runs after the physics trials, so GPU rendering work does not overlap timed CPU trials.

Suite presets are intended starting points, not claims that every engine can finish every largest case. Timeouts and errors remain in the report, with their logs. No automatic backend fallback, accuracy downgrade, capacity reduction or engine substitution is used to manufacture a passing result.


## Isaac Sim

The Isaac adapter creates the canonical scene as USD prims (Z-up, metres, kilograms): cubes scaled to the box half extents, spheres and Z-axis capsules with collision APIs, explicit mass, diagonal inertia and zero centre-of-mass offset, and one physics material per friction value (static = dynamic friction, zero restitution). Hinges are revolute joints anchored to the world frame, with the joint X axis rotated onto the hinge axis. PhysX runs TGS with 4 position and 1 velocity iteration per body, stabilization and CCD disabled, and a zero sleep threshold, so bodies never sleep. With `backend=cuda`, dynamics and broadphase run on the GPU and USD write-back is disabled; GPU contact, patch and pair buffers are sized from the total body count. Environments are offset on a grid spaced by twice the scene's extent plus 1 m, so grounds never overlap; poses are reported relative to each environment's origin. Large environment counts therefore place bodies kilometres from the origin, where float32 positions are coarser. Initial velocities go through the tensor API in world frame. Device time, when CUPTI is available, is the union of all CUDA activity in the process recorded by the PyTorch profiler, PhysX kernels included. This configuration follows PhysX defaults and has not been run on NVIDIA hardware in this repository.

## Brick structures

`brick_ring` and `pyramids` port the `brick_ring_110k` and `pyramids_500k` stress scenes from the Nexus `rbd-performance` branch (`crates/examples3d/stress_scenes`). Bricks are 1 × 0.5 × 0.5 m boxes of density 1 (0.25 kg) with friction 0.5. The ring is laid in running bond, 10 bricks deep at the bottom and tapering linearly to one at the top; a sphere of radius courses/5 and density 10 rolls into it. The pyramids are triangular walls 30 bricks wide at the base, 465 bricks each, on a 16:68 grid; two spheres of radius 30/7 and density 50 roll down two wall columns. Spheres start resting on the ground with 30 m/s linear velocity and the matching rolling angular velocity, which every adapter applies explicitly (MuJoCo, MJX and Genesis take free-joint angular velocity in the body frame; Newton derives joint velocities from world-frame body twists).

Size counts dynamic bodies, spheres included, and is exact: 110,000 reproduces the ring (radius ≈ 80 m, 40 courses), and 505,922 reproduces the 16 × 68 pyramid field. Other sizes scale the ring's course count with the square root of the size and solve for the radius; pyramids add or remove walls, shrinking the base only below one wall. The last course or wall is partial. Compared with the source, the ground top is at z = 0 instead of 0.5, gravity is 9.81 instead of 10 m/s², and sleeping is disabled. The floor-penetration gate is 0.5 m (one brick height) instead of 0.15 m: the 110K ring's 21-tonne sphere outweighs a brick 86,000:1 and crushed bricks up to 0.28 m into the floor on Nexus, while a brick sunk entirely below the surface still fails. Genesis's collision-pair budget is raised to four pairs per body (its default of 150 overflows on brick walls). The 110K–500K suites do not capture replays: one frame of 500K poses is about 70 MB of JSON.

## MPM workloads

MPM is separate from rigid-body scoring. Only Nexus, Genesis and Newton participate; each trial has exactly one environment. A suite may declare `baseline = "genesis"`; rigid suites retain native MuJoCo as the default. Unsupported engines are filtered before the MPM suite is recorded. Missing or failed baseline cases withhold the overall score. No kernel-only measurements enter that score.

The shared scene contains an `mpm` block. Its dynamic `bodies` entries represent material points: position, velocity, mass and a sphere radius used only for common-renderer visualization and particle volume. They are never instantiated as rigid colliders. Captures reuse the pose transport with identity quaternions as padding; they do not represent material deformation tensors. Every point is rendered. Static box entries define the same floor and four tank walls in every adapter.

The three workloads are an elastic cube drop, a granular column collapse and a fluid dam break. All contain 0.064 cubic metres at 1,000 kg/m³ (64 kg total). Size must be an even cube; particle spacing is 0.4 divided by its cube root, with two points per grid-cell edge. Scaling refines a fixed physical scene instead of enlarging the material. Domain padding keeps Genesis's internal numerical boundary outside the common colliders. Nexus and Newton use dynamic sparse grids where supported; Genesis uses a bounded grid. Allocation strategy remains an engine difference and is disclosed.

Presets range from 512 to 195,112 particles (58³); the same 0.4 m cube of material is refined, so the grid cell shrinks with the particle spacing. CPIC rigid coupling is disabled in Nexus and Genesis so that every engine resolves the tank walls through its grid boundary handling; Nexus re-enables CPIC whenever it rebuilds its rigid coupling, so its adapter checks the flag again after the timed rollout.

Every adapter receives the same frame timestep and 20 substeps. The presets use 1/240 s frames; a conservative shared acoustic CFL check rejects unsuitable combinations. The common elastic parameters are Young's modulus 10 kPa and Poisson ratio 0.2. Sand uses a 30-degree friction angle and zero cohesion/hardening. Liquid has zero viscosity and a 10 kPa initial pressure tangent: Nexus uses a Tait law (gamma 7, coefficient K/7), Genesis uses its lambda J(J-1) law with lambda=K, and Newton uses its implicit yield-surface rheology with bulk modulus K. These are initial-parameter comparisons, not identical constitutive laws. Nexus/Genesis elasticity is Neo-Hookean; Newton's implicit elasticity differs. Report settings preserve each actual material/solver configuration.

Genesis reserves exactly N points, then explicitly sets all positions, velocities and activation flags before saving its reset state. Its adapter checks every particle's mass individually; Genesis's float32 total drifts beyond a 1e-5 relative tolerance near 200K particles. Nexus rebuilds the canonical particle state after warmup. Newton rebuilds solver state after warmup, outside timing, to clear internal history. Setup, JIT, reset, final readback, validation and rendering are excluded from stepping. Per-frame transfers, solver work and Newton's native collision projection are included. Profiling and memory collection use the same independent-replay design as rigid trials. Genesis/Metal device timestamps remain unavailable; Nexus uses native GPU timestamps, and Newton/CUDA uses Warp device activities when available.

MPM validation checks particle count, activation count, finite positions, normalized transport quaternions, domain escape, and floor penetration no larger than one grid cell. These are broad failure gates; they do not quantify volume conservation, stress accuracy, convergence, or material fidelity. All particles stay active, with no sleeping or adaptive particle removal. Dense Newton/CPU is useful for integration checks but is excluded from the requested Mac GPU benchmark round.

Implementation references are the pinned [Nexus MPM pipeline](https://github.com/dimforge/nexus/blob/1cfbd768db6ac770f302a7f432a43df42953eb67/src_mpm/pipeline.rs), [Genesis MPM materials](https://github.com/Genesis-Embodied-AI/Genesis/tree/02908fe6474d8b073b662347a8794614fa250658/genesis/engine/materials/MPM), and [Newton implicit MPM solver](https://github.com/newton-physics/newton/blob/dd2dace79fd47dd6c198f4a7a2f894c2aa47dcc4/newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py).

### Menagerie robotics

Robots are only benchmarked standing on the ground: every robot is floating-base,
starts crouched with its feet 3 mm above the floor, and stands under position
servos, so floor contacts are active throughout. Two models are available:
`robot_go1` (quadruped, 12 actuators) and `robot_t1` (humanoid, 23 actuators).
Job validation rejects robot scenes whose base does not start below its
standing-height ceiling.

`robot_t1` imports the Booster T1 from the same Menagerie commit: 24 links,
23 position actuators, box feet. It starts from its `home` keyframe with hips,
knees and ankles further flexed (hip pitch -0.6, knee 1.2, ankle pitch -0.6 rad,
feet level) and holds the `home` actuator targets. Menagerie's kp = 75, kv = 5
servos assume a balancing policy; with no controller, MuJoCo probes showed the
robot tipping over after about 1.5 s, so all servos use kp = 300, kv = 20. Class
defaults, `inheritrange` and joint `actuatorfrcrange` are resolved into explicit
`ctrlrange`/`forcerange` on each actuator, keeping Menagerie's torque limits,
because not every importer supports those attributes. The base must finish
between 0.55 and 0.75 m, upright and within 1 m of its start.

`robot_go1` imports the Unitree Go1 from the `menagerie` commit in `engines.lock`.
Every environment contains one free-floating, 13-link robot with 12 position
actuators. The benchmark starts from a crouch (hip 1.2 rad, knee -2.4 rad) with
feet 3 mm above the ground, and continuously holds the original MJCF `home`
keyframe's actuator targets (0, 0.9, -1.8 rad per leg). The two-second rollout
includes the rise and subsequent standing motion; it is not a locomotion policy.

`tools/robot_scene.py` retains the source's inertias, joint parameters, collision
primitives and actuator limits. It removes visual meshes, adds the ground, and
excludes self-contacts consistently across engines. Rendered links use the
original collision primitives, with poses at MJCF link origins. The source model
and resulting portable MJCF are recorded in each job. Rapier and Nexus use the
pinned Rapier MJCF loader and reduced-coordinate multibody joints; MuJoCo and
Genesis use their native MJCF loaders. The current Nexus robotics preset uses four
substeps and four internal PGS iterations per substep, explicit Coriolis forces, and
`friction_in_bias_pass = false`. Both `substep_refresh` and
`substep_refresh_light` are false. It reserves 64 collision pairs and 32
multibody contact-constraint slots per environment, with automatic growth
enabled. Each contact point consumes one normal and two friction slots.
Robot results record the applied settings, requested capacities, final total
capacities and the latest asynchronous demand counters. The latter are not
peak measurements. Untimed capture replays also read actual GPU demand after
each step into `capacity-samples.json` and reject capacity overflow. Neither
the timing nor the memory replay performs those extra diagnostic readbacks.
These settings apply to every robot workload.
Solver/contact approximations differ;
these are comparable workloads, not claims of identical dynamics.

Scene size is **robots per environment** (currently one); `worlds` sweeps independent
robots up to 4,096. Each Go1 environment must finish upright, with base height between
0.22 and 0.50 m and horizontal drift below 1 m after at least one simulated second.
All 13 links must remain present and awake. Failing robots are excluded from timing
rankings. The initial positions are checked against MuJoCo forward kinematics.
Sleeping remains disabled in every engine.

### Full round resource policy

`cargo run --release -- round` (`tools/full_round.py`) runs all parts serially, with
Rapier parallelism enabled and up to 8 workers requested. Backends follow the
detected hardware (see the README); Newton, MJX and Isaac Sim run only on CUDA, and
engines that fail a two-step probe trial are skipped and listed, never substituted.
The plan, including probe outcomes, is saved in `round-plan.json` and reused on
`--resume` so resumed suites keep their backends. CPU subprocesses are limited to
120 seconds including setup, warmup and rollout; GPU budgets are declared in each
preset. On macOS, a physical-footprint cap of 56% of RAM limits
each adapter. `skip_after_resource_limit` records subsequent identical/larger
environment batches as skipped after a timeout or memory-cap error. Accuracy-gate
failures remain distinct from resource failures.

Liquid MPM boundary friction is 0.01, the minimum accepted by Genesis's rigid
boundary material, and is shared by all MPM adapters. Liquid viscosity remains zero.

Newton MPM uses the Q1 velocity, P0 strain and particle collider basis combination from its reference dam-break example. The initial B2/default-strain combination produced non-finite sand/liquid states in CPU probes and is not used.
