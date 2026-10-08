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

Nexus collects each step exactly once, in batches of eight independent query managers. It synchronizes and harvests every batch so pending readbacks cannot silently skip steps. The adapter rejects empty, invalid or potentially exhausted query buffers. Raw per-step durations and totals per pass label are saved in `gpu-samples.json`. Profiling pass boundaries and batch barriers can change GPU scheduling; these samples describe the instrumented replay, not a subtraction from the uninstrumented wall time. On the M4 Max, per-pass counter sampling has significant overhead: GPU sample totals can exceed the uninstrumented synchronized duration. There is no assertion that the GPU curve must be lower. Query-buffer allocation adapts to supported sizes (up to 2048 pairs, the observed 32 KiB Metal limit), and records the selected capacity.

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

## Repeatability and scope

Record full source pins, Python/Rust locks, adapter build fingerprint, scene/job hashes, harness revision/dirty status, machine/OS/GPU description, settings and raw samples. The harness fingerprint includes uncommitted source content; publish results only with the corresponding source revision and lockfiles. Timings are hardware-, OS-, power-, temperature- and driver-dependent. Running unrelated heavy work affects them. Thread limits are requested through Rayon and common runtime environment variables; those are not a promise of identical framework worker counts.

The report renderer consumes a separate capture process; it uses common geometry, camera rules, colours and video timing for all engines. It never calls an engine renderer. Capture files store only world 0 to bound output size. Rendering costs are excluded and renderer failures remain visible in `render-status.json`.

Suite presets are intended starting points, not claims that every engine can finish every largest case. Timeouts and errors remain in the report, with their logs. No automatic backend fallback, accuracy downgrade, capacity reduction or engine substitution is used to manufacture a passing result.
