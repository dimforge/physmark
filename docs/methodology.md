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

## Repeatability and scope

Record full source pins, Python/Rust locks, adapter build fingerprint, scene/job hashes, harness revision/dirty status, machine/OS/GPU description, settings and raw samples. The harness fingerprint includes uncommitted source content; publish results only with the corresponding source revision and lockfiles. Timings are hardware-, OS-, power-, temperature- and driver-dependent. Running unrelated heavy work affects them. Thread limits are requested through Rayon and common runtime environment variables; those are not a promise of identical framework worker counts.

The report renderer consumes a separate capture process; it uses common geometry, camera rules, colours and video timing for all engines. It never calls an engine renderer. Capture files store only world 0 to bound output size. Rendering costs are excluded and renderer failures remain visible in `render-status.json`.

Suite presets are intended starting points, not claims that every engine can finish every largest case. Timeouts and errors remain in the report, with their logs. No automatic backend fallback, accuracy downgrade, capacity reduction or engine substitution is used to manufacture a passing result.
