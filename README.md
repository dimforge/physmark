<p align="center">
  <img src="assets/branding/physmark-logo.svg" alt="PhysMark." width="600">
</p>

# PhysMark

PhysMark benchmarks physics engines against each other on identical scenes. Each
engine is built from a pinned upstream commit, runs the same scene with the same
timestep and mass properties, and is timed over a warmed, synchronized rollout in
its own process. Each trial is checked for physical sanity, replayed to measure
GPU time and RAM/GPU memory, and rendered with a shared renderer. The result is an
offline HTML report with performance curves and tables, memory tables, and a
replay video per trial.

Engines: **Rapier**, **Nexus**, **Genesis**, **Newton**, **MuJoCo**, **MJX** and
**Isaac Sim**.

## Run the full benchmark

Prerequisites: Git, Rust, Python 3, [`uv`](https://docs.astral.sh/uv/), CMake,
Ninja and FFmpeg (plus the Xcode command-line tools on macOS):

```sh
# Ubuntu/Debian
sudo apt install git python3 cmake ninja-build ffmpeg libgl-dev
# macOS
xcode-select --install && brew install cmake ninja ffmpeg
# uv (both); make sure ~/.local/bin is on your PATH afterwards
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On NVIDIA GPUs, Nexus compiles its shaders with Khal's `cargo cuda`, which also
needs the CUDA toolkit (`CUDA_PATH` pointing at it) and a one-time install of the
Rust-CUDA codegen backend. See the [Khal README](https://github.com/dimforge/khal#readme)
for its native requirements (notably LLVM 7.1 for `LLVM_CONFIG`):

```sh
cargo install --git https://github.com/dimforge/khal cargo-cuda --force
LLVM_CONFIG=$HOME/llvm-7/bin/llvm-config cargo cuda install
```

```sh
cargo run --release -- round results/full
```

This fetches and builds the engines, runs every suite, and writes
`results/full/index.html`, a landing page that links one report per part.
Each part is ranked separately:

| Part | What it measures |
| --- | --- |
| Rigid-body scale | Eight rigid scenes (free fall, piles, stacks, dominoes, pendulums, friction ramp) at 8 to 10,000 bodies |
| Brick structures | A brick ring and rows of brick pyramids smashed by rolling spheres, 1,000 to 28,000 bodies |
| Large brick structures | The same scenes at 110,000 and 505,922 bodies |
| Material points (MPM) | Elastic and sand, 512 to 195,112 particles |
| Robots | Go1 quadruped and Booster T1 humanoid standing up on the ground, 1 to 4,096 environments |

The round detects the hardware and picks a backend for each engine. Engines that
can't run on the machine are skipped and listed on the landing page, never
swapped for another backend:

| Engine | Apple GPU | NVIDIA GPU | Neither |
| --- | --- | --- | --- |
| Rapier, MuJoCo | CPU | CPU | CPU |
| Nexus | Metal | CUDA | WebGPU |
| Genesis | Metal | CUDA | CPU |
| Newton, MJX | skipped | CUDA | skipped |
| Isaac Sim | skipped | CUDA (Linux/Windows) | skipped |

Isaac Sim also requires accepting NVIDIA's EULA yourself before the run
(`export OMNI_KIT_ACCEPT_EULA=YES`); otherwise it is skipped. Every selected engine
first runs a two-step probe trial, and any that fails is skipped too.

Use `--resume` to continue an interrupted round, `--threads N` to set CPU workers
(default: up to 8), and `--skip-build` to reuse the current build.

## Publish rounds from several machines

The reports have a platform selector. To compare rounds from different machines,
add each round to a static site directory:

```sh
cargo run --release -- site results/site results/full
```

Each round becomes a self-contained `results/site/<platform>/` directory, listed in
`results/site/platforms.json`. The platform name and label come from the round's
hardware (override them with `--id` and `--label`). Per-trial evidence files over
1 MB (trajectories, scenes, profiler traces) are left out unless you pass
`--max-evidence-mb`; the reports hide their links.

To add a round from another machine to a deployed site, copy the deployed
`platforms.json` into the local site directory first, then upload only the new
`<platform>/` directory and the updated `platforms.json`. The other platforms'
directories stay untouched. Serve the site over HTTP: opened from disk, a report
shows only its own platform.

To publish to the PhysMark website, run `./publish-linux.sh` on the Linux + NVIDIA
machine and `./publish-macos.sh` on the Mac, each after its round (pass the round
directory if it is not `result/full` or `results/full`). Each script uploads only its
own platform directory with rsync, merges `platforms.json` with the online copy and
leaves the other platform untouched. Only what visitors see is uploaded: pages,
charts, gallery images and videos (`site --report-only`). Reports made before the
platform selector must be regenerated with `cargo run --release -- report` first.

## Other commands

```sh
# One suite (see suites/), e.g. a quick check of every engine:
cargo run --release -- run --suite suites/smoke.toml

# Override suite dimensions or backends:
cargo run --release -- run --suite suites/stacks-three-scales.toml --engines rapier,nexus --backend nexus=webgpu

# Rebuild a report without rerunning physics:
cargo run --release -- report results/<run>

# Check installed engines and pins:
cargo run --release -- doctor

# Export a scene as JSON:
cargo run --release -- scene brick_ring --size 1000 --output scene.json
```

A run exits nonzero if any trial fails; failed trials stay in the report. Add
`--output <dir> --resume` to continue an interrupted run.

## Engine versions

`engines.lock` pins every engine to a full commit SHA (Isaac Sim to its tagged
release wheel); `requirements.lock` and the adapter `Cargo.lock` files pin their
dependencies. To move an engine:

```sh
cargo run --release -- pin genesis <tag-or-commit>
cargo run --release -- build
cargo run --release -- run --suite suites/smoke.toml
```

The runner refuses to start if a checkout no longer matches its pin or the build
is out of date.

## Documentation

- [Methodology](docs/methodology.md): scenes, solver settings per engine, timing,
  memory, validation gates and what the numbers do and don't establish.
- Robot models come from the pinned MuJoCo Menagerie; their notices are in
  [docs/licenses](docs/licenses) (Unitree Go1: BSD-3-Clause, Booster T1:
  Apache-2.0). Keep them with redistributed robot jobs and reports.

## Development

```sh
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
.physmark/envs/python/bin/python -m pytest tests
```

`crates/physmark-core` holds the scene generator and subprocess protocol,
`crates/physmark` the controller, `adapters/<engine>` one adapter per engine
(with shared code in `adapters/common`), `suites/` the benchmark presets and
`tools/` the build, round, report and site scripts.
