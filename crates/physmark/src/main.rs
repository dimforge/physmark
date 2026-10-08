use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use physmark_core::{AdapterResult, ENGINES, Job, SCENARIOS, hash, scene, validate_result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(version, about = "Reproducible cross-engine physics benchmarks")]
struct Cli {
    #[arg(long, global = true, default_value = ".")]
    root: PathBuf,
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Fetch exact commits from engines.lock into the private source cache.
    Fetch,
    /// Resolve an explicit upstream ref and update its pin. Does not build or benchmark.
    Pin { engine: String, reference: String },
    /// Build native adapters and install Python engines from pinned source.
    Build {
        #[arg(long, default_value = "all")]
        engines: String,
    },
    /// Inspect installed tools, source pins and adapter availability.
    Doctor,
    /// Write a deterministic engine-neutral scene as JSON.
    Scene {
        scenario: String,
        #[arg(long, default_value_t = 8)]
        size: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 0.008333333333333333)]
        dt: f64,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Run serial, isolated trials. Capture is a separate, untimed replay.
    Run {
        #[arg(long, default_value = "suites/smoke.toml")]
        suite: PathBuf,
        #[arg(long, value_delimiter = ',')]
        engines: Option<Vec<String>>,
        #[arg(long, value_delimiter = ',')]
        scenarios: Option<Vec<String>>,
        #[arg(long, value_delimiter = ',')]
        sizes: Option<Vec<usize>>,
        #[arg(long, value_delimiter = ',')]
        worlds: Option<Vec<usize>>,
        #[arg(long)]
        steps: Option<usize>,
        #[arg(long)]
        warmup: Option<usize>,
        #[arg(long)]
        repeats: Option<usize>,
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Comma-separated overrides, e.g. genesis=metal,newton=cpu,mjx=cpu.
        #[arg(long, value_delimiter = ',')]
        backend: Vec<String>,
        #[arg(long)]
        no_capture: bool,
        #[arg(long)]
        timeout: Option<u64>,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Regenerate offline HTML, performance curves, screenshots and videos.
    Report {
        directory: PathBuf,
        #[arg(long)]
        no_video: bool,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Suite {
    name: String,
    engines: Vec<String>,
    scenarios: Vec<String>,
    sizes: Vec<usize>,
    worlds: Vec<usize>,
    steps: usize,
    warmup: usize,
    repeats: usize,
    seeds: Vec<u64>,
    dt: f64,
    capture: bool,
    capture_stride: usize,
    timeout_seconds: u64,
}
fn command_output(root: &Path, program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}
fn checked(root: &Path, program: &str, args: &[&str]) -> Result<()> {
    ensure!(
        Command::new(program)
            .args(args)
            .current_dir(root)
            .status()?
            .success(),
        "{program} failed"
    );
    Ok(())
}
fn lock(root: &Path) -> Result<toml::Value> {
    Ok(toml::from_str(&fs::read_to_string(
        root.join("engines.lock"),
    )?)?)
}
fn sources(lock: &toml::Value) -> Vec<(String, String, String)> {
    let mut out = vec![];
    for group in ["engines", "dependencies"] {
        if let Some(table) = lock[group].as_table() {
            for (name, v) in table {
                if v.get("source").is_none() {
                    out.push((
                        name.clone(),
                        v["repository"].as_str().unwrap().into(),
                        v["commit"].as_str().unwrap().into(),
                    ));
                }
            }
        }
    }
    out
}
fn fetch(root: &Path) -> Result<()> {
    for (name, url, rev) in sources(&lock(root)?) {
        ensure!(
            rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()),
            "pin for {name} must be a full commit SHA"
        );
        let dir = root.join(".physmark/sources").join(&name);
        fs::create_dir_all(&dir)?;
        if !dir.join(".git").exists() {
            checked(&dir, "git", &["init", "-q"])?;
            checked(&dir, "git", &["remote", "add", "origin", &url])?;
        }
        ensure!(
            command_output(&dir, "git", &["remote", "get-url", "origin"]) == url,
            "unexpected remote for {name}"
        );
        ensure!(
            command_output(&dir, "git", &["diff", "HEAD", "--"]).is_empty(),
            "source {name} has tracked modifications"
        );
        if command_output(&dir, "git", &["rev-parse", "HEAD"]) != rev {
            checked(&dir, "git", &["fetch", "--depth", "1", "origin", &rev])?;
            checked(&dir, "git", &["checkout", "--detach", &rev])?;
        }
        println!("{name} {rev}");
    }
    Ok(())
}
fn verify_sources(root: &Path) -> Result<()> {
    for (name, _, rev) in sources(&lock(root)?) {
        let dir = root.join(".physmark/sources").join(&name);
        ensure!(
            command_output(&dir, "git", &["rev-parse", "HEAD"]) == rev,
            "{name} checkout does not match engines.lock; run fetch and build"
        );
        ensure!(
            command_output(&dir, "git", &["diff", "HEAD", "--"]).is_empty(),
            "{name} source has tracked modifications"
        );
    }
    Ok(())
}
fn stamp(root: &Path) -> Result<String> {
    fn visit(root: &Path, path: &Path, out: &mut BTreeMap<String, String>) -> Result<()> {
        for e in fs::read_dir(path)? {
            let p = e?.path();
            if p.is_dir() {
                if !["target", "__pycache__"]
                    .contains(&p.file_name().unwrap().to_str().unwrap_or(""))
                {
                    visit(root, &p, out)?;
                }
            } else {
                out.insert(
                    p.strip_prefix(root)?.to_string_lossy().into(),
                    hash(&fs::read(p)?),
                );
            }
        }
        Ok(())
    }
    let mut out = BTreeMap::new();
    for p in ["adapters", "crates", "tools"] {
        visit(root, &root.join(p), &mut out)?;
    }
    for p in [
        "engines.lock",
        "Cargo.lock",
        "requirements.lock",
        "rust-toolchain.toml",
    ] {
        if let Ok(v) = fs::read(root.join(p)) {
            out.insert(p.into(), hash(&v));
        }
    }
    Ok(hash(&out))
}
fn python(root: &Path) -> PathBuf {
    root.join(".physmark/envs/python/bin/python")
}
fn adapter_command(root: &Path, engine: &str) -> Command {
    if ["rapier", "nexus"].contains(&engine) {
        Command::new(root.join(format!(
            "adapters/{engine}/target/release/physmark-{engine}"
        )))
    } else {
        let mut c = Command::new(python(root));
        c.arg(root.join("adapters/python/adapter.py"));
        c
    }
}
fn subprocess(
    mut cmd: Command,
    root: &Path,
    job: &Path,
    result: &Path,
    log: &Path,
    timeout: u64,
    threads: usize,
) -> Result<()> {
    let file = fs::File::create(log)?;
    cmd.args([job, result])
        .current_dir(root)
        .stdout(file.try_clone()?)
        .stderr(file);
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ] {
        cmd.env(key, threads.to_string());
    }
    cmd.env("JAX_ENABLE_X64", "false")
        .env("PYTHONUNBUFFERED", "1");
    let mut child = cmd.spawn().context("starting engine adapter")?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            ensure!(
                status.success(),
                "adapter exited {status}; see {}",
                log.display()
            );
            return Ok(());
        }
        if start.elapsed() > Duration::from_secs(timeout) {
            let _ = child.kill();
            let _ = child.wait();
            bail!("adapter timed out after {timeout}s; see {}", log.display());
        }
        thread::sleep(Duration::from_millis(20));
    }
}
fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}
fn run(
    root: &Path,
    suite: Suite,
    threads: usize,
    backends: Vec<String>,
    out: Option<PathBuf>,
) -> Result<()> {
    verify_sources(root)?;
    ensure!(
        fs::read_to_string(root.join(".physmark/build.stamp")).unwrap_or_default() == stamp(root)?,
        "adapter sources, pins or dependencies changed; run `cargo run --release -- build` first"
    );
    ensure!(
        !suite.engines.is_empty()
            && !suite.scenarios.is_empty()
            && !suite.sizes.is_empty()
            && !suite.worlds.is_empty()
            && !suite.seeds.is_empty(),
        "suite dimensions must not be empty"
    );
    ensure!(
        suite.repeats > 0
            && suite.steps > 0
            && suite.capture_stride > 0
            && suite.timeout_seconds > 0
            && threads > 0,
        "counts must be positive"
    );
    let mut backend_map: BTreeMap<String, String> = ENGINES
        .iter()
        .map(|e| {
            (
                e.to_string(),
                if cfg!(target_os = "macos") && ["nexus", "genesis"].contains(e) {
                    "metal"
                } else if *e == "nexus" {
                    "webgpu"
                } else {
                    "cpu"
                }
                .into(),
            )
        })
        .collect();
    for spec in backends {
        let (e, b) = spec
            .split_once('=')
            .context("backend must be engine=backend")?;
        ensure!(ENGINES.contains(&e), "unknown engine {e}");
        backend_map.insert(e.into(), b.into());
    }
    let unique: std::collections::BTreeSet<_> = suite.engines.iter().collect();
    ensure!(unique.len() == suite.engines.len(), "duplicate engines");
    for e in &suite.engines {
        ensure!(ENGINES.contains(&e.as_str()), "unknown engine {e}");
    }
    for s in &suite.scenarios {
        ensure!(SCENARIOS.contains(&s.as_str()), "unknown scenario {s}");
    }
    let epoch = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let out = out.unwrap_or_else(|| root.join(format!("results/{}-{epoch}", suite.name)));
    let out = if out.is_absolute() {
        out
    } else {
        root.join(out)
    };
    ensure!(
        !out.exists(),
        "output already exists; choose a new directory to preserve results"
    );
    fs::create_dir_all(out.join("runs"))?;
    let metadata = json!({"schema_version":1,"created_unix":epoch,"suite":suite,"backends":backend_map,"threads_requested":threads,"source_pins":lock(root)?,"lock_hash":hash(&fs::read(root.join("engines.lock"))?),"build_stamp":stamp(root)?,"harness_commit":command_output(root,"git",&["rev-parse","HEAD"]),"harness_status":command_output(root,"git",&["status","--porcelain"]),"hardware":{"uname":command_output(root,"uname",&["-a"]),"cpu":command_output(root,"sysctl",&["-n","machdep.cpu.brand_string"]),"memory_bytes":command_output(root,"sysctl",&["-n","hw.memsize"]),"os":command_output(root,"sw_vers",&[]),"gpu":command_output(root,"system_profiler",&["SPDisplaysDataType"])},"rust":command_output(root,"rustc",&["--version"]),"python_packages":command_output(root,python(root).to_str().unwrap(),&["-c","import importlib.metadata,json;print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))"]),"timing":"synchronized warmed fixed-horizon rollouts; setup, JIT, reset, readback and rendering excluded; separate process per trial; serial scheduling","quality":"baseline sanity gates, not accuracy-equated solvers","thread_note":"thread environment limits are requests; GPU and framework worker counts remain engine dependent"});
    atomic_json(&out.join("metadata.json"), &metadata)?;
    let mut records: Vec<Value> = vec![];
    let mut order = vec![];
    for scenario in &suite.scenarios {
        for &size in &suite.sizes {
            for &worlds in &suite.worlds {
                for &seed in &suite.seeds {
                    for repeat in 0..suite.repeats {
                        for e in &suite.engines {
                            order.push((scenario.clone(), size, worlds, seed, repeat, e.clone()));
                        }
                    }
                }
            }
        }
    }
    // Deterministic shuffle reduces order/thermal bias without concurrent timed trials.
    let mut rng = 42u64;
    for i in (1..order.len()).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        order.swap(i, rng as usize % (i + 1));
    }
    let total = order.len();
    for (idx, (scenario, size, worlds, seed, repeat, engine)) in order.into_iter().enumerate() {
        let id = format!("{engine}-{scenario}-n{size}-w{worlds}-s{seed}-r{repeat}");
        let dir = out.join("runs").join(&id);
        fs::create_dir(&dir)?;
        let job = Job {
            schema_version: 1,
            scene: scene(&scenario, size, seed, suite.dt)?,
            engine: engine.clone(),
            backend: backend_map[&engine].clone(),
            mode: "timing".into(),
            worlds,
            steps: suite.steps,
            warmup: suite.warmup,
            capture_stride: suite.capture_stride,
            threads,
        };
        job.validate()?;
        atomic_json(&dir.join("job.json"), &job)?;
        println!("[{}/{}] {id} ({})", idx + 1, total, job.backend);
        let trial = (|| -> Result<AdapterResult> {
            subprocess(
                adapter_command(root, &engine),
                root,
                &dir.join("job.json"),
                &dir.join("result.json"),
                &dir.join("adapter.log"),
                suite.timeout_seconds,
                threads,
            )?;
            Ok(serde_json::from_slice(&fs::read(dir.join("result.json"))?)?)
        })();
        let mut record = json!({"id":id,"engine":engine,"backend":job.backend,"scenario":scenario,"size":size,"worlds":worlds,"seed":seed,"repeat":repeat,"steps":suite.steps,"dt":suite.dt,"scene_hash":hash(&job.scene),"job_hash":hash(&job),"directory":format!("runs/{id}")});
        match trial {
            Ok(r) => {
                let v = validate_result(&job, &r);
                record["status"] = json!(if v.passed { "passed" } else { "invalid" });
                record["validation"] = serde_json::to_value(v)?;
                record["elapsed_seconds"] = json!(r.elapsed_seconds);
                record["ms_per_step"] = json!(r.elapsed_seconds * 1000. / suite.steps as f64);
                record["world_steps_per_second"] =
                    json!(suite.steps as f64 * worlds as f64 / r.elapsed_seconds);
                record["setup_seconds"] = json!(r.setup_seconds);
                record["warmup_seconds"] = json!(r.warmup_seconds);
                record["settings"] = r.settings;
            }
            Err(e) => {
                record["status"] = json!("error");
                record["error"] = json!(format!("{e:#}"));
            }
        }
        println!("  {}", record["status"]);
        if suite.capture && repeat == 0 && seed == suite.seeds[0] && record["status"] == "passed" {
            let capture = Job {
                mode: "capture".into(),
                ..job.clone()
            };
            atomic_json(&dir.join("capture-job.json"), &capture)?;
            match subprocess(
                adapter_command(root, &engine),
                root,
                &dir.join("capture-job.json"),
                &dir.join("capture.json"),
                &dir.join("capture.log"),
                suite.timeout_seconds,
                threads,
            ) {
                Ok(()) => {
                    let replay: Result<AdapterResult> = (|| {
                        Ok(serde_json::from_slice(&fs::read(
                            dir.join("capture.json"),
                        )?)?)
                    })();
                    match replay {
                        Ok(r) => {
                            let v = validate_result(&capture, &r);
                            if v.passed && !r.frames.is_empty() {
                                record["capture"] = json!(format!("runs/{id}/capture.json"));
                            } else {
                                record["capture_error"] =
                                    json!(format!("capture failed validation: {:?}", v.reasons));
                            }
                        }
                        Err(e) => {
                            record["capture_error"] = json!(e.to_string());
                        }
                    }
                }
                Err(e) => {
                    record["capture_error"] = json!(format!("{e:#}"));
                }
            }
        }
        records.push(record);
        atomic_json(&out.join("results.json"), &records)?;
    }
    println!("Results: {}", out.display());
    report(root, &out, false)?;
    let failed = records
        .iter()
        .filter(|r| r["status"] != "passed" || r.get("capture_error").is_some())
        .count();
    ensure!(
        failed == 0,
        "{failed} / {} trials failed; retained in report",
        records.len()
    );
    Ok(())
}
fn report(root: &Path, dir: &Path, no_video: bool) -> Result<()> {
    let mut c = Command::new(python(root));
    c.arg(root.join("tools/report.py"))
        .arg(dir)
        .current_dir(root);
    if no_video {
        c.arg("--no-video");
    }
    ensure!(c.status()?.success(), "report generation failed");
    Ok(())
}
fn main() -> Result<()> {
    let c = Cli::parse();
    let root = c.root.canonicalize()?;
    ensure!(
        root.join("engines.lock").exists(),
        "run from the PhysMark directory or use --root"
    );
    match c.command {
        Action::Fetch => fetch(&root),
        Action::Pin { engine, reference } => {
            let mut l = lock(&root)?;
            let group = if l["engines"].get(&engine).is_some() {
                "engines"
            } else {
                "dependencies"
            };
            let entry = l[group].get(&engine).context("unknown engine/dependency")?;
            ensure!(
                entry.get("source").is_none(),
                "MJX shares the MuJoCo pin; pin mujoco instead"
            );
            let dir = root.join(".physmark/sources").join(&engine);
            checked(
                &dir,
                "git",
                &["fetch", "--depth", "1", "origin", &reference],
            )?;
            let rev = command_output(&dir, "git", &["rev-parse", "FETCH_HEAD"]);
            l[group][&engine]["commit"] = toml::Value::String(rev.clone());
            if engine == "mujoco" {
                l["engines"]["mjx"]["commit"] = toml::Value::String(rev.clone());
            }
            fs::write(root.join("engines.lock"), toml::to_string_pretty(&l)?)?;
            println!("Pinned {engine} to {rev}. Run fetch, build, then smoke tests.");
            Ok(())
        }
        Action::Build { engines } => {
            fetch(&root)?;
            let status = Command::new("python3")
                .arg(root.join("tools/build.py"))
                .arg("--engines")
                .arg(engines)
                .current_dir(&root)
                .status()?;
            ensure!(status.success(), "build failed");
            fs::write(root.join(".physmark/build.stamp"), stamp(&root)?)?;
            Ok(())
        }
        Action::Doctor => {
            println!("Source check: {:?}", verify_sources(&root));
            for e in ENGINES {
                let p = if ["rapier", "nexus"].contains(e) {
                    root.join(format!("adapters/{e}/target/release/physmark-{e}"))
                } else {
                    python(&root)
                };
                println!("{e:8} {}", if p.exists() { "installed" } else { "missing" });
            }
            if python(&root).exists() {
                checked(
                    &root,
                    python(&root).to_str().unwrap(),
                    &[
                        "-c",
                        "import importlib.metadata as m; print({n:m.version(n) for n in ['mujoco','mujoco-mjx','newton','mujoco-warp','genesis-world','jax','warp-lang','quadrants']})",
                    ],
                )?;
            }
            println!(
                "Build fingerprint: {}",
                if fs::read_to_string(root.join(".physmark/build.stamp")).unwrap_or_default()
                    == stamp(&root)?
                {
                    "current"
                } else {
                    "needs build"
                }
            );
            Ok(())
        }
        Action::Scene {
            scenario,
            size,
            seed,
            dt,
            output,
        } => atomic_json(&output, &scene(&scenario, size, seed, dt)?),
        Action::Run {
            suite,
            engines,
            scenarios,
            sizes,
            worlds,
            steps,
            warmup,
            repeats,
            threads,
            backend,
            no_capture,
            timeout,
            output,
        } => {
            let mut s: Suite = toml::from_str(&fs::read_to_string(root.join(suite))?)?;
            if let Some(v) = engines {
                s.engines = v;
            }
            if let Some(v) = scenarios {
                s.scenarios = v;
            }
            if let Some(v) = sizes {
                s.sizes = v;
            }
            if let Some(v) = worlds {
                s.worlds = v;
            }
            if let Some(v) = steps {
                s.steps = v;
            }
            if let Some(v) = warmup {
                s.warmup = v;
            }
            if let Some(v) = repeats {
                s.repeats = v;
            }
            if let Some(v) = timeout {
                s.timeout_seconds = v;
            }
            if no_capture {
                s.capture = false;
            }
            run(&root, s, threads, backend, output)
        }
        Action::Report {
            directory,
            no_video,
        } => report(&root, &directory, no_video),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn process_failure_and_timeout_are_errors() {
        let root =
            std::env::temp_dir().join(format!("physmark-process-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let mut fail = Command::new("sh");
        fail.args(["-c", "exit 7"]);
        assert!(
            subprocess(
                fail,
                &root,
                &root.join("job"),
                &root.join("result"),
                &root.join("fail.log"),
                2,
                1
            )
            .is_err()
        );
        let mut slow = Command::new("sh");
        slow.args(["-c", "exec sleep 10"]);
        let start = Instant::now();
        let error = subprocess(
            slow,
            &root,
            &root.join("job"),
            &root.join("result"),
            &root.join("timeout.log"),
            1,
            1,
        )
        .unwrap_err();
        assert!(error.to_string().contains("timed out"));
        assert!(start.elapsed() < Duration::from_secs(3));
        fs::remove_dir_all(root).unwrap();
    }
}
