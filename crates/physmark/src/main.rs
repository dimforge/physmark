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
// Parsed once per invocation; Windows paths make `Run` larger than the other variants.
#[allow(clippy::large_enum_variant)]
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
        /// Continue an existing output, retaining completed trials and recording new provenance.
        #[arg(long, requires = "output")]
        resume: bool,
        /// Skip remaining trials of engine:scenario:size:worlds=reason; retained as unranked outcomes.
        #[arg(long = "skip-case")]
        skip_cases: Vec<String>,
        /// Run cases that earlier runs on this machine recorded as out of memory.
        #[arg(long)]
        retry_known_failures: bool,
        #[arg(long)]
        timeout: Option<u64>,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Run the complete benchmark round on this machine: detect Metal/CUDA/Isaac support, build
    /// what can run, probe every engine/backend, run all suites and index their reports.
    Round {
        output: PathBuf,
        /// Continue an interrupted round with its recorded plan.
        #[arg(long)]
        resume: bool,
        /// CPU workers per trial (default: up to 8).
        #[arg(long)]
        threads: Option<usize>,
        /// Use the engines already built.
        #[arg(long)]
        skip_build: bool,
        /// Run cases that earlier runs on this machine recorded as out of memory.
        #[arg(long)]
        retry_known_failures: bool,
    },
    /// Re-run only the memory replay of listed passing trials and replace their memory results,
    /// keeping their timings (the previous measurement is kept in the record).
    RemeasureMemory {
        directory: PathBuf,
        #[arg(long, value_delimiter = ',', required = true)]
        ids: Vec<String>,
        /// Why the memory is remeasured, stored with each record.
        #[arg(long)]
        reason: String,
    },
    /// Regenerate offline HTML, performance curves, screenshots and videos.
    Report {
        directory: PathBuf,
        #[arg(long)]
        no_video: bool,
    },
    /// Add a finished round to a static site of per-platform reports (or replace its platform),
    /// so rounds from different machines can be compared and uploaded independently.
    Site {
        site: PathBuf,
        round: PathBuf,
        /// Platform directory name (default: the round's platform.json).
        #[arg(long)]
        id: Option<String>,
        /// Platform selector label.
        #[arg(long)]
        label: Option<String>,
        /// Leave out larger per-trial evidence files; negative publishes everything.
        #[arg(long, default_value_t = 1.0, allow_hyphen_values = true)]
        max_evidence_mb: f64,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cpu_timeout_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    skip_after_resource_limit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    baseline: Option<String>,
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
                // Shared checkouts (MJX) and binary distributions (Isaac Sim) are not fetched.
                if v.get("source").is_none() && v.get("distribution").is_none() {
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
            if name == "menagerie" {
                checked(&dir, "git", &["config", "remote.origin.promisor", "true"])?;
                checked(
                    &dir,
                    "git",
                    &["config", "remote.origin.partialclonefilter", "blob:none"],
                )?;
                checked(&dir, "git", &["sparse-checkout", "init", "--no-cone"])?;
            }
        }
        if name == "menagerie" {
            // Only the portable robot collision models and their licenses are needed.
            // Reapplied on every fetch so existing checkouts gain newly benchmarked robots.
            checked(
                &dir,
                "git",
                &[
                    "sparse-checkout",
                    "set",
                    "--no-cone",
                    "/unitree_go1/go1.xml",
                    "/unitree_go1/LICENSE",
                    "/booster_t1/t1.xml",
                    "/booster_t1/LICENSE",
                    "/README.md",
                ],
            )?;
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
            if name == "menagerie" {
                checked(
                    &dir,
                    "git",
                    &[
                        "fetch",
                        "--filter=blob:none",
                        "--depth",
                        "1",
                        "origin",
                        &rev,
                    ],
                )?;
            } else {
                checked(&dir, "git", &["fetch", "--depth", "1", "origin", &rev])?;
            }
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
/// The interpreter of a virtual environment under `.physmark/envs`.
fn venv_python(root: &Path, name: &str) -> PathBuf {
    let env = root.join(".physmark/envs").join(name);
    if cfg!(windows) {
        env.join("Scripts/python.exe")
    } else {
        env.join("bin/python")
    }
}
fn python(root: &Path) -> PathBuf {
    venv_python(root, "python")
}
/// Isaac Sim pins its own Python packages, so it gets a separate environment.
fn isaac_python(root: &Path) -> PathBuf {
    venv_python(root, "isaac")
}
/// The system interpreter running the build and round helpers (standard library only).
fn system_python() -> &'static str {
    if cfg!(windows) { "python" } else { "python3" }
}
fn native_adapter(root: &Path, engine: &str) -> PathBuf {
    root.join(format!(
        "adapters/{engine}/target/release/physmark-{engine}{}",
        std::env::consts::EXE_SUFFIX
    ))
}
fn load_scene(
    root: &Path,
    name: &str,
    size: usize,
    seed: u64,
    dt: f64,
) -> Result<physmark_core::Scene> {
    if !name.starts_with("robot_") {
        return scene(name, size, seed, dt);
    }
    let output = Command::new(python(root))
        .arg(root.join("tools/robot_scene.py"))
        .arg("--root")
        .arg(root)
        .arg("--scenario")
        .arg(name)
        .arg("--size")
        .arg(size.to_string())
        .arg("--seed")
        .arg(seed.to_string())
        .arg("--dt")
        .arg(dt.to_string())
        .output()?;
    ensure!(
        output.status.success(),
        "robot scene import failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}
fn adapter_command(root: &Path, engine: &str) -> Command {
    if ["rapier", "nexus"].contains(&engine) {
        Command::new(native_adapter(root, engine))
    } else {
        let mut c = Command::new(if engine == "isaac" {
            isaac_python(root)
        } else {
            python(root)
        });
        c.arg(root.join(format!("adapters/{engine}/{engine}_adapter.py")));
        c
    }
}
fn memory_limit_bytes() -> Result<Option<u64>> {
    let Some(value) = std::env::var_os("PHYSMARK_MEMORY_LIMIT_GIB") else {
        return Ok(None);
    };
    ensure!(
        cfg!(any(target_os = "macos", target_os = "linux")),
        "memory monitoring currently requires macOS or Linux"
    );
    let gib: u64 = value.to_string_lossy().parse()?;
    ensure!(gib > 0, "memory limit must be positive");
    Ok(Some(
        gib.checked_mul(1 << 30).context("memory limit overflow")?,
    ))
}
#[cfg(target_os = "macos")]
fn process_footprint(pid: u32) -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v0>::uninit();
    // SAFETY: flavor V0 writes the matching initialized structure into this buffer.
    let status = unsafe {
        libc::proc_pid_rusage(pid as i32, libc::RUSAGE_INFO_V0, usage.as_mut_ptr().cast())
    };
    if status == 0 {
        // SAFETY: a successful proc_pid_rusage initialized every field above.
        Some(unsafe { usage.assume_init() }.ri_phys_footprint)
    } else {
        None
    }
}
/// Anonymous plus shared resident memory of the adapter's whole process group (it leads its
/// own group, so JIT and worker descendants count too). File-backed pages are excluded: they
/// are reclaimable, and CUDA/Torch libraries alone map gigabytes of them.
#[cfg(target_os = "linux")]
fn process_footprint(pid: u32) -> Option<u64> {
    let mut total = 0;
    let mut found = false;
    for entry in fs::read_dir("/proc").ok()?.flatten() {
        let path = entry.path();
        let Ok(stat) = fs::read_to_string(path.join("stat")) else {
            continue;
        };
        // The command name may contain spaces; fields after its closing parenthesis are fixed.
        let Some(group) = stat
            .rsplit_once(')')
            .and_then(|(_, rest)| rest.split_whitespace().nth(2))
        else {
            continue;
        };
        if group != pid.to_string() {
            continue;
        }
        let Ok(status) = fs::read_to_string(path.join("status")) else {
            continue;
        };
        found = true;
        for line in status.lines() {
            if let Some(kib) = line
                .strip_prefix("RssAnon:")
                .or_else(|| line.strip_prefix("RssShmem:"))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            {
                total += kib * 1024;
            }
        }
    }
    found.then_some(total)
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn process_footprint(_pid: u32) -> Option<u64> {
    None
}
/// The last line of an adapter log reporting an allocation failure or an engine refusing a
/// size it cannot allocate (numpy, MuJoCo, JAX/XLA, PyTorch, CUDA, Rust, Genesis guards).
fn out_of_memory_line(log: &Path) -> Option<String> {
    const MARKERS: [&str; 9] = [
        "memoryerror",
        "unable to allocate",
        "could not allocate memory",
        "out of memory",
        "resource_exhausted",
        "outofmemoryerror",
        "cuda_error_out_of_memory",
        "memory allocation of",
        "failed to allocate",
    ];
    let text = fs::read(log).ok()?;
    let tail = String::from_utf8_lossy(&text[text.len().saturating_sub(1 << 16)..]).into_owned();
    tail.lines().rev().find_map(|line| {
        let lower = line.to_lowercase();
        let refused_size = lower.contains("shape (") && lower.contains("is too large");
        (refused_size || MARKERS.iter().any(|m| lower.contains(m))).then(|| {
            // Drop terminal colour codes some engines log with.
            let mut clean = String::new();
            let mut chars = line.chars();
            while let Some(c) = chars.next() {
                if c == '\u{1b}' {
                    chars.by_ref().find(|c| c.is_ascii_alphabetic());
                } else {
                    clean.push(c);
                }
            }
            clean.trim().chars().take(300).collect()
        })
    })
}
/// `oom` for memory-cap kills and allocation failures, `timeout` for timeouts.
fn resource_failure(error: &str) -> Option<&'static str> {
    if error.contains("adapter memory limit exceeded") || error.contains("adapter out of memory") {
        Some("OOM")
    } else if error.contains("adapter timed out") {
        Some("timeout")
    } else {
        None
    }
}
/// Engine source identity for known failures: they are retried after a pin change.
fn engine_pin(lock: &toml::Value, engine: &str) -> String {
    let entry = &lock["engines"][engine];
    [entry.get("commit"), entry.get("distribution")]
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}
fn known_failures_path(root: &Path) -> PathBuf {
    root.join(".physmark/known-resource-failures.json")
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
    let memory_limit = memory_limit_bytes()?;
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
    cmd.env(
        "JAX_COMPILATION_CACHE_DIR",
        root.join(".physmark/cache/jax"),
    );
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd.spawn().context("starting engine adapter")?;
    let start = Instant::now();
    let mut last_memory_check = start;
    loop {
        if let Some(status) = child.try_wait()? {
            if !status.success()
                && let Some(line) = out_of_memory_line(log)
            {
                bail!("adapter out of memory: {line}; see {}", log.display());
            }
            ensure!(
                status.success(),
                "adapter exited {status}; see {}",
                log.display()
            );
            // PhysX logs GPU failures and carries on: a full contact buffer drops contacts, and a
            // failed device allocation leaves every solver kernel unlaunched, so nothing moves and
            // the trial would pass with an impossible step time.
            let text = fs::read(log).unwrap_or_default();
            let text = String::from_utf8_lossy(&text);
            let first = |needles: &[&str]| {
                text.lines()
                    .find(|l| needles.iter().any(|n| l.to_lowercase().contains(n)))
                    .map(|l| l.trim().chars().take(300).collect::<String>())
            };
            if let Some(line) = first(&["failed to allocate memory", "fail to launch kernel"]) {
                bail!("adapter out of memory: GPU allocation failed: {line}; see {}", log.display());
            }
            // PhysX contact buffers, and MuJoCo Warp's constraint and contact buffers
            // ("nefc overflow - please increase njmax"), drop what does not fit.
            if let Some(line) = first(&["buffer overflow detected", "overflow - please increase"]) {
                bail!("adapter reported a GPU buffer overflow: {line}; see {}", log.display());
            }
            return Ok(());
        }
        if let Some(limit) = memory_limit
            && last_memory_check.elapsed() >= Duration::from_millis(200)
        {
            last_memory_check = Instant::now();
            if let Some(bytes) = process_footprint(child.id())
                && bytes > limit
            {
                #[cfg(unix)]
                // SAFETY: this group belongs to the adapter process spawned above.
                unsafe {
                    libc::kill(-(child.id() as i32), libc::SIGKILL);
                }
                let _ = child.kill();
                let _ = child.wait();
                bail!(
                    "adapter memory limit exceeded: memory footprint {bytes} bytes > {limit} bytes; see {}",
                    log.display()
                );
            }
        }
        if start.elapsed() > Duration::from_secs(timeout) {
            #[cfg(unix)]
            // SAFETY: this process group was created for this owned child above.
            // Kill JIT compiler descendants too, so they cannot contaminate later trials.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
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
/// Independent memory replay of a passing trial; fills `record["memory"]` (and its validation).
fn memory_replay(
    root: &Path,
    engine: &str,
    dir: &Path,
    job: &Job,
    timeout: u64,
    threads: usize,
    record: &mut Value,
) -> Result<()> {
    let profile = Job {
        mode: "memory".into(),
        ..job.clone()
    };
    atomic_json(&dir.join("memory-job.json"), &profile)?;
    let measurement = (|| -> Result<physmark_core::memory::MemoryMeasurement> {
        subprocess(
            adapter_command(root, engine),
            root,
            &dir.join("memory-job.json"),
            &dir.join("memory-result.json"),
            &dir.join("memory.log"),
            timeout,
            threads,
        )?;
        let result: AdapterResult = serde_json::from_slice(&fs::read(dir.join("memory-result.json"))?)?;
        let validation = validate_result(&profile, &result);
        ensure!(
            validation.passed,
            "memory replay failed physics validation: {:?}",
            validation.reasons
        );
        let memory = result.memory.context("adapter omitted memory measurement")?;
        memory.validate(&profile)?;
        record["memory_validation"] = serde_json::to_value(validation)?;
        Ok(memory)
    })();
    record["memory"] = match measurement {
        Ok(memory) => serde_json::to_value(memory)?,
        Err(error) => json!({"status":"error", "reason":format!("{error:#}")}),
    };
    Ok(())
}
/// Re-runs only the memory replay of listed passing trials of a run, keeping their timings.
fn remeasure_memory(root: &Path, out: &Path, ids: &[String], reason: &str) -> Result<()> {
    verify_sources(root)?;
    ensure!(
        fs::read_to_string(root.join(".physmark/build.stamp")).unwrap_or_default() == stamp(root)?,
        "adapter sources, pins or dependencies changed; run `cargo run --release -- build` first"
    );
    let metadata: Value = serde_json::from_slice(&fs::read(out.join("metadata.json"))?)?;
    let suite: Suite = serde_json::from_value(metadata["suite"].clone())?;
    let threads = metadata["threads_requested"].as_u64().context("metadata lacks threads")? as usize;
    let mut records: Vec<Value> = serde_json::from_slice(&fs::read(out.join("results.json"))?)?;
    let epoch = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    for id in ids {
        let index = records
            .iter()
            .position(|r| r["id"] == id.as_str())
            .with_context(|| format!("no trial {id}"))?;
        ensure!(records[index]["status"] == "passed", "{id} did not pass; nothing to remeasure");
        let dir = out.join(records[index]["directory"].as_str().unwrap_or_default());
        let job: Job = serde_json::from_slice(&fs::read(dir.join("job.json"))?)?;
        let timeout = if job.backend == "cpu" {
            suite.cpu_timeout_seconds.unwrap_or(suite.timeout_seconds)
        } else {
            suite.timeout_seconds
        };
        let engine = job.engine.clone();
        let previous = records[index]["memory"].clone();
        println!("remeasuring memory of {id}");
        memory_replay(root, &engine, &dir, &job, timeout, threads, &mut records[index])?;
        records[index]["memory_remeasured"] = json!({"reason":reason,"created_unix":epoch,
            "build_stamp":stamp(root)?,"previous_memory":previous});
        println!("  memory {}", records[index]["memory"]["status"]);
        atomic_json(&out.join("results.json"), &records)?;
    }
    let history_path = out.join("remeasure-history.json");
    let mut history: Vec<Value> = fs::read(&history_path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    history.push(json!({"created_unix":epoch,"ids":ids,"reason":reason,"build_stamp":stamp(root)?}));
    atomic_json(&history_path, &history)
}
fn validate_resume_metadata(previous: &Value, current: &Value) -> Result<()> {
    // Scheduling may change: it only orders the remaining trials, and each resume's metadata
    // is kept in resume-history.json.
    for key in ["memory_limit_bytes"] {
        ensure!(
            previous.get(key) == current.get(key),
            "cannot resume: {key} changed"
        );
    }
    // Timeouts may change: they decide which slow cases finish, not how a finished trial was
    // measured. Engines may be added, keeping the original ones and their backends, so a run
    // can be extended with newly wired engines. Everything else in the suite must match.
    let suite = |m: &Value| {
        let mut s = m.get("suite").cloned().unwrap_or(Value::Null);
        if let Some(o) = s.as_object_mut() {
            o.remove("timeout_seconds");
            o.remove("cpu_timeout_seconds");
            o.remove("engines");
        }
        s
    };
    ensure!(
        previous.get("suite").is_some() && suite(previous) == suite(current),
        "cannot resume: suite changed"
    );
    let engines = |m: &Value| m["suite"]["engines"].as_array().cloned().unwrap_or_default();
    for engine in engines(previous) {
        ensure!(engines(current).contains(&engine), "cannot resume: engine {engine} removed");
        let name = engine.as_str().unwrap_or_default();
        ensure!(
            previous["backends"][name] == current["backends"][name],
            "cannot resume: backend of {name} changed"
        );
    }
    for key in [
        "schema_version",
        "threads_requested",
        "source_pins",
        "lock_hash",
        "python_packages",
        "hardware",
        "rust",
    ] {
        ensure!(
            previous.get(key).is_some() && previous[key] == current[key],
            "cannot resume: {key} changed"
        );
    }
    Ok(())
}
/// Replaces one pin's commit in the lockfile text, keeping its layout and comments.
/// MJX shares MuJoCo's commit line, so pinning MuJoCo moves both.
fn repin(text: &str, group: &str, engine: &str, rev: &str) -> Result<String> {
    ensure!(
        rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()),
        "resolved commit must be a full SHA"
    );
    let before: toml::Value = toml::from_str(text)?;
    let old = before[group][engine]["commit"]
        .as_str()
        .context("pin has no commit")?;
    let updated = text.replace(
        &format!("commit = \"{old}\""),
        &format!("commit = \"{rev}\""),
    );
    let after: toml::Value = toml::from_str(&updated)?;
    // Only the requested pin (and MJX alongside MuJoCo) may change.
    for group in ["engines", "dependencies"] {
        for (name, entry) in before[group].as_table().into_iter().flatten() {
            let expected = if name == engine || (engine == "mujoco" && name == "mjx") {
                rev
            } else {
                entry["commit"].as_str().unwrap_or_default()
            };
            ensure!(
                after[group][name]["commit"].as_str() == Some(expected),
                "pinning {engine} would also change {name}; edit engines.lock by hand"
            );
        }
    }
    Ok(updated)
}
fn parse_skip_cases(specs: &[String]) -> Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    for spec in specs {
        let (key, reason) = spec
            .split_once('=')
            .context("skip-case requires engine:scenario:size:worlds=reason")?;
        let parts: Vec<_> = key.split(':').collect();
        ensure!(
            parts.len() == 4 && ENGINES.contains(&parts[0]) && SCENARIOS.contains(&parts[1]),
            "invalid skip-case key"
        );
        ensure!(
            parts[2].parse::<usize>()? > 0
                && parts[3].parse::<usize>()? > 0
                && !reason.trim().is_empty(),
            "invalid skip-case counts/reason"
        );
        ensure!(
            result.insert(key.into(), reason.into()).is_none(),
            "duplicate skip-case"
        );
    }
    Ok(result)
}
fn run(
    root: &Path,
    mut suite: Suite,
    threads: usize,
    backends: Vec<String>,
    out: Option<PathBuf>,
    resume: bool,
    skip_cases: Vec<String>,
    retry_known_failures: bool,
) -> Result<()> {
    if suite.scenarios.iter().any(|s| s.starts_with("mpm_")) {
        ensure!(
            suite.scenarios.iter().all(|s| s.starts_with("mpm_")),
            "run MPM and rigid-body benchmarks in separate suites"
        );
        ensure!(
            suite.worlds == [1],
            "MPM benchmarks require exactly one environment"
        );
        suite
            .engines
            .retain(|engine| physmark_core::mpm::ENGINES.contains(&engine.as_str()));
        ensure!(!suite.engines.is_empty(), "no selected engine supports MPM");
        if suite.baseline.is_none() {
            suite.baseline = Some("genesis".into());
        }
    }
    let skip_cases = parse_skip_cases(&skip_cases)?;
    verify_sources(root)?;
    let pins = lock(root)?;
    let memory_limit = memory_limit_bytes()?;
    // Machine-local record of out-of-memory cases, consulted unless retrying them.
    let mut known_failures: Vec<Value> = fs::read(known_failures_path(root))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
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
            && suite.cpu_timeout_seconds.is_none_or(|v| v > 0)
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
                } else if *e == "isaac" {
                    // Isaac Sim requires an NVIDIA GPU; PhysX GPU dynamics is its native mode.
                    "cuda"
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
        out.exists() == resume,
        "use a new output directory, or --resume with an existing output"
    );
    fs::create_dir_all(out.join("runs"))?;
    let metadata = json!({"schema_version":1,"created_unix":epoch,"suite":suite,"backends":backend_map,"threads_requested":threads,"source_pins":lock(root)?,"lock_hash":hash(&fs::read(root.join("engines.lock"))?),"build_stamp":stamp(root)?,"harness_commit":command_output(root,"git",&["rev-parse","HEAD"]),"harness_status":command_output(root,"git",&["status","--porcelain"]),"hardware":{"uname":command_output(root,"uname",&["-a"]),"cpu":command_output(root,"sysctl",&["-n","machdep.cpu.brand_string"]),"memory_bytes":command_output(root,"sysctl",&["-n","hw.memsize"]),"os":command_output(root,"sw_vers",&[]),"gpu":command_output(root,"system_profiler",&["SPDisplaysDataType"])},"rust":command_output(root,"rustc",&["--version"]),"python_packages":command_output(root,python(root).to_str().unwrap(),&["-c","import importlib.metadata,json;print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))"]),"isaac_python_packages":command_output(root,isaac_python(root).to_str().unwrap(),&["-c","import importlib.metadata,json;print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))"]),"timing":"synchronized warmed fixed-horizon rollouts; setup, JIT, reset, readback and rendering excluded; separate process per trial; serial scheduling","gpu_timing":"independent warmed profiling process per passing GPU trial; device durations only; unavailable is never replaced by wall time; raw method and scope per trial; performance tables remain synchronized wall time","memory":"independent replay per passing trial; OS process RAM high-water and sampled per-process GPU allocation counters; 10 ms target cadence plus synchronized phase boundaries; ends before readback; RAM and GPU can overlap on unified-memory systems","quality":"baseline sanity gates, not accuracy-equated solvers","thread_note":"thread environment limits are requests; GPU and framework worker counts remain engine dependent"});
    let mut metadata = metadata;
    metadata["memory_limit_bytes"] = json!(memory_limit_bytes()?);
    metadata["scheduling"] = json!(
        "ascending environment count, then size; deterministic shuffled trials within each; serial adapters"
    );
    metadata["memory_monitor"] = json!(
        "when enabled: macOS adapter physical footprint sampled once per second by the controller; exceeding the limit terminates the trial"
    );
    let mut records: Vec<Value> = if resume {
        let mut previous: Value = serde_json::from_slice(&fs::read(out.join("metadata.json"))?)?;
        validate_resume_metadata(&previous, &metadata)?;
        // A resume that adds engines extends the run: record them, keeping the original
        // provenance (each resume's full metadata is in resume-history.json).
        if previous["suite"]["engines"] != metadata["suite"]["engines"] {
            previous["suite"]["engines"] = metadata["suite"]["engines"].clone();
            previous["backends"] = metadata["backends"].clone();
            atomic_json(&out.join("metadata.json"), &previous)?;
        }
        let history_path = out.join("resume-history.json");
        let mut history: Vec<Value> = if history_path.exists() {
            serde_json::from_slice(&fs::read(&history_path)?)?
        } else {
            vec![]
        };
        history.push(json!({"metadata":metadata,"skip_cases":skip_cases}));
        atomic_json(&history_path, &history)?;
        let mut records: Vec<Value> = serde_json::from_slice(&fs::read(out.join("results.json"))?)?;
        // Errors recorded before allocation failures were recognized are reclassified from
        // their logs, so they feed the resource policy and the known failures too.
        for r in &mut records {
            let Some(error) = r["error"].as_str().filter(|e| r["status"] == "error" && e.starts_with("adapter exited")) else {
                continue;
            };
            let log = out.join(r["directory"].as_str().unwrap_or_default()).join("adapter.log");
            if let Some(line) = out_of_memory_line(&log) {
                r["error_before_reclassification"] = json!(error);
                r["error"] = json!(format!("adapter out of memory: {line}; see {}", log.display()));
            }
        }
        for r in &records {
            if r["status"] == "error" && r["error"].as_str().and_then(resource_failure) == Some("OOM")
                && !known_failures.iter().any(|k| k["trial"] == r["id"] && k["run"] == out.display().to_string())
            {
                let capped = r["error"].as_str().is_some_and(|e| e.contains("adapter memory limit exceeded"));
                known_failures.push(json!({"engine":r["engine"],"backend":r["backend"],"scenario":r["scenario"],
                    "size":r["size"],"worlds":r["worlds"],"pin":engine_pin(&pins, r["engine"].as_str().unwrap_or_default()),
                    "memory_limit_bytes":if capped { json!(memory_limit) } else { Value::Null },
                    "trial":r["id"],"run":out.display().to_string(),"error":r["error"],"created_unix":epoch}));
            }
        }
        atomic_json(&known_failures_path(root), &known_failures)?;
        atomic_json(&out.join("results.json"), &records)?;
        records
    } else {
        atomic_json(&out.join("metadata.json"), &metadata)?;
        vec![]
    };
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
    // Establish lower-memory cases first, keeping shuffled trials within each scale, so a
    // resource failure is known before the larger cases it rules out.
    order.sort_by_key(|case| (case.2, case.1));
    let total = order.len();
    for (idx, (scenario, size, worlds, seed, repeat, engine)) in order.into_iter().enumerate() {
        let id = format!("{engine}-{scenario}-n{size}-w{worlds}-s{seed}-r{repeat}");
        if records.iter().any(|r| r["id"] == id) {
            continue;
        }
        let case = format!("{engine}:{scenario}:{size}:{worlds}");
        let backend = &backend_map[&engine];
        // A case at least as large, in bodies and environments, as one that ran out of memory
        // or time on the same engine and scene is assumed to fail the same way.
        let automatic_skip = if suite.skip_after_resource_limit {
            records.iter().find_map(|r| {
                let kind = resource_failure(r["error"].as_str()?)?;
                (r["engine"] == engine && r["scenario"] == scenario && r["status"] == "error"
                    && r["size"].as_u64()? <= size as u64 && r["worlds"].as_u64()? <= worlds as u64)
                    .then(|| format!("skipped (likely {kind}): {} failed with {kind} at {} × {}", r["id"].as_str().unwrap_or_default(), r["size"], r["worlds"]))
            })
        } else {
            None
        };
        let known_skip = known_failures.iter().filter(|_| !retry_known_failures).find_map(|k| {
            let covers = k["engine"] == engine && k["backend"] == backend.as_str()
                && k["scenario"] == scenario && k["pin"] == engine_pin(&pins, &engine)
                && k["size"].as_u64()? <= size as u64 && k["worlds"].as_u64()? <= worlds as u64
                // A memory-cap kill is retried once the cap is raised above the recorded one.
                && match (k["memory_limit_bytes"].as_u64(), memory_limit) {
                    (Some(recorded), Some(current)) => current <= recorded,
                    _ => true,
                };
            covers.then(|| format!("skipped (likely OOM): known failure {} ({}); rerun with --retry-known-failures", k["trial"].as_str().unwrap_or(""), k["run"].as_str().unwrap_or("")))
        });
        let skip = skip_cases.get(&case).or(automatic_skip.as_ref()).or(known_skip.as_ref());
        let trial_timeout = if backend_map[&engine] == "cpu" {
            suite.cpu_timeout_seconds.unwrap_or(suite.timeout_seconds)
        } else {
            suite.timeout_seconds
        };
        let dir = out.join("runs").join(&id);
        if dir.exists() && skip.is_none() {
            let archive = out.join("interrupted");
            fs::create_dir_all(&archive)?;
            fs::rename(&dir, archive.join(format!("{id}-{epoch}")))?;
        }
        fs::create_dir_all(&dir)?;
        let job = Job {
            schema_version: 1,
            scene: load_scene(root, &scenario, size, seed, suite.dt)?,
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
        let mut record = json!({"id":id,"engine":engine,"backend":job.backend,"scenario":scenario,"size":size,"worlds":worlds,"seed":seed,"repeat":repeat,"steps":suite.steps,"dt":suite.dt,"scene_hash":hash(&job.scene),"job_hash":hash(&job),"directory":format!("runs/{id}")});
        record["harness_build_stamp"] = metadata["build_stamp"].clone();
        if let Some(reason) = skip {
            record["status"] = json!("skipped");
            record["error"] = json!(reason);
            if let Some(label) = reason.split_once(')').filter(|_| reason.starts_with("skipped (")) {
                record["outcome_label"] = json!(format!("{})", label.0));
            }
            record["gpu_timing"] = json!({"status":"not_collected","reason":"trial skipped"});
            record["memory"] = json!({"status":"not_collected","reason":"trial skipped"});
            atomic_json(
                &dir.join("skip.json"),
                &json!({"reason":reason,"created_unix":epoch}),
            )?;
            if !dir.join("adapter.log").exists() {
                fs::write(dir.join("adapter.log"), format!("Skipped: {reason}\n"))?;
            }
            println!("  skipped: {reason}");
            records.push(record);
            atomic_json(&out.join("results.json"), &records)?;
            continue;
        }
        let trial = (|| -> Result<AdapterResult> {
            subprocess(
                adapter_command(root, &engine),
                root,
                &dir.join("job.json"),
                &dir.join("result.json"),
                &dir.join("adapter.log"),
                trial_timeout,
                threads,
            )?;
            Ok(serde_json::from_slice(&fs::read(dir.join("result.json"))?)?)
        })();

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
                if resource_failure(&format!("{e:#}")) == Some("OOM") {
                    let capped = format!("{e:#}").contains("adapter memory limit exceeded");
                    known_failures.push(json!({"engine":engine,"backend":backend,"scenario":scenario,
                        "size":size,"worlds":worlds,"pin":engine_pin(&pins, &engine),
                        "memory_limit_bytes":if capped { json!(memory_limit) } else { Value::Null },
                        "trial":id,"run":out.display().to_string(),"error":record["error"],"created_unix":epoch}));
                    atomic_json(&known_failures_path(root), &known_failures)?;
                }
            }
        }
        record["gpu_timing"] = json!({"status":if job.backend == "cpu" {"not_applicable"} else {"not_collected"},
            "reason":if job.backend == "cpu" {"CPU backend"} else {"wall-time trial did not pass"}});
        if job.backend != "cpu" && record["status"] == "passed" {
            let profile = Job {
                mode: "gpu_timing".into(),
                ..job.clone()
            };
            atomic_json(&dir.join("gpu-job.json"), &profile)?;
            let measurement = (|| -> Result<physmark_core::GpuTiming> {
                subprocess(
                    adapter_command(root, &engine),
                    root,
                    &dir.join("gpu-job.json"),
                    &dir.join("gpu-result.json"),
                    &dir.join("gpu.log"),
                    trial_timeout,
                    threads,
                )?;
                let r: AdapterResult =
                    serde_json::from_slice(&fs::read(dir.join("gpu-result.json"))?)?;
                ensure!(
                    r.schema_version == 1 && r.engine == job.engine && r.backend == job.backend,
                    "GPU replay identity mismatch"
                );
                let timing = r
                    .gpu_timing
                    .as_ref()
                    .context("adapter omitted GPU timing status")?;
                timing.validate(&profile)?;
                if timing.status == "available" {
                    let validation = validate_result(&profile, &r);
                    ensure!(
                        validation.passed,
                        "GPU replay failed physics validation: {:?}",
                        validation.reasons
                    );
                    record["gpu_validation"] = serde_json::to_value(validation)?;
                    record["gpu_replay_elapsed_seconds"] = json!(r.elapsed_seconds);
                }
                Ok(timing.clone())
            })();
            match measurement {
                Ok(timing) => {
                    record["gpu_ms_per_step"] =
                        json!(timing.seconds.map(|s| s * 1000. / job.steps as f64));
                    record["gpu_timing"] = serde_json::to_value(timing)?;
                }
                Err(e) => {
                    record["gpu_timing"] = json!({"status":"error","reason":format!("{e:#}")})
                }
            }
        }
        record["memory"] =
            json!({"status":"not_collected", "reason":"wall-time trial did not pass"});
        if record["status"] == "passed" {
            memory_replay(root, &engine, &dir, &job, trial_timeout, threads, &mut record)?;
        }
        println!(
            "  {} / GPU {} / memory {}",
            record["status"], record["gpu_timing"]["status"], record["memory"]["status"]
        );
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
                trial_timeout,
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
        .filter(|r| {
            r["status"] != "passed"
                || r.get("capture_error").is_some()
                || r["gpu_timing"]["status"] == "error"
                || r["memory"]["status"] == "error"
        })
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
            let l = lock(&root)?;
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
            ensure!(
                entry.get("distribution").is_none(),
                "{engine} is installed from binary wheels; edit its tag commit and distribution in engines.lock"
            );
            let dir = root.join(".physmark/sources").join(&engine);
            checked(
                &dir,
                "git",
                &["fetch", "--depth", "1", "origin", &reference],
            )?;
            let rev = command_output(&dir, "git", &["rev-parse", "FETCH_HEAD"]);
            let path = root.join("engines.lock");
            let text = repin(&fs::read_to_string(&path)?, group, &engine, &rev)?;
            fs::write(path, text)?;
            println!("Pinned {engine} to {rev}. Run fetch, build, then smoke tests.");
            Ok(())
        }
        Action::Build { engines } => {
            fetch(&root)?;
            let status = Command::new(system_python())
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
                    native_adapter(&root, e)
                } else if *e == "isaac" {
                    isaac_python(&root)
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
        } => atomic_json(&output, &load_scene(&root, &scenario, size, seed, dt)?),
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
            resume,
            skip_cases,
            retry_known_failures,
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
            run(
                &root,
                s,
                threads,
                backend,
                output,
                resume,
                skip_cases,
                retry_known_failures,
            )
        }
        Action::Report {
            directory,
            no_video,
        } => report(&root, &directory, no_video),
        Action::RemeasureMemory {
            directory,
            ids,
            reason,
        } => {
            let out = std::env::current_dir()?.join(directory);
            remeasure_memory(&root, &out, &ids, &reason)
        }
        Action::Round {
            output,
            resume,
            threads,
            skip_build,
            retry_known_failures,
        } => {
            let mut c = Command::new(system_python());
            c.arg(root.join("tools/full_round.py"))
                .arg(std::env::current_dir()?.join(output))
                .current_dir(&root)
                .env("PHYSMARK_BIN", std::env::current_exe()?);
            if resume {
                c.arg("--resume");
            }
            if let Some(threads) = threads {
                c.arg("--threads").arg(threads.to_string());
            }
            if skip_build {
                c.arg("--skip-build");
            }
            if retry_known_failures {
                c.arg("--retry-known-failures");
            }
            ensure!(c.status()?.success(), "full round failed");
            Ok(())
        }
        Action::Site {
            site,
            round,
            id,
            label,
            max_evidence_mb,
        } => {
            let cwd = std::env::current_dir()?;
            let mut c = Command::new(system_python());
            c.arg(root.join("tools/site.py"))
                .arg(cwd.join(site))
                .arg(cwd.join(round))
                .arg("--max-evidence-mb")
                .arg(max_evidence_mb.to_string())
                .current_dir(&root);
            if let Some(id) = id {
                c.arg("--id").arg(id);
            }
            if let Some(label) = label {
                c.arg("--label").arg(label);
            }
            ensure!(c.status()?.success(), "site publishing failed");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_failures_are_classified_as_out_of_memory() {
        let dir = std::env::temp_dir().join(format!("physmark-oom-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let log = dir.join("adapter.log");
        for (text, oom) in [
            ("numpy._core._exceptions._ArrayMemoryError: Unable to allocate 150. GiB for an array", true),
            ("ValueError: Error: engine error: Could not allocate memory", true),
            ("\u{1b}[38;5;9m[Genesis] [ERROR] GenesisException: Mass matrix shape (n_dofs=60000, n_dofs=60000, n_envs=1) is too large.\u{1b}[0m", true),
            ("jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate", true),
            ("memory allocation of 123456789 bytes failed", true),
            ("ValueError: unsupported Genesis backend", false),
        ] {
            fs::write(&log, format!("setup\n{text}\n")).unwrap();
            let line = out_of_memory_line(&log);
            assert_eq!(line.is_some(), oom, "{text}");
            if let Some(line) = line {
                assert!(!line.contains('\u{1b}'), "{line}");
                assert_eq!(resource_failure(&format!("adapter out of memory: {line}")), Some("OOM"));
            }
        }
        assert_eq!(resource_failure("adapter timed out after 300s"), Some("timeout"));
        assert_eq!(resource_failure("adapter exited exit status: 1"), None);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn known_failures_follow_the_engine_pin() {
        let lock: toml::Value = toml::from_str(
            "[engines.mjx]\ncommit = \"abc\"\n[engines.isaac]\ncommit = \"def\"\ndistribution = \"isaacsim==6\"\n",
        )
        .unwrap();
        assert_eq!(engine_pin(&lock, "mjx"), "abc");
        assert_eq!(engine_pin(&lock, "isaac"), "def isaacsim==6");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn physical_footprint_can_read_current_process() {
        assert!(process_footprint(std::process::id()).unwrap() > 0);
    }
    #[test]
    #[cfg(target_os = "linux")]
    fn memory_footprint_covers_the_adapter_process_group() {
        use std::os::unix::process::CommandExt;
        // A shell leading its own group, holding ~64 MiB of anonymous memory in a child.
        let mut child = Command::new("sh")
            .args(["-c", "python3 -c 'b=bytearray(64<<20); import time; time.sleep(5)' & wait"])
            .process_group(0)
            .spawn()
            .unwrap();
        thread::sleep(Duration::from_secs(2));
        let bytes = process_footprint(child.id()).unwrap();
        let _ = child.kill();
        let _ = child.wait();
        assert!(bytes > 60 << 20, "{bytes}");
    }
    #[test]
    fn resume_requires_identical_workload_pins_and_machine() {
        let original = json!({"schema_version":1,"suite":{"sizes":[8,128,1024],"repeats":3,"engines":["nexus"]},
            "backends":{"nexus":"metal"},"threads_requested":1,"source_pins":{"nexus":"abc"},
            "lock_hash":"abc","python_packages":"packages","hardware":{"cpu":"test"},"rust":"1.98"});
        validate_resume_metadata(&original, &original).unwrap();
        let mut current = original.clone();
        current["build_stamp"] = json!("new controller");
        validate_resume_metadata(&original, &current).unwrap();
        for key in [
            "suite",
            "backends",
            "threads_requested",
            "source_pins",
            "lock_hash",
            "hardware",
            "python_packages",
        ] {
            let mut changed = original.clone();
            changed[key] = json!("changed");
            assert!(validate_resume_metadata(&original, &changed).is_err());
        }
        let mut retimed = original.clone();
        retimed["suite"]["timeout_seconds"] = json!(300);
        retimed["scheduling"] = json!("ascending environment count, then size");
        validate_resume_metadata(&original, &retimed).unwrap();
        let mut extended = original.clone();
        extended["suite"]["engines"] = json!(["nexus", "isaac"]);
        extended["backends"] = json!({"nexus":"metal","isaac":"cuda"});
        validate_resume_metadata(&original, &extended).unwrap();
        let mut rebacked = original.clone();
        rebacked["backends"] = json!({"nexus":"cuda"});
        assert!(validate_resume_metadata(&original, &rebacked).is_err());
        let mut resized = original.clone();
        resized["suite"]["repeats"] = json!(1);
        assert!(validate_resume_metadata(&original, &resized).is_err());
        assert!(parse_skip_cases(&["mjx:stacks:1024:1=memory limit".into()]).is_ok());
        assert!(parse_skip_cases(&["mjx:stacks:1024:1=".into()]).is_err());
    }
    #[test]
    fn repin_changes_only_the_commit_and_keeps_layout() {
        let a = "a".repeat(40);
        let b = "b".repeat(40);
        let c = "c".repeat(40);
        let text = format!(
            "# kept\n[engines.mujoco]\ncommit = \"{a}\"\n\n[engines.mjx]\ncommit = \"{a}\"\nsource = \"mujoco\"\n\n[dependencies.menagerie]\ncommit = \"{b}\"\n"
        );
        let updated = repin(&text, "engines", "mujoco", &c).unwrap();
        assert_eq!(updated, text.replace(&a, &c));
        let updated = repin(&text, "dependencies", "menagerie", &c).unwrap();
        assert_eq!(updated, text.replace(&b, &c));
        assert!(repin(&text, "engines", "mujoco", "abc").is_err());
    }
    #[test]
    fn binary_distributions_and_shared_checkouts_are_not_fetched() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let names: Vec<_> = sources(&lock(&root).unwrap())
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();
        assert!(names.contains(&"mujoco".into()) && names.contains(&"menagerie".into()));
        assert!(!names.contains(&"isaac".into()) && !names.contains(&"mjx".into()));
    }
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
