#!/usr/bin/env python3
"""Run the complete benchmark round on this machine and index its independently ranked reports.

Detects Metal, NVIDIA CUDA and Isaac Sim support, builds the engines that can run here,
probes each engine/backend with a tiny trial, then runs every suite with what passed.
Engines that cannot run are skipped, never replaced by another backend, and listed on the
landing page. Usually started through `cargo run --release -- round <output>`.
Standard library only: it runs before the benchmark Python environment exists.
"""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from capabilities import candidates, detect, nexus_feature
from site_nav import THEME_HEAD, bar, default_platform

ROOT = Path(__file__).resolve().parents[1]
EXE = ".exe" if platform.system() == "Windows" else ""
RIGID = ["rapier", "mujoco", "nexus", "genesis", "newton", "mjx", "isaac"]
PARTS = [
    ("rigid", "Rigid-body scale", "full-rigid.toml", RIGID,
     "Eight scenes at 8, 128, 1,024 and 10,000 bodies; one environment."),
    ("stacks", "Brick structures", "full-stacks.toml", RIGID,
     "Brick ring and pyramid walls smashed by rolling spheres at 1,000, 10,000 and 28,000 bodies; one environment."),
    ("stacks-large", "Large brick structures", "full-stacks-large.toml", ["rapier", "mujoco", "nexus", "genesis", "isaac"],
     "The Nexus 110K brick ring and 500K pyramids, each also at the other's scale; no replay capture."),
    ("mpm", "Material points", "full-mpm.toml", ["nexus", "genesis", "newton"],
     "Elastic and sand from 512 to 195,112 particles (8³ to 58³); one environment; CPIC disabled."),
    ("robot", "Robot environments", "robot.toml", ["rapier", "mujoco", "nexus", "genesis", "newton", "isaac"],
     "One Go1 quadruped (12 actuators) or Booster T1 humanoid (23 actuators) standing on the ground per environment; 1 through 4,096 independent environments."),
]
NAMES = dict(rapier="Rapier", mujoco="MuJoCo", nexus="Nexus", genesis="Genesis", newton="Newton", mjx="MJX", isaac="Isaac Sim")
REPOSITORY = "https://github.com/dimforge/physmark"
# Project websites, or repositories for engines without one.
HOMEPAGES = dict(
    rapier="https://rapier.rs",
    mujoco="https://mujoco.org",
    nexus="https://nexus.dimforge.com",
    genesis="https://genesis-embodied-ai.github.io",
    newton="https://github.com/newton-physics/newton",
    mjx="https://mujoco.readthedocs.io/en/stable/mjx.html",
    isaac="https://developer.nvidia.com/isaac/sim",
)


def controller():
    return os.environ.get("PHYSMARK_BIN") or str(ROOT / f"target/release/physmark{EXE}")


def venv_python(name):
    env = ROOT / ".physmark/envs" / name
    return env / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def adapter_command(engine):
    if engine in ("rapier", "nexus"):
        return [str(ROOT / f"adapters/{engine}/target/release/physmark-{engine}{EXE}")]
    python = venv_python("isaac" if engine == "isaac" else "python")
    return [str(python), str(ROOT / f"adapters/{engine}/{engine}_adapter.py")]


def build(backends):
    env = dict(os.environ)
    if "nexus" in backends:
        env["PHYSMARK_NEXUS_FEATURES"] = nexus_feature(backends["nexus"])
    command = [controller(), "--root", str(ROOT), "build", "--engines", ",".join(sorted(backends))]
    print("BUILD", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def probe(engine, backend, directory, threads):
    """Runs two steps of a two-body scene; returns None on success or the failure reason."""
    directory.mkdir(parents=True, exist_ok=True)
    scene = directory / "scene.json"
    if not scene.exists():
        subprocess.run([controller(), "--root", str(ROOT), "scene", "free_fall", "--size", "2", "--output", str(scene)],
                       cwd=ROOT, check=True, capture_output=True)
    job = dict(schema_version=1, scene=json.loads(scene.read_text()), engine=engine, backend=backend, mode="timing",
               worlds=1, steps=2, warmup=2, capture_stride=1, threads=threads)
    job_path, result_path = directory / f"{engine}-job.json", directory / f"{engine}-result.json"
    job_path.write_text(json.dumps(job))
    result_path.unlink(missing_ok=True)
    log = directory / f"{engine}.log"
    try:
        with log.open("w") as output:
            # Generous: the first Genesis/JAX compile or Isaac Sim start can take minutes.
            status = subprocess.run(adapter_command(engine) + [str(job_path), str(result_path)], cwd=ROOT,
                                    stdout=output, stderr=subprocess.STDOUT, timeout=1200).returncode
    except FileNotFoundError:
        return "adapter not installed"
    except subprocess.TimeoutExpired:
        return "probe timed out after 1200 s"
    if status != 0:
        tail = log.read_text(errors="replace").strip().splitlines()[-1:] or ["no output"]
        return f"probe failed: {tail[0][:300]}"
    if not result_path.exists():
        # Isaac Sim's Kit app swallows Python exceptions and still exits 0.
        return f"probe wrote no result; see {log.name}"
    result = json.loads(result_path.read_text())
    if result.get("backend") != backend or len(result.get("final_poses", [])) != 2:
        return "probe returned an incomplete result"
    return None


def plan_round(output, threads, skip_build):
    caps = detect()
    backends, skipped = candidates(caps)
    print("DETECTED", json.dumps({k: caps[k] for k in ("system", "machine", "cpu", "metal", "cuda", "nvidia_gpus")}), flush=True)
    for engine, reason in skipped.items():
        print(f"SKIP {engine}: {reason}", flush=True)
    if not skip_build:
        build(backends)
    for engine, backend in list(backends.items()):
        reason = probe(engine, backend, output / "probes", threads)
        print(f"PROBE {engine}/{backend}: {reason or 'ok'}", flush=True)
        if reason:
            skipped[engine] = f"{backend}: {reason}"
            del backends[engine]
    memory_limit = None
    if caps["system"] in ("Darwin", "Linux") and caps["memory_bytes"]:
        # The controller samples the adapter's physical footprint (macOS) or the resident
        # anonymous memory of its process group (Linux) and kills it above this cap.
        memory_limit = int(os.environ.get("PHYSMARK_MEMORY_LIMIT_GIB") or int(caps["memory_bytes"] / 2**30 * 0.5625))
    return dict(created_unix=time.time(), capabilities=caps, backends=backends, skipped=skipped,
                threads=threads, memory_limit_gib=memory_limit)


def hardware_line(plan):
    caps = plan["capabilities"]
    gpu = ", ".join(g.split(",")[0] for g in caps["nvidia_gpus"]) or ("Apple GPU (Metal)" if caps["metal"] else "no CUDA/Metal GPU")
    return f"{caps['cpu']} · {gpu} · {caps['system']} {caps['machine']}"


def index(output, plan, statuses):
    cards = []
    for key, title, _, _, description in PARTS:
        part_status = statuses.get(key, {})
        directory = part_status.get("report_directory", key)
        path = output / directory / "results.json"
        records = json.loads(path.read_text()) if path.exists() else []
        counts = Counter(r["status"] for r in records)
        count_text = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "Pending"
        state = part_status.get("state", "pending")
        engines = part_status.get("engines")
        engine_text = f'<p class="counts">{html.escape(", ".join(NAMES[e] for e in engines))}</p>' if engines else ""
        link = f'<a href="{html.escape(directory, quote=True)}/index.html">Open report →</a>' if (output / directory / "index.html").exists() else ""
        cards.append(f'<article><div class="eyebrow">{html.escape(state)}</div><h2>{title}</h2><p>{description}</p>{engine_text}<p class="counts">{count_text}</p>{link}</article>')
    notes = [f"{NAMES[e]} skipped: {reason}" for e, reason in sorted(plan["skipped"].items())]
    notes_path = output / "round-notes.json"
    if notes_path.exists():
        notes += json.loads(notes_path.read_text())
    if notes:
        cards.insert(0, '<article style="grid-column:1/-1"><h2>Round status</h2>' + "".join(f"<p>{html.escape(n)}</p>" for n in notes) + "</article>")
    for source in (ROOT / "assets").rglob("*"):
        target = output / "assets" / source.relative_to(ROOT / "assets")
        if source.is_file() and (not target.exists() or target.stat().st_size != source.stat().st_size):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    # This round is one platform of a published site (see site_nav); `physmark site` may relabel it.
    platform_path = output / "platform.json"
    if not platform_path.exists():
        platform_id, label = default_platform(plan)
        platform_path.write_text(json.dumps(dict(id=platform_id, label=label), indent=2))
    platform_label = json.loads(platform_path.read_text())["label"]
    backends = " · ".join(
        f'<a href="{HOMEPAGES[e]}">{NAMES[e]}</a> {html.escape(b.upper() if b in ("cpu", "cuda") else b.capitalize())}'
        for e, b in plan["backends"].items())
    limit = f" Adapter memory cap: {plan['memory_limit_gib']} GiB." if plan.get("memory_limit_gib") else ""
    (output / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhysMark · Full round</title><link rel="icon" href="assets/branding/physmark-icon.svg">
''' + THEME_HEAD + '''<style>*{box-sizing:border-box}body{background:var(--bg);color:var(--text);font:16px/1.6 system-ui,sans-serif;margin:0}main{max-width:1120px;margin:auto;padding:38px}header{display:flex;justify-content:space-between;align-items:center;gap:16px;border-bottom:1px solid var(--line);padding-bottom:28px}.brand{flex-shrink:0;max-width:100%;line-height:0}.brand img{display:block;width:288px;max-width:100%;height:auto}.tag{display:inline-block;border:1px solid var(--line);padding:3px 9px;border-radius:16px;font-size:12px}.muted{color:var(--muted)}h2{font-size:22px;margin:5px 0}p{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px;margin:32px 0}article{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px}a{color:var(--accent);text-decoration:none;font-weight:600}.eyebrow{text-transform:uppercase;font-size:11px;letter-spacing:1.5px;color:var(--accent);margin-top:26px}.counts{font-size:14px;color:var(--text)}.notice{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:10px;padding:16px 22px;margin:24px 0 0}.notice h2{font-size:17px;margin:0 0 6px}.notice p{margin:6px 0}small{color:var(--muted)}@media(max-width:650px){main{padding:20px}header{display:block}}</style><main>
<header><div class="brand"><img class="logo-dark" src="assets/branding/physmark-logo.svg" width="288" height="65" alt="PhysMark."><img class="logo-light" src="assets/branding/physmark-logo-light.svg" width="288" height="65" alt="PhysMark."></div><div class="muted">Physics, measured. <span class="tag">FULL BENCHMARK ROUND</span></div></header>''' + bar(0, platform_label) + '''
<div class="eyebrow">''' + html.escape(hardware_line(plan)) + '''</div><p>''' + backends + f" · {plan['threads']} CPU workers" + '''</p><section class="notice"><h2>Read before drawing conclusions</h2><p><strong>These benchmarks are AI-generated</strong>, including the scenes and the integration code of every engine.</p><p>Readers and physics engine authors are encouraged to double-check how each engine is integrated, in the <a href="''' + REPOSITORY + '''/tree/main/adapters">adapters directory</a>; corrections are welcome.</p><p>Take the results with a grain of salt: they only cover the specific scenarios below, each run with a specific solver configuration (timestep, substeps, iterations and the other settings listed in each report). Other scenes or settings can rank the engines differently.</p></section><div class="grid">''' + "".join(cards) + '''</div><small>Each report includes synchronized timings, device timestamps where exposed, RAM/GPU memory, validation outcomes, scene previews and untimed replay videos. Performance tables use synchronized time and highlight the fastest CPU and GPU engine per case. Missing data and resource-limit skips remain visible. CPU subprocess budget: 120 seconds.''' + html.escape(limit) + '''</small></main></html>''')
    (output / "round-status.json").write_text(json.dumps(statuses, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("output", type=Path)
    p.add_argument("--resume", action="store_true", help="continue an interrupted round with its recorded plan")
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--skip-build", action="store_true", help="use the engines already built")
    p.add_argument("--retry-known-failures", action="store_true",
                   help="run cases that earlier runs on this machine recorded as out of memory")
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        p.error("use a new output directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "round-plan.json"
    if args.resume and plan_path.exists():
        # Resumed suites must keep their backends, so the recorded plan is reused as is.
        plan = json.loads(plan_path.read_text())
        if not args.skip_build:
            build(plan["backends"])
    else:
        plan = plan_round(output, args.threads, args.skip_build)
        plan_path.write_text(json.dumps(plan, indent=2))
    previous = output / "round-status.json"
    statuses = json.loads(previous.read_text()) if args.resume and previous.exists() else {}
    env = dict(os.environ)
    if plan.get("memory_limit_gib"):
        env["PHYSMARK_MEMORY_LIMIT_GIB"] = str(plan["memory_limit_gib"])
    else:
        env.pop("PHYSMARK_MEMORY_LIMIT_GIB", None)
    index(output, plan, statuses)
    for key, title, suite, wanted, _ in PARTS:
        if statuses.get(key, {}).get("state") in ("complete", "complete with exclusions", "skipped"):
            continue
        engines = [e for e in wanted if e in plan["backends"]]
        if not engines:
            statuses[key] = dict(state="skipped", reason="no engine for this part runs on this machine")
            index(output, plan, statuses)
            continue
        if shutil.disk_usage(output).free < 12 * 1024**3:
            raise RuntimeError("Less than 12 GiB free; preserve results and free space before --resume")
        command = [controller(), "--root", str(ROOT), "run", "--suite", f"suites/{suite}", "--threads", str(plan["threads"]),
                   "--engines", ",".join(engines),
                   "--backend", ",".join(f"{e}={plan['backends'][e]}" for e in engines),
                   "--output", str(output / key)]
        if (output / key).exists():
            command.append("--resume")
        if args.retry_known_failures:
            command.append("--retry-known-failures")
        skip_file = output / "skip-cases.json"
        if skip_file.exists():
            for case, reason in json.loads(skip_file.read_text()).items():
                command += ["--skip-case", f"{case}={reason}"]
        statuses[key] = dict(state="running", started_unix=time.time(), command=command, engines=engines)
        index(output, plan, statuses)
        print(f"RUNNING {title}: {', '.join(engines)}", flush=True)
        with (output / f"{key}.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(10)
                index(output, plan, statuses)
        complete = False
        if (output / key / "index.html").exists():
            metadata = json.loads((output / key / "metadata.json").read_text())["suite"]
            records = json.loads((output / key / "results.json").read_text())
            expected = metadata["repeats"]
            for dimension in ["engines", "scenarios", "sizes", "worlds", "seeds"]:
                expected *= len(metadata[dimension])
            complete = len(records) == expected
        statuses[key].update(state=("complete" if process.returncode == 0 else "complete with exclusions") if complete else "failed",
                             exit_code=process.returncode, finished_unix=time.time())
        index(output, plan, statuses)
        print(f"{title}: {statuses[key]['state']}", flush=True)
        if not complete:
            raise RuntimeError(f"{title} did not produce its report; inspect {key}.log and resume")
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    sys.exit(main())
