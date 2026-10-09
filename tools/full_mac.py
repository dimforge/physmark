#!/usr/bin/env python3
"""Run the complete Mac round serially and index its independently ranked reports."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PARTS = [
    ("rigid", "Rigid-body scale", "full-mac-rigid.toml", "Eight scenes at 8, 128, 1,024 and 10,000 bodies; one environment."),
    ("mpm", "Material points", "full-mac-mpm.toml", "Elastic, sand and liquid at 512, 4,096 and 32,768 particles; one environment."),
    ("environments", "Stack environments", "full-mac-environments.toml", "8- and 128-box stacks across 1, 4, 16, 64, 256, 1,024 and 4,096 environments."),
    ("robot", "Go1 robot environments", "robot.toml", "One 13-link, 12-actuator Go1 per environment; 1 through 4,096 independent environments."),
]


def index(output, statuses):
    cards = []
    for key, title, _, description in PARTS:
        part_status = statuses.get(key, {})
        directory = part_status.get("report_directory", key)
        path = output / directory / "results.json"
        records = json.loads(path.read_text()) if path.exists() else []
        counts = Counter(r["status"] for r in records)
        count_text = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "Pending"
        state = part_status.get("state", "pending")
        link = f'<a href="{html.escape(directory, quote=True)}/index.html">Open report →</a>' if (output / directory / 'index.html').exists() else ''
        cards.append(f'<article><div class="eyebrow">{html.escape(state)}</div><h2>{title}</h2><p>{description}</p><p class="counts">{count_text}</p>{link}</article>')
    notes_path = output / 'round-notes.json'
    if notes_path.exists():
        notes = ''.join(f'<p>{html.escape(note)}</p>' for note in json.loads(notes_path.read_text()))
        cards.insert(0, '<article style="grid-column:1/-1"><h2>Round status</h2>' + notes + '</article>')
    logo = ROOT / 'assets'
    if not (output/'assets').exists():
        shutil.copytree(logo, output/'assets')
    (output / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhysMark · Full Mac round</title><link rel="icon" href="assets/branding/physmark-icon.svg">
<style>:root{color-scheme:dark}body{background:#0b121b;color:#e8f0f6;font:16px/1.6 system-ui;margin:0}main{max-width:1120px;margin:60px auto;padding:0 28px}h1{font-size:42px;letter-spacing:-1.4px;line-height:1.15}h1 span{color:#66e3b5}h2{font-size:22px;margin:5px 0}p{color:#a8bac9}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px;margin:32px 0}article{background:#111d29;border:1px solid #25384a;border-radius:14px;padding:24px}a{color:#66e3b5;text-decoration:none;font-weight:600}.eyebrow{text-transform:uppercase;font-size:11px;letter-spacing:1.5px;color:#66e3b5}.counts{font-size:14px;color:#d4dee7}small{color:#9cb0c0}</style><main>
<div class="eyebrow">M4 Max · Full benchmark round</div><h1><img src="assets/branding/physmark-logo.svg" alt="physmark." style="width:290px;max-width:100%"></h1><p>Rapier CPU with 8 workers · MuJoCo CPU · Nexus Metal · Genesis Metal<br>MPM compares Nexus and Genesis. Newton and MJX are excluded from this Mac round.</p><div class="grid">''' + ''.join(cards) + '''</div><small>Each report includes synchronized timings, device timestamps where exposed, RAM/GPU memory, validation outcomes, scene previews and untimed replay videos. Rankings require complete valid coverage and use each suite’s declared baseline. Missing data and resource-limit skips remain visible. CPU subprocess budget: 120 seconds. Physical-footprint cap: 72 GiB.</small></main></html>''')
    (output / 'round-status.json').write_text(json.dumps(statuses, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        p.error('use a new output directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    previous = output / 'round-status.json'
    statuses = json.loads(previous.read_text()) if args.resume and previous.exists() else {}
    env = {**os.environ, 'PHYSMARK_MEMORY_LIMIT_GIB':'72'}
    index(output, statuses)
    for key, title, suite, _ in PARTS:
        if statuses.get(key, {}).get('state') in ('complete', 'complete with exclusions'):
            continue
        if shutil.disk_usage(output).free < 12 * 1024**3:
            raise RuntimeError('Less than 12 GiB free; preserve results and free space before --resume')
        command = [str(ROOT/'target/release/physmark'), 'run', '--suite', f'suites/{suite}', '--threads', '8', '--output', str(output/key)]
        if (output/key).exists():
            command.append('--resume')
        skip_file = output / 'skip-cases.json'
        if skip_file.exists():
            for case, reason in json.loads(skip_file.read_text()).items():
                command += ['--skip-case', f'{case}={reason}']
        if key == 'environments':
            for n in [256, 1024, 4096]:
                command += ['--skip-case', f'genesis:stacks:128:{n}=Previously stalled on Metal at 256 environments; larger medium batches excluded by user request']
        statuses[key] = dict(state='running', started_unix=time.time(), command=command)
        index(output, statuses)
        print(f'RUNNING {title}', flush=True)
        with (output/f'{key}.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(10)
                index(output, statuses)
        complete = False
        if (output/key/'index.html').exists():
            metadata = json.loads((output/key/'metadata.json').read_text())["suite"]
            records = json.loads((output/key/'results.json').read_text())
            expected = metadata["repeats"]
            for dimension in ["engines", "scenarios", "sizes", "worlds", "seeds"]:
                expected *= len(metadata[dimension])
            complete = len(records) == expected
        statuses[key].update(state=('complete' if process.returncode == 0 else 'complete with exclusions') if complete else 'failed',
                             exit_code=process.returncode, finished_unix=time.time())
        index(output, statuses)
        print(f'{title}: {statuses[key]["state"]}', flush=True)
        if not complete:
            raise RuntimeError(f'{title} did not produce its report; inspect {key}.log and resume')
    print(output/'index.html', flush=True)


if __name__ == '__main__':
    main()
