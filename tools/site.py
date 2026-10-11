#!/usr/bin/env python3
"""Add a finished round to a static site of per-platform reports, or replace that platform's copy.

    site/platforms.json      every published platform, in selector order
    site/index.html          opens the visitor's last platform, else the first one
    site/<platform id>/      the round's landing page, suite reports and evidence

Each platform directory is self-contained: a site built on another machine can upload its own
platform directory together with an updated platforms.json without touching the others. To add
a platform to a deployed site, place the deployed platforms.json in the site directory first.
Large evidence files (trajectories, scenes, profiler traces) are left out above --max-evidence-mb;
reports hide links to them. --report-only publishes just what visitors see: pages, charts, gallery
images and videos, without results, metadata or evidence files.
Usually started through `cargo run --release -- site <site> <round>`, or tools/publish.sh.
Standard library only.
"""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

from full_round import index
from site_nav import THEME_HEAD, default_platform, slug

# Files a suite report needs besides its runs; everything else at its top level is bookkeeping.
REPORT_FILES = ["index.html", "results.json", "summary.json", "metadata.json", "render-status.json"]
# Gallery media is always published, whatever its size.
MEDIA = {"simulation.png", "simulation.mp4"}

ROOT_PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhysMark</title><link rel="icon" href="__FIRST__/assets/branding/physmark-icon.svg">
__THEME_HEAD__<style>body{background:var(--bg);color:var(--text);font:16px/1.6 system-ui,sans-serif;margin:0}main{max-width:720px;margin:80px auto;padding:0 28px}a{color:var(--accent);font-weight:600;text-decoration:none}li{margin:8px 0}</style>
<main><h1>PhysMark</h1><p>Choose the platform the benchmark ran on:</p><ul id="platforms">__LINKS__</ul></main>
<script>(async()=>{let list;try{const r=await fetch('platforms.json',{cache:'no-cache'});if(!r.ok)return;list=(await r.json()).platforms}catch(e){return}
if(!Array.isArray(list)||!list.length)return;let last=null;try{last=localStorage.getItem('physmark-platform')}catch(e){}
const target=list.find(p=>p.id===last)||list[0];location.replace(encodeURIComponent(target.id)+'/'+location.hash)})()</script></html>"""


def publish_suite(source, destination, limit, report_only):
    destination.mkdir(parents=True)
    for name in ["index.html"] if report_only else REPORT_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    for licence in source.glob("LICENSE.*"):
        shutil.copy2(licence, destination / licence.name)
    for folder in ["curves", "assets"]:
        if (source / folder).is_dir():
            # Pages show the SVG charts; the PNG copies are for other uses.
            shutil.copytree(source / folder, destination / folder,
                            ignore=shutil.ignore_patterns("*.png") if report_only and folder == "curves" else None)
    if report_only:
        size = 0
        for path in sorted((source / "runs").glob("*/simulation.*")) if (source / "runs").is_dir() else []:
            if path.name in MEDIA:
                (destination / path.relative_to(source)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination / path.relative_to(source))
                size += path.stat().st_size
        (destination / "omitted.json").write_text(json.dumps("all"))
        return size, 0
    omitted, size = [], 0
    for path in sorted((source / "runs").rglob("*")) if (source / "runs").is_dir() else []:
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.name in MEDIA or limit is None or path.stat().st_size <= limit:
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / relative)
            size += path.stat().st_size
        else:
            omitted.append(relative.as_posix())
    (destination / "omitted.json").write_text(json.dumps(omitted))
    return size, len(omitted)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("site", type=Path, help="site directory, created if missing")
    p.add_argument("round", type=Path, help="round output directory (cargo run --release -- round <output>)")
    p.add_argument("--id", help="platform directory name, e.g. linux-rtx-5080 (default: the round's platform.json)")
    p.add_argument("--label", help="selector label, e.g. 'Linux · NVIDIA GeForce RTX 5080'")
    p.add_argument("--max-evidence-mb", type=float, default=1.0,
                   help="leave out larger per-trial evidence files; negative publishes everything (default: 1)")
    p.add_argument("--report-only", action="store_true",
                   help="publish only what visitors see: pages, charts, gallery images and videos")
    args = p.parse_args()
    source, site = args.round.resolve(), args.site.resolve()
    plan_path = source / "round-plan.json"
    if not plan_path.exists():
        p.error(f"{source} is not a round output (no round-plan.json)")
    plan = json.loads(plan_path.read_text())
    platform_path = source / "platform.json"
    platform = json.loads(platform_path.read_text()) if platform_path.exists() else dict(zip(("id", "label"), default_platform(plan)))
    label = args.label or platform["label"]
    platform_id = slug(args.id) if args.id else platform["id"]
    destination = site / platform_id
    if source == destination or source in destination.parents or site in source.parents or site == source:
        p.error("the site and the round must be separate directories")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in [] if args.report_only else ["round-plan.json", "round-status.json", "round-notes.json"]:
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    (destination / "platform.json").write_text(json.dumps(dict(id=platform_id, label=label), indent=2))
    limit = None if args.max_evidence_mb < 0 else int(args.max_evidence_mb * 2**20)
    statuses = json.loads((source / "round-status.json").read_text()) if (source / "round-status.json").exists() else {}
    total, omitted = 0, 0
    for suite in sorted(d for d in source.iterdir() if (d / "results.json").exists() and (d / "index.html").exists()):
        size, count = publish_suite(suite, destination / suite.name, limit, args.report_only)
        total, omitted = total + size, omitted + count
    # Regenerate the landing page so it carries the current navigation and label.
    index(destination, plan, statuses, source=source)

    manifest_path = site / "platforms.json"
    platforms = json.loads(manifest_path.read_text())["platforms"] if manifest_path.exists() else []
    entry = dict(id=platform_id, label=label)
    platforms = [entry if q["id"] == platform_id else q for q in platforms]
    if entry not in platforms:
        platforms.append(entry)
    manifest_path.write_text(json.dumps(dict(platforms=platforms), indent=2))
    links = "".join(f'<li><a href="{html.escape(q["id"], quote=True)}/">{html.escape(q["label"])}</a></li>' for q in platforms)
    (site / "index.html").write_text(ROOT_PAGE.replace("__THEME_HEAD__", THEME_HEAD).replace("__LINKS__", links).replace("__FIRST__", html.escape(platforms[0]["id"], quote=True)))
    if args.report_only:
        print(f"Published {label} to {destination} ({total / 2**20:,.0f} MiB of gallery media, report only)")
    else:
        print(f"Published {label} to {destination} ({total / 2**20:,.0f} MiB of evidence, {omitted:,} large files left out)")
    print(f"Upload {destination.name}/ and platforms.json (and index.html for a new site) from {site}")


if __name__ == "__main__":
    sys.exit(main())
