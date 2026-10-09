#!/usr/bin/env python3
"""Offline reports and engine-neutral replay rendering, never called during timing."""

import argparse
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

COLORS = dict(
    rapier="#00b8a9",
    nexus="#ee9f34",
    genesis="#e86582",
    newton="#9467df",
    mujoco="#579df2",
    mjx="#b9c44a",
)


def count_unit(scenario):
    return "robots" if scenario == "robot_go1" else "particles" if scenario.startswith("mpm_") else "bodies"


def count_label(scenario, count):
    unit = count_unit(scenario)
    if count == 1:
        unit = {"robots": "robot", "particles": "particle", "bodies": "body"}[unit]
    return f"{count:,} {unit}"


def summarize(records, suite=None):
    groups = {}
    for r in records:
        key = (r["engine"], r["backend"], r["scenario"], r["size"], r["worlds"])
        groups.setdefault(key, []).append(r)
    if suite:
        import itertools

        for engine, scenario, size, worlds in itertools.product(
            suite["engines"], suite["scenarios"], suite["sizes"], suite["worlds"]
        ):
            if not any(
                key[0] == engine and key[2:] == (scenario, size, worlds)
                for key in groups
            ):
                groups[(engine, "unavailable", scenario, size, worlds)] = []
    rows = []
    for (engine, backend, scenario, size, worlds), runs in sorted(groups.items()):
        good = [r for r in runs if r["status"] == "passed"]
        values = [r["ms_per_step"] for r in good]
        expected = len(suite["seeds"]) * suite["repeats"] if suite else len(runs)
        complete = len(good) == expected and expected > 0
        if suite:
            complete = complete and {(r["seed"], r["repeat"]) for r in good} == {
                (seed, repeat)
                for seed in suite["seeds"]
                for repeat in range(suite["repeats"])
            }
        gpu = [
            r
            for r in good
            if r.get("gpu_timing", {}).get("status") == "available"
            and isinstance(r.get("gpu_ms_per_step"), (int, float))
            and math.isfinite(r["gpu_ms_per_step"])
            and r["gpu_ms_per_step"] > 0
        ]
        gpu_values = [r["gpu_ms_per_step"] for r in gpu]
        methods = {
            (r["gpu_timing"].get("method"), r["gpu_timing"].get("scope")) for r in gpu
        }
        gpu_complete = complete and len(gpu) == expected and len(methods) == 1
        if methods and any(not method or not scope for method, scope in methods):
            gpu_complete = False
        rows.append(
            dict(
                engine=engine,
                backend=backend,
                scenario=scenario,
                size=size,
                worlds=worlds,
                passed=len(good),
                total=expected,
                eligible=complete,
                median_ms=float(np.median(values)) if values else None,
                p25_ms=float(np.percentile(values, 25)) if values else None,
                p75_ms=float(np.percentile(values, 75)) if values else None,
                min_ms=min(values) if values else None,
                max_ms=max(values) if values else None,
                gpu_available=len(gpu),
                gpu_eligible=gpu_complete,
                gpu_method=gpu[0]["gpu_timing"].get("method") if gpu else None,
                gpu_scope=gpu[0]["gpu_timing"].get("scope") if gpu else None,
                gpu_median_ms=float(np.median(gpu_values)) if gpu_values else None,
                gpu_p25_ms=float(np.percentile(gpu_values, 25)) if gpu_values else None,
                gpu_p75_ms=float(np.percentile(gpu_values, 75)) if gpu_values else None,
            )
        )
        rows[-1]["memory"] = memory_statistics(good, expected, complete)
    return rows


def memory_statistics(runs, expected, complete):
    summary = {}
    for name in ["ram", "gpu", "footprint"]:
        metrics = [
            r["memory"]["metrics"][name]
            for r in runs
            if r.get("memory", {}).get("status") == "complete"
            and isinstance(r.get("steps"), int)
            and r["steps"] > 0
            and r["memory"].get("completed_steps") == r["steps"]
            and r["memory"].get("metrics", {}).get(name, {}).get("status")
            == "available"
        ]
        metrics = [
            m
            for m in metrics
            if isinstance(m.get("peak_bytes"), int) and m["peak_bytes"] > 0
        ]
        methods = {
            (m.get("method"), m.get("scope"), m.get("peak_kind")) for m in metrics
        }
        values = [m["peak_bytes"] for m in metrics]
        summary[name] = dict(
            available=len(metrics),
            eligible=bool(
                complete
                and len(metrics) == expected
                and len(methods) == 1
                and all(all(m) for m in methods)
            ),
            method=metrics[0].get("method") if metrics else None,
            scope=metrics[0].get("scope") if metrics else None,
            peak_kind=metrics[0].get("peak_kind") if metrics else None,
            median_bytes=float(np.median(values)) if values else None,
            p25_bytes=float(np.percentile(values, 25)) if values else None,
            p75_bytes=float(np.percentile(values, 75)) if values else None,
        )
    return summary


def leaderboard(rows, engines, baseline_engine="mujoco"):
    cases = {(r["scenario"], r["size"], r["worlds"]) for r in rows}
    baseline = {
        (r["scenario"], r["size"], r["worlds"]): r
        for r in rows
        if r["engine"] == baseline_engine and r["eligible"]
    }
    leaders = []
    for engine in engines:
        subset = [r for r in rows if r["engine"] == engine]
        complete = len(subset) == len(cases) and all(r["eligible"] for r in subset)
        ratios = [
            baseline[(r["scenario"], r["size"], r["worlds"])]["median_ms"]
            / r["median_ms"]
            for r in subset
            if r["eligible"] and (r["scenario"], r["size"], r["worlds"]) in baseline
        ]
        score = (
            math.exp(sum(map(math.log, ratios)) / len(ratios))
            if complete and len(ratios) == len(cases) and ratios
            else None
        )
        leaders.append(
            dict(
                engine=engine,
                backend=subset[0]["backend"] if subset else "?",
                passed=sum(r["passed"] for r in subset),
                total=sum(r["total"] for r in subset),
                complete=complete,
                score=score,
            )
        )
    return sorted(leaders, key=lambda r: -(r["score"] or 0))


def curves(rows, out, records=(), memory_metric=None):
    files = []
    memory_names = {
        "ram": "RAM · peak resident set",
        "gpu": "GPU / VRAM · sampled peak allocations",
        "footprint": "RAM · peak physical footprint",
    }
    if memory_metric:
        rows = [
            dict(
                r,
                eligible=r["memory"][memory_metric]["eligible"],
                gpu_eligible=False,
                **{
                    f"{key}_ms": r["memory"][memory_metric][f"{key}_bytes"] / 2**20
                    if r["memory"][memory_metric][f"{key}_bytes"] is not None
                    else None
                    for key in ["median", "p25", "p75"]
                },
            )
            for r in rows
        ]
        if not any(r["eligible"] for r in rows):
            return []
    batching = len({r["worlds"] for r in rows}) > 1
    axis, fixed = ("worlds", "size") if batching else ("size", "worlds")
    for scenario, count in sorted({(r["scenario"], r[fixed]) for r in rows}):
        if not any(
            r["eligible"]
            for r in rows
            if r["scenario"] == scenario and r[fixed] == count
        ):
            continue
        previews = sorted(
            [
                r
                for r in records
                if r["scenario"] == scenario
                and r[fixed] == count
                and r["status"] == "passed"
                and "capture" in r
                and (out / r["directory"] / "simulation.png").is_file()
            ],
            key=lambda r: (
                list(COLORS).index(r["engine"]),
                r["worlds"],
                -r["size"],
                r["seed"],
                r["repeat"],
            ),
        )
        preview = previews[0] if previews else None
        screenshot = (
            plt.imread(out / preview["directory"] / "simulation.png")
            if preview
            else None
        )
        for throughput in [False, True] if (batching or scenario.startswith("mpm_")) and not memory_metric else [False]:
            fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")

            def plot_values(rs, prefix, style, label, color):
                def value(r, key):
                    ms = r[prefix + key + "_ms"]
                    return (r["size"] if scenario.startswith("mpm_") else r["worlds"]) * 1000 / ms if throughput else ms

                x = [r[axis] for r in rs]
                ax.plot(
                    x, [value(r, "median") for r in rs], style, label=label, color=color
                )
                lo, hi = ("p75", "p25") if throughput else ("p25", "p75")
                ax.fill_between(
                    x,
                    [value(r, lo) for r in rs],
                    [value(r, hi) for r in rs],
                    color=color,
                    alpha=0.15,
                )

            for engine in COLORS:
                rs = sorted(
                    [
                        r
                        for r in rows
                        if r["engine"] == engine
                        and r["scenario"] == scenario
                        and r[fixed] == count
                        and r["eligible"]
                    ],
                    key=lambda r: r[axis],
                )
                if not rs:
                    continue
                if memory_metric:
                    signatures = {
                        (
                            r["memory"][memory_metric]["method"],
                            r["memory"][memory_metric]["scope"],
                            r["memory"][memory_metric]["peak_kind"],
                        )
                        for r in rs
                    }
                    for signature in sorted(signatures):
                        matching = [
                            r
                            for r in rs
                            if (
                                r["memory"][memory_metric]["method"],
                                r["memory"][memory_metric]["scope"],
                                r["memory"][memory_metric]["peak_kind"],
                            )
                            == signature
                        ]
                        label = f"{engine} / {matching[0]['backend']}"
                        if len(signatures) > 1:
                            label += f" · {signature[0]} / {signature[1]}"
                        plot_values(matching, "", "o-", label, COLORS[engine])
                else:
                    plot_values(
                        rs,
                        "",
                        "o-",
                        f"{engine} / {rs[0]['backend']} · sync",
                        COLORS[engine],
                    )
                device = [r for r in rs if r.get("gpu_eligible")]
                # Keep different profiler methods and scopes in separate curves.
                for method, scope in sorted(
                    {(r["gpu_method"], r["gpu_scope"]) for r in device}
                ):
                    ds = [
                        r
                        for r in device
                        if (r["gpu_method"], r["gpu_scope"]) == (method, scope)
                    ]
                    plot_values(
                        ds,
                        "gpu_",
                        "s--",
                        f"{engine} / {ds[0]['backend']} · GPU timestamps",
                        COLORS[engine],
                    )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Environments" if batching else {"robots": "Robots per environment", "particles": "Particles per environment", "bodies": "Dynamic bodies per world"}[count_unit(scenario)],
                ylabel="Memory (MiB)"
                if memory_metric
                else ("Particle-steps / second" if scenario.startswith("mpm_") else "Environment-steps / second")
                if throughput
                else "Time / step across all environments (ms)",
                title=(memory_names[memory_metric] + "\n" if memory_metric else "")
                + f"{scenario.replace('_', ' ')} · "
                + (f"{count_label(scenario, count)} / environment" if batching else f"{count} world(s)")
                + (" · higher is better" if throughput else " · lower is better"),
            )
            requested = sorted(
                {
                    r[axis]
                    for r in rows
                    if r["scenario"] == scenario and r[fixed] == count
                }
            )
            ax.set_xticks(requested, [f"{value:,}" for value in requested])
            left = requested[0] / 1.25
            # Reserve a right-hand band so the preview and legend cannot hide
            # the final data points. This padding is logarithmic like the x axis.
            right = requested[-1] * (requested[-1] / left) ** 0.4
            ax.set_xlim(left, right)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8, loc="upper right")
            if preview is not None:
                thumbnail = ax.inset_axes([0.795, 0.025, 0.19, 0.25], zorder=5)
                thumbnail.imshow(screenshot)
                thumbnail.set_anchor("SE")
                thumbnail.set_xticks([])
                thumbnail.set_yticks([])
                thumbnail.set_title(
                    f"Scene preview · {count_label(scenario, preview['size'])} · {preview['engine']}",
                    fontsize=6,
                    color="#596574",
                    pad=3,
                )
                for spine in thumbnail.spines.values():
                    spine.set_edgecolor("#9eabb8")
                    spine.set_linewidth(0.6)
            suffix = f"n{count}-environments" if batching else f"w{count}"
            if memory_metric:
                suffix += f"-memory-{memory_metric}"
            filename = (
                f"curves/{scenario}-{suffix}{'-throughput' if throughput else ''}"
            )
            fig.savefig(out / (filename + ".svg"))
            fig.savefig(out / (filename + ".png"), dpi=150)
            plt.close(fig)
            files.append(filename + ".svg")
    return files


def rotation(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def geometry(body):
    if body.get("parts"):
        return [face @ rotation(part["rotation"]).T + part["position"]
                for part in body["parts"] for face in geometry(part)]
    size = body["size"]
    if body["shape"] == "box":
        v = np.array(
            [
                [x, y, z]
                for x in [-size[0], size[0]]
                for y in [-size[1], size[1]]
                for z in [-size[2], size[2]]
            ]
        )
        return [
            v[ids]
            for ids in [
                [0, 1, 3, 2],
                [4, 6, 7, 5],
                [0, 4, 5, 1],
                [2, 3, 7, 6],
                [0, 2, 6, 4],
                [1, 5, 7, 3],
            ]
        ]
    if body["shape"] == "cylinder":
        angles = np.linspace(0, 2*np.pi, 13)
        rings = [np.array([[size[0]*np.cos(a), size[0]*np.sin(a), z] for a in angles]) for z in [-size[2], size[2]]]
        # Triangulate caps as quads with a repeated centre vertex, matching VTK's fixed face width.
        faces = [np.array([rings[0][i], rings[0][i+1], rings[1][i+1], rings[1][i]]) for i in range(12)]
        faces += [np.array([ring[i], ring[i+1], [0,0,z], [0,0,z]]) for ring,z in zip(rings, [-size[2],size[2]]) for i in range(12)]
        return faces
    r = size[0]
    h = size[2] if body["shape"] == "capsule" else 0
    lat = np.linspace(-np.pi / 2, np.pi / 2, 9)
    lon = np.linspace(0, 2 * np.pi, 13)
    v = np.array(
        [
            [
                [
                    r * np.cos(a) * np.cos(b),
                    r * np.cos(a) * np.sin(b),
                    r * np.sin(a) + (h if a >= 0 else -h),
                ]
                for b in lon
            ]
            for a in lat
        ]
    )
    return [
        np.array([v[i, j], v[i + 1, j], v[i + 1, j + 1], v[i, j + 1]])
        for i in range(8)
        for j in range(12)
    ]


# Bump when replay rendering changes so reports do not reuse obsolete media.
RENDER_VERSION = 5


class GroundPlane(Poly3DCollection):
    def do_3d_projection(self):
        super().do_3d_projection()
        # The fixed camera looks down onto the ground. Its large near edge must
        # not make the whole plane paint over above-ground objects. Other
        # collections retain Matplotlib's normal depth sorting.
        return float("inf")


def render(record, out, no_video=False):
    capture = out / record["capture"]
    data = json.loads(capture.read_text())
    job = json.loads((out / record["directory"] / "job.json").read_text())
    scene = job["scene"]
    frames = data["frames"]
    if not frames:
        raise ValueError("capture contains no frames")
    bodies = [b for b in scene["bodies"] if b["dynamic"]]
    if scene.get("robot"):
        bodies = [{**b, "parts": parts} for b, parts in zip(bodies, scene["robot"]["render_parts"])]
    if any(len(f["poses"]) != len(bodies) for f in frames):
        raise ValueError("capture pose count mismatch")
    positions = np.array([f["poses"] for f in frames])[:, :, :3]
    if not np.isfinite(positions).all():
        raise ValueError("capture contains non-finite positions")
    margin = 0.12 if scene.get("mpm") or scene.get("robot") else 1.0
    lo = positions.min(axis=(0, 1)) - margin
    hi = positions.max(axis=(0, 1)) + margin
    if scene["scenario"] != "free_fall":
        lo[2] = -0.05 if scene.get("mpm") else -0.3
    center = (hi + lo) / 2
    span = max(hi - lo)
    lo = center - span / 2
    hi = center + span / 2
    if scene.get("mpm") or scene.get("robot") or len(bodies) >= 128:
        from render_large import render_large

        return render_large(record, out, scene, frames, bodies, (lo, hi),
                            geometry, rotation, COLORS[record["engine"]], RENDER_VERSION, no_video)
    fig = plt.figure(figsize=(7.2, 5.4), dpi=100, facecolor="#101923")
    ax = fig.add_subplot(111, projection="3d", facecolor="#101923")
    fig.subplots_adjust(0, 0, 1, 0.92)
    ax.set_axis_off()
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=-62)
    if scene["scenario"] != "free_fall":
        floor = np.array(
            [
                [lo[0] - span, lo[1] - span, 0],
                [hi[0] + span, lo[1] - span, 0],
                [hi[0] + span, hi[1] + span, 0],
                [lo[0] - span, hi[1] + span, 0],
            ]
        )
        ax.add_collection3d(
            GroundPlane([floor], facecolors="#263746", edgecolors="#486172", alpha=1.0)
        )
    for b in scene["bodies"]:
        if not b["dynamic"] and b["position"][2] > 0:
            faces = [
                v @ rotation(b["rotation"]).T + np.array(b["position"])
                for v in geometry(b)
            ]
            ax.add_collection3d(
                Poly3DCollection(
                    faces, facecolors="#64778a", edgecolors="#a3afbb", linewidths=0.4
                )
            )
    meshes = [geometry(b) for b in bodies]
    collection = Poly3DCollection(
        [],
        facecolors=COLORS[record["engine"]],
        edgecolors="#102a34",
        linewidths=0.35,
        alpha=1.0,
    )
    ax.add_collection3d(collection)
    caption = fig.text(0.04, 0.965, "", color="white", size=11, va="top")

    def draw(frame):
        faces = []
        for mesh, pose in zip(meshes, frame["poses"]):
            R = rotation(pose[3:])
            p = np.array(pose[:3])
            faces.extend(v @ R.T + p for v in mesh)
        collection.set_verts(faces)
        caption.set_text(
            f"{record['engine'].upper()} / {record['backend']}   •   {scene['scenario'].replace('_', ' ')}   •   n={len(bodies)}\nWorld 0   ·   t={frame['step'] * scene['dt']:.3f} s   ·   untimed replay"
        )
        fig.canvas.draw()

    media = out / record["directory"]
    draw(frames[-1])
    fig.savefig(media / "simulation.png", facecolor=fig.get_facecolor())
    if not no_video and shutil.which("ffmpeg"):
        width, height = fig.canvas.get_width_height()
        duration = frames[-1]["step"] * scene["dt"]
        fps = 30
        times = np.array([f["step"] * scene["dt"] for f in frames])
        indices = np.searchsorted(
            times, np.linspace(0, duration, max(2, int(duration * fps) + 1))
        ).clip(max=len(frames) - 1)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgba",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(media / "simulation.mp4"),
        ]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in indices:
                draw(frames[i])
                process.stdin.write(fig.canvas.buffer_rgba())
        finally:
            process.stdin.close()
            status = process.wait()
        if status:
            raise RuntimeError(f"ffmpeg exited {status}")
    plt.close(fig)
    (media / "render-version.json").write_text(
        json.dumps(
            {
                "png": RENDER_VERSION,
                "mp4": RENDER_VERSION
                if not no_video and shutil.which("ffmpeg")
                else None,
            }
        )
    )


BRAND_FILES = (
    "physmark-logo.svg",
    "physmark-icon.svg",
    "favicon-32.png",
    "apple-touch-icon.png",
)
BRAND_LOGO = '<div class="brand"><img src="assets/physmark-logo.svg" width="288" height="65" alt="PhysMark."></div>'
BRAND_HEAD = """<!-- physmark-branding:start -->
<link rel="icon" type="image/png" sizes="32x32" href="assets/favicon-32.png">
<link rel="icon" type="image/svg+xml" sizes="any" href="assets/physmark-icon.svg">
<link rel="apple-touch-icon" sizes="180x180" href="assets/apple-touch-icon.png">
<meta name="theme-color" content="#0d141b">
<style>.brand{flex-shrink:0;max-width:100%;line-height:0}.brand img{display:block;width:288px;max-width:100%;height:auto}</style>
<!-- physmark-branding:end -->"""


def brand_report(page, out):
    """Package local assets and brand current or legacy HTML without rerunning physics."""
    source = Path(__file__).resolve().parents[1] / "assets" / "branding"
    destination = out / "assets"
    destination.mkdir(exist_ok=True)
    for name in BRAND_FILES:
        shutil.copy2(source / name, destination / name)
    if "__BRAND_HEAD__" in page:
        page = page.replace("__BRAND_HEAD__", BRAND_HEAD)
    else:
        page = re.sub(
            r"<!-- physmark-branding:start -->.*?<!-- physmark-branding:end -->\n?",
            "",
            page,
            flags=re.S,
        )
        page = page.replace("<style>", BRAND_HEAD + "\n<style>", 1)
    if "__BRAND_LOGO__" in page:
        return page.replace("__BRAND_LOGO__", BRAND_LOGO)
    return re.sub(
        r'<div class="brand">.*?</div>', lambda _: BRAND_LOGO, page, count=1, flags=re.S
    )


HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhysMark — __TITLE__</title>
__BRAND_HEAD__
<style>:root{color-scheme:dark;--bg:#0d141b;--panel:#17222e;--line:#304252;--text:#eaf1f7;--muted:#a5b8c7;--accent:#41dcc3}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:38px}header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:28px}.brand{font-size:30px;font-weight:750;letter-spacing:-1px}small,.muted{color:var(--muted)}h1{font-size:42px;letter-spacing:-1.5px;margin:30px 0 8px}h2{font-size:22px;margin-top:38px}p{max-width:970px}a{color:var(--accent)}.stats,.leaders,.gallery,.charts{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px}.number{font-size:30px;font-weight:700}.leaders .score{font-size:34px;font-weight:750;color:var(--accent)}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:11px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:500}select,input{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:10px;font:inherit;margin:0 8px 12px 0}.scroll{overflow:auto}.passed{color:#52dfb1}.error,.invalid{color:#ff8d95}.charts{grid-template-columns:repeat(auto-fit,minmax(400px,1fr))}.charts img{width:100%;background:white;border-radius:8px}.gallery{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.gallery video,.gallery img{width:100%;border-radius:8px}.gallery .card{padding:12px}.scene-section{margin-top:28px;padding-top:20px;border-top:1px solid var(--line)}.scene-section h3{font-size:20px;margin:0 0 4px}.scene-section>p{margin:0 0 16px}.tag{display:inline-block;border:1px solid var(--line);padding:3px 9px;border-radius:16px;font-size:12px}details{margin-top:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted);font-size:12px}footer{margin-top:50px;border-top:1px solid var(--line);padding-top:22px;color:var(--muted)}@media(max-width:650px){main{padding:20px}h1{font-size:32px}.charts{grid-template-columns:1fr}header{display:block}}</style>
<main><header>__BRAND_LOGO__<div class="muted">Physics, measured. <span class="tag">LOCAL BENCHMARK</span></div></header><h1>__TITLE__</h1><p class="muted">__HARDWARE__ · __DATE__<br>Synchronized physics stepping. Rendering and compilation excluded. Sleeping disabled.</p><div class="stats" id="stats"></div>__RUN_NOTES__<h2>Leaderboard · synchronized time</h2><p class="muted">Geometric mean speed relative to __BASELINE__ across every case in this suite (higher is faster). A score requires every requested trial and every baseline case to pass. These are baseline sanity checks, not matched-accuracy results. CPU and GPU entries are labelled separately.</p><div class="leaders" id="leaders"></div><h2>Performance curves</h2><p class="muted">Solid lines: synchronized wall time. Dashed lines: device timestamps from an independent warmed replay. Time covers a simulation step across all environments. Throughput is environments × 1,000 / milliseconds per batch step. Shading shows the timing 25th–75th percentiles across repeats and seeds, inverted for throughput. Each metric requires complete coverage. GPU scopes differ (pass sums, activity sums or activity unions); consult the table below. Profiling may alter scheduling; instrumented GPU times can exceed unprofiled synchronized times. The leaderboard uses only synchronized times.</p><div class="charts">__CURVES__</div><h2>GPU timing coverage</h2><div class="scroll">__GPU_STATUS__</div><h2>RAM and GPU memory · lower is better</h2><p class="muted">Separate memory replays preserve the timing measurements. RAM is the operating system’s process resident-set high-water mark; macOS physical footprint is also reported. GPU / VRAM is the sampled peak of native per-process device counters (10 ms target polling plus synchronized phase boundaries). Peaks cover setup, warmup, reset and rollout, ending before final readback and rendering. They include runtime, compilation caches and allocator pools, not just body data. Sampling can miss brief GPU peaks. Apple Silicon uses unified memory: GPU allocations and process RAM overlap and must not be added. Metal reports allocated resources; NVIDIA NVML reports per-process device memory, including context overhead. Different counter scopes are not interchangeable. Missing measurements are never inferred from timings or old results.</p><div class="charts">__MEMORY_CURVES__</div><div class="scroll">__MEMORY_STATUS__</div><h2>Trials</h2><select id="engine"><option value="">All engines</option></select><select id="scenario"><option value="">All scenarios</option></select><select id="status"><option value="">All outcomes</option><option>passed</option><option>invalid</option><option>error</option><option>skipped</option></select><div class="scroll"><table><thead><tr><th>Engine / backend</th><th>Scenario</th><th>__COUNT_UNIT__ × worlds</th><th>Seed / repeat</th><th>Sync ms / step</th><th>GPU ms / step</th><th>GPU timing</th><th>World steps / s</th><th>Peak RAM RSS MiB</th><th>Peak footprint MiB</th><th>GPU peak MiB (sampled)</th><th>Memory</th><th>Outcome</th><th>Evidence</th></tr></thead><tbody id="trials"></tbody></table></div><h2>Simulation gallery</h2><p class="muted">Independent untimed replay, rendered by the same renderer and camera rules. Batched cases show world 0. Videos play at simulated time.</p><div id="simulation-gallery">__GALLERY__</div><details><summary>Methodology, source pins and machine metadata</summary><pre id="metadata"></pre></details><footer><a href="results.json">Raw trials</a> · <a href="summary.json">Summary statistics</a> · <a href="metadata.json">Provenance</a> · PhysMark protocol v1__ROBOT_LICENSE__<br>No single score establishes physical accuracy. Inspect solver settings, validation errors, coverage and simulations alongside timing.</footer></main>
<script>const DATA=__DATA__;const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>x==null?'—':Number(x).toLocaleString(undefined,{maximumSignificantDigits:4});
document.querySelector('#stats').innerHTML=[['Engines',DATA.leaders.length],['Trials',DATA.records.length],['Passed',DATA.records.filter(r=>r.status==='passed').length],['Cases',new Set(DATA.rows.map(r=>[r.scenario,r.size,r.worlds].join('/'))).size]].map(([label,n])=>`<div class="card"><small>${label}</small><div class="number">${n}</div></div>`).join('');
document.querySelector('#leaders').innerHTML=DATA.leaders.map((r,i)=>`<div class="card"><small>${r.score==null?'UNRANKED':'#'+(i+1)} · ${esc(r.backend)}</small><h3>${esc(r.engine)}</h3><div class="score">${r.score==null?'—':fmt(r.score)+'×'}</div><small>${r.passed}/${r.total} trials passed${r.score==null?' · incomplete coverage or missing baseline':''}</small></div>`).join('');
for(const field of ['engine','scenario']){const el=document.querySelector('#'+field);for(const v of [...new Set(DATA.records.map(r=>r[field]))].sort()){const opt=document.createElement('option');opt.textContent=v;el.append(opt)}}
function memoryCell(r,key){const m=r.memory?.metrics?.[key];return m?.status==='available'?fmt(m.peak_bytes/1048576):'—'}
function draw(){document.querySelector('#trials').innerHTML=DATA.records.filter(r=>['engine','scenario','status'].every(k=>!document.querySelector('#'+k).value||r[k]===document.querySelector('#'+k).value)).map(r=>`<tr><td>${esc(r.engine)} <small>${esc(r.backend)}</small></td><td>${esc(r.scenario)}</td><td>${r.size} × ${r.worlds}</td><td>${r.seed} / ${r.repeat}</td><td>${fmt(r.ms_per_step)}</td><td>${fmt(r.gpu_ms_per_step)}</td><td title="${esc(r.gpu_timing?.reason||[r.gpu_timing?.method,r.gpu_timing?.scope].filter(Boolean).join(' / '))}">${esc(r.gpu_timing?.status||'not_collected')}</td><td>${fmt(r.world_steps_per_second)}</td><td>${memoryCell(r,'ram')}</td><td>${memoryCell(r,'footprint')}</td><td>${memoryCell(r,'gpu')}</td><td title="${esc(r.memory?.reason||'Independent replay; see memory result for counter coverage')}">${esc(r.memory?.status||'not_collected')}</td><td class="${r.status}" title="${esc(r.error||r.validation?.reasons.join('; ')||'')}">${r.status}</td><td><a href="${r.directory}/job.json">job</a> · <a href="${r.directory}/adapter.log">log</a>${['passed','invalid'].includes(r.status)?` · <a href="${r.directory}/result.json">result</a>`:''}${['available','unavailable','error'].includes(r.gpu_timing?.status)?` · <a href="${r.directory}/gpu.log">GPU log</a>`:''}${['available','unavailable'].includes(r.gpu_timing?.status)?` · <a href="${r.directory}/gpu-result.json">GPU result</a>`:''}${['complete','error'].includes(r.memory?.status)?` · <a href="${r.directory}/memory.log">memory log</a>`:''}${r.memory?.status==='complete'?` · <a href="${r.directory}/memory-result.json">memory result</a>`:''}</td></tr>`).join('')}
for(const el of document.querySelectorAll('select'))el.addEventListener('change',draw);draw();document.querySelector('#metadata').textContent=JSON.stringify(DATA.metadata,null,2);</script></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    out = args.directory.resolve()
    records = json.loads((out / "results.json").read_text())
    metadata = json.loads((out / "metadata.json").read_text())
    rows = summarize(records, metadata["suite"])
    baseline = metadata["suite"].get("baseline") or "mujoco"
    leaders = leaderboard(rows, metadata["suite"]["engines"], baseline)
    (out / "summary.json").write_text(
        json.dumps(dict(cases=rows, leaderboard=leaders), indent=2)
    )
    (out / "curves").mkdir(exist_ok=True)
    gallery = {}
    failures = []
    engine_order = {
        engine: index for index, engine in enumerate(metadata["suite"]["engines"])
    }
    for r in sorted(
        records,
        key=lambda r: (engine_order[r["engine"]], r["backend"], r["repeat"]),
    ):
        if "capture" not in r:
            continue
        try:
            media_dir = out / r["directory"]
            version_path = media_dir / "render-version.json"
            version = (
                json.loads(version_path.read_text()) if version_path.exists() else {}
            )
            if (
                version.get("png") != RENDER_VERSION
                or not (media_dir / "simulation.png").exists()
                or (
                    not args.no_video
                    and (
                        version.get("mp4") != RENDER_VERSION
                        or not (media_dir / "simulation.mp4").exists()
                    )
                )
            ):
                render(r, out, args.no_video)
            version = json.loads(version_path.read_text())
            base = r["directory"]
            title = f"{r['engine']} / {r['backend']}"
            if r["status"] != "passed":
                title += " · diagnostic (excluded trial)"
            media = (
                f'<video controls loop muted preload="none" poster="{base}/simulation.png" src="{base}/simulation.mp4"></video>'
                if version.get("mp4") == RENDER_VERSION
                and (out / base / "simulation.mp4").exists()
                else f'<img loading="lazy" src="{base}/simulation.png" alt="{html.escape(title)}">'
            )
            scene_key = (r["scenario"], r["size"], r["worlds"], r["seed"])
            gallery.setdefault(scene_key, []).append(
                f'<div class="card">{media}<small>{html.escape(title)}</small><br><a href="{base}/capture.json">Trajectory</a> · <a href="{base}/simulation.png">PNG</a></div>'
            )
        except Exception as e:
            failures.append(dict(id=r["id"], error=str(e)))
            print(f"Render failed for {r['id']}: {e}", file=sys.stderr)
    gallery_sections = []
    for index, ((scenario, size, worlds, seed), cards) in enumerate(
        sorted(gallery.items())
    ):
        heading = f"{scenario.replace('_', ' ').capitalize()} · {count_label(scenario, size)}"
        details = f"{worlds} world{'s' if worlds != 1 else ''} · Seed {seed}"
        gallery_sections.append(
            f'<section class="scene-section" aria-labelledby="scene-{index}">'
            f'<h3 id="scene-{index}">{html.escape(heading)}</h3>'
            f'<p class="muted">{html.escape(details)}</p>'
            f'<div class="gallery">{"".join(cards)}</div></section>'
        )
    (out / "render-status.json").write_text(
        json.dumps(dict(failures=failures, ffmpeg=shutil.which("ffmpeg")), indent=2)
    )
    # Generate plots after captures so a fresh report can embed its scene previews.
    files = curves(rows, out, records)
    memory_files = [
        f
        for metric in ["ram", "gpu", "footprint"]
        for f in curves(rows, out, records, metric)
    ]
    memory_coverage = {}
    for record in records:
        memory = record.get(
            "memory",
            {"status": "not_collected", "reason": "Run predates memory collection"},
        )
        for name in ["ram", "gpu", "footprint"]:
            metric = memory.get("metrics", {}).get(name, memory)
            if (
                name == "footprint"
                and memory.get("status") == "complete"
                and name not in memory.get("metrics", {})
            ):
                continue
            key = (
                record["engine"],
                record["backend"],
                name,
                metric["status"],
                metric.get("method", ""),
                metric.get("scope", ""),
                metric.get("peak_kind", ""),
                metric.get("reason", ""),
            )
            memory_coverage[key] = memory_coverage.get(key, 0) + 1
    memory_status = "<table><thead><tr><th>Engine / backend</th><th>Metric</th><th>Trials</th><th>Status</th><th>Counter / scope / reason</th></tr></thead><tbody>"
    for (engine, backend, name, status, method, scope, kind, reason), count in sorted(
        memory_coverage.items()
    ):
        detail = reason or f"{method} / {scope} / {kind}"
        memory_status += f'<tr><td>{html.escape(engine)} / {html.escape(backend)}</td><td>{name}</td><td>{count}</td><td>{html.escape(status)}</td><td style="white-space:normal">{html.escape(detail)}</td></tr>'
    memory_status += "</tbody></table>"
    import datetime

    data = json.dumps(
        dict(records=records, metadata=metadata, rows=rows, leaders=leaders),
        allow_nan=False,
    ).replace("<", "\\u003c")
    coverage = {}
    for record in records:
        timing = record.get(
            "gpu_timing",
            {"status": "not_collected", "reason": "Run predates GPU timing collection"},
        )
        key = (
            record["engine"],
            record["backend"],
            timing["status"],
            timing.get("method") or "—",
            timing.get("scope") or "—",
            timing.get("reason") or "",
        )
        coverage[key] = coverage.get(key, 0) + 1
    gpu_status = "<table><thead><tr><th>Engine / backend</th><th>Trials</th><th>Status</th><th>Method / scope / reason</th></tr></thead><tbody>"
    for (engine, backend, status, method, scope, reason), count in sorted(
        coverage.items()
    ):
        detail = reason or f"{method} / {scope}"
        gpu_status += f'<tr><td>{html.escape(engine)} / {html.escape(backend)}</td><td>{count}</td><td>{html.escape(status)}</td><td style="white-space:normal">{html.escape(detail)}</td></tr>'
    gpu_status += "</tbody></table>"
    notes_path = out / "run-notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else []
    if any(s.startswith("mpm_") for s in metadata["suite"]["scenarios"]):
        notes = ["Single-environment MPM. Size is material-point count. Every engine receives the same particle positions, masses, grid spacing, gravity, substeps and simulated horizon. Constitutive laws, grids and collision discretizations differ; this is not an accuracy-equated comparison. The baseline is " + baseline + ".",
                 "MPM throughput plots show particle-steps per second. Snapshots display material points as spheres; their rigid orientation fields are protocol padding. Newton MPM uses SolverImplicitMPM, independently of its rigid-body MuJoCo Warp backend.", *notes]
    robot_license = ""
    if "robot_go1" in metadata["suite"]["scenarios"]:
        license_source = Path(__file__).resolve().parents[1] / "docs/licenses/unitree-go1.txt"
        shutil.copy2(license_source, out / "LICENSE.unitree-go1.txt")
        robot_license = ' · <a href="LICENSE.unitree-go1.txt">Unitree Go1 model license</a>'
        notes = ["One scene unit is one 13-link, 12-actuator Unitree Go1 from the pinned MuJoCo Menagerie. The robot starts in a crouch and receives its MJCF home-keyframe actuator controls. Self-contact is disabled consistently; floor contacts remain enabled. Each environment is independent. The standing-height, orientation and drift gates apply to every environment; gallery videos show environment 0.", *notes]
    units = {count_unit(scenario) for scenario in metadata["suite"]["scenarios"]}
    table_count_unit = next(iter(units)).capitalize() if len(units) == 1 else "Scene size"
    notes_html = (
        "<h2>Run notes</h2>" + "".join(f"<p>{html.escape(note)}</p>" for note in notes)
        if notes
        else ""
    )
    page = (
        HTML
        .replace("__BASELINE__", html.escape("native MuJoCo" if baseline == "mujoco" else baseline))
        .replace("__COUNT_UNIT__", table_count_unit)
        .replace("__ROBOT_LICENSE__", robot_license)
        .replace("__TITLE__", html.escape(metadata["suite"]["name"]))
        .replace("__GPU_STATUS__", gpu_status)
        .replace("__MEMORY_STATUS__", memory_status)
        .replace(
            "__MEMORY_CURVES__",
            "".join(
                f'<a href="{f}"><img src="{f}" alt="Memory curve, lower is better"></a>'
                for f in memory_files
            )
            or '<p class="muted">No complete memory measurements in this run.</p>',
        )
        .replace("__RUN_NOTES__", notes_html)
        .replace(
            "__HARDWARE__",
            html.escape(metadata["hardware"]["cpu"] or metadata["hardware"]["uname"]),
        )
        .replace(
            "__DATE__",
            datetime.datetime.fromtimestamp(
                metadata["created_unix"], datetime.timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC"),
        )
        .replace(
            "__CURVES__",
            "".join(
                f'<a href="{f}"><img src="{f}" alt="Performance curve"></a>'
                for f in files
            ) or '<p class="muted">No cases have complete valid timing coverage.</p>',
        )
        .replace(
            "__GALLERY__",
            "".join(gallery_sections)
            or '<p class="muted">No successful captures in this run.</p>',
        )
        .replace("__DATA__", data)
    )
    (out / "index.html").write_text(brand_report(page, out))
    print(f"Report: {out / 'index.html'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
