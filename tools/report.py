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

from site_nav import THEME_HEAD, bar, default_platform

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, FuncFormatter, NullFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

COLORS = dict(
    rapier="#00b8a9",
    nexus="#ee9f34",
    genesis="#e86582",
    newton="#9467df",
    mujoco="#579df2",
    mjx="#b9c44a",
    isaac="#2f9e5a",
)


ROBOT_NOTES = {
    "robot_go1": ("Unitree Go1", "unitree-go1.txt", "robot_go1: one 13-link, 12-actuator Unitree Go1 quadruped from the pinned MuJoCo Menagerie, receiving its MJCF home-keyframe actuator controls."),
    "robot_t1": ("Booster T1", "booster-t1.txt", "robot_t1: one 24-link, 23-actuator Booster T1 humanoid from the pinned MuJoCo Menagerie, standing to its home keyframe. Its servos are stiffened from kp 75 to kp 300, kv 20 because no balance controller runs; Menagerie torque limits are kept."),
}


def count_unit(scenario):
    return "robots" if scenario.startswith("robot_") else "particles" if scenario.startswith("mpm_") else "bodies"


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
        rows[-1]["startup"] = startup_statistics(good, complete)
    return rows


def startup_statistics(runs, complete):
    """Setup + warmup seconds: time from building the engine to the first measured step.

    Engines compile at different points (scene build, first steps), so only the sum compares."""
    def seconds(r, key):
        v = r.get(key)
        return v if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 else None

    totals = [seconds(r, "setup_seconds") + seconds(r, "warmup_seconds") for r in runs
              if seconds(r, "setup_seconds") is not None and seconds(r, "warmup_seconds") is not None]
    stat = lambda vs, q: float(np.percentile(vs, q)) if vs else None
    return dict(
        eligible=complete and len(totals) == len(runs),
        median_s=stat(totals, 50), p25_s=stat(totals, 25), p75_s=stat(totals, 75),
        setup_median_s=stat([r["setup_seconds"] for r in runs if seconds(r, "setup_seconds") is not None], 50),
        warmup_median_s=stat([r["warmup_seconds"] for r in runs if seconds(r, "warmup_seconds") is not None], 50),
    )


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


def failure_tag(record):
    """Short reason a trial did not pass. A trailing ? marks a case skipped after a smaller one failed."""
    reasons = (record.get("validation") or {}).get("reasons") or []
    text = " ".join([record.get("outcome_label") or "", record.get("error") or "", *reasons]).lower()
    if "out of memory" in text or "memory limit" in text or "likely oom" in text:
        tag = "OOM"
    elif "timed out" in text or "likely timeout" in text:
        tag = "timeout"
    elif "nan" in re.findall(r"[a-z]+", text) or "non-finite" in text or "diverged" in text:
        tag = "NaN"
    else:
        return "invalid" if record["status"] == "invalid" else "skipped" if record["status"] == "skipped" else "error"
    return tag + "?" if record["status"] == "skipped" else tag


# Table column order; CPU backends come first, then GPU backends, each in this order.
ENGINE_ORDER = ["rapier", "mujoco", "nexus", "genesis", "isaac", "mjx", "newton"]


def timing_value(record):
    v = record.get("ms_per_step")
    return v if record["status"] == "passed" and isinstance(v, (int, float)) and math.isfinite(v) and v > 0 else None


def startup_value(record):
    """Setup + warmup seconds of a passing trial (see startup_statistics)."""
    parts = [record.get(key) for key in ("setup_seconds", "warmup_seconds")]
    if record["status"] != "passed" or not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in parts):
        return None
    return sum(parts)


def memory_value(name):
    """Peak MiB of one memory metric, from a complete replay of a passing trial."""
    def value(record):
        memory = record.get("memory") or {}
        metric = (memory.get("metrics") or {}).get(name) or {}
        if (record["status"] != "passed" or memory.get("status") != "complete"
                or metric.get("status") != "available" or not isinstance(metric.get("peak_bytes"), int)
                or metric["peak_bytes"] <= 0):
            return None
        return metric["peak_bytes"] / 2**20
    return value


def case_table(records, engines, value, throughput=False):
    """Median value per case and engine. Engines are grouped CPU then GPU; the lowest median of
    each group is marked best. A cell without values carries the most common reason."""
    batching = len({r["worlds"] for r in records}) > 1
    cases = sorted({(r["scenario"], r["size"], r["worlds"]) for r in records})
    backends = {e: next((r["backend"] for r in records if r["engine"] == e), None) for e in engines}
    engines = [e for e in engines if backends[e]]
    order = sorted(engines, key=lambda e: (backends[e] != "cpu", ENGINE_ORDER.index(e) if e in ENGINE_ORDER else len(ENGINE_ORDER), engines.index(e)))
    rows = []
    for scenario, size, worlds in cases:
        cells = {}
        for engine in order:
            runs = [r for r in records if (r["engine"], r["scenario"], r["size"], r["worlds"]) == (engine, scenario, size, worlds)]
            measured = [(r, value(r)) for r in runs if value(r) is not None]
            tags = [failure_tag(r) if r["status"] != "passed" else "not measured" for r in runs if value(r) is None]
            throughputs = [r["world_steps_per_second"] for r, _ in measured if isinstance(r.get("world_steps_per_second"), (int, float))]
            cells[engine] = dict(
                passed=len(measured),
                total=len(runs),
                median=float(np.median([v for _, v in measured])) if measured else None,
                world_steps_per_second=float(np.median(throughputs)) if throughput and batching and throughputs else None,
                # Measured failures take precedence over cases skipped because of them.
                failure=max(sorted(set(tags)), key=lambda t: (not t.endswith("?"), tags.count(t))) if tags else None,
            )
        for gpu in (False, True):
            group = [e for e in order if (backends[e] != "cpu") == gpu and cells[e]["median"] is not None]
            best = min(group, key=lambda e: cells[e]["median"]) if group else None
            for e in order:
                if (backends[e] != "cpu") == gpu:
                    cells[e]["best"] = e == best
        rows.append(dict(scenario=scenario, size=size, worlds=worlds, cells=cells))
    return dict(engines=order, backends={e: backends[e] for e in order}, batching=batching, rows=rows)


def case_table_html(table):
    batching, order, rows = table["batching"], table["engines"], table["rows"]
    if not rows or not order:
        return '<p class="muted">No trials yet.</p>'
    show_size = not batching or len({r["size"] for r in rows}) > 1
    gpu = [e for e in order if table["backends"][e] != "cpu"]
    split = order[len(order) - len(gpu)] if gpu and len(gpu) < len(order) else None

    def number(v):
        return f"{v:,.0f}" if v >= 1000 else f"{v:.4g}"

    def edge(e):
        return ' class="split"' if e == split else ""

    leading = 1 + show_size + batching
    groups = f'<tr><th colspan="{leading}"></th>'
    if len(gpu) < len(order):
        groups += f'<th colspan="{len(order) - len(gpu)}" class="group-head">CPU</th>'
    if gpu:
        groups += f'<th colspan="{len(gpu)}" class="group-head{" split" if split else ""}">GPU</th>'
    groups += "</tr>"
    head = "<tr><th>Scenario</th>" + (f"<th>{html.escape(count_unit(rows[0]['scenario']).capitalize())}</th>" if show_size else "") \
        + ("<th>Environments</th>" if batching else "")
    for e in order:
        head += f'<th{edge(e)}>{html.escape(e)} <small>{html.escape(table["backends"][e])}</small></th>'
    head += "</tr>"
    body = ""
    for index, row in enumerate(rows):
        span = sum(1 for r in rows if r["scenario"] == row["scenario"])
        first = index == 0 or rows[index - 1]["scenario"] != row["scenario"]
        cells = (f'<th rowspan="{span}" scope="rowgroup">{html.escape(row["scenario"].replace("_", " "))}</th>' if first else "")
        if show_size:
            cells += f"<td>{row['size']:,}</td>"
        if batching:
            cells += f"<td>{row['worlds']:,}</td>"
        for e in order:
            c = row["cells"][e]
            classes = ["split"] if e == split else []
            if c["median"] is None:
                text = html.escape(c["failure"]) if c["failure"] else "·"
                title = f"{c['total']} trial(s), none measured" if c["total"] else "not run"
                cells += f'<td class="{" ".join(classes + ["fail"])}" title="{title}">{text}</td>'
                continue
            text = number(c["median"])
            if c["world_steps_per_second"] is not None:
                text += f' <small>({number(c["world_steps_per_second"])}/s)</small>'
            if c["passed"] < c["total"]:
                text += f' <small class="fail">({c["passed"]}/{c["total"]}, {html.escape(c["failure"])})</small>'
            if c["best"]:
                classes.append("best")
            attr = f' class="{" ".join(classes)}"' if classes else ""
            cells += f"<td{attr}>{text}</td>"
        body += f'<tr class="{"group" if first else ""}">{cells}</tr>'
    return f'<div class="scroll"><table class="perf"><thead>{groups}{head}</thead><tbody>{body}</tbody></table></div>'


MEMORY_TABLES = [
    ("ram", "Peak RAM · resident set (MiB)", False),
    ("gpu", "Peak GPU / VRAM · sampled (MiB)", True),
    ("footprint", "Peak physical footprint · macOS (MiB)", False),
]


def memory_tables(records, engines):
    """One table per memory metric that has any measurement; the GPU table lists GPU engines only."""
    tables = {}
    for name, _, gpu_only in MEMORY_TABLES:
        table = case_table(records, engines, memory_value(name))
        if gpu_only:
            keep = [e for e in table["engines"] if table["backends"][e] != "cpu"]
            table["engines"], table["backends"] = keep, {e: table["backends"][e] for e in keep}
        if any(row["cells"][e]["median"] is not None for row in table["rows"] for e in table["engines"]):
            tables[name] = table
    return tables


def memory_tables_html(tables):
    return "".join(f"<h3>{html.escape(title)}</h3>{case_table_html(tables[name])}"
                   for name, title, _ in MEMORY_TABLES if name in tables) \
        or '<p class="muted">No complete memory measurements in this run.</p>'


def curves(rows, out, records=(), memory_metric=None, startup=False):
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
    if startup:
        rows = [
            dict(r, eligible=r["startup"]["eligible"], gpu_eligible=False,
                 **{f"{key}_ms": r["startup"][f"{key}_s"] for key in ["median", "p25", "p75"]})
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
        for throughput in [False, True] if (batching or scenario.startswith("mpm_")) and not memory_metric and not startup else [False]:
            fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
            plotted = []

            def plot_values(rs, prefix, style, label, color):
                def value(r, key):
                    ms = r[prefix + key + "_ms"]
                    return (r["size"] if scenario.startswith("mpm_") else r["worlds"]) * 1000 / ms if throughput else ms

                x = [r[axis] for r in rs]
                ax.plot(
                    x, [value(r, "median") for r in rs], style, label=label, color=color
                )
                lo, hi = ("p75", "p25") if throughput else ("p25", "p75")
                plotted.extend(value(r, key) for r in rs for key in ("median", lo, hi))
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
                        f"{engine} / {rs[0]['backend']}" + ("" if startup else " · sync"),
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
                else "Setup + warmup (s)"
                if startup
                else ("Particle-steps / second" if scenario.startswith("mpm_") else "Environment-steps / second")
                if throughput
                else "Time / step across all environments (ms)",
                title=(memory_names[memory_metric] + "\n" if memory_metric else "")
                + ("Time to first measured step · setup + warmup\n" if startup else "")
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
            if startup:
                suffix += "-startup"
            filename = (
                f"curves/{scenario}-{suffix}{'-throughput' if throughput else ''}"
            )
            fig.savefig(out / (filename + "-log.svg"))
            fig.savefig(out / (filename + "-log.png"), dpi=150)
            # The same figure on linear axes, starting at zero. The right-hand band again
            # keeps the preview and legend clear of the final data points.
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.xaxis.set_major_locator(AutoLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 100 else f"{v:g}"))
            ax.set_xlim(0, requested[-1] * 1.4)
            ax.set_ylim(0, max(v for v in plotted if v is not None and math.isfinite(v)) * 1.08)
            fig.savefig(out / (filename + "-linear.svg"))
            fig.savefig(out / (filename + "-linear.png"), dpi=150)
            plt.close(fig)
            files.append(filename)
    return files


def chart_images(files, alt, empty):
    """Chart thumbnails for both axis scales; the page's axis toggle swaps them. Clicking enlarges."""
    return "".join(
        f'<a class="zoom" href="{f}-linear.svg"><img src="{f}-linear.svg" data-linear="{f}-linear.svg" '
        f'data-log="{f}-log.svg" alt="{html.escape(alt)}"></a>'
        for f in files
    ) or f'<p class="muted">{html.escape(empty)}</p>'


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
RENDER_VERSION = 6


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
    if scene.get("mpm"):
        # Frame the whole tank, not just the material in it.
        for b in scene["bodies"]:
            if not b["dynamic"]:
                lo = np.minimum(lo, np.array(b["position"]) - b["size"])
                hi = np.maximum(hi, np.array(b["position"]) + b["size"])
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
    "physmark-logo-light.svg",
    "physmark-icon.svg",
    "favicon-32.png",
    "apple-touch-icon.png",
)
BRAND_LOGO = ('<div class="brand"><img class="logo-dark" src="assets/physmark-logo.svg" width="288" height="65" alt="PhysMark.">'
              '<img class="logo-light" src="assets/physmark-logo-light.svg" width="288" height="65" alt="PhysMark."></div>')
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
__THEME_HEAD__
<style>*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:38px}header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:28px}.brand{font-size:30px;font-weight:750;letter-spacing:-1px}small,.muted{color:var(--muted)}h1{font-size:42px;letter-spacing:-1.5px;margin:30px 0 8px}h2{font-size:22px;margin-top:38px}p{max-width:970px}a{color:var(--accent)}.stats,.gallery,.charts{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px}.number{font-size:30px;font-weight:700}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:11px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:500}select,input{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:10px;font:inherit;margin:0 8px 12px 0}.scroll{overflow:auto}.passed{color:var(--good)}.error,.invalid{color:var(--bad)}.charts{grid-template-columns:repeat(auto-fit,minmax(400px,1fr))}.charts img{width:100%;background:white;border-radius:8px}.gallery{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.gallery video,.gallery img{width:100%;border-radius:8px}.gallery .card{padding:12px}.scene-section{margin-top:28px;padding-top:20px;border-top:1px solid var(--line)}.scene-section h3{font-size:20px;margin:0 0 4px}.scene-section>p{margin:0 0 16px}.tag{display:inline-block;border:1px solid var(--line);padding:3px 9px;border-radius:16px;font-size:12px}details{margin-top:24px}details.trials>summary{font-size:22px;font-weight:700;cursor:pointer;margin:38px 0 14px}details.trials>summary small{font-size:15px;font-weight:400;color:var(--muted)}.zoom{display:block;cursor:zoom-in}.no-evidence th:last-child,.no-evidence td:last-child{display:none}.perf th,.perf td{padding:8px 10px}.perf td.best{font-weight:750;color:var(--accent)}.perf .fail{color:var(--bad)}.perf tr.group>*{border-top:2px solid var(--line)}.perf .split{border-left:2px solid var(--muted)}.perf .group-head{text-align:center;color:var(--text);font-weight:600;letter-spacing:1px;border-bottom:none}h3{font-size:17px;margin:26px 0 8px}.perf th[scope=rowgroup]{color:var(--text);font-weight:600;vertical-align:top}#lightbox{position:fixed;inset:0;z-index:50;display:flex;align-items:center;justify-content:center;padding:3vmin;background:#000c;cursor:zoom-out}#lightbox[hidden]{display:none}#lightbox img{max-width:100%;max-height:100%;height:auto;border-radius:8px;background:white;box-shadow:0 10px 40px #000a}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted);font-size:12px}footer{margin-top:50px;border-top:1px solid var(--line);padding-top:22px;color:var(--muted)}@media(max-width:650px){main{padding:20px}h1{font-size:32px}.charts{grid-template-columns:1fr}header{display:block}}</style>
<main><header>__BRAND_LOGO__<div class="muted">Physics, measured. <span class="tag">LOCAL BENCHMARK</span></div></header>__NAV__<h1>__TITLE__</h1><p class="muted">__HARDWARE__ · __DATE__<br>Synchronized physics stepping. Rendering and compilation excluded. Sleeping disabled.</p><div class="stats" id="stats"></div>__RUN_NOTES__<h2>Performance curves</h2><p class="muted">Solid lines: synchronized wall time. Dashed lines: device timestamps from an independent warmed replay. Time covers a simulation step across all environments. Throughput is environments × 1,000 / milliseconds per batch step. Shading shows the timing 25th–75th percentiles across repeats and seeds, inverted for throughput. Each metric requires complete coverage. GPU scopes differ (pass sums, activity sums or activity unions); each trial's method and scope appear in the trials table. Profiling may alter scheduling; instrumented GPU times can exceed unprofiled synchronized times. The axes toggle at the top switches every chart between linear and logarithmic scales; click a chart to enlarge it.</p><div class="charts">__CURVES__</div><h2>Performance table</h2><p class="muted">Median synchronized milliseconds per step across all environments (lower is better)__THROUGHPUT_NOTE__. Bold marks the fastest CPU engine and the fastest GPU engine of each case. (k/n, reason) marks cases where only k of n trials passed. Failure reasons: invalid (validation gate), OOM, timeout, NaN (diverged) and error; a trailing ? marks a case skipped after a smaller one failed that way. · means not run. These are baseline sanity checks, not matched-accuracy results; CPU and GPU backends are labelled.</p>__PERF_TABLE__<h2>RAM and GPU memory · lower is better</h2><p class="muted">Separate memory replays preserve the timing measurements. RAM is the operating system’s process resident-set high-water mark; macOS physical footprint is also reported. GPU / VRAM is the sampled peak of native per-process device counters (10 ms target polling plus synchronized phase boundaries). Peaks cover setup, warmup, reset and rollout, ending before final readback and rendering. They include runtime, compilation caches and allocator pools, not just body data. Sampling can miss brief GPU peaks. Apple Silicon uses unified memory: GPU allocations and process RAM overlap and must not be added. Metal reports allocated resources; NVIDIA NVML reports per-process device memory, including context overhead. Different counter scopes are not interchangeable. Missing measurements are never inferred from timings or old results.</p><div class="charts">__MEMORY_CURVES__</div><p class="muted">Median peak per case and engine across repeats, in MiB (lower is better). Bold marks the lowest CPU engine and the lowest GPU engine of each case. (k/n, reason) marks cases where only k of n trials have a measurement; reasons are those of the performance table, or not measured when the trial passed but its memory replay has no reading. · means not run. Each trial's counter, scope and reason appear in the trials table and its memory result.</p>__MEMORY_TABLES__<h2>Time to first measured step · lower is better</h2><p class="muted">Setup (building the scene in a fresh process, including engine imports and any compilation done at build time) plus warmup (the untimed trajectory before the measured one, where lazily compiling engines compile their kernels). Engines compile at different points, so only the sum compares across engines. Process launch and interpreter start are excluded. Compilation caches persist between trials, so cold and warm starts mix; shading shows the 25th–75th percentiles across repeats.</p><div class="charts">__STARTUP_CURVES__</div><p class="muted">Median setup + warmup seconds per case and engine across repeats (lower is better). Bold marks the quickest CPU engine and the quickest GPU engine of each case. (k/n, reason) and · read as in the performance table.</p>__STARTUP_TABLE__<details class="trials"><summary>Trials <small>__TRIAL_COUNT__ · every repeat with its timings, memory and evidence</small></summary><select id="engine"><option value="">All engines</option></select><select id="scenario"><option value="">All scenarios</option></select><select id="status"><option value="">All outcomes</option><option>passed</option><option>invalid</option><option>error</option><option>skipped</option></select><div class="scroll"><table><thead><tr><th>Engine / backend</th><th>Scenario</th><th>__COUNT_UNIT__ × worlds</th><th>Seed / repeat</th><th>Sync ms / step</th><th>GPU ms / step</th><th>GPU timing</th><th>World steps / s</th><th>Setup s</th><th>Warmup s</th><th>Peak RAM RSS MiB</th><th>Peak footprint MiB</th><th>GPU peak MiB (sampled)</th><th>Memory</th><th>Outcome</th><th>Evidence</th></tr></thead><tbody id="trials"></tbody></table></div></details><h2>Simulation gallery</h2><p class="muted">Independent untimed replay, rendered by the same renderer and camera rules. Batched cases show world 0. Videos play at simulated time.</p><div id="simulation-gallery">__GALLERY__</div><details><summary>Methodology, source pins and machine metadata</summary><pre id="metadata"></pre></details><footer><span class="evidence" data-href="results.json"><a href="results.json">Raw trials</a> · </span><span class="evidence" data-href="summary.json"><a href="summary.json">Summary statistics</a> · </span><span class="evidence" data-href="metadata.json"><a href="metadata.json">Provenance</a> · </span>PhysMark protocol v1__ROBOT_LICENSE__<br>No single score establishes physical accuracy. Inspect solver settings, validation errors, coverage and simulations alongside timing.</footer></main><div id="lightbox" hidden><img alt=""></div>
<script>const DATA=__DATA__;const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>x==null?'—':Number(x).toLocaleString(undefined,{maximumSignificantDigits:4});
document.querySelector('#stats').innerHTML=[['Engines',DATA.metadata.suite.engines.length],['Trials',DATA.records.length],['Passed',DATA.records.filter(r=>r.status==='passed').length],['Cases',new Set(DATA.rows.map(r=>[r.scenario,r.size,r.worlds].join('/'))).size]].map(([label,n])=>`<div class="card"><small>${label}</small><div class="number">${n}</div></div>`).join('');
for(const field of ['engine','scenario']){const el=document.querySelector('#'+field);for(const v of [...new Set(DATA.records.map(r=>r[field]))].sort()){const opt=document.createElement('option');opt.textContent=v;el.append(opt)}}
let OMITTED=new Set();
function memoryCell(r,key){const m=r.memory?.metrics?.[key];return m?.status==='available'?fmt(m.peak_bytes/1048576):'—'}
function draw(){document.querySelector('#trials').innerHTML=DATA.records.filter(r=>['engine','scenario','status'].every(k=>!document.querySelector('#'+k).value||r[k]===document.querySelector('#'+k).value)).map(r=>`<tr><td>${esc(r.engine)} <small>${esc(r.backend)}</small></td><td>${esc(r.scenario)}</td><td>${r.size} × ${r.worlds}</td><td>${r.seed} / ${r.repeat}</td><td>${fmt(r.ms_per_step)}</td><td>${fmt(r.gpu_ms_per_step)}</td><td title="${esc(r.gpu_timing?.reason||[r.gpu_timing?.method,r.gpu_timing?.scope].filter(Boolean).join(' / '))}">${esc(r.gpu_timing?.status||'not_collected')}</td><td>${fmt(r.world_steps_per_second)}</td><td>${fmt(r.setup_seconds)}</td><td>${fmt(r.warmup_seconds)}</td><td>${memoryCell(r,'ram')}</td><td>${memoryCell(r,'footprint')}</td><td>${memoryCell(r,'gpu')}</td><td title="${esc(r.memory?.reason||'Independent replay; see memory result for counter coverage')}">${esc(r.memory?.status||'not_collected')}</td><td class="${r.status}" title="${esc(r.error||r.validation?.reasons.join('; ')||'')}">${esc(r.outcome_label||r.status)}</td><td>${[['job.json','job',true],['adapter.log','log',true],['result.json','result',['passed','invalid'].includes(r.status)],['gpu.log','GPU log',['available','unavailable','error'].includes(r.gpu_timing?.status)],['gpu-result.json','GPU result',['available','unavailable'].includes(r.gpu_timing?.status)],['memory.log','memory log',['complete','error'].includes(r.memory?.status)],['memory-result.json','memory result',r.memory?.status==='complete']].filter(([f,,shown])=>shown&&!OMITTED.has(r.directory+'/'+f)).map(([f,label])=>`<a href="${r.directory}/${f}">${label}</a>`).join(' · ')||'—'}</td></tr>`).join('')}
for(const el of document.querySelectorAll('details.trials select'))el.addEventListener('change',draw);draw();
// A published site may leave out evidence files ("all" for a report-only site, see physmark site); hide their links.
fetch('omitted.json').then(r=>r.ok?r.json():[]).then(list=>{OMITTED=list==='all'?{has:()=>true}:new Set(list);if(list==='all')document.querySelector('details.trials table').classList.add('no-evidence');for(const e of document.querySelectorAll('.evidence'))if(OMITTED.has(e.dataset.href))e.remove();draw()},()=>{});
const box=document.querySelector('#lightbox'),boxImg=box.querySelector('img');boxImg.addEventListener('load',()=>{const w=boxImg.naturalWidth,h=boxImg.naturalHeight;if(w&&h)boxImg.style.width=w*Math.min(innerWidth*.94/w,innerHeight*.94/h)+'px'});document.addEventListener('click',e=>{const z=e.target.closest('.zoom');if(!z||e.button||e.metaKey||e.ctrlKey||e.shiftKey||e.altKey)return;e.preventDefault();const img=z.tagName==='IMG'?z:z.querySelector('img');boxImg.src=z.tagName==='A'?z.href:img.currentSrc||img.src;boxImg.alt=img.alt;box.hidden=false});box.addEventListener('click',()=>{box.hidden=true;boxImg.removeAttribute('src')});document.addEventListener('keydown',e=>{if(e.key==='Escape')box.click()});</script></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    out = args.directory.resolve()
    records = json.loads((out / "results.json").read_text())
    metadata = json.loads((out / "metadata.json").read_text())
    # Engines present in the records but missing from an older metadata file come last.
    for engine in dict.fromkeys(r["engine"] for r in records):
        if engine not in metadata["suite"]["engines"]:
            metadata["suite"]["engines"].append(engine)
    rows = summarize(records, metadata["suite"])
    performance = case_table(records, metadata["suite"]["engines"], timing_value, throughput=True)
    memory = memory_tables(records, metadata["suite"]["engines"])
    startup = case_table(records, metadata["suite"]["engines"], startup_value)
    (out / "summary.json").write_text(
        json.dumps(dict(cases=rows, performance_table=performance, memory_tables=memory, startup_table=startup), indent=2)
    )
    # Curves are regenerated below; drop those of an older layout or of removed cases.
    shutil.rmtree(out / "curves", ignore_errors=True)
    (out / "curves").mkdir()
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
                else f'<img class="zoom" loading="lazy" src="{base}/simulation.png" alt="{html.escape(title)}">'
            )
            scene_key = (r["scenario"], r["size"], r["worlds"], r["seed"])
            gallery.setdefault(scene_key, []).append(
                f'<div class="card">{media}<small>{html.escape(title)}</small><br><span class="evidence" data-href="{base}/capture.json"><a href="{base}/capture.json">Trajectory</a> · </span><a href="{base}/simulation.png">PNG</a></div>'
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
    startup_files = curves(rows, out, records, startup=True)
    import datetime

    data = json.dumps(
        dict(records=records, metadata=metadata, rows=rows),
        allow_nan=False,
    ).replace("<", "\\u003c")
    notes_path = out / "run-notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else []
    remeasured = [r for r in records if r.get("memory_remeasured")]
    for reason in sorted({r["memory_remeasured"]["reason"] for r in remeasured}):
        ids = [r["id"] for r in remeasured if r["memory_remeasured"]["reason"] == reason]
        notes.append(f"Memory of {len(ids)} trial(s) was remeasured after their timing run, in a separate replay: {reason} "
                     f"Their timings are unchanged; the earlier measurement is kept in results.json ({', '.join(ids)}).")
    if any(s.startswith("mpm_") for s in metadata["suite"]["scenarios"]):
        notes = ["Single-environment MPM. Size is material-point count. Every engine receives the same particle positions, masses, grid spacing, gravity, substeps and simulated horizon. Constitutive laws, grids and collision discretizations differ; this is not an accuracy-equated comparison.",
                 "MPM throughput plots show particle-steps per second. Snapshots display material points as spheres; their rigid orientation fields are protocol padding. Newton MPM uses SolverImplicitMPM, independently of its rigid-body MuJoCo Warp backend.", *notes]
    robot_license = ""
    for scenario, (title, license_file, note) in ROBOT_NOTES.items():
        if scenario in metadata["suite"]["scenarios"]:
            license_source = Path(__file__).resolve().parents[1] / "docs/licenses" / license_file
            shutil.copy2(license_source, out / f"LICENSE.{license_file}")
            robot_license += f' · <a href="LICENSE.{license_file}">{title} model license</a>'
            notes = [note, *notes]
    if robot_license:
        notes = ["Robots stand on a ground plane: every environment starts crouched with its feet touching the floor and stands under its position servos. Self-contact is disabled consistently; floor contacts remain enabled. Each environment is independent. The standing-height, orientation and drift gates apply to every environment; gallery videos show environment 0.", *notes]
    units = {count_unit(scenario) for scenario in metadata["suite"]["scenarios"]}
    table_count_unit = next(iter(units)).capitalize() if len(units) == 1 else "Scene size"
    notes_html = (
        "<h2>Run notes</h2>" + "".join(f"<p>{html.escape(note)}</p>" for note in notes)
        if notes
        else ""
    )
    # Inside a round, the report sits one level below its platform directory (see site_nav).
    in_round = (out.parent / "round-plan.json").exists()
    platform_label = ""
    if in_round:
        platform_file = out.parent / "platform.json"
        platform_label = (json.loads(platform_file.read_text()) if platform_file.exists()
                          else dict(zip(("id", "label"), default_platform(json.loads((out.parent / "round-plan.json").read_text())))))["label"]
    page = (
        HTML
        .replace("__THEME_HEAD__", THEME_HEAD)
        .replace("__NAV__", bar(1 if in_round else None, platform_label, axes=True, home=in_round))
        .replace("__PERF_TABLE__", case_table_html(performance))
        .replace("__MEMORY_TABLES__", memory_tables_html(memory))
        .replace("__STARTUP_TABLE__", case_table_html(startup))
        .replace("__THROUGHPUT_NOTE__", "; environment-steps per second in parentheses" if performance["batching"] else "")
        .replace("__TRIAL_COUNT__", f"{len(records):,} trial{'s' if len(records) != 1 else ''}")
        .replace("__COUNT_UNIT__", table_count_unit)
        .replace("__ROBOT_LICENSE__", robot_license)
        .replace("__TITLE__", html.escape(metadata["suite"]["name"]))
        .replace(
            "__MEMORY_CURVES__",
            chart_images(memory_files, "Memory curve, lower is better", "No complete memory measurements in this run."),
        )
        .replace(
            "__STARTUP_CURVES__",
            chart_images(startup_files, "Setup and warmup curve, lower is better", "No case has complete passing coverage yet."),
        )
        .replace("__RUN_NOTES__", notes_html)
        .replace(
            "__HARDWARE__",
            html.escape(metadata["hardware"]["cpu"] or platform_label or metadata["hardware"]["uname"]),
        )
        .replace(
            "__DATE__",
            datetime.datetime.fromtimestamp(
                metadata["created_unix"], datetime.timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC"),
        )
        .replace(
            "__CURVES__",
            chart_images(files, "Performance curve", "No cases have complete valid timing coverage."),
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
