#!/usr/bin/env python3
"""Resolve the pinned Menagerie Go1 into a portable collision-only MJCF scene."""
import argparse
import json
import subprocess
import tomllib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np


def numbers(values):
    return " ".join(format(float(x), ".17g") for x in values)


def make_scene(root, size=1, seed=42, dt=1 / 240):
    if size != 1:
        raise ValueError("robot_go1 uses size=1 robot per environment; scale with worlds")
    root = Path(root)
    pin = tomllib.loads((root / "engines.lock").read_text())["dependencies"]["menagerie"]["commit"]
    checkout = root / ".physmark/sources/menagerie"
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != pin:
        raise ValueError("Menagerie checkout does not match engines.lock; fetch the pinned commit")
    source = checkout / "unitree_go1/go1.xml"
    xml = ET.parse(source).getroot()
    # Rendering uses the original collision primitives. Remove visual-only assets;
    # preserve link masses, inertias, joints and actuator definitions from Menagerie.
    for parent in xml.iter():
        for child in list(parent):
            if child.tag in ("asset", "light", "camera", "sensor", "site") or (
                child.tag == "geom" and (child.get("mesh") or child.get("class") == "visual")
            ):
                parent.remove(child)
        parent.attrib.pop("material", None)
    compiler = xml.find("compiler")
    compiler.attrib.pop("meshdir", None)
    option = xml.find("option")
    option.set("timestep", str(dt))
    option.set("gravity", "0 0 -9.81")
    option.set("iterations", "50")
    ET.SubElement(option, "flag", sleep="disable")
    world = xml.find("worldbody")
    # Direct masks express the complete policy without pair-exclusion synthesis.
    # All robot geoms are type 1, affinity 2; the ground is type 2, affinity 1.
    for geom in xml.iter("geom"):
        geom.set("contype", "1")
        geom.set("conaffinity", "2")
    contact = xml.find("contact")
    if contact is not None:
        xml.remove(contact)
    ET.SubElement(world, "geom", name="physmark_floor", type="plane", size="2 2 .1",
        friction=".8 .02 .01", condim="6", contype="2", conaffinity="1")
    model = mj.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    home = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    qpos = model.key_qpos[home].copy()
    control = model.key_ctrl[home].copy()
    assert model.nq == 19 and model.nu == 12
    qpos[7:] = [0, 1.2, -2.4] * 4
    qpos[2] = 0
    data = mj.MjData(model)
    data.qpos[:] = qpos
    mj.mj_forward(model, data)
    lowest = float("inf")
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        s = model.geom_size[g]
        rot = data.geom_xmat[g].reshape(3, 3)
        typ = model.geom_type[g]
        if typ == mj.mjtGeom.mjGEOM_BOX:
            extent = np.abs(rot[2]) @ s
        elif typ == mj.mjtGeom.mjGEOM_CAPSULE:
            extent = s[0] + abs(rot[2, 2]) * s[1]
        elif typ == mj.mjtGeom.mjGEOM_CYLINDER:
            extent = s[0] * np.linalg.norm(rot[2, :2]) + abs(rot[2, 2]) * s[1]
        else:
            extent = s[0]
        lowest = min(lowest, data.geom_xpos[g, 2] - extent)
    qpos[2] = -lowest + .003
    ET.SubElement(xml.find("keyframe"), "key", name="physmark_crouch", qpos=numbers(qpos), ctrl=numbers(control))
    standalone = ET.tostring(xml, encoding="unicode")
    model = mj.MjModel.from_xml_string(standalone)
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, model.nkey - 1)
    mj.mj_forward(model, data)
    bodies = [dict(id=0, dynamic=False, shape="box", size=[2,2,.05], position=[0,0,-.05], rotation=[0,0,0,1], velocity=[0,0,0], mass=0, inertia=[0,0,0], friction=.8)]
    parts = []
    body_names = []
    for i in range(1, model.nbody):
        body_names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i))
        bodies.append(dict(id=i, dynamic=True, shape="box", size=[.01]*3,
            position=data.xpos[i].tolist(), rotation=data.xquat[i][[1,2,3,0]].tolist(), velocity=[0,0,0],
            mass=float(model.body_mass[i]), inertia=model.body_inertia[i].tolist(), friction=.8))
        link_parts = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != i:
                continue
            shape = {2:"sphere",3:"capsule",5:"cylinder",6:"box"}[int(model.geom_type[g])]
            s = model.geom_size[g].tolist()
            if shape in ("capsule", "cylinder"):
                s = [s[0], 0, s[1]]
            link_parts.append(dict(shape=shape, size=s, position=model.geom_pos[g].tolist(), rotation=model.geom_quat[g][[1,2,3,0]].tolist()))
        parts.append(link_parts)
    return dict(schema_version=1, scenario="robot_go1", size=len(body_names), seed=seed,
        gravity=[0,0,-9.81], dt=dt, bodies=bodies, hinges=[], robot=dict(
            model="Unitree Go1", source_commit=pin, source_path="unitree_go1/go1.xml", xml=standalone,
            initial_keyframe="physmark_crouch", target_keyframe="home", qpos=qpos.tolist(), ctrl=control.tolist(),
            joint_names=[mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(j)) for j in model.actuator_trnid[:,0]],
            body_names=body_names, render_parts=parts, initial_height=float(qpos[2]), minimum_standing_height=.22,
            self_contacts=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dt", type=float, default=1/240)
    a = p.parse_args()
    print(json.dumps(make_scene(a.root, a.size, a.seed, a.dt)))
