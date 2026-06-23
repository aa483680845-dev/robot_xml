#!/usr/bin/env python3
"""
csv_to_mjcf.py — Convert a SolidWorks-URDF-exporter style CSV into a MuJoCo MJCF XML.

Usage:
    python csv_to_mjcf.py [input.csv] [-o output.xml] [--meshdir ../meshes/]

If arguments are omitted, defaults are:
    input  : urdf/A02L-MP4-HT_defeature.csv  (relative to this script)
    output : <robot_name>.xml                (next to the script)
    meshdir: ../meshes/                       (relative to the output XML)
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from xml.dom import minidom
from xml.etree import ElementTree as ET


# ---------- helpers ---------------------------------------------------------

def to_float(s, default=0.0):
    s = (s or "").strip()
    if s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def fmt(v):
    """Format a float for XML — strip trailing zeros, keep precision."""
    if isinstance(v, str):
        return v
    if v == 0:
        return "0"
    return f"{v:.10g}"


def vec3(x, y, z):
    return f"{fmt(x)} {fmt(y)} {fmt(z)}"


def strip_pkg_path(path):
    """Drop ROS package:// prefix and return only the file basename."""
    p = (path or "").strip()
    if not p:
        return ""
    if "package://" in p:
        p = p.split("/")[-1]
    return os.path.basename(p)


def mesh_asset_name(filename):
    """Mesh asset name = filename without extension."""
    base = strip_pkg_path(filename)
    name, _ = os.path.splitext(base)
    return name


# ---------- CSV parsing -----------------------------------------------------

def parse_csv(path):
    """Return a list of row dicts with all numeric fields converted."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [[c.strip() for c in r] for r in reader if any(c.strip() for c in r)]

    header = rows[0]
    data_rows = rows[1:]

    links = []
    for raw in data_rows:
        # pad short rows
        if len(raw) < len(header):
            raw = raw + [""] * (len(header) - len(raw))
        d = dict(zip(header, raw))

        link = {
            "name":         d.get("Link Name", "").strip(),
            "com":          (to_float(d.get("Center of Mass X")),
                             to_float(d.get("Center of Mass Y")),
                             to_float(d.get("Center of Mass Z"))),
            "com_rpy":      (to_float(d.get("Center of Mass Roll")),
                             to_float(d.get("Center of Mass Pitch")),
                             to_float(d.get("Center of Mass Yaw"))),
            "mass":         to_float(d.get("Mass")),
            "ixx":          to_float(d.get("Moment Ixx")),
            "ixy":          to_float(d.get("Moment Ixy")),
            "ixz":          to_float(d.get("Moment Ixz")),
            "iyy":          to_float(d.get("Moment Iyy")),
            "iyz":          to_float(d.get("Moment Iyz")),
            "izz":          to_float(d.get("Moment Izz")),
            "visual_xyz":   (to_float(d.get("Visual X")),
                             to_float(d.get("Visual Y")),
                             to_float(d.get("Visual Z"))),
            "visual_rpy":   (to_float(d.get("Visual Roll")),
                             to_float(d.get("Visual Pitch")),
                             to_float(d.get("Visual Yaw"))),
            "mesh":         d.get("Mesh Filename", "").strip(),
            "color":        (to_float(d.get("Color Red"), 1.0),
                             to_float(d.get("Color Green"), 1.0),
                             to_float(d.get("Color Blue"), 1.0),
                             to_float(d.get("Color Alpha"), 1.0)),
            "col_xyz":      (to_float(d.get("Collision X")),
                             to_float(d.get("Collision Y")),
                             to_float(d.get("Collision Z"))),
            "col_rpy":      (to_float(d.get("Collision Roll")),
                             to_float(d.get("Collision Pitch")),
                             to_float(d.get("Collision Yaw"))),
            "col_mesh":     d.get("Collision Mesh Filename", "").strip(),
            "joint_name":   d.get("Joint Name", "").strip(),
            "joint_type":   d.get("Joint Type", "").strip(),
            "joint_xyz":    (to_float(d.get("Joint Origin X")),
                             to_float(d.get("Joint Origin Y")),
                             to_float(d.get("Joint Origin Z"))),
            "joint_rpy":    (to_float(d.get("Joint Origin Roll")),
                             to_float(d.get("Joint Origin Pitch")),
                             to_float(d.get("Joint Origin Yaw"))),
            "parent":       d.get("Parent", "").strip(),
            "axis":         (to_float(d.get("Joint Axis X")),
                             to_float(d.get("Joint Axis Y")),
                             to_float(d.get("Joint Axis Z"))),
            "effort":       to_float(d.get("Limit Effort")),
            "velocity":     to_float(d.get("Limit Velocity")),
            "lower":        to_float(d.get("Limit Lower")),
            "upper":        to_float(d.get("Limit Upper")),
            "damping":      to_float(d.get("Dynamics Damping")),
            "friction":     to_float(d.get("Dynamics Friction")),
        }
        if link["name"]:
            links.append(link)
    return links


# ---------- MJCF generation -------------------------------------------------

URDF_TO_MJC_JOINT = {
    "revolute":   "hinge",
    "continuous": "hinge",
    "prismatic":  "slide",
    "fixed":      None,
}


def add_inertial(body_el, link):
    if link["mass"] <= 0:
        return
    inertial = ET.SubElement(body_el, "inertial")
    inertial.set("pos", vec3(*link["com"]))
    # MuJoCo uses quat or euler; rpy = (roll, pitch, yaw) maps to euler attribute.
    rpy = link["com_rpy"]
    if any(abs(v) > 1e-12 for v in rpy):
        inertial.set("euler", vec3(*rpy))
    inertial.set("mass", fmt(link["mass"]))
    # fullinertia order: ixx iyy izz ixy ixz iyz
    inertial.set("fullinertia",
                 f"{fmt(link['ixx'])} {fmt(link['iyy'])} {fmt(link['izz'])} "
                 f"{fmt(link['ixy'])} {fmt(link['ixz'])} {fmt(link['iyz'])}")


def add_geom(body_el, link, kind):
    """kind = 'visual' or 'collision'"""
    if kind == "visual":
        mesh = link["mesh"]
        xyz, rpy = link["visual_xyz"], link["visual_rpy"]
    else:
        mesh = link["col_mesh"]
        xyz, rpy = link["col_xyz"], link["col_rpy"]
    if not mesh:
        return
    g = ET.SubElement(body_el, "geom")
    g.set("name", f"{link['name']}_{kind}")
    g.set("type", "mesh")
    g.set("mesh", mesh_asset_name(mesh))
    if any(abs(v) > 1e-12 for v in xyz):
        g.set("pos", vec3(*xyz))
    if any(abs(v) > 1e-12 for v in rpy):
        g.set("euler", vec3(*rpy))
    if kind == "visual":
        # visual-only geom: no contact, no dynamics, just rendering
        g.set("contype", "0")
        g.set("conaffinity", "0")
        g.set("group", "1")
        g.set("rgba", f"{fmt(link['color'][0])} {fmt(link['color'][1])} "
                      f"{fmt(link['color'][2])} {fmt(link['color'][3])}")
    else:
        g.set("group", "3")
        g.set("rgba", "0.5 0.6 0.7 0.3")


def add_joint(body_el, link):
    jt = link["joint_type"].lower()
    # No joint info (root link) or fixed joint → emit no <joint>
    if not jt or jt == "fixed" or not link["joint_name"]:
        return
    jtype = URDF_TO_MJC_JOINT.get(jt, "hinge")
    if jtype is None:
        return
    j = ET.SubElement(body_el, "joint")
    j.set("name", link["joint_name"])
    j.set("type", jtype)
    j.set("pos", "0 0 0")
    j.set("axis", vec3(*link["axis"]))
    # range only meaningful for non-continuous
    if link["joint_type"].lower() != "continuous" and (link["lower"] or link["upper"]):
        j.set("range", f"{fmt(link['lower'])} {fmt(link['upper'])}")
        j.set("limited", "true")
    if link["damping"]:
        j.set("damping", fmt(link["damping"]))
    if link["friction"]:
        j.set("frictionloss", fmt(link["friction"]))


def build_body(parent_el, link, children_of):
    body = ET.SubElement(parent_el, "body")
    body.set("name", link["name"])
    # Joint origin (in URDF) becomes the body pos/euler relative to parent.
    if link["parent"]:
        body.set("pos", vec3(*link["joint_xyz"]))
        if any(abs(v) > 1e-12 for v in link["joint_rpy"]):
            body.set("euler", vec3(*link["joint_rpy"]))
    else:
        body.set("pos", "0 0 0")

    add_inertial(body, link)
    add_joint(body, link)
    add_geom(body, link, "visual")
    add_geom(body, link, "collision")

    for child in children_of.get(link["name"], []):
        build_body(body, child, children_of)
    return body


def build_mjcf(links, robot_name, meshdir):
    by_name = {l["name"]: l for l in links}
    children_of = defaultdict(list)
    roots = []
    for l in links:
        if l["parent"] and l["parent"] in by_name:
            children_of[l["parent"]].append(l)
        else:
            roots.append(l)

    mujoco = ET.Element("mujoco", {"model": robot_name})

    ET.SubElement(mujoco, "compiler", {
        "angle":     "radian",
        "meshdir":   meshdir,
        "autolimits": "true",
        # URDF rpy is extrinsic XYZ (R = Rz(yaw) * Ry(pitch) * Rx(roll)).
        # MuJoCo's default eulerseq is "xyz" (intrinsic), which silently
        # disagrees whenever two or more of (r, p, y) are nonzero.
        # Uppercase "XYZ" selects extrinsic XYZ to match URDF exactly.
        "eulerseq":  "XYZ",
    })
    ET.SubElement(mujoco, "option", {"integrator": "implicitfast"})

    defaults = ET.SubElement(mujoco, "default")
    ET.SubElement(defaults, "joint", {"damping": "0.1", "armature": "0.01"})
    ET.SubElement(defaults, "geom",  {"contype": "1", "conaffinity": "1"})

    # asset: one <mesh> per unique mesh file referenced by visual or collision
    asset = ET.SubElement(mujoco, "asset")
    seen = set()
    for l in links:
        for mesh in (l["mesh"], l["col_mesh"]):
            if mesh and mesh not in seen:
                seen.add(mesh)
                ET.SubElement(asset, "mesh", {
                    "name": mesh_asset_name(mesh),
                    "file": strip_pkg_path(mesh),
                })

    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(worldbody, "light", {
        "pos": "0 0 3", "dir": "0 0 -1", "diffuse": "0.8 0.8 0.8"
    })
    for root in roots:
        build_body(worldbody, root, children_of)

    # actuators for every non-fixed joint
    actuator = ET.SubElement(mujoco, "actuator")
    for l in links:
        if URDF_TO_MJC_JOINT.get(l["joint_type"].lower()) is None:
            continue
        if not l["joint_name"]:
            continue
        attrs = {
            "name":   f"{l['joint_name']}_motor",
            "joint":  l["joint_name"],
            "gear":   "1",
        }
        if l["effort"]:
            attrs["ctrlrange"] = f"{fmt(-l['effort'])} {fmt(l['effort'])}"
            attrs["ctrllimited"] = "true"
        ET.SubElement(actuator, "motor", attrs)

    return mujoco


def pretty_xml(elem):
    rough = ET.tostring(elem, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ")


# ---------- main ------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "urdf", "A02L-MP4-HT_defeature.csv")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default=default_csv,
                   help="Input CSV path")
    p.add_argument("-o", "--output", default=None,
                   help="Output MJCF XML path")
    p.add_argument("--meshdir", default="meshes/",
                   help="Mesh directory referenced by the MJCF compiler "
                        "(relative to the output XML). Default: meshes/")
    p.add_argument("--name", default=None,
                   help="Robot/model name (defaults to CSV file stem)")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"CSV not found: {args.csv}")

    robot_name = args.name or os.path.splitext(os.path.basename(args.csv))[0]
    out_path = args.output or os.path.join(script_dir, f"{robot_name}.xml")

    links = parse_csv(args.csv)
    if not links:
        sys.exit("No links parsed from CSV.")

    root = build_mjcf(links, robot_name, args.meshdir)
    xml_str = pretty_xml(root)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"Parsed {len(links)} links from: {args.csv}")
    print(f"MJCF written to: {out_path}")
    print(f"Mesh dir (in MJCF compiler): {args.meshdir}")


if __name__ == "__main__":
    main()
