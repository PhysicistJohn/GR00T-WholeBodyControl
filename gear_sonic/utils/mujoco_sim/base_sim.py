"""MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

import os
import json
import math
import pathlib
from pathlib import Path
import pickle
import re
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot

try:
    # Optional: lets the console live-tune SENSOR_NOISE_* without a restart.
    # Not a hard dependency -- handsim_bus lives in the sibling unitree-g1-handsim
    # repo (importable in the deployed container via PYTHONPATH; may be absent
    # in standalone/test invocations of this module, which is fine -- settings
    # just stay at their static config values).
    import handsim_bus
except ImportError:
    handsim_bus = None

try:
    # Optional, same PYTHONPATH story as handsim_bus above: lets a world's
    # scene.json point "mjcf_room" at a standalone, fully-authored MJCF room
    # (e.g. keysight_lab_mujoco/keysight_lab.xml) instead of describing it as
    # primitive-box landmarks. Absent in standalone invocations -- mjcf_room
    # scenes just fall back to their (still-supported) landmark boxes.
    import handsim_world
except ImportError:
    handsim_world = None

_bootstrap_import_error = None
try:
    from g1_mcp.bootstrap import (
        BootstrapPhase,
        JointBootstrapCommand,
        SimBootstrapConfig,
        SimBootstrapController,
    )
    from robot_registry import DEFAULT_29
except ImportError as exc:
    _bootstrap_import_error = exc
    BootstrapPhase = None
    JointBootstrapCommand = None
    SimBootstrapConfig = None
    SimBootstrapController = None
    DEFAULT_29 = None

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _handsim_scene_anchor_path() -> Path:
    return Path(os.environ.get("HANDSIM_SCENE_ANCHOR_PATH", "/tmp/handsim_scene_anchor.json"))


def _safe_xml_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_") or "item"


def _xml_num(value: float) -> str:
    return f"{float(value):.5g}"


def _xml_attr(value: Path | str) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _is_robot_hand_contact(body_name: str, geom_name: str) -> bool:
    """Recognize hand links without matching the `handsim_*` namespace."""
    label = f"{body_name} {geom_name}".lower()
    return bool(
        re.search(r"(^|[_\s])hand($|[_\s])", label)
        or any(token in label for token in ("finger", "thumb", "wrist"))
    )


def _default_handsim_world_config_path() -> Path:
    world = os.environ.get("HANDSIM_WORLD", "keysight_lab").strip() or "keysight_lab"
    repo = (os.environ.get("HANDSIM_REPO", "").strip()
            or os.environ.get("DROID_REPO", "").strip()
            or str(Path.home() / "Github" / "unitree-g1-handsim"))
    return Path(repo).expanduser() / "assets" / "render" / "worlds" / world / "scene.json"


def _load_handsim_collision_scene_config() -> tuple[Path, dict] | None:
    config_path = os.environ.get("HANDSIM_COLLISION_SCENE_CONFIG", "").strip()
    path = Path(config_path).expanduser() if config_path else _default_handsim_world_config_path()
    if not path.exists():
        return None
    try:
        with open(path) as f:
            scene = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[handsim-collision] scene config unavailable: {path} ({exc})", flush=True)
        return None
    if not isinstance(scene, dict):
        print(f"[handsim-collision] scene config ignored: {path} is not an object", flush=True)
        return None
    return path, scene


def _handsim_repo_root(config_path: Path) -> Path:
    repo = os.environ.get("HANDSIM_REPO", "").strip() or os.environ.get("DROID_REPO", "").strip()
    if repo:
        return Path(repo).expanduser()
    # repo/assets/render/worlds/outdoor_park/scene.json -> repo (5 segments
    # up from the file: outdoor_park, worlds, render, assets, then repo itself)
    try:
        return config_path.parents[4]
    except IndexError:
        return Path.home() / "Github" / "unitree-g1-handsim"


def _initial_root_xy(xml_path: Path) -> tuple[float, float]:
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        if model.nq >= 2:
            return float(data.qpos[0]), float(data.qpos[1])
    except Exception as exc:
        print(f"[handsim-collision] using origin anchor; base scene pre-load failed: {exc}", flush=True)
    return 0.0, 0.0


def _write_handsim_scene_anchor(anchor_xy: tuple[float, float]) -> None:
    path = _handsim_scene_anchor_path()
    payload = {"root_pos": [float(anchor_xy[0]), float(anchor_xy[1]), 0.0]}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[handsim-collision] could not write scene anchor {path}: {exc}", flush=True)


def _iter_handsim_scene_items(scene: dict):
    for section in ("landmarks", "decor"):
        items = scene.get(section, {})
        if isinstance(items, dict):
            yield from items.items()


def _read_abo_extents(repo: Path, model_id: str, scale: float) -> tuple[float, float, float] | None:
    meta_path = repo / "external" / "abo_render_assets" / model_id / "metadata.json"
    try:
        with open(meta_path) as f:
            extents = json.load(f).get("extents_xyz_mujoco")
        if isinstance(extents, list) and len(extents) >= 3:
            return tuple(float(v) * scale for v in extents[:3])
    except (OSError, ValueError, TypeError):
        pass
    return None


def _box_geom(name: str, x: float, y: float, z: float, sx: float, sy: float, sz: float, yaw: float = 0.0) -> str:
    sx = max(float(sx), 0.015)
    sy = max(float(sy), 0.015)
    sz = max(float(sz), 0.015)
    return (
        f'      <geom name="{name}" type="box" '
        f'pos="{_xml_num(x)} {_xml_num(y)} {_xml_num(z)}" '
        f'size="{_xml_num(sx)} {_xml_num(sy)} {_xml_num(sz)}" '
        f'euler="0 0 {_xml_num(yaw)}" contype="1" conaffinity="1" '
        'friction="1.0 0.01 0.001" rgba="0.85 0.22 0.16 0.16"/>\n'
    )


def _local_xy(x: float, y: float, yaw: float, lx: float, ly: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * lx - s * ly, y + s * lx + c * ly


def _box_geom_local(
    name: str,
    x: float,
    y: float,
    yaw: float,
    lx: float,
    ly: float,
    z: float,
    sx: float,
    sy: float,
    sz: float,
) -> str:
    wx, wy = _local_xy(x, y, yaw, lx, ly)
    return _box_geom(name, wx, wy, z, sx, sy, sz, yaw)


def _shadow_half_extents(cfg: dict, default: tuple[float, float]) -> tuple[float, float]:
    shadow = cfg.get("shadow", default)
    if not isinstance(shadow, list | tuple) or len(shadow) < 2:
        return default
    return max(float(shadow[0]), 0.04), max(float(shadow[1]), 0.04)


def _is_panel_like(raw_name: str, cfg: dict) -> bool:
    label = f"{raw_name} {cfg.get('kind', '')} {cfg.get('model', '')}".lower()
    return any(word in label for word in ("monitor", "display", "tv", "wall"))


def _is_tabletop_prop(cfg: dict) -> bool:
    return float(cfg.get("z", 0.0)) >= 0.35


def _scene_object_proxy_geom(raw_name: str, cfg: dict, x: float, y: float, yaw: float, extents: tuple[float, float, float] | None = None) -> str:
    key = "handsim_col_" + _safe_xml_name(str(raw_name)) + "_proxy"
    scale = float(cfg.get("mesh_scale", cfg.get("scale", 1.0)))
    base_z = float(cfg.get("z", 0.0))
    sx, sy = _shadow_half_extents(cfg, (0.35, 0.25))
    sx = max(sx * 0.92, 0.08)
    sy = max(sy * 0.92, 0.08)

    if _is_panel_like(str(raw_name), cfg):
        sx = max(sx, 0.55 * scale)
        sy = max(sy, 0.12)
        sz = max(0.36 * scale, 0.45)
        center_z = base_z if base_z >= 0.35 else max(sz, 1.05 * scale)
    elif _is_tabletop_prop(cfg):
        sz = max(0.05 * scale, 0.04)
        center_z = base_z + sz
    elif extents:
        sx = max(sx, extents[0] * 0.45)
        sy = max(sy, extents[1] * 0.45)
        sz = max(extents[2] * 0.48, 0.10)
        center_z = base_z + sz
    else:
        sz = max(0.12 * scale, 0.08)
        if sx >= 0.45 or sy >= 0.45 or scale >= 2.0:
            sz = max(sz, min(0.36 * scale, 1.45))
        center_z = base_z + sz

    return _box_geom(key, x, y, center_z, sx, sy, sz, yaw)


def _capsule_geom(name: str, x: float, y: float, z: float, radius: float, half_height: float) -> str:
    return (
        f'      <geom name="{name}" type="capsule" '
        f'pos="{_xml_num(x)} {_xml_num(y)} {_xml_num(z)}" '
        f'size="{_xml_num(max(radius, 0.015))} {_xml_num(max(half_height, 0.03))}" '
        'contype="1" conaffinity="1" friction="1.0 0.01 0.001" '
        'rgba="0.85 0.22 0.16 0.16"/>\n'
    )


def _mesh_collision_asset(mesh_name: str, obj_path: Path, scale: float) -> str:
    return (
        f'    <mesh name="{mesh_name}" file="{_xml_attr(obj_path)}" '
        f'scale="{_xml_num(scale)} {_xml_num(scale)} {_xml_num(scale)}"/>\n'
    )


def _mesh_collision_body(name: str, mesh_name: str, x: float, y: float, z: float, yaw: float) -> str:
    return (
        f'    <body name="{name}_body" pos="{_xml_num(x)} {_xml_num(y)} {_xml_num(z)}" '
        f'euler="0 0 {_xml_num(yaw)}">\n'
        f'      <geom name="{name}" type="mesh" mesh="{mesh_name}" '
        'contype="1" conaffinity="1" friction="1.0 0.01 0.001" '
        'rgba="0.85 0.22 0.16 0.18"/>\n'
        '    </body>\n'
    )


def _scene_mesh_path(repo: Path, kind: str, model_name: str) -> Path | None:
    if not model_name:
        return None
    if kind == "abo":
        return repo / "external" / "abo_render_assets" / model_name / "model.obj"
    if kind == "scanned":
        return repo / "external" / "mujoco_scanned_objects" / "models" / model_name / "model.obj"
    return None


def _strip_joints(elem: ET.Element) -> None:
    """Drop <joint>/<freejoint> elements from an injected collision fragment.
    These decorative DOFs (e.g. bench_full's unplug-the-cable sliders) are
    never actuated here, but their mere presence shifts every later joint's
    qpos address in the physics model -- if one lands ahead of the robot's
    own free joint, index-0-based free-base assumptions elsewhere can break."""
    for child in list(elem):
        if child.tag in ("joint", "freejoint"):
            elem.remove(child)
        else:
            _strip_joints(child)


def _prepare_handsim_mjcf_collision(elem: ET.Element, prefix: str, mesh_map: dict[str, str] | None = None) -> None:
    name = elem.get("name")
    if name:
        elem.set("name", f"{prefix}_{name}")
    if elem.tag == "geom":
        elem.attrib.pop("material", None)
        elem.set("contype", "1")
        elem.set("conaffinity", "1")
        elem.set("friction", "1.0 0.01 0.001")
        elem.set("rgba", "0.85 0.22 0.16 0.16")
        mesh_ref = elem.get("mesh")
        if mesh_ref and mesh_map and mesh_ref in mesh_map:
            elem.set("mesh", mesh_map[mesh_ref])
    for child in list(elem):
        _prepare_handsim_mjcf_collision(child, prefix, mesh_map)


def _handsim_mjcf_collision_body(
    repo: Path, raw_name: str, cfg: dict, x: float, y: float, yaw: float
) -> tuple[list[str], str] | None:
    if cfg.get("kind") != "benchtop_setup":
        return None
    source = repo / "assets" / "render" / "benchtop" / "keysight_bench_setup.xml"
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as exc:
        print(f"[handsim-collision] benchtop setup missing: {source} ({exc})", flush=True)
        return None
    worldbody = root.find("worldbody")
    if worldbody is None:
        return None

    prefix = "handsim_col_" + _safe_xml_name(str(raw_name))
    # The source MJCF's own geoms can reference real <mesh> assets (e.g. the
    # RF amplifier CAD in keysight_bench_full) -- those definitions live in
    # <asset>, not <worldbody>, so cloning worldbody alone drops them and
    # MuJoCo fails to compile with "mesh '...' not found". Carry them along,
    # renamed to this instance's prefix and pointed at absolute file paths
    # (this collision scene is written to a different directory than the
    # source, so the source's own relative meshdir won't resolve).
    mesh_assets = []
    mesh_map: dict[str, str] = {}
    asset_elem = root.find("asset")
    if asset_elem is not None:
        for mesh_elem in asset_elem.findall("mesh"):
            old_name = mesh_elem.get("name")
            file_attr = mesh_elem.get("file")
            if not old_name or not file_attr:
                continue
            new_name = f"{prefix}_{old_name}"
            mesh_map[old_name] = new_name
            mesh_scale = mesh_elem.get("scale", "1 1 1")
            mesh_assets.append(
                f'    <mesh name="{new_name}" '
                f'file="{_xml_attr(source.parent / "uploads" / file_attr)}" '
                f'scale="{mesh_scale}"/>\n'
            )

    children = []
    for child in list(worldbody):
        if child.tag == "light":
            continue
        if child.tag == "geom" and child.get("type") == "plane":
            continue
        cloned = ET.fromstring(ET.tostring(child, encoding="unicode"))
        _strip_joints(cloned)
        _prepare_handsim_mjcf_collision(cloned, prefix, mesh_map)
        children.append(ET.tostring(cloned, encoding="unicode"))
    if not children:
        return None
    z = float(cfg.get("z", 0.0))
    body = "\n      ".join(children)
    return (
        mesh_assets,
        f'    <body name="{prefix}" pos="{_xml_num(x)} {_xml_num(y)} {_xml_num(z)}" '
        f'euler="0 0 {_xml_num(yaw)}">\n'
        f"      {body}\n"
        "    </body>\n",
    )


def _handsim_mjcf_room_body(scene: dict, repo: Path, anchor_xy: tuple[float, float]) -> tuple[list[str], str] | None:
    if handsim_world is None:
        return None
    room_path = handsim_world.room_path(scene, repo)
    if room_path is None:
        return None
    asset_xml, body_xml = handsim_world.room_fragment(room_path, "handsim_room", strip_joints=True)
    if not body_xml:
        return None
    room_offset = scene.get("mjcf_room_offset", [0.0, 0.0])
    ox = float(room_offset[0]) if isinstance(room_offset, list | tuple) and len(room_offset) > 0 else 0.0
    oy = float(room_offset[1]) if isinstance(room_offset, list | tuple) and len(room_offset) > 1 else 0.0
    yaw = float(scene.get("mjcf_room_yaw", 0.0))
    x, y = anchor_xy[0] + ox, anchor_xy[1] + oy
    assets = [asset_xml] if asset_xml else []
    body = (
        f'    <body name="handsim_room" pos="{_xml_num(x)} {_xml_num(y)} 0" euler="0 0 {_xml_num(yaw)}">\n'
        f"      {body_xml}\n"
        "    </body>\n"
    )
    return assets, body


def _handsim_collision_assets_and_geoms(scene: dict, repo: Path, anchor_xy: tuple[float, float]) -> tuple[list[str], list[str]]:
    assets = []
    geoms = []
    ax, ay = anchor_xy

    room = _handsim_mjcf_room_body(scene, repo, anchor_xy)
    if room is not None:
        room_assets, room_body = room
        assets.extend(room_assets)
        geoms.append(room_body)

    for raw_name, cfg in _iter_handsim_scene_items(scene):
        if not isinstance(cfg, dict):
            continue
        kind = str(cfg.get("kind", "")).strip()
        if kind in {"", "empty", "path", "floor_zone"}:
            continue

        offset = cfg.get("offset", [0.0, 0.0])
        if not isinstance(offset, list | tuple) or len(offset) < 2:
            offset = [0.0, 0.0]
        x = ax + float(offset[0])
        y = ay + float(offset[1])
        yaw = float(cfg.get("yaw", 0.0))
        scale = float(cfg.get("mesh_scale", cfg.get("scale", 1.0)))
        key = "handsim_col_" + _safe_xml_name(str(raw_name))

        if kind == "benchtop_setup":
            result = _handsim_mjcf_collision_body(repo, str(raw_name), cfg, x, y, yaw)
            if result:
                bench_assets, body = result
                assets.extend(bench_assets)
                geoms.append(body)
        elif kind == "wall":
            size = cfg.get("size", [1.0, 0.06])
            height = float(cfg.get("height", 1.5))
            sx = float(size[0]) if isinstance(size, list | tuple) and len(size) > 0 else 1.0
            sy = float(size[1]) if isinstance(size, list | tuple) and len(size) > 1 else 0.06
            geoms.append(_box_geom(key, x, y, height, sx, max(sy, 0.12), height, yaw))
        elif kind == "door_frame":
            s = scale
            geoms.append(_box_geom_local(key + "_left", x, y, yaw, -0.78 * s, 0.0, 1.35 * s, 0.08 * s, 0.10 * s, 1.35 * s))
            geoms.append(_box_geom_local(key + "_right", x, y, yaw, 0.78 * s, 0.0, 1.35 * s, 0.08 * s, 0.10 * s, 1.35 * s))
            geoms.append(_box_geom_local(key + "_top", x, y, yaw, 0.0, 0.0, 2.62 * s, 0.86 * s, 0.10 * s, 0.08 * s))
            geoms.append(_box_geom_local(key + "_threshold", x, y, yaw, 0.0, -0.09 * s, 0.06 * s, 0.95 * s, 0.04 * s, 0.04 * s))
        elif kind == "booth":
            s = scale
            geoms.append(_box_geom_local(key + "_back", x, y, yaw, 0.0, 0.92 * s, 1.10 * s, 1.55 * s, 0.09 * s, 1.10 * s))
            geoms.append(_box_geom_local(key + "_side", x, y, yaw, -1.55 * s, 0.0, 1.10 * s, 0.09 * s, 0.90 * s, 1.10 * s))
            geoms.append(_box_geom_local(key + "_front_rail", x, y, yaw, 0.0, 0.855 * s, 1.76 * s, 1.20 * s, 0.04 * s, 0.16 * s))
        elif kind == "instrument_rack":
            s = scale
            geoms.append(_box_geom(key, x, y, 0.84 * s, 0.62 * s, 0.34 * s, 0.78 * s, yaw))
        elif kind == "fence":
            length = float(cfg.get("length", 4.0)) * scale
            geoms.append(_box_geom(key, x, y, 0.45 * scale, length * 0.5, 0.055 * scale, 0.45 * scale, yaw))
        elif kind == "lamp":
            s = scale
            geoms.append(_capsule_geom(key + "_post", x, y, 0.75 * s, 0.035 * s, 0.75 * s))
        elif kind == "abo":
            model_name = str(cfg.get("model", "")).strip()
            obj_path = _scene_mesh_path(repo, kind, model_name)
            ext = _read_abo_extents(repo, str(cfg.get("model", "")), scale)
            if obj_path is not None and obj_path.exists():
                mesh_name = "handsim_mesh_" + _safe_xml_name(str(raw_name))
                assets.append(_mesh_collision_asset(mesh_name, obj_path, scale))
                geoms.append(_mesh_collision_body(key, mesh_name, x, y, float(cfg.get("z", 0.0)), yaw))
                geoms.append(_scene_object_proxy_geom(str(raw_name), cfg, x, y, yaw, ext))
                continue
            shadow = cfg.get("shadow", [0.45, 0.35])
            if not isinstance(shadow, list | tuple) or len(shadow) < 2:
                shadow = [0.45, 0.35]
            if ext:
                sx = max(float(shadow[0]) * 0.65, ext[0] * 0.5)
                sy = max(float(shadow[1]) * 0.65, ext[1] * 0.5)
                sz = max(ext[2] * 0.5, 0.04)
            else:
                sx, sy, sz = float(shadow[0]) * 0.75, float(shadow[1]) * 0.75, 0.35 * scale
            base_z = float(cfg.get("z", 0.0))
            center_z = base_z + sz if base_z < 0.35 else base_z
            geoms.append(_box_geom(key, x, y, center_z, sx, sy, sz, yaw))
        elif kind == "scanned":
            model_name = str(cfg.get("model", "")).strip()
            obj_path = _scene_mesh_path(repo, kind, model_name)
            if obj_path is not None and obj_path.exists():
                mesh_name = "handsim_mesh_" + _safe_xml_name(str(raw_name))
                assets.append(_mesh_collision_asset(mesh_name, obj_path, scale))
                geoms.append(_mesh_collision_body(key, mesh_name, x, y, float(cfg.get("z", 0.0)), yaw))
                geoms.append(_scene_object_proxy_geom(str(raw_name), cfg, x, y, yaw))
                continue
            shadow = cfg.get("shadow", [0.35, 0.25])
            if not isinstance(shadow, list | tuple) or len(shadow) < 2:
                shadow = [0.35, 0.25]
            base_z = float(cfg.get("z", 0.0))
            sx = max(float(shadow[0]) * 0.85, 0.04)
            sy = max(float(shadow[1]) * 0.85, 0.04)
            sz = 0.08 * scale if base_z >= 0.35 else max(0.12 * scale, 0.05)
            center_z = base_z + sz
            geoms.append(_box_geom(key, x, y, center_z, sx, sy, sz, yaw))
        else:
            shadow = cfg.get("shadow", [0.4, 0.3])
            if isinstance(shadow, list | tuple) and len(shadow) >= 2:
                sx, sy = float(shadow[0]), float(shadow[1])
            else:
                sx, sy = 0.35 * scale, 0.25 * scale
            geoms.append(_box_geom(key, x, y, 0.35 * scale, sx, sy, 0.35 * scale, yaw))

    return assets, geoms


def _handsim_dynamic_object_bodies(scene: dict, anchor_xy: tuple[float, float]) -> list[str]:
    """Build separately owned free bodies after the robot joint hierarchy.

    Room fragments intentionally have their decorative joints stripped.  A
    manipulation object is different: physics must own its free joint, but it
    must be appended after every robot/hand joint so the controller's existing
    qpos layout remains unchanged.
    """
    objects = scene.get("manipulation_objects", {})
    if not isinstance(objects, dict):
        return []
    out = []
    for raw_name, cfg in objects.items():
        if not isinstance(cfg, dict):
            continue
        position = cfg.get("position", [0.0, 0.0, 0.5])
        quat = cfg.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        if not isinstance(position, list | tuple) or len(position) < 3:
            continue
        if not isinstance(quat, list | tuple) or len(quat) < 4:
            quat = [1.0, 0.0, 0.0, 0.0]
        name = _safe_xml_name(str(raw_name))
        x = anchor_xy[0] + float(position[0])
        y = anchor_xy[1] + float(position[1])
        z = float(position[2])
        geoms = cfg.get("geoms", [])
        if not isinstance(geoms, list) or not geoms:
            continue
        geom_xml = []
        for index, geom in enumerate(geoms):
            if not isinstance(geom, dict):
                continue
            shape = str(geom.get("shape", "sphere"))
            if shape not in {"sphere", "cylinder", "box"}:
                continue
            local = geom.get("position", [0.0, 0.0, 0.0])
            size = geom.get("size", [0.05])
            if not isinstance(local, list | tuple) or len(local) < 3:
                local = [0.0, 0.0, 0.0]
            if not isinstance(size, list | tuple) or not size:
                size = [0.05]
            expected = 3 if shape == "box" else (2 if shape == "cylinder" else 1)
            size_values = [max(0.002, float(v)) for v in size[:expected]]
            if len(size_values) != expected:
                continue
            mass = max(0.001, float(geom.get("mass", 0.1)))
            friction = geom.get("friction", [1.0, 0.01, 0.003])
            if not isinstance(friction, list | tuple) or len(friction) < 3:
                friction = [1.0, 0.01, 0.003]
            condim = max(1, min(6, int(geom.get("condim", 4))))
            material = _safe_xml_name(str(geom.get("material", "mat_ball")))
            geom_name = _safe_xml_name(str(geom.get("name", index)))
            geom_xml.append(
                f'      <geom name="handsim_object_{name}_{geom_name}" type="{shape}" '
                f'pos="{" ".join(_xml_num(v) for v in local[:3])}" '
                f'material="handsim_room_{material}" '
                f'size="{" ".join(_xml_num(v) for v in size_values)}" '
                f'mass="{_xml_num(mass)}" '
                f'friction="{" ".join(_xml_num(v) for v in friction[:3])}" '
                f'condim="{condim}"/>\n'
            )
        if not geom_xml:
            continue
        out.append((
            f'    <body name="handsim_object_{name}" '
            f'pos="{_xml_num(x)} {_xml_num(y)} {_xml_num(z)}" '
            f'quat="{" ".join(_xml_num(v) for v in quat[:4])}">\n'
            f'      <freejoint name="handsim_object_{name}_free"/>\n'
        ) + "".join(geom_xml) + '    </body>\n')
    return out


def _inject_handsim_collision_scene(xml_path: Path) -> Path:
    loaded = _load_handsim_collision_scene_config()
    if loaded is None:
        return xml_path

    config_path, scene = loaded
    repo = _handsim_repo_root(config_path)
    anchor_xy = _initial_root_xy(xml_path)
    _write_handsim_scene_anchor(anchor_xy)
    assets, geoms = _handsim_collision_assets_and_geoms(scene, repo, anchor_xy)
    dynamic_objects = _handsim_dynamic_object_bodies(scene, anchor_xy)
    if not geoms and not dynamic_objects:
        print(f"[handsim-collision] no colliders generated from {config_path}", flush=True)
        return xml_path

    with open(xml_path) as f:
        xml = f.read()
    if scene.get("mjcf_room"):
        # The authored room supplies its own bounded floor. Keeping the base
        # robot scene's infinite floor duplicates support contacts and lets
        # range sensors observe fictitious ground beyond the room envelope.
        xml = re.sub(
            r'<geom\s+name="floor"[^>]*/>\s*', "", xml,
            count=1, flags=re.S)
    if assets:
        asset_block = "\n" + "".join(assets)
        if "</asset>" in xml:
            xml = xml.replace("</asset>", asset_block + "  </asset>", 1)
        elif "<worldbody>" in xml:
            xml = xml.replace("<worldbody>", "<asset>\n" + "".join(assets) + "  </asset>\n\n  <worldbody>", 1)
    block = (
        "\n    <body name=\"handsim_collision_scene\" pos=\"0 0 0\">\n"
        + "".join(geoms)
        + "    </body>\n"
        + "".join(dynamic_objects)
    )
    if "</worldbody>" not in xml:
        print(f"[handsim-collision] scene has no worldbody close tag: {xml_path}", flush=True)
        return xml_path
    xml = xml.replace("</worldbody>", block + "  </worldbody>", 1)
    generated_path = xml_path.with_name(f".{xml_path.stem}_handsim_collision.xml")
    with open(generated_path, "w") as f:
        f.write(xml)
    print(
        f"[handsim-collision] injected {len(geoms)} static colliders "
        f"and {len(dynamic_objects)} dynamic manipulation objects "
        f"({len(assets)} mesh assets) from {config_path} "
        f"at anchor ({anchor_xy[0]:.2f}, {anchor_xy[1]:.2f})",
        flush=True,
    )
    return generated_path


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)

        # Optional sensor-noise layer (default off): the sim is otherwise
        # ground-truth-perfect on IMU/encoders/camera every step, which is
        # unrealistic and (if this ever feeds training/eval) hides sim2real
        # brittleness. No real noise-magnitude spec exists in this repo, so
        # these are conservative placeholders, not measured values.
        self.sensor_noise_enable = bool(self.config.get("SENSOR_NOISE_ENABLE", False))
        self._noise_rng = np.random.default_rng(self.config.get("SENSOR_NOISE_SEED"))
        self._joint_pos_noise_std = float(self.config.get("SENSOR_NOISE_JOINT_POS_STD", 0.001))
        self._joint_vel_noise_std = float(self.config.get("SENSOR_NOISE_JOINT_VEL_STD", 0.01))
        self._imu_quat_noise_std = float(self.config.get("SENSOR_NOISE_IMU_QUAT_STD", 0.001))
        self._imu_quat_bias_std = float(self.config.get("SENSOR_NOISE_IMU_QUAT_BIAS_STD", 0.002))
        self._imu_vel_noise_std = float(self.config.get("SENSOR_NOISE_IMU_VEL_STD", 0.01))
        self._camera_noise_std = float(self.config.get("SENSOR_NOISE_CAMERA_STD", 2.0))
        self._imu_quat_bias_rotvec = np.zeros(3)
        if self.sensor_noise_enable:
            self._resample_imu_bias()
        self._settings_poll_interval = 0.5  # seconds; avoid hammering redis at physics rate
        self._last_settings_poll = 0.0
        self._poll_live_settings(force=True)
        self.camera_configs = camera_configs

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen

        self.init_scene()
        self.bootstrap_controller = None
        self._bootstrap_phase = None
        if self.config.get("ENABLE_BOOTSTRAP_CONTROLLER", False):
            if SimBootstrapController is None or DEFAULT_29 is None:
                raise RuntimeError(
                    "ENABLE_BOOTSTRAP_CONTROLLER requires unitree-g1-handsim/src "
                    f"and scripts on PYTHONPATH; import failed: {_bootstrap_import_error!r}"
                ) from _bootstrap_import_error
            if len(DEFAULT_29) != self.num_body_dof:
                raise RuntimeError(
                    f"bootstrap stand pose has {len(DEFAULT_29)} joints; "
                    f"simulator expects {self.num_body_dof}"
                )
            bootstrap_config = SimBootstrapConfig(
                damping_seconds=float(os.environ.get("DROID_BOOT_DAMPING_SECONDS", "0.5")),
                stand_seconds=float(os.environ.get("DROID_BOOT_STAND_SECONDS", "2.5")),
                external_stable_seconds=float(
                    os.environ.get("DROID_BOOT_EXTERNAL_STABLE_SECONDS", "0.1")
                ),
                handover_seconds=float(os.environ.get("DROID_BOOT_HANDOVER_SECONDS", "1.0")),
                hold_kp=float(os.environ.get("DROID_BOOT_HOLD_KP", "24.0")),
                hold_kd=float(os.environ.get("DROID_BOOT_HOLD_KD", "3.0")),
                damping_kd=float(os.environ.get("DROID_BOOT_DAMPING_KD", "4.0")),
            )
            self.bootstrap_controller = SimBootstrapController(
                np.asarray(DEFAULT_29, dtype=float), bootstrap_config
            )
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
            latency_ms=float(self.config.get("TRANSPORT_LATENCY_MS", 0.0)),
            jitter_ms=float(self.config.get("TRANSPORT_JITTER_MS", 0.0)),
            drop_prob=float(self.config.get("TRANSPORT_DROP_PROB", 0.0)),
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"]
        xml_path = _inject_handsim_collision_scene(xml_path)
        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band
        self.elastic_band = None  # [MULTI-48] referenced unconditionally below; init or headless band-off crashes
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand()
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body("pelvis").id

        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        configured_left_hand = self.config.get("LEFT_HAND_JOINT_NAMES")
        configured_right_hand = self.config.get("RIGHT_HAND_JOINT_NAMES")
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
            elif configured_left_hand is None and "left_hand" in name:
                self.left_hand_index.append(i)
            elif configured_right_hand is None and "right_hand" in name:
                self.right_hand_index.append(i)

        if configured_left_hand is not None:
            self.left_hand_index = [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in configured_left_hand
            ]
        if configured_right_hand is not None:
            self.right_hand_index = [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in configured_right_hand
            ]

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS
        assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
        assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(self.left_hand_index)
        self.right_hand_index = np.array(self.right_hand_index)

        def actuator_ids(joint_ids):
            result = []
            transmission_joints = np.asarray(self.mj_model.actuator_trnid)[:, 0]
            for joint_id in joint_ids:
                matches = np.flatnonzero(transmission_joints == int(joint_id))
                if len(matches) != 1:
                    joint_name = self.mj_model.joint(int(joint_id)).name
                    raise ValueError(
                        f"joint {joint_name!r} has {len(matches)} actuators; expected exactly one"
                    )
                result.append(int(matches[0]))
            return np.asarray(result, dtype=np.int64)

        self.body_actuator_index = actuator_ids(self.body_joint_index)
        self.left_hand_actuator_index = actuator_ids(self.left_hand_index)
        self.right_hand_actuator_index = actuator_ids(self.right_hand_index)
        self.torques = np.zeros(self.mj_model.nu, dtype=np.float64)
        if np.all(self.mj_model.actuator_ctrllimited):
            self.torque_limit = np.max(np.abs(self.mj_model.actuator_ctrlrange), axis=1)
        elif len(self.torque_limit) != self.mj_model.nu:
            raise ValueError(
                f"torque-limit vector has {len(self.torque_limit)} entries for {self.mj_model.nu} actuators"
            )

        # Dynamic manipulation objects are appended after every robot/hand
        # joint. Discover all addresses by name; never extend the controller's
        # positional qpos assumptions to these free bodies.
        self.manipulation_objects = {}
        loaded_scene = _load_handsim_collision_scene_config()
        object_cfgs = loaded_scene[1].get("manipulation_objects", {}) if loaded_scene else {}
        if isinstance(object_cfgs, dict):
            for raw_name, cfg in object_cfgs.items():
                if not isinstance(cfg, dict):
                    continue
                object_id = str(raw_name)
                safe_name = _safe_xml_name(object_id)
                body_name = f"handsim_object_{safe_name}"
                joint_name = f"handsim_object_{safe_name}_free"
                body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if body_id < 0 or joint_id < 0:
                    continue
                geom_ids = {
                    gid for gid in range(self.mj_model.ngeom)
                    if int(self.mj_model.geom_bodyid[gid]) == int(body_id)
                }
                self.manipulation_objects[object_id] = {
                    "body_id": int(body_id),
                    "joint_id": int(joint_id),
                    "qpos_adr": int(self.mj_model.jnt_qposadr[joint_id]),
                    "dof_adr": int(self.mj_model.jnt_dofadr[joint_id]),
                    "geom_ids": geom_ids,
                    "support_geom": str(cfg.get("support_geom", "")),
                    "reset_qpos": self.mj_data.qpos[
                        int(self.mj_model.jnt_qposadr[joint_id]):
                        int(self.mj_model.jnt_qposadr[joint_id]) + 7
                    ].copy(),
                }
        self._manipulation_last_command_id = None
        self._manipulation_next_poll = 0.0
        self._manipulation_next_publish = 0.0
        if self.manipulation_objects:
            print(f"[handsim-manipulation] objects={list(self.manipulation_objects)}", flush=True)

    def _poll_manipulation_command(self) -> None:
        if handsim_bus is None or not self.manipulation_objects:
            return
        now = time.time()
        if now < self._manipulation_next_poll:
            return
        self._manipulation_next_poll = now + 0.02
        try:
            command = handsim_bus.read_state(handsim_bus.MANIPULATION_COMMAND)
        except Exception:
            return
        if not isinstance(command, dict):
            return
        command_id = command.get("command_id")
        if not command_id or command_id == self._manipulation_last_command_id:
            return
        self._manipulation_last_command_id = command_id
        if command.get("action") != "reset":
            return
        requested_id = str(command.get("object_id", ""))
        reset_ids = list(self.manipulation_objects) if requested_id == "*" else [requested_id]
        reset_ids = [name for name in reset_ids if name in self.manipulation_objects]
        if not reset_ids:
            return
        for object_id in reset_ids:
            item = self.manipulation_objects[object_id]
            pose = command.get("pose_world") if len(reset_ids) == 1 else None
            qpos = item["reset_qpos"].copy()
            if isinstance(pose, list | tuple) and len(pose) >= 7:
                try:
                    candidate = np.asarray(pose[:7], dtype=np.float64)
                    if np.isfinite(candidate).all() and np.linalg.norm(candidate[3:7]) > 1e-6:
                        candidate[3:7] /= np.linalg.norm(candidate[3:7])
                        qpos = candidate
                except (TypeError, ValueError):
                    pass
            qadr, dadr = item["qpos_adr"], item["dof_adr"]
            self.mj_data.qpos[qadr:qadr + 7] = qpos
            self.mj_data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        print(f"[handsim-manipulation] reset {reset_ids} command={command_id}", flush=True)

    def _publish_manipulation_state(self) -> None:
        if handsim_bus is None or not self.manipulation_objects:
            return
        now = time.time()
        if now < self._manipulation_next_publish:
            return
        self._manipulation_next_publish = now + 1.0 / 30.0
        contacts_by_object = {name: [] for name in self.manipulation_objects}
        geom_to_object = {
            gid: name
            for name, item in self.manipulation_objects.items()
            for gid in item["geom_ids"]
        }
        for index in range(self.mj_data.ncon):
            contact = self.mj_data.contact[index]
            object_id = geom_to_object.get(int(contact.geom1))
            object_geom = int(contact.geom1)
            other_geom = int(contact.geom2)
            if object_id is None:
                object_id = geom_to_object.get(int(contact.geom2))
                object_geom = int(contact.geom2)
                other_geom = int(contact.geom1)
            if object_id is None:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, index, force)
            other_body_id = int(self.mj_model.geom_bodyid[other_geom])
            contacts_by_object[object_id].append({
                "object_geom": self.mj_model.geom(object_geom).name,
                "other_geom": self.mj_model.geom(other_geom).name,
                "other_body": self.mj_model.body(other_body_id).name,
                "position_world": [float(v) for v in contact.pos],
                "normal_force_n": float(abs(force[0])),
            })
        payload = {"ts": now, "objects": {}}
        for object_id, item in self.manipulation_objects.items():
            body_id = item["body_id"]
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY,
                body_id, velocity, 0)
            contacts = contacts_by_object[object_id]
            hand_contacts = [
                c for c in contacts
                if _is_robot_hand_contact(c["other_body"], c["other_geom"])
            ]
            support_name = item["support_geom"]
            payload["objects"][object_id] = {
                "position_world": [float(v) for v in self.mj_data.xpos[body_id]],
                "quat_wxyz": [float(v) for v in self.mj_data.xquat[body_id]],
                "linear_velocity_world": [float(v) for v in velocity[3:6]],
                "angular_velocity_world": [float(v) for v in velocity[0:3]],
                "supported": bool(support_name and any(c["other_geom"] == support_name for c in contacts)),
                "contacts": contacts,
                "hand_contacts": hand_contacts,
                "contact_count": len(contacts),
                "hand_contact_count": len(hand_contacts),
            }
        try:
            handsim_bus.publish_state(handsim_bus.MANIPULATION_OBJECTS, payload)
        except Exception:
            pass

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        q = self.mj_data.qpos[self.body_joint_index + self.qpos_offset - 1]
        dq = self.mj_data.qvel[self.body_joint_index + self.qvel_offset - 1]
        if self.bootstrap_controller is not None:
            snapshot = self.unitree_bridge.body_command_snapshot()
            external = None
            max_age = float(os.environ.get("DROID_BOOT_EXTERNAL_MAX_AGE_SECONDS", "0.1"))
            min_kp = float(os.environ.get("DROID_BOOT_EXTERNAL_MIN_KP", "1.0"))
            min_active = int(os.environ.get("DROID_BOOT_EXTERNAL_MIN_ACTIVE_JOINTS", "12"))
            command_is_active = (
                snapshot is not None
                and int(np.count_nonzero(snapshot["kp"] >= min_kp)) >= min_active
            )
            if snapshot is not None and snapshot["age_s"] <= max_age and command_is_active:
                external = JointBootstrapCommand(
                    q=snapshot["q"],
                    dq=snapshot["dq"],
                    kp=snapshot["kp"],
                    kd=snapshot["kd"],
                    tau=snapshot["tau"],
                )
            handover_allowed = os.path.exists(
                os.environ.get("DROID_BOOT_HANDOVER_PATH", "/tmp/bootstrap_handover")
            )
            command = self.bootstrap_controller.step(
                self.mj_data.time,
                q,
                dq,
                external,
                handover_allowed=handover_allowed,
            )
            if self.bootstrap_controller.phase != self._bootstrap_phase:
                self._bootstrap_phase = self.bootstrap_controller.phase
                print(f"[base_sim] bootstrap -> {self._bootstrap_phase.value}", flush=True)
            if self.bootstrap_controller.phase is BootstrapPhase.READY:
                # Permanently relinquish startup authority. Runtime LowCmd
                # freshness and safety now belong to SONIC's normal path.
                self.bootstrap_controller = None
            return command.tau + command.kp * (command.q - q) + command.kd * (command.dq - dq)

        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        root_quat_wxyz = self.mj_data.body("torso_link").xquat.copy()
        # scipy Rotation needs [x,y,z,w]; every other quaternion in this module
        # (rt/lowstate imu_state, rt/secondary_imu) stays MuJoCo-native [w,x,y,z],
        # so convert back before returning instead of leaking the scipy order out.
        root_quat_xyzw = root_quat_wxyz[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat_xyzw).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat_wxyz))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        self._poll_live_settings()
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            # IMU contract (matches the real G1 lowstate): accelerometer is
            # SPECIFIC FORCE in the body frame -- reads +g up at rest --
            # not world-frame coordinate acceleration (qacc, zero at rest).
            # Consumers: deploy state logger, and lidar-inertial odometry via
            # the ros-bridge (docs/real_u8_autonomy_contract.md), whose
            # gravity model breaks under qacc semantics.
            acc = np.asarray(self.mj_data.qacc[:6], dtype=np.float64).copy()
            rot9 = np.zeros(9)
            mujoco.mju_quat2Mat(rot9, self.mj_data.qpos[3:7])
            acc[:3] = rot9.reshape(3, 3).T @ (
                self.mj_data.qacc[:3] - self.mj_model.opt.gravity)
            obs["floating_base_acc"] = acc
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_actuator_index]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_actuator_index]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]

        if self.sensor_noise_enable:
            rng = self._noise_rng
            imu_rotvec = self._imu_quat_bias_rotvec + rng.normal(0.0, self._imu_quat_noise_std, size=3)
            obs["secondary_imu_quat"] = self._apply_small_rotation_wxyz(
                obs["secondary_imu_quat"], imu_rotvec
            )
            obs["secondary_imu_vel"] = obs["secondary_imu_vel"] + rng.normal(
                0.0, self._imu_vel_noise_std, size=6
            )
            obs["body_q"] = obs["body_q"] + rng.normal(0.0, self._joint_pos_noise_std, size=obs["body_q"].shape)
            obs["body_dq"] = obs["body_dq"] + rng.normal(0.0, self._joint_vel_noise_std, size=obs["body_dq"].shape)
            if self.num_hand_dof > 0:
                for side in ("left_hand_q", "right_hand_q"):
                    obs[side] = obs[side] + rng.normal(0.0, self._joint_pos_noise_std, size=obs[side].shape)
                for side in ("left_hand_dq", "right_hand_dq"):
                    obs[side] = obs[side] + rng.normal(0.0, self._joint_vel_noise_std, size=obs[side].shape)
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_actuator_index]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self._poll_manipulation_command()
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.elastic_band:
            # [MULTI-48] headless band-through-handoff: hold the robot while the
            # policy enters CONTROL. Treat /tmp/release_band as a live toggle so
            # the web console can reattach the band after an unstable generated
            # motion without restarting MuJoCo.
            should_enable_band = not os.path.exists("/tmp/release_band")
            if self.elastic_band.enable != should_enable_band:
                if should_enable_band:
                    band_pos = self.mj_data.xpos[self.band_attached_link].copy()
                    self.elastic_band.point = np.array([band_pos[0], band_pos[1], 1.0])
                    self.mj_data.qvel[: self.qvel_offset] = 0
                self.elastic_band.enable = should_enable_band
                state = "attached" if should_enable_band else "released"
                print(f"[base_sim] /tmp/release_band -> elastic band {state}", flush=True)
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        self.torques.fill(0.0)
        self.torques[self.body_actuator_index] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_actuator_index] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_actuator_index] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)
        self._publish_manipulation_state()

        self.check_fall()

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            elif "mjcf_name" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["mjcf_name"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            image = renderer.render()
            if self.sensor_noise_enable:
                noise = self._noise_rng.normal(0.0, self._camera_noise_std, size=image.shape)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            render_caches[camera_name + "_image"] = image

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def check_fall(self):
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def _resample_imu_bias(self):
        """Resample the per-episode fixed IMU orientation bias (small-angle rotvec, rad)."""
        self._imu_quat_bias_rotvec = self._noise_rng.normal(0.0, self._imu_quat_bias_std, size=3)

    def _poll_live_settings(self, force: bool = False):
        """Pull live-tunable sensor-noise settings from the console (handsim_bus),
        throttled to avoid a redis round-trip every physics tick. Fail-safe: any
        error (redis down, bad payload) just keeps the last-known values -- this
        is a nice-to-have, not safety-critical, and must never disrupt the sim
        loop. imu_quat_bias_std only takes effect on the next reset (the bias
        itself is meant to stay fixed for an episode, not jitter mid-run)."""
        if handsim_bus is None:
            return
        now = time.time()
        if not force and (now - self._last_settings_poll) < self._settings_poll_interval:
            return
        self._last_settings_poll = now
        try:
            settings = handsim_bus.read_state(handsim_bus.SIM_SETTINGS)
        except Exception:
            return
        if not settings:
            return
        self.sensor_noise_enable = bool(settings.get("sensor_noise_enable", self.sensor_noise_enable))
        self._joint_pos_noise_std = float(settings.get("joint_pos_std", self._joint_pos_noise_std))
        self._joint_vel_noise_std = float(settings.get("joint_vel_std", self._joint_vel_noise_std))
        self._imu_quat_noise_std = float(settings.get("imu_quat_std", self._imu_quat_noise_std))
        self._imu_quat_bias_std = float(settings.get("imu_quat_bias_std", self._imu_quat_bias_std))
        self._imu_vel_noise_std = float(settings.get("imu_vel_std", self._imu_vel_noise_std))
        self._camera_noise_std = float(settings.get("camera_std", self._camera_noise_std))

    @staticmethod
    def _apply_small_rotation_wxyz(quat_wxyz: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
        """Return quat_wxyz rotated by a small-angle rotvec (rad), left-multiplied. Non-destructive."""
        angle = np.linalg.norm(rotvec)
        if angle < 1e-12:
            return quat_wxyz.copy()
        axis = rotvec / angle
        half = angle * 0.5
        delta = np.array([np.cos(half), *(axis * np.sin(half))])
        w0, x0, y0, z0 = delta
        w1, x1, y1, z1 = quat_wxyz
        out = np.array([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ])
        return out / np.linalg.norm(out)

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.sensor_noise_enable:
            self._resample_imu_bias()


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Only 'default' is supported in this minimal build."
            )

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                self.sim_env.sim_step()
                now = time.time()
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)
