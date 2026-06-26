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

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _handsim_scene_anchor_path() -> Path:
    return Path(os.environ.get("HANDSIM_SCENE_ANCHOR_PATH", "/tmp/handsim_scene_anchor.json"))


def _safe_xml_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_") or "item"


def _xml_num(value: float) -> str:
    return f"{float(value):.5g}"


def _xml_attr(value: Path | str) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _load_handsim_collision_scene_config() -> tuple[Path, dict] | None:
    config_path = os.environ.get("HANDSIM_COLLISION_SCENE_CONFIG", "").strip()
    if not config_path:
        return None
    path = Path(config_path).expanduser()
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
    repo = os.environ.get("HANDSIM_REPO", "").strip()
    if repo:
        return Path(repo).expanduser()
    # assets/render/worlds/outdoor_park/scene.json -> repo root
    try:
        return config_path.parents[3]
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


def _handsim_collision_assets_and_geoms(scene: dict, repo: Path, anchor_xy: tuple[float, float]) -> tuple[list[str], list[str]]:
    assets = []
    geoms = []
    ax, ay = anchor_xy

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

        if kind == "wall":
            size = cfg.get("size", [1.0, 0.06])
            height = float(cfg.get("height", 1.5))
            geoms.append(_box_geom(key, x, y, height * 0.5, float(size[0]), float(size[1]), height * 0.5, yaw))
        elif kind == "door_frame":
            s = scale
            geoms.append(_box_geom(key + "_left", x - 0.86 * s, y, 1.05 * s, 0.055 * s, 0.10 * s, 1.05 * s, yaw))
            geoms.append(_box_geom(key + "_right", x + 0.86 * s, y, 1.05 * s, 0.055 * s, 0.10 * s, 1.05 * s, yaw))
            geoms.append(_box_geom(key + "_top", x, y, 2.07 * s, 0.92 * s, 0.10 * s, 0.07 * s, yaw))
        elif kind == "booth":
            s = scale
            geoms.append(_box_geom(key + "_back", x, y, 0.78 * s, 1.15 * s, 0.08 * s, 0.78 * s, yaw))
            dx = math.cos(yaw + math.pi * 0.5) * 0.72 * s
            dy = math.sin(yaw + math.pi * 0.5) * 0.72 * s
            geoms.append(_box_geom(key + "_side", x + dx, y + dy, 0.72 * s, 0.08 * s, 0.78 * s, 0.72 * s, yaw))
        elif kind == "instrument_rack":
            s = scale
            geoms.append(_box_geom(key, x, y, 0.75 * s, 0.45 * s, 0.32 * s, 0.75 * s, yaw))
        elif kind == "fence":
            length = float(cfg.get("length", 4.0)) * scale
            geoms.append(_box_geom(key, x, y, 0.45 * scale, length * 0.5, 0.055 * scale, 0.45 * scale, yaw))
        elif kind == "lamp":
            s = scale
            geoms.append(_capsule_geom(key + "_post", x, y, 0.75 * s, 0.035 * s, 0.75 * s))
        elif kind == "abo":
            model_name = str(cfg.get("model", "")).strip()
            obj_path = _scene_mesh_path(repo, kind, model_name)
            if obj_path is not None and obj_path.exists():
                mesh_name = "handsim_mesh_" + _safe_xml_name(str(raw_name))
                assets.append(_mesh_collision_asset(mesh_name, obj_path, scale))
                geoms.append(_mesh_collision_body(key, mesh_name, x, y, float(cfg.get("z", 0.0)), yaw))
                continue
            shadow = cfg.get("shadow", [0.45, 0.35])
            if not isinstance(shadow, list | tuple) or len(shadow) < 2:
                shadow = [0.45, 0.35]
            ext = _read_abo_extents(repo, str(cfg.get("model", "")), scale)
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


def _inject_handsim_collision_scene(xml_path: Path) -> Path:
    loaded = _load_handsim_collision_scene_config()
    if loaded is None:
        return xml_path

    config_path, scene = loaded
    repo = _handsim_repo_root(config_path)
    anchor_xy = _initial_root_xy(xml_path)
    _write_handsim_scene_anchor(anchor_xy)
    assets, geoms = _handsim_collision_assets_and_geoms(scene, repo, anchor_xy)
    if not geoms:
        print(f"[handsim-collision] no colliders generated from {config_path}", flush=True)
        return xml_path

    with open(xml_path) as f:
        xml = f.read()
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
        self.camera_configs = camera_configs

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen

        self.init_scene()
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
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
            elif "left_hand" in name:
                self.left_hand_index.append(i)
            elif "right_hand" in name:
                self.right_hand_index.append(i)

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS
        assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
        assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(self.left_hand_index)
        self.right_hand_index = np.array(self.right_hand_index)

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
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
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body("torso_link").xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

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
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
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
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
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
        # -1: actuator array is 0-based while joint indices from the model are 1-based
        self.torques[self.body_joint_index - 1] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

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
            render_caches[camera_name + "_image"] = renderer.render()

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

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)


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
