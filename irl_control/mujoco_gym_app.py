from abc import ABC, abstractmethod
from typing import Dict

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from irl_control.constants import IRL_CONTROL_BASE_DIR
from irl_control.device import Device
from irl_control.osc import OSC
from irl_control.robot import Robot
from irl_control.utils import stderr_redirected

MUJOCO_FRAME_SKIP = 3


class MujocoGymApp(MujocoEnv, ABC):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 33,
    }

    def __init__(
        self,
        robot_config_file,
        scene_file,
        observation_space,
        action_space,
        osc_device_pairs=[("base", "base_osc"), ("ur5right", "arm_osc"), ("ur5left", "arm_osc")],
        osc_use_admittance=True,
        robot_name="DualUR5",
        render_mode="human",
        width=480,
        height=360,
        frameskip=MUJOCO_FRAME_SKIP,
        hide_mjpy_warnings=False,
    ):
        full_model_path = IRL_CONTROL_BASE_DIR / "assets" / scene_file
        assert full_model_path.exists(), f"Scene file {full_model_path} does not exist!"

        create_mjpy_env = lambda: MujocoEnv.__init__(
            self,
            full_model_path.as_posix(),
            frame_skip=frameskip,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
        )

        if hide_mjpy_warnings:
            with stderr_redirected():
                create_mjpy_env()
        else:
            create_mjpy_env()

        # Tmp fix: _create_overlay throws error about the *solver_iter* attribute
        # You can change the gymnasium source code, but this shouldn't be necessary
        gym_major_version = int(gym.__version__.split(".")[1])
        if gym_major_version >= 27:
            viewer = self.mujoco_renderer._get_viewer(render_mode)
        else:
            viewer = self._get_viewer(render_mode)

        viewer._create_overlay = lambda: None
        self._viewer_setup(viewer)

        # Reset the action space to the "true" action space (not based on mujoco's xml)
        self.ctrl_action_space = self.action_space
        self.action_space = action_space

        robot_config_path = IRL_CONTROL_BASE_DIR / "robot_configs" / robot_config_file

        with open(robot_config_path, "r") as file:
            self._irl_robot_cfg = yaml.safe_load(file)

        self._irl_devices = self.__get_devices(self.model, self.data, self._irl_robot_cfg)

        # Specify the controller configuations that should be used for the corresponding devices
        osc_device_configs = [
            (device_name, self.__get_controller_config(osc_name))
            for device_name, osc_name in osc_device_pairs
        ]

        # Get the configuration for the nullspace controller
        self.robot = self.__get_robot(robot_name=robot_name)
        nullspace_config = self.__get_controller_config("nullspace")

        self.controller = OSC(
            self.robot,
            self.data,
            osc_device_configs,
            nullspace_config,
            admittance=osc_use_admittance,
            default_start_pt=self.default_start_pt,
        )

    def _viewer_setup(self, viewer):
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -25
        viewer.cam.lookat[0] = 0.1
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = 0.0
        viewer.cam.distance = 2

    def __get_devices(self, mj_model, mj_data, yaml_cfg, use_sim=True):
        all_devices = np.array(
            [Device(dev, mj_model, mj_data, use_sim) for dev in yaml_cfg["devices"]]
        )
        robots = np.array([])
        all_robot_device_idxs = np.array([], dtype=np.int32)
        for robot_cfg in yaml_cfg["robots"]:
            robot_device_idxs = robot_cfg["device_ids"]
            all_robot_device_idxs = np.hstack([all_robot_device_idxs, robot_device_idxs])
            robot = Robot(
                all_devices[robot_device_idxs], robot_cfg["name"], mj_model, mj_data, use_sim
            )
            robots = np.append(robots, robot)

        all_idxs = np.arange(len(all_devices))
        keep_idxs = np.setdiff1d(all_idxs, all_robot_device_idxs)
        devices = np.hstack([all_devices[keep_idxs], robots])
        return devices

    def __get_robot(self, robot_name: str) -> Robot:
        for device in self._irl_devices:
            if type(device) == Robot:
                if device.name == robot_name:
                    return device

    def __get_controller_config(self, name: str) -> Dict:
        ctrlr_conf = self._irl_robot_cfg["controller_configs"]
        for entry in ctrlr_conf:
            if entry["name"] == name:
                return entry

    @property
    @abstractmethod
    def default_start_pt(self):
        raise NotImplementedError

    def set_free_joint_qpos(self, free_joint_name, quat=None, pos=None):
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, free_joint_name)
        offset = self.model.jnt_qposadr[jnt_id]

        if quat is not None:
            quat_idxs = np.arange(offset + 3, offset + 7)  # Quaternion indices
            self.data.qpos[quat_idxs] = quat

        if pos is not None:
            pos_idxs = np.arange(offset, offset + 3)  # Position indices
            self.data.qpos[pos_idxs] = pos

    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl, n_frames)


class MujocoGymAppHighFidelity(MujocoGymApp):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 111,
    }

    def __init__(
        self,
        robot_config_file,
        scene_file,
        observation_space,
        action_space,
        osc_device_pairs=[("base", "base_osc"), ("ur5right", "arm_osc"), ("ur5left", "arm_osc")],
        osc_use_admittance=True,
        robot_name="DualUR5",
        render_mode="human",
        width=480,
        height=360,
        frameskip=MUJOCO_FRAME_SKIP,
        hide_mjpy_warnings=False,
    ):
        MujocoGymApp.__init__(
            self,
            robot_config_file,
            scene_file,
            observation_space,
            action_space,
            osc_device_pairs,
            osc_use_admittance,
            robot_name,
            render_mode,
            width,
            height,
            frameskip,
            hide_mjpy_warnings,
        )
