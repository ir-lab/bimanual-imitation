from abc import ABC, abstractmethod

import numpy as np
import yaml

from irl_environments.constants import IRL_ENVIRONMENTS_BASE_DIR
from irl_environments.core.state import State, State2D, State3D, StateMujoco3D
from irl_environments.core.target_course import (
    TargetCourse,
    TargetCourse2D,
    TargetCourse3D,
    TargetCourseMujoco3D,
)
from irl_environments.core.utils import ActionGroup, GripType, ObservationGroup, get_enum_value


class Track(ABC):
    def __init__(
        self,
        course: TargetCourse,
        state: State,
        device_name: str,
        device_config: dict,
        spaces: dict,
    ):
        assert isinstance(course, TargetCourse)
        assert isinstance(state, State)
        self.__device_config = device_config
        self.__course = course
        self.__state = state
        self.__device_name = device_name
        self.__observation_space, self.__action_space = get_gym_space_enums(
            self, device_config, spaces
        )
        expert_target_speed = self._get_device_value("expert_target_speed")
        noise = self._get_device_value("expert_target_speed_noise")
        if noise is not None:
            assert len(noise) == 2
            expert_target_speed += np.random.uniform(low=noise[0], high=noise[1])
        self.__expert_target_speed = expert_target_speed

    def _get_device_value(self, key):
        if key in self.__device_config.keys():
            return self.__device_config[key]
        else:
            print(f'Error: Attribute "{key}" not found in the device config!')
            raise KeyError

    def set_initial_state(self, state):
        self.__state = state

    @property
    def device_config(self):
        return self.__device_config

    @property
    def course(self):
        return self.__course

    @property
    def state(self):
        return self.__state

    @property
    def device_name(self):
        return self.__device_name

    @property
    def action_space(self):
        return self.__action_space

    @property
    def observation_space(self):
        return self.__observation_space

    @property
    def expert_target_speed(self):
        return self.__expert_target_speed

    @abstractmethod
    def get_valid_observation_groups() -> set:
        raise NotImplementedError

    @abstractmethod
    def get_valid_action_groups() -> set:
        raise NotImplementedError


class Track2D(Track):
    def __init__(
        self,
        course: TargetCourse2D,
        state: State2D,
        device_name: str,
        device_config: dict,
        spaces: dict,
    ):
        assert isinstance(course, TargetCourse2D)
        assert isinstance(state, State2D)
        super().__init__(course, state, device_name, device_config, spaces)

    @staticmethod
    def get_valid_observation_groups() -> set:
        return set([ObservationGroup.POSITION_2D, ObservationGroup.YAW])

    @staticmethod
    def get_valid_action_groups() -> set:
        return set([ActionGroup.SPEED, ActionGroup.DELTA_STEER])

    @property
    def delta_bounds(self):
        return self._get_device_value("delta_bounds")

    @property
    def enforce_delta_bounds(self):
        return self._get_device_value("enforce_delta_bounds")

    @property
    def velocity_bounds(self):
        return self._get_device_value("velocity_bounds")

    @property
    def enforce_velocity_bounds(self):
        return self._get_device_value("enforce_velocity_bounds")


class Track3D(Track):
    def __init__(
        self,
        course: TargetCourse3D,
        state: State3D,
        device_name: str,
        device_config: dict,
        spaces: dict,
    ):
        assert isinstance(course, TargetCourse3D)
        assert isinstance(state, State3D)
        super().__init__(course, state, device_name, device_config, spaces)

    @staticmethod
    def get_valid_action_groups() -> set:
        return set(
            [
                ActionGroup.DELTA_POSITION,
            ]
        )

    @staticmethod
    def get_valid_observation_groups() -> set:
        return set(
            [
                ObservationGroup.POSITION,
                ObservationGroup.DELTA_POSITION_POLAR,
            ]
        )

    @property
    def delta_xyz_bounds(self):
        assert ActionGroup.DELTA_POSITION in self.action_space
        return (
            self._get_device_value("delta_x_bounds"),
            self._get_device_value("delta_y_bounds"),
            self._get_device_value("delta_z_bounds"),
        )

    @property
    def enforce_delta_xyz_bounds(self):
        assert ActionGroup.DELTA_POSITION in self.action_space
        return self._get_device_value("enforce_delta_xyz_bounds")


class TrackMujoco3D(Track):
    def __init__(
        self,
        course: TargetCourseMujoco3D,
        state: StateMujoco3D,
        device_name: str,
        device_config: dict,
        spaces: dict,
    ):
        assert isinstance(course, TargetCourseMujoco3D)
        assert isinstance(state, StateMujoco3D)
        super().__init__(course, state, device_name, device_config, spaces)

        noise_file = self._get_device_value("noise_file")
        noise_config_path = IRL_ENVIRONMENTS_BASE_DIR / f"param/noise_configs/{noise_file}"

        with open(noise_config_path, "r", encoding="utf8") as file:
            noise_config = yaml.safe_load(file)

        valid_noise_degrees = ["zero", "low", "medium", "high", "orig"]
        action_noise_degree = self._get_device_value("action_noise")

        assert action_noise_degree in valid_noise_degrees
        if action_noise_degree == "zero":
            self._action_noise = None
        else:
            self._action_noise = noise_config["action_noise"][action_noise_degree]
            assert self._action_noise is not None

        observation_noise_degree = self._get_device_value("observation_noise")
        assert observation_noise_degree in valid_noise_degrees
        if observation_noise_degree == "zero":
            self._observation_noise = None
        else:
            self._observation_noise = noise_config["observation_noise"][observation_noise_degree]
            assert self._observation_noise is not None

    @staticmethod
    def get_valid_action_groups() -> set:
        return set(
            [
                ActionGroup.DELTA_POSITION,
                ActionGroup.DELTA_SIX_DOF,
                ActionGroup.DELTA_QUAT,
                ActionGroup.DELTA_EULER,
            ]
        )

    @staticmethod
    def get_valid_observation_groups() -> set:
        return set(
            [
                ObservationGroup.POSITION,
                ObservationGroup.DELTA_POSITION_POLAR,
                ObservationGroup.DELTA_TARGET_POS,
                ObservationGroup.MALE_OBJ_POS,
                ObservationGroup.MALE_OBJ_SIX_DOF,
                ObservationGroup.FEMALE_OBJ_POS,
                ObservationGroup.FEMALE_OBJ_SIX_DOF,
                ObservationGroup.DELTA_OBJS_POS,
                ObservationGroup.DELTA_OBJS_SIX_DOF,
                ObservationGroup.DELTA_POS_QUAD_PEG_LEFT,
                ObservationGroup.DELTA_POS_QUAD_PEG_LEFT_CBRT,
                ObservationGroup.DELTA_POS_NIST_PEG_LEFT_CBRT,
                ObservationGroup.DELTA_POS_QUAD_PEG_FRONT_LEFT_CBRT,
                ObservationGroup.DELTA_POS_QUAD_PEG_LEFT_POLAR,
                ObservationGroup.DELTA_POS_QUAD_PEG_RIGHT,
                ObservationGroup.DELTA_POS_QUAD_PEG_RIGHT_CBRT,
                ObservationGroup.DELTA_POS_NIST_PEG_RIGHT_CBRT,
                ObservationGroup.DELTA_POS_QUAD_PEG_FRONT_RIGHT_CBRT,
                ObservationGroup.DELTA_POS_QUAD_PEG_RIGHT_POLAR,
                ObservationGroup.DUAL_PEG_DELTA_POSITIONS,
                ObservationGroup.SIX_DOF,
                ObservationGroup.QUAT,
                ObservationGroup.EULER,
                ObservationGroup.TARGET_SIX_DOF,
                ObservationGroup.DELTA_TARGET_SIX_DOF,
                ObservationGroup.TARGET_QUAT,
                ObservationGroup.DELTA_TARGET_QUAT,
                ObservationGroup.TARGET_EULER,
                ObservationGroup.GRIP_FORCE,
                ObservationGroup.GRIP_FORCE_EWMA,
                ObservationGroup.GRIP_TORQUE,
                ObservationGroup.GRIP_TORQUE_EWMA,
                ObservationGroup.POSITION_DIFF_NORM,
                ObservationGroup.BASE_ANGLE,
            ]
        )

    @property
    def orientation_groups(self) -> set:
        return [
            ActionGroup.DELTA_SIX_DOF,
            ActionGroup.DELTA_QUAT,
            ActionGroup.DELTA_EULER,
        ]

    @property
    def gripper_step_action(self):
        grip_type_str = self._get_device_value("gripper_step_action")
        return get_enum_value(grip_type_str, GripType)

    @property
    def gripper_reset_action(self):
        grip_type_str = self._get_device_value("gripper_reset_action")
        return get_enum_value(grip_type_str, GripType)

    @property
    def gripper_idx(self):
        return self._get_device_value("gripper_idx")

    @property
    def grip_forces(self):
        return {
            GripType.OPEN: self._get_device_value("open_gripper_force"),
            GripType.CLOSE: self._get_device_value("close_gripper_force"),
        }

    @property
    def xyz_bounds(self):
        return (
            self._get_device_value("x_bounds"),
            self._get_device_value("y_bounds"),
            self._get_device_value("z_bounds"),
        )

    @property
    def enforce_xyz_bounds(self):
        return self._get_device_value("enforce_xyz_bounds")

    @property
    def delta_xyz_bounds(self):
        assert ActionGroup.DELTA_POSITION in self.action_space
        return (
            self._get_device_value("delta_x_bounds"),
            self._get_device_value("delta_y_bounds"),
            self._get_device_value("delta_z_bounds"),
        )

    @property
    def enforce_delta_xyz_bounds(self):
        assert ActionGroup.DELTA_POSITION in self.action_space
        return self._get_device_value("enforce_delta_xyz_bounds")

    @property
    def action_noise(self):
        return self._action_noise

    @property
    def observation_noise(self):
        return self._observation_noise

    @property
    def ewma_alpha(self):
        _ewma_alpha = self._get_device_value("ewma_alpha")
        assert _ewma_alpha < 1.0 and _ewma_alpha > 0.0
        return _ewma_alpha


def get_gym_space_enums(track_instance: Track, device_config: dict, env_spaces):
    space_name = device_config["space"]
    assert space_name in env_spaces.keys()

    action_space_strs = env_spaces[space_name]["action_space"]
    action_space = []
    if action_space_strs is not None:
        # enforce unique action groups
        assert len(set(action_space_strs)) == len(action_space_strs)
        assert isinstance(track_instance.get_valid_action_groups(), set)
        for action_str in action_space_strs:
            action_enum = get_enum_value(action_str, ActionGroup)
            assert action_enum in track_instance.get_valid_action_groups()
            action_space.append(action_enum)
    action_space = tuple(action_space)  # make immutable

    obs_space_strs = env_spaces[space_name]["observation_space"]
    observation_space = []
    # enforce unique observation groups
    assert len(set(obs_space_strs)) == len(obs_space_strs)
    assert isinstance(track_instance.get_valid_observation_groups(), set)
    for obs_str in obs_space_strs:
        obs_enum = get_enum_value(obs_str, ObservationGroup)
        assert obs_enum in track_instance.get_valid_observation_groups()
        observation_space.append(obs_enum)
    observation_space = tuple(observation_space)  # make immutable

    return observation_space, action_space
