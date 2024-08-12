from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml
from gymnasium.spaces.box import Box

from irl_data import proto_logger
from irl_environments.core.state import State
from irl_environments.core.target_course import TargetCourse
from irl_environments.core.track import Track, get_gym_space_enums
from irl_environments.core.utils import (
    PursuitType,
    get_action_group_dim,
    get_enum_value,
    get_observation_group_dim,
)


class Runner(ABC):
    def __init__(self, proto_file: Path, randomize_track_length=False):
        assert self.yaml_param_file.exists()
        if proto_file is not None:
            assert proto_file.exists()
            expert_traj = proto_logger.load_trajs(proto_file)[0].obs_T_Do
        else:
            expert_traj = None

        with open(self.yaml_param_file, "r", encoding="utf8") as file:
            config = yaml.safe_load(file)

        assert "run_config" in config.keys()
        run_config = config["run_config"]
        self.__expert_config = run_config["expert"]
        self.__path_env_config = run_config["path_env"]
        self.__pursuit_type = get_enum_value(run_config["pursuit_type"], PursuitType)
        self.__default_start_pt = {}
        self.__tracks: OrderedDict[str, Track] = {}

        num_expert_samples = self.get_expert_param("num_samples")
        if num_expert_samples is None and not randomize_track_length:
            resample = False
            self.__expert_num_samples = expert_traj.shape[0]
        elif num_expert_samples is None and randomize_track_length:
            resample = True
            expert_num_samples = expert_traj.shape[0]
            lower_scale = float(self.get_expert_param("rand_num_samples_lower_scale"))
            upper_scale = float(self.get_expert_param("rand_num_samples_upper_scale"))
            self.__expert_num_samples = np.random.randint(
                low=int(expert_num_samples * lower_scale),
                high=int(expert_num_samples * upper_scale),
            )
        elif num_expert_samples is not None and randomize_track_length:
            warning = (
                "**Warning: Randomizing Track Length. "
                f"Overriding yaml param `num samples` (value {num_expert_samples}) "
                "with randomized value.**"
            )
            print(warning)
            resample = True
            expert_num_samples = expert_traj.shape[0]
            lower_scale = float(self.get_expert_param("rand_num_samples_lower_scale"))
            upper_scale = float(self.get_expert_param("rand_num_samples_upper_scale"))
            self.__expert_num_samples = np.random.randint(
                low=int(expert_num_samples * lower_scale),
                high=int(expert_num_samples * upper_scale),
            )
        else:  # num_expert_samples is defined and we do not randomize track length
            resample = True
            self.__expert_num_samples = num_expert_samples

        self.__last_course_idx = self.expert_num_samples - 2

        for device_name in self.get_all_devices():
            # Assume all expert demonstrations contain the xyz and quaternion structure (length 7)
            device_slice, default_pt = self.get_slice_and_default_pt(expert_traj, device_name)
            # Set the default start pt for any non-active devices
            self.__default_start_pt[device_name] = default_pt

            if device_name in config["devices"].keys():
                device_config = config["devices"][device_name]
                device_course = self.get_course_from_expert(
                    device_slice,
                    device_config["lookahead_gain"],
                    device_config["lookahead_dist"],
                    device_name,
                    resample,
                )
                device_state = self.get_initial_state(device_config, device_course)
                device_track = self.get_track_type()(
                    device_course, device_state, device_name, device_config, config["spaces"]
                )
                self.tracks[device_name] = device_track

        self.__keys = self.tracks.keys()

    def get_expert_param(self, key: str):
        keys = key.split(".")
        value = self.__expert_config
        for k in keys:
            assert k in value, print(f"Key {k} not found in {value}")
            value = value[k]
        return value

    def get_path_env_param(self, key):
        keys = key.split(".")
        value = self.__path_env_config
        for k in keys:
            assert k in value, print(f"Key {k} not found in {value}")
            value = value[k]
        return value

    @abstractmethod
    def get_slice_and_default_pt(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_course_from_expert(
        self, device_slice, lookahead_gain, lookahead_dist, device_name, resample=None
    ) -> TargetCourse:
        raise NotImplementedError

    @abstractmethod
    def get_initial_state(self, device_config, course) -> State:
        raise NotImplementedError

    @abstractmethod
    def get_track_type() -> Track:
        raise NotImplementedError

    @abstractmethod
    def get_all_devices():
        raise NotImplementedError

    @property
    @abstractmethod
    def yaml_param_file(self) -> Path:
        raise NotImplementedError

    @property
    def default_start_pt(self):
        return self.__default_start_pt

    @property
    def pursuit_type(self):
        return self.__pursuit_type

    @property
    def tracks(self):
        return self.__tracks

    @property
    def expert_num_samples(self):
        return self.__expert_num_samples

    @property
    def last_course_idx(self):
        return self.__last_course_idx

    @property
    def keys(self):
        return self.__keys


def get_gym_spaces(runner_instance: Runner, pred_horizon, obs_horizon):
    with open(runner_instance.yaml_param_file, "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    obs_size = 0
    act_size = 0
    env_spaces = config["spaces"]
    for device_name in runner_instance.get_all_devices():
        if device_name in config["devices"].keys():
            device_config = config["devices"][device_name]
            track_instance = runner_instance.get_track_type()
            obs_enums, act_enums = get_gym_space_enums(track_instance, device_config, env_spaces)
            for obs_enum in obs_enums:
                obs_size += get_observation_group_dim(obs_enum)
            for act_enum in act_enums:
                act_size += get_action_group_dim(act_enum)

    obs_size *= obs_horizon
    act_size *= pred_horizon

    observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
    action_space = Box(low=-100, high=100, shape=(act_size,), dtype=np.float32)
    return observation_space, action_space
