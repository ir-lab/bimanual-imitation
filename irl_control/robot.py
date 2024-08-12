import copy
import time
from enum import Enum
from threading import Lock
from typing import Any, Dict, List

import mujoco
import numpy as np

from irl_control.device import Device, DeviceState


class RobotState(Enum):
    M = "INERTIA"
    DQ = "DQ"
    J = "JACOBIAN"
    G = "GRAVITY"


class Robot:
    def __init__(
        self, sub_devices: List[Device], robot_name, model, data, use_sim, collect_hz=1000
    ):
        self._data = data
        self._model = model

        self._sub_devices = sub_devices
        self._sub_devices_dict: Dict[str, Device] = dict()
        for dev in self._sub_devices:
            self._sub_devices_dict[dev.name] = dev

        self._name = robot_name
        self._num_scene_joints = self._model.nv

        self._all_joint_ids = np.array([], dtype=np.int32)
        for dev in self._sub_devices:
            self._all_joint_ids = np.hstack([self._all_joint_ids, dev.all_joint_ids])
        self._all_joint_ids = np.sort(np.unique(self._all_joint_ids))

        self._num_joints_total = len(self._all_joint_ids)

        self._data_collect_hz = collect_hz
        self.__use_sim = use_sim
        self.__running = False

        self.__state_locks: Dict[RobotState, Lock] = dict([(key, Lock()) for key in RobotState])
        self.__state_var_map: Dict[RobotState, function] = {
            RobotState.M: lambda: self.__get_M(),
            RobotState.DQ: lambda: self.__get_dq(),
            RobotState.J: lambda: self.__get_jacobian(),
            RobotState.G: lambda: self.__get_gravity(),
        }
        self.__state: Dict[RobotState, Any] = dict()

    @property
    def name(self):
        return self._name

    @property
    def sub_devices(self):
        return self._sub_devices

    @property
    def sub_devices_dict(self):
        return self._sub_devices_dict

    @property
    def all_joint_ids(self):
        return self._all_joint_ids

    def __get_gravity(self):
        return self._data.qfrc_bias

    def __get_jacobian(self):
        """
        Return the Jacobians for all of the devices,
        so that OSC can stack them according to provided the target entries
        """
        Js = dict()
        J_idxs = dict()
        start_idx = 0
        for name, device in self._sub_devices_dict.items():
            J_sub = device.get_state(DeviceState.J)
            J_idxs[name] = np.arange(start_idx, start_idx + J_sub.shape[0])
            start_idx += J_sub.shape[0]
            J_sub = J_sub[:, self._all_joint_ids]
            Js[name] = J_sub
        return Js, J_idxs

    def __get_dq(self):
        dq = np.zeros(self._all_joint_ids.shape)
        for dev in self._sub_devices:
            dq[dev.all_joint_ids] = dev.get_state(DeviceState.DQ)
        return dq

    def __get_M(self):
        M = np.zeros((self._num_scene_joints, self._num_scene_joints))
        mujoco.mj_fullM(self._model, M, self._data.qM)
        M = M[np.ix_(self._all_joint_ids, self._all_joint_ids)]
        return M

    def get_state(self, state_var: RobotState):
        if self.__use_sim:
            func = self.__state_var_map[state_var]
            state = copy.copy(func())
        else:
            self.__state_locks[state_var].acquire()
            state = copy.copy(self.__state[state_var])
            self.__state_locks[state_var].release()
        return state

    def __set_state(self, state_var: RobotState):
        assert self.__use_sim is False
        self.__state_locks[state_var].acquire()
        func = self.__state_var_map[state_var]
        value = func()
        # Make sure to copy (or else reference will stick to Dict value)
        self.__state[state_var] = copy.copy(value)
        self.__state_locks[state_var].release()

    def is_running(self):
        return self.__running

    def is_using_sim(self):
        return self.__use_sim

    def __update_state(self):
        assert self.__use_sim is False
        for var in RobotState:
            self.__set_state(var)

    def start(self):
        assert self.__running is False and self.__use_sim is False
        self.__running = True
        interval = float(1.0 / float(self._data_collect_hz))
        prev_time = time.time()
        while self.__running:
            for dev in self._sub_devices:
                dev.update_state()
            self.__update_state()
            curr_time = time.time()
            diff = curr_time - prev_time
            delay = max(interval - diff, 0)
            time.sleep(delay)
            prev_time = curr_time

    def stop(self):
        assert self.__running is True and self.__use_sim is False
        self.__running = False

    def get_device(self, device_name: str) -> Device:
        return self._sub_devices_dict[device_name]

    def get_all_states(self):
        """
        Get's the state of all the devices connected plus the robot states
        """
        state = {}
        for device_name, device in self._sub_devices_dict.items():
            state[device_name] = device.get_all_states()

        for key in RobotState:
            state[key] = self.get_state(key)

        return state

    def get_device_states(self):
        """
        Get's the state of all the devices connected
        """
        state = {}
        for device_name, device in self._sub_devices_dict.items():
            state[device_name] = device.get_all_states()
        return state
