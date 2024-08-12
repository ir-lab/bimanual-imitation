import copy
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict

import mujoco
import numpy as np


class DeviceState(Enum):
    Q = "Q"
    Q_ACTUATED = "Q_ACTUATED"
    DQ = "DQ"
    DQ_ACTUATED = "DQ_ACTUATED"
    DDQ = "DDQ"
    EE_XYZ = "EE_XYZ"
    EE_XYZ_VEL = "EE_XYZ_VEL"
    EE_QUAT = "EE_QUAT"
    FORCE = "FORCE"
    TORQUE = "TORQUE"
    J = "JACOBIAN"


class Device:
    """
    The Device class encapsulates the device parameters specified in the yaml file
    that is passed to MujocoApp. It collects data from the simulator, obtaining the
    desired device states.
    """

    def __init__(self, device_yml: Dict, model, data, use_sim: bool):
        self._data = data
        self._model = model
        self.__use_sim = use_sim
        # Assign all of the yaml parameters
        self._name = device_yml["name"]
        self._max_vel = device_yml.get("max_vel")
        self._EE = device_yml["EE"]
        self._ctrlr_dof_xyz = device_yml["ctrlr_dof_xyz"]
        self._ctrlr_dof_abg = device_yml["ctrlr_dof_abg"]
        self._ctrlr_dof = np.hstack([self._ctrlr_dof_xyz, self._ctrlr_dof_abg])

        if "start_angles" in device_yml.keys():
            self._start_angles = np.array(device_yml["start_angles"])
        else:
            self._start_angles = None

        self._num_gripper_joints = device_yml["num_gripper_joints"]

        try:
            # Check if the user specifies a start body for the while loop to terminte at
            start_body_name = device_yml["start_body"]
            start_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, start_body_name)
        except:
            start_body = 0

        # Reference: ABR Control
        # Get the joint ids, using the specified EE / start body
        # start with the end-effector (EE) and work back to the world body
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._EE)
        joint_ids = []
        joint_names = []

        while model.body_parentid[body_id] != 0 and model.body_parentid[body_id] != start_body:
            jntadrs_start = model.body_jntadr[body_id]
            tmp_ids = []
            tmp_names = []
            for ii in range(model.body_jntnum[body_id]):
                tmp_ids.append(jntadrs_start + ii)
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, tmp_ids[-1])
                tmp_names.append(joint_name)
            joint_ids += tmp_ids[::-1]
            joint_names += tmp_names[::-1]
            body_id = model.body_parentid[body_id]

        # Flip the list so it starts with the base of the arm / first joint
        self._joint_names = joint_names[::-1]
        self._joint_ids = np.array(joint_ids[::-1])

        gripper_start_idx = self._joint_ids[-1] + 1
        self._gripper_ids = np.arange(
            gripper_start_idx, gripper_start_idx + self._num_gripper_joints
        )
        self._all_joint_ids = np.hstack([self._joint_ids, self._gripper_ids])

        # Find the actuator and control indices
        actuator_trnids = model.actuator_trnid[:, 0]
        self._ctrl_idxs = np.intersect1d(actuator_trnids, self._all_joint_ids, return_indices=True)[
            1
        ]
        self._actuator_trnids = actuator_trnids[self._ctrl_idxs]

        self.reset_start_angles()

        # Check that the
        if np.sum(np.hstack([self._ctrlr_dof_xyz, self._ctrlr_dof_abg])) > len(self._joint_ids):
            print("Fewer DOF than specified")

        # Initialize dicts to keep track of the state variables and locks
        self.__state_var_map: Dict[DeviceState, Callable[[], np.ndarray]] = {
            DeviceState.Q: lambda: data.qpos[self._all_joint_ids],
            DeviceState.Q_ACTUATED: lambda: data.qpos[self._joint_ids],
            DeviceState.DQ: lambda: data.qvel[self._all_joint_ids],
            DeviceState.DQ_ACTUATED: lambda: data.qvel[self._joint_ids],
            DeviceState.DDQ: lambda: data.qacc[self._all_joint_ids],
            DeviceState.EE_XYZ: lambda: data.xpos[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._EE)
            ],
            DeviceState.EE_XYZ_VEL: lambda: data.cvel[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._EE), :3
            ],
            DeviceState.EE_QUAT: lambda: data.xquat[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._EE)
            ],
            DeviceState.FORCE: lambda: self.__get_force(),
            DeviceState.TORQUE: lambda: self.__get_torque(),
            DeviceState.J: lambda: self.__get_jacobian(),
        }

        self.__state: Dict[DeviceState, Any] = dict()
        self.__state_locks: Dict[DeviceState, Lock] = dict([(key, Lock()) for key in DeviceState])

        # These are the that keys we should use when returning data from get_all_states()
        self._concise_state_vars = [
            DeviceState.Q_ACTUATED,
            DeviceState.DQ_ACTUATED,
            DeviceState.EE_XYZ,
            DeviceState.EE_XYZ_VEL,
            DeviceState.EE_QUAT,
            DeviceState.FORCE,
            DeviceState.TORQUE,
        ]

    @property
    def name(self):
        return self._name

    @property
    def all_joint_ids(self):
        return self._all_joint_ids

    @property
    def ctrlr_dof(self):
        return self._ctrlr_dof

    @property
    def ctrlr_dof_xyz(self):
        return self._ctrlr_dof_xyz

    @property
    def ctrlr_dof_abg(self):
        return self._ctrlr_dof_abg

    @property
    def max_vel(self):
        return self._max_vel

    @property
    def ctrl_idxs(self):
        return self._ctrl_idxs

    @property
    def actuator_trnids(self):
        return self._actuator_trnids

    def reset_start_angles(self):
        if self._start_angles is not None:
            qpos_data = self._data.qpos

            if self._name in ["ur5right", "ur5left", "base"]:
                qpos_data[self._joint_ids] = np.copy(self._start_angles)

            mujoco.mj_step(self._model, self._data)  # Perform a simulation step

    def __get_jacobian(self, full=False):
        """
        NOTE: Returns either:
        1) The full jacobian (of the Device, using its EE), if full==True
        2) The full jacobian evaluated at the controlled DoF, if full==False
        The parameter, full=False, is added in case we decide for the get methods
        to take in arguments (currently not supported).
        """
        J = np.array([])
        EE_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._EE)
        J = np.zeros((3, self._model.nv))
        Jr = np.zeros((3, self._model.nv))
        mujoco.mj_jacBody(self._model, self._data, jacp=J, jacr=Jr, body=EE_id)
        J = np.vstack([J, Jr]) if J.size else Jr
        if full == False:
            J = J[self._ctrlr_dof]
        return J

    def __get_R(self):
        """
        Get rotation matrix for device's ft_frame
        """
        if self._name == "ur5right":
            site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "ft_frame_ur5right")
        if self._name == "ur5left":
            site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "ft_frame_ur5left")

        xmat = self._data.site_xmat[site_id].reshape(3, 3)
        return xmat

    def __get_force(self):
        """
        Get the external forces, used (for admittance control) acting upon
        the gripper sensors
        """
        if self._name == "ur5right":
            force = np.matmul(self.__get_R(), self._data.sensordata[0:3])
            return force
        if self._name == "ur5left":
            force = np.matmul(self.__get_R(), self._data.sensordata[6:9])
            return force
        else:
            return np.zeros(3)

    def __get_torque(self):
        """
        Get the external torques, used (for admittance control) acting upon
        the gripper sensors
        """
        if self._name == "ur5right":
            force = np.matmul(self.__get_R(), self._data.sensordata[3:6])
            return force
        if self._name == "ur5left":
            force = np.matmul(self.__get_R(), self._data.sensordata[9:12])
            return force
        else:
            return np.zeros(3)

    def __set_state(self, state_var: DeviceState):
        """
        Set the state of the device corresponding to the key value (if exists)
        """
        assert self.__use_sim is False
        self.__state_locks[state_var].acquire()

        ########################################
        # NOTE: This is only a placeholder for non-simulated/mujoco devices
        # To get a realtime value, you'd need to communicate with a device driver/api
        # Then set the var_value to what the device driver/api returns
        var_func = self.__state_var_map[state_var]
        var_value = var_func()
        ########################################

        # (for Mujoco) Make sure to copy (or else reference will stick to Dict value)
        self.__state[state_var] = copy.copy(var_value)
        self.__state_locks[state_var].release()

    def get_state(self, state_var: DeviceState):
        """
        Get the state of the device corresponding to the key value (if exists)
        """
        if self.__use_sim:
            func = self.__state_var_map[state_var]
            state = copy.copy(func())
        else:
            self.__state_locks[state_var].acquire()
            state = copy.copy(self.__state[state_var])
            self.__state_locks[state_var].release()
        return state

    def get_all_states(self):
        return dict([(key, self.get_state(key)) for key in self._concise_state_vars])

    def update_state(self):
        """
        This should running in a thread: Robot.start()
        """
        assert self.__use_sim is False
        for var in DeviceState:
            self.__set_state(var)
