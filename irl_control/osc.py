from typing import Any, Dict, Tuple

import numpy as np
from transforms3d.derivations.quaternions import qmult
from transforms3d.euler import quat2euler
from transforms3d.quaternions import qinverse

from irl_control.device import Device, DeviceState
from irl_control.robot import Robot, RobotState
from irl_control.utils.target import Target

DUAL_UR5_MASK = [True] * 7 + [False] * 6 + [True] * 6 + [False] * 6
ADMITTANCE_GAIN = 0.01


class ControllerConfig:
    def __init__(self, ctrlr_dict):
        self.ctrlr_dict = ctrlr_dict

    def __getitem__(self, __name: str) -> Any:
        return self.ctrlr_dict[__name]

    def get_params(self, keys):
        return [self.ctrlr_dict[key] for key in keys]

    def __setitem__(self, __name: str, __value: Any) -> None:
        self.ctrlr_dict[__name] = __value


class OSC:
    """
    OSC provides Operational Space Control for a given Robot.
    This controller accepts targets as a input, and generates a control signal
    for the devices that are linked to the targets.
    """

    def __init__(
        self,
        robot: Robot,
        sim,
        input_device_configs: Tuple[str, Dict],
        nullspace_config: Dict = None,
        use_g=True,
        admittance=False,
        default_start_pt=None,
    ):
        self.sim = sim
        self.robot = robot

        # Create a dict, device_configs, which maps a device name to a
        # ControllerConfig. ControllerConfig is a lightweight wrapper
        # around the dict class to add some desired methods
        self.device_configs = dict()
        for dcnf in input_device_configs:
            self.device_configs[dcnf[0]] = ControllerConfig(dcnf[1])
        self.nullspace_config = nullspace_config
        self.use_g = use_g
        self.admittance = admittance
        self.default_start_pt = default_start_pt

        # Obtain the controller configuration parameters
        # and calculate the task space gains
        for device_name in self.device_configs.keys():
            kv, kp, ko = self.device_configs[device_name].get_params(["kv", "kp", "ko"])
            task_space_gains = np.array([kp] * 3 + [ko] * 3)
            self.device_configs[device_name]["task_space_gains"] = task_space_gains
            self.device_configs[device_name]["lamb"] = task_space_gains / kv

    def __Mx(self, J, M):
        """
        Returns the inverse of the task space inertia matrix
        Parameters
        ----------
        J: Jacobian matrix
        M: inertia matrix
        """
        M_inv = self.__svd_solve(M)
        Mx_inv = np.dot(J, np.dot(M_inv, J.T))
        threshold = 1e-4
        if abs(np.linalg.det(Mx_inv)) >= threshold:
            Mx = self.__svd_solve(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=threshold * 0.1)
        return Mx, M_inv

    def __svd_solve(self, A):
        """
        Use the SVD Method to calculate the inverse of a matrix
        Parameters
        ----------
        A: Matrix
        """
        u, s, v = np.linalg.svd(A)
        Ainv = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
        return Ainv

    def __limit_vel(self, u_task: np.ndarray, device: Device):
        """
        Limit the velocity of the task space control vector
        Parameters
        ----------
        u_task: array of length 6 corresponding to the task space control
        """
        if device.max_vel is not None:
            kv, kp, ko, lamb = self.device_configs[device.name].get_params(
                ["kv", "kp", "ko", "lamb"]
            )
            scale = np.ones(6)

            # Apply the sat gains to the x,y,z components
            norm_xyz = np.linalg.norm(u_task[:3])
            sat_gain_xyz = device.max_vel[0] / kp * kv
            scale_xyz = device.max_vel[0] / kp * kv
            if norm_xyz > sat_gain_xyz:
                scale[:3] *= scale_xyz / norm_xyz

            # Apply the sat gains to the a,b,g components
            norm_abg = np.linalg.norm(u_task[3:])
            sat_gain_abg = device.max_vel[1] / ko * kv
            scale_abg = device.max_vel[1] / ko * kv
            if norm_abg > sat_gain_abg:
                scale[3:] *= scale_abg / norm_abg
            u_task = kv * scale * lamb * u_task
        else:
            print("Device max_vel must be set in the yaml file!")
            raise Exception

        return u_task

    def calc_error(self, target: Target, device: Device):
        """
        Compute the difference between the target and device EE
        for the x,y,z and a,b,g components
        """
        u_task = np.zeros(6)
        # Calculate x,y,z error
        if np.sum(device.ctrlr_dof_xyz) > 0:
            diff = device.get_state(DeviceState.EE_XYZ) - target.get_xyz()
            u_task[:3] = diff

        # Calculate a,b,g error
        if np.sum(device.ctrlr_dof_abg) > 0:
            q_r = np.array(
                qmult(device.get_state(DeviceState.EE_QUAT), qinverse(target.get_quat()))
            )
            u_task[3:] = quat2euler(q_r)
        return u_task

    def generate(self, targets: Dict[str, Target]):
        """
        Generate forces for the corresponding devices which are in the
        robot's sub-devices. Accepts a dictionary of device names (keys),
        which map to a Target.
        Parameters
        ----------
        targets: dict of device names mapping to Target objects
        """

        # auto_targets gets rid of the base target for automatic base control
        auto_targets: Dict[str, Target] = dict()
        for device_name in ["ur5right", "ur5left"]:
            if device_name in targets.keys():
                auto_targets[device_name] = targets[device_name]
            else:
                if self.default_start_pt is not None:
                    auto_targets[device_name] = Target()
                    pos = self.default_start_pt[device_name][:3]
                    quat = self.default_start_pt[device_name][3:7]
                    auto_targets[device_name].set_xyz(pos)
                    auto_targets[device_name].set_quat(quat)
                else:
                    print(f"Error: Must Provide a Target Value for {device_name}!")
                    raise ValueError

        targets = auto_targets

        if self.robot.is_using_sim() is False:
            assert self.robot.is_running(), "Robot must be running!"

        robot_state = self.robot.get_all_states()

        # Get the Jacobian for the all of devices passed in
        Js, J_idxs = robot_state[RobotState.J]
        J = np.array([])
        for device_name in targets.keys():
            J = np.vstack([J, Js[device_name]]) if J.size else Js[device_name]

        mask = DUAL_UR5_MASK
        J = J[:, mask]
        M = robot_state[RobotState.M]

        M = M[mask]
        M = M[:, mask]

        # Compute the inverse matrices used for task space operations
        Mx, M_inv = self.__Mx(J, M)

        # Initialize the control vectors and sim data needed for control calculations
        dq = robot_state[RobotState.DQ]
        dq = dq[mask]
        dx = np.dot(J, dq)
        uv_all = np.dot(M, dq)
        u_all = np.zeros(len(self.robot.all_joint_ids[mask]))
        u_task_all = np.array([])
        ext_f = np.array([])

        for device_name, target in targets.items():
            device = self.robot.get_device(device_name)

            # Calculate the error from the device EE to target
            u_task = self.calc_error(target, device)
            stiffness = np.array(self.device_configs[device_name]["k"] + [1] * 3)
            damping = np.array(self.device_configs[device_name]["d"] + [1] * 3)

            # Apply gains to the error terms
            if device.max_vel is not None:
                u_task = self.__limit_vel(u_task, device)
                u_task *= stiffness
            else:
                task_space_gains = self.device_configs[device.name]["task_space_gains"]
                u_task *= task_space_gains * stiffness

            # Apply kv gain
            kv = self.device_configs[device.name]["kv"]
            target_vel = np.hstack([target.get_xyz_vel(), target.get_abg_vel()])
            if np.all(target_vel == 0):
                ist, c1, c2 = np.intersect1d(
                    device.all_joint_ids, self.robot.all_joint_ids[mask], return_indices=True
                )
                u_all[c2] = -1 * kv * uv_all[c2]
            else:
                diff = dx[J_idxs[device_name]] - np.array(target_vel)[device.ctrlr_dof]
                u_task[device.ctrlr_dof] += kv * diff * damping[device.ctrlr_dof]

            force = np.append(
                robot_state[device_name][DeviceState.FORCE],
                robot_state[device_name][DeviceState.TORQUE],
            )
            ext_f = np.append(ext_f, force[device.ctrlr_dof])
            u_task_all = np.append(u_task_all, u_task[device.ctrlr_dof])

        # Transform task space signal to joint space
        if self.admittance is True:
            u_all -= np.dot(J.T, np.dot(Mx, u_task_all + ADMITTANCE_GAIN * ext_f))
        else:
            u_all -= np.dot(J.T, np.dot(Mx, u_task_all))

        # Apply gravity forces
        if self.use_g:
            qfrc_bias = robot_state[RobotState.G]
            u_all += qfrc_bias[self.robot.all_joint_ids[mask]]

        # Apply the nullspace controller using the specified parameters
        # (if passed to constructor / initialized)
        if self.nullspace_config is not None:
            damp_kv = self.nullspace_config["kv"]
            u_null = np.dot(M, -damp_kv * dq)
            Jbar = np.dot(M_inv, np.dot(J.T, Mx))
            null_filter = np.eye(len(self.robot.all_joint_ids[mask])) - np.dot(J.T, Jbar.T)
            u_all += np.dot(null_filter, u_null)

        # Return the forces and indices to apply the forces
        forces = []
        force_idxs = []
        for dev in self.robot.sub_devices:
            ist, c1, c2 = np.intersect1d(
                dev.actuator_trnids, self.robot.all_joint_ids[mask], return_indices=True
            )
            forces.append(u_all[c2])
            ist2, c12, c22 = np.intersect1d(dev.actuator_trnids, ist, return_indices=True)
            force_idxs.append(dev.ctrl_idxs[c22])

        return force_idxs, forces
