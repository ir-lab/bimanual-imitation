import time
from collections import OrderedDict
from pathlib import Path

import mujoco
import numpy as np
import yaml
from transforms3d.affines import compose
from transforms3d.euler import quat2euler, quat2mat
from transforms3d.quaternions import qinverse, qmult

from irl_control.device import DeviceState
from irl_control.mujoco_gym_app import MujocoGymAppHighFidelity
from irl_control.utils.target import Target
from irl_environments.core.utils import ActionGroup, PursuitType, quat2sd
from irl_environments.expert_bimanual_quad_insert import BaseQuadInsertionTask
from irl_environments.path_envs.bimanual_mj_path_env import BimanualMjPathEnv

PEG_DX = 0.09
PEG_DY = 0.33
PEG_FRONT_DY = 0.13
PEG_DZ = 0.025

PEG_XY_TOL = 0.002
PEG_Z_TOL = 0.0005


class QuadInsertPathEnv(BimanualMjPathEnv, MujocoGymAppHighFidelity, BaseQuadInsertionTask):
    def __init__(
        self,
        expert_proto,
        scene_file,
        observation_space,
        action_space,
        robot_config_file,
        render_mode="rgb_array",
        randomize_track_length=False,
    ):
        BimanualMjPathEnv.__init__(
            self,
            expert_proto,
            scene_file,
            observation_space,
            action_space,
            robot_config_file,
            render_mode=render_mode,
            randomize_track_length=randomize_track_length,
            gym_app="high_fidelity",
        )

        assert isinstance(expert_proto, Path)
        init_config_filename = expert_proto.parent / f"{expert_proto.stem}.yaml"
        BaseQuadInsertionTask.__init__(self, init_config_filename)

        with open(init_config_filename, "r") as init_config_file:
            mj_obj_config = yaml.safe_load(init_config_file)

        action_objects = "action_objects"
        self.__female_obj = mj_obj_config[action_objects]["female_object"]
        self.__male_obj = mj_obj_config[action_objects]["male_object"]
        self.__done_simple_pursuit = False
        self.__insert_integral_state = dict([(key, np.zeros(3)) for key in self.keys])
        self.__insert_differential_state = dict([(key, np.zeros(3)) for key in self.keys])

        self.__insert_pre_p_gain = np.array(self.get_expert_param("insert_pursuit.pre_p_gain"))
        self.__insert_p_gain = self.get_expert_param("insert_pursuit.p_gain")
        self.__insert_i_gain = self.get_expert_param("insert_pursuit.i_gain")
        self.__insert_d_gain = self.get_expert_param("insert_pursuit.d_gain")
        self.__insert_i_min = self.get_expert_param("insert_pursuit.i_min")
        self.__insert_i_max = self.get_expert_param("insert_pursuit.i_max")

    def insert_pursuit(self):
        # Insert Pursuit has *no action noise* ==> Must be precise
        ctrl = []

        joint_name = self.__female_obj["joint_name"]
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        start_idx = self.model.jnt_qposadr[joint_id]
        female_quat = self.data.qpos[start_idx + 3 : start_idx + 7]

        joint_name = self.__male_obj["joint_name"]
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        start_idx = self.model.jnt_qposadr[joint_id]
        male_quat = self.data.qpos[start_idx + 3 : start_idx + 7]

        action_noise_bool = self.get_action_noise_bool()

        for key in self.keys:
            track = self.tracks[key]
            for action_group in track.action_space:
                if action_group == ActionGroup.DELTA_POSITION:
                    if key == "ur5right":
                        f_high = self.__get_peg_pos(peg_name="right", obj_name="female")
                        m_high = self.__get_peg_pos(peg_name="right", obj_name="male")
                        error = f_high - m_high
                    elif key == "ur5left":
                        f_low = self.__get_peg_pos(peg_name="left", obj_name="female")
                        m_low = self.__get_peg_pos(peg_name="left", obj_name="male")
                        error = f_low - m_low
                    else:
                        raise NotImplementedError

                    error *= self.__insert_pre_p_gain

                    self.__insert_integral_state[key] += error

                    gt_idxs = self.__insert_integral_state[key] > self.__insert_i_max
                    if np.any(gt_idxs):
                        self.__insert_integral_state[key][gt_idxs] = [
                            self.__insert_i_max
                        ] * np.argwhere(gt_idxs).size

                    lt_idxs = self.__insert_integral_state[key] < self.__insert_i_min
                    if np.any(lt_idxs):
                        self.__insert_integral_state[key][lt_idxs] = [
                            self.__insert_i_min
                        ] * np.argwhere(lt_idxs).size

                    dpos = (
                        self.__insert_p_gain * error
                        + self.__insert_integral_state[key] * self.__insert_i_gain
                        + (error - self.__insert_differential_state[key]) * self.__insert_d_gain
                    )
                    self.__insert_differential_state[key] = error
                    dpos = self._apply_pos_noise(dpos, track.action_noise, action_noise_bool)
                    ctrl.append(dpos)
                elif action_group == ActionGroup.DELTA_SIX_DOF:
                    dquat = qmult(female_quat, qinverse(male_quat))
                    dquat = self._apply_quat_noise(dquat, track.action_noise, action_noise_bool)
                    dsd = quat2sd(dquat)
                    ctrl.append(dsd)
                elif action_group == ActionGroup.DELTA_QUAT:
                    dquat = qmult(female_quat, qinverse(male_quat))
                    dquat = self._apply_quat_noise(dquat, track.action_noise, action_noise_bool)
                    ctrl.append(dquat)
                elif action_group == ActionGroup.DELTA_EULER:
                    dquat = qmult(female_quat, qinverse(male_quat))
                    dquat = self._apply_quat_noise(dquat, track.action_noise, action_noise_bool)
                    deuler = np.array(quat2euler(dquat))
                    ctrl.append(deuler)
                else:
                    raise NotImplementedError

        ctrl = np.concatenate(ctrl)
        return ctrl

    def get_action_noise_bool(self):
        # You can use boolean logic here to decide when to apply noise
        # (e.g. based on the current state of the environment)
        # NOTE: Depending on the noise config, there may be no noise applied
        apply_noise = True
        return apply_noise

    def get_obs_noise_bool(self):
        # You can use boolean logic here to decide when to apply noise
        # (e.g. based on the current state of the environment)
        # NOTE: Depending on the noise config, there may be no noise applied
        apply_noise = True
        return apply_noise

    def get_delta_objs_pos(self):
        female_peg_pos = self.get_obj_pos(obj_name="female")
        male_peg_pos = self.get_obj_pos(obj_name="male")
        peg_pos_diff = female_peg_pos - male_peg_pos
        return peg_pos_diff

    def get_delta_objs_quat(self):
        female_quat = self.get_obj_quat("female")
        male_quat = self.get_obj_quat("male")
        dquat = qmult(female_quat, qinverse(male_quat))
        return dquat

    def get_obj_quat(self, obj_name):
        if obj_name == "female":
            joint_name = self.__female_obj["joint_name"]
        elif obj_name == "male":
            joint_name = self.__male_obj["joint_name"]
        else:
            raise ValueError
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        start_idx = self.model.jnt_qposadr[joint_id]
        quat = self.data.qpos[start_idx + 3 : start_idx + 7]
        return quat

    def get_obj_pos(self, obj_name):
        if obj_name == "female":
            joint_name = self.__female_obj["joint_name"]
        elif obj_name == "male":
            joint_name = self.__male_obj["joint_name"]
        else:
            raise ValueError
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        start_idx = self.model.jnt_qposadr[joint_id]
        pos = self.data.qpos[start_idx : start_idx + 3]
        return pos

    def get_delta_peg_pos(self, peg_name):
        female_peg_pos = self.__get_peg_pos(peg_name=peg_name, obj_name="female")
        male_peg_pos = self.__get_peg_pos(peg_name=peg_name, obj_name="male")
        peg_pos_diff = female_peg_pos - male_peg_pos
        return peg_pos_diff

    def __get_peg_pos(self, peg_name, obj_name):
        assert peg_name in ["left", "front_left", "right", "front_right"]
        assert obj_name in ["male", "female"]

        peg_pos = None
        if obj_name == "female":
            joint_name = self.__female_obj["joint_name"]
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            start_idx = self.model.jnt_qposadr[joint_id]
            female_pos = self.data.qpos[start_idx : start_idx + 3]
            female_quat = self.data.qpos[start_idx + 3 : start_idx + 7]
            f1 = compose(female_pos, quat2mat(female_quat), [1, 1, 1])
            if peg_name == "right":
                f2 = compose([-PEG_DX, -PEG_DY, PEG_DZ], np.eye(3), [1, 1, 1])
                f12 = np.matmul(f1, f2)
                f_high = f12[:3, -1].flatten()
                peg_pos = f_high
            elif peg_name == "front_right":
                f2 = compose([PEG_DX, -PEG_FRONT_DY, PEG_DZ], np.eye(3), [1, 1, 1])
                f12 = np.matmul(f1, f2)
                f_high = f12[:3, -1].flatten()
                peg_pos = f_high
            elif peg_name == "left":
                f3 = compose([-PEG_DX, PEG_DY, PEG_DZ], np.eye(3), [1, 1, 1])
                f13 = np.matmul(f1, f3)
                f_low = f13[:3, -1].flatten()
                peg_pos = f_low
            elif peg_name == "front_left":
                f3 = compose([PEG_DX, PEG_FRONT_DY, PEG_DZ], np.eye(3), [1, 1, 1])
                f13 = np.matmul(f1, f3)
                f_low = f13[:3, -1].flatten()
                peg_pos = f_low
            else:
                raise ValueError
        elif obj_name == "male":
            joint_name = self.__male_obj["joint_name"]
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            start_idx = self.model.jnt_qposadr[joint_id]
            male_pos = self.data.qpos[start_idx : start_idx + 3]
            male_quat = self.data.qpos[start_idx + 3 : start_idx + 7]

            m1 = compose(male_pos, quat2mat(male_quat), [1, 1, 1])
            if peg_name == "right":
                m2 = compose([-PEG_DX, -PEG_DY, 0.00], np.eye(3), [1, 1, 1])
                m12 = np.matmul(m1, m2)
                m_high = m12[:3, -1].flatten()
                peg_pos = m_high
            elif peg_name == "front_right":
                m2 = compose([PEG_DX, -PEG_FRONT_DY, 0.00], np.eye(3), [1, 1, 1])
                m12 = np.matmul(m1, m2)
                m_high = m12[:3, -1].flatten()
                peg_pos = m_high
            elif peg_name == "left":
                m3 = compose([-PEG_DX, PEG_DY, 0.00], np.eye(3), [1, 1, 1])
                m13 = np.matmul(m1, m3)
                m_low = m13[:3, -1].flatten()
                peg_pos = m_low
            elif peg_name == "front_left":
                m3 = compose([PEG_DX, PEG_FRONT_DY, 0.00], np.eye(3), [1, 1, 1])
                m13 = np.matmul(m1, m3)
                m_low = m13[:3, -1].flatten()
                peg_pos = m_low
            else:
                raise ValueError
        else:
            raise ValueError

        return peg_pos

    def done_insert_pursuit(self):
        f_high = self.__get_peg_pos(peg_name="right", obj_name="female")
        m_high = self.__get_peg_pos(peg_name="right", obj_name="male")
        high_diffs = np.abs(f_high - m_high)
        done_xh = high_diffs[0] < PEG_XY_TOL
        done_yh = high_diffs[1] < PEG_XY_TOL
        done_zh = high_diffs[2] < PEG_Z_TOL

        f_low = self.__get_peg_pos(peg_name="left", obj_name="female")
        m_low = self.__get_peg_pos(peg_name="left", obj_name="male")
        low_diffs = np.abs(f_low - m_low)
        done_xl = low_diffs[0] < PEG_XY_TOL
        done_yl = low_diffs[1] < PEG_XY_TOL
        done_zl = low_diffs[2] < PEG_Z_TOL

        return np.all([done_xh, done_yh, done_zh, done_xl, done_yl, done_zl])

    def dagger_expert_policy_fn(self):
        return self.run_pursuit()

    def run_pursuit(self):
        if self.pursuit_type == PursuitType.SIMPLE:
            if self.__done_simple_pursuit:
                ctrl = self.insert_pursuit()
            else:
                ctrl = self.simple_pursuit_expert()
        else:
            raise NotImplementedError
        return ctrl

    def exec_done(self, time, mj_done_func):
        done = False
        if not self.__done_simple_pursuit:
            done_simple_pursuit = super().exec_done(time, mj_done_func)
            self.__done_simple_pursuit = done_simple_pursuit or self.done_insert_pursuit()
        else:
            done = self.done_insert_pursuit()
        return done

    def mj_reset_func(
        self,
        targets: "OrderedDict[str, Target]",
        reset_devices,
        gripper_ctrl=None,
        device_markers=None,
    ):

        if device_markers is not None:
            raise NotImplementedError
            # device_markers was created for mujoco-py (below)
            # for marker_name, marker_pos, marker_quat in device_markers:
            #     self.sim.data.set_mocap_pos(marker_name, marker_pos)
            #     self.sim.data.set_mocap_quat(marker_name, marker_quat)

        for device in self.robot.sub_devices:
            device.reset_start_angles()

        self.run_insertion_reset_sequence()

        error = np.inf
        start_time = time.time()
        while (error > 0.01) and (time.time() - start_time < 5.0):
            ctrlr_output = self.controller.generate(targets)
            ctrl = np.zeros(self.ctrl_action_space.shape)
            for force_idx, force in zip(*ctrlr_output):
                ctrl[force_idx] = force

            if gripper_ctrl is not None:
                for g_idx, g_force in gripper_ctrl:
                    ctrl[g_idx] = g_force

            self.do_simulation(ctrl, 1)

            if self.render_mode == "human":
                self.render()
            state = self.robot.get_device_states()
            errs = []
            for device_name in reset_devices:
                pos = state[device_name][DeviceState.EE_XYZ]
                errs.append(pos - targets[device_name].pos)
                quat = state[device_name][DeviceState.EE_QUAT]
                errs.append(qmult(quat, qinverse(targets[device_name].quat))[1:])

            if len(errs) == 0:
                print(
                    'Warning: "reset_pos" and "reset_quat" are set to False for all devices in the Device YAML file!'
                )
                return

            error = np.linalg.norm(np.concatenate(errs))

        robot_state = self.robot.get_device_states()
        return robot_state
