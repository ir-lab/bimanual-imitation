from collections import OrderedDict

import numpy as np
from transforms3d.euler import euler2mat, euler2quat, quat2euler
from transforms3d.quaternions import mat2quat, qinverse, qmult
from transforms3d.utils import normalized_vector as normed

from irl_control.device import DeviceState
from irl_control.utils.target import Target
from irl_environments.core.utils import (
    ActionGroup,
    ObservationGroup,
    PursuitType,
    cart2polar,
    get_action_group_dim,
    get_observation_group_dim,
    quat2sd,
    sd2quat,
)
from irl_environments.path_envs.base_path_env import PathEnv
from irl_environments.runners.bimanual_runner import BimanualRunner


class BimanualPathEnv(BimanualRunner, PathEnv):
    def __init__(self, expert_proto, randomize_track_length=False):
        BimanualRunner.__init__(self, expert_proto, randomize_track_length)
        PathEnv.__init__(self)

        self.__min_T = self.get_expert_param("min_T")
        self.__integral_state = dict([(key, np.zeros(3)) for key in self.keys])
        self.__differential_state = dict([(key, np.zeros(3)) for key in self.keys])
        self.__p_gain = self.get_expert_param("path_pursuit.p_gain")
        self.__i_gain = self.get_expert_param("path_pursuit.i_gain")
        self.__d_gain = self.get_expert_param("path_pursuit.d_gain")
        self.__i_min = self.get_expert_param("path_pursuit.i_min")
        self.__i_max = self.get_expert_param("path_pursuit.i_max")

        # Storage for grip_force/torque_ewma features
        self.__grip_forces_ewma = dict([(key, None) for key in self.keys])
        self.__grip_torques_ewma = dict([(key, None) for key in self.keys])

    def exec_reset(self, mj_reset_func):
        targets = self.get_initial_targets()
        grip_ctrl = self.get_grip_ctrl("reset")

        robot_state = mj_reset_func(
            targets,
            reset_devices=self.get_all_devices(),
            gripper_ctrl=grip_ctrl,
            device_markers=None,
        )

        for key in self.keys:
            track = self.tracks[key]
            pos = robot_state[track.device_name][DeviceState.EE_XYZ]
            quat = robot_state[track.device_name][DeviceState.EE_QUAT]
            v_all = robot_state[track.device_name][DeviceState.EE_XYZ_VEL]
            v = np.linalg.norm(v_all)
            track.state.update(*pos, *quat, v)

    def exec_constrain_action(self, action):
        action_idx = 0
        constrained_action = np.copy(action)
        for key in self.keys:
            track = self.tracks[key]
            for action_group in track.action_space:
                action_len = get_action_group_dim(action_group)
                if action_group == ActionGroup.DELTA_POSITION:
                    if track.enforce_delta_xyz_bounds:
                        dx_bounds, dy_bounds, dz_bounds = track.delta_xyz_bounds
                        constrained_action[action_idx] = np.clip(action[action_idx], *dx_bounds)
                        constrained_action[action_idx + 1] = np.clip(
                            action[action_idx + 1], *dy_bounds
                        )
                        constrained_action[action_idx + 2] = np.clip(
                            action[action_idx + 2], *dz_bounds
                        )
                elif action_group == ActionGroup.DELTA_SIX_DOF:
                    # DO NOTHING
                    pass
                elif action_group == ActionGroup.DELTA_QUAT:
                    constrained_action[action_idx : action_idx + action_len] = normed(
                        action[action_idx : action_idx + action_len]
                    )
                elif action_group == ActionGroup.DELTA_EULER:
                    # DO NOTHING
                    pass
                else:
                    raise NotImplementedError
                action_idx += action_len
        return constrained_action

    def exec_update_states(self, action, time, dt, sim_step_func):
        if self.debug_mode:
            print("################### exec update states ###################")
        targets: OrderedDict = {}
        action_idx = 0
        for key in self.keys:
            track = self.tracks[key]

            # Set the targets to the default position
            targets[key] = Target()
            default_pos = self.default_start_pt[key][:3]
            targets[key].set_xyz(default_pos)
            default_quat = self.default_start_pt[key][3:7]
            targets[key].set_quat(default_quat)
            if self.debug_mode:
                print(f"{key} action space: {track.action_space}")

            for action_group in track.action_space:
                action_len = get_action_group_dim(action_group)
                if action_group == ActionGroup.DELTA_POSITION:
                    x = track.state.x + dt * action[action_idx]
                    y = track.state.y + dt * action[action_idx + 1]
                    z = track.state.z + dt * action[action_idx + 2]
                    targets[key].set_xyz([x, y, z])
                elif action_group == ActionGroup.DELTA_SIX_DOF:
                    if not np.any(action[action_idx : action_idx + action_len]):
                        dquat = normed(np.random.normal(size=4))
                        print("Warning: All Zeros (Delta SixDoF)")
                    else:
                        dquat = sd2quat(action[action_idx : action_idx + action_len])
                    target_quat = qmult(dquat, track.state.quat)
                    targets[key].set_quat(target_quat)
                elif action_group == ActionGroup.DELTA_QUAT:
                    dquat = action[action_idx : action_idx + action_len]
                    target_quat = qmult(dquat, track.state.quat)
                    targets[key].set_quat(target_quat)
                elif action_group == ActionGroup.DELTA_EULER:
                    deuler = action[action_idx : action_idx + action_len]
                    dquat = np.array(euler2quat(*deuler))
                    target_quat = qmult(dquat, track.state.quat)
                    targets[key].set_quat(target_quat)
                else:
                    raise NotImplementedError
                if self.debug_mode:
                    print(
                        f"Assigning {action_group} for {key} : {action[action_idx:action_idx+action_len]}"
                    )
                action_idx += action_len

        grip_ctrl = self.get_grip_ctrl("step")
        sim_err, robot_state = sim_step_func(targets, gripper_ctrl=grip_ctrl)
        out_of_bounds = False
        for key in self.keys:
            track = self.tracks[key]
            if track.enforce_xyz_bounds:
                out_of_bounds = out_of_bounds or targets[track.device_name].check_ob(
                    *track.xyz_bounds, set=True
                )
        for key in self.keys:
            track = self.tracks[key]
            pos = robot_state[track.device_name][DeviceState.EE_XYZ]
            quat = robot_state[track.device_name][DeviceState.EE_QUAT]
            v_all = robot_state[track.device_name][DeviceState.EE_XYZ_VEL]
            v = np.linalg.norm(v_all)
            if not sim_err:
                # Update the state of the track to the robot's resulting position/rotation/velocity
                track.state.update(*pos, *quat, v)
        time += dt
        return sim_err, out_of_bounds, time

    def exec_obs(self, mj_obs_func):
        if self.debug_mode:
            print("################### exec obs ###################")
        state = mj_obs_func()
        observations = []
        obs_noise_bool = self.get_obs_noise_bool()

        for key in self.keys:
            track = self.tracks[key]
            if self.debug_mode:
                print(f"{key} observation space: {track.observation_space}")
            for observation_group in track.observation_space:
                if observation_group == ObservationGroup.POSITION:
                    pos = state[track.device_name][DeviceState.EE_XYZ]
                    pos = self._apply_pos_noise(pos, track.observation_noise, obs_noise_bool)
                    observations.append(pos)
                elif observation_group == ObservationGroup.DELTA_TARGET_POS:
                    pos = state[track.device_name][DeviceState.EE_XYZ]
                    pos = self._apply_pos_noise(pos, track.observation_noise, obs_noise_bool)
                    dtarget_pos = track.course.pos[self.last_course_idx] - pos
                    observations.append(dtarget_pos)
                elif observation_group == ObservationGroup.MALE_OBJ_POS:
                    pos = self.get_obj_pos("male")
                    pos = self._apply_pos_noise(pos, track.observation_noise, obs_noise_bool)
                    observations.append(pos)
                elif observation_group == ObservationGroup.MALE_OBJ_SIX_DOF:
                    quat = self.get_obj_quat("male")
                    quat = self._apply_quat_noise(quat, track.observation_noise, obs_noise_bool)
                    obs = quat2sd(quat)
                    observations.append(obs)
                elif observation_group == ObservationGroup.FEMALE_OBJ_POS:
                    pos = self.get_obj_pos("female")
                    pos = self._apply_pos_noise(pos, track.observation_noise, obs_noise_bool)
                    observations.append(pos)
                elif observation_group == ObservationGroup.FEMALE_OBJ_SIX_DOF:
                    quat = self.get_obj_quat("female")
                    quat = self._apply_quat_noise(quat, track.observation_noise, obs_noise_bool)
                    obs = quat2sd(quat)
                    observations.append(obs)
                elif observation_group == ObservationGroup.DELTA_OBJS_POS:
                    pos = self.get_delta_objs_pos()
                    pos = self._apply_pos_noise(pos, track.observation_noise, obs_noise_bool)
                    observations.append(pos)
                elif observation_group == ObservationGroup.DELTA_OBJS_SIX_DOF:
                    dquat = self.get_delta_objs_quat()
                    dquat = self._apply_quat_noise(dquat, track.observation_noise, obs_noise_bool)
                    obs = quat2sd(dquat)
                    observations.append(obs)
                elif observation_group == ObservationGroup.DELTA_POS_QUAD_PEG_LEFT:
                    dpos_peg_left = self.get_delta_peg_pos("left")
                    dpos_peg_left = self._apply_pos_noise(
                        dpos_peg_left, track.observation_noise, obs_noise_bool
                    )
                    observations.append(dpos_peg_left)
                elif observation_group == ObservationGroup.DELTA_POS_QUAD_PEG_LEFT_CBRT:
                    dpos_peg_left = np.cbrt(self.get_delta_peg_pos("left"))
                    dpos_peg_left = self._apply_pos_noise(
                        dpos_peg_left, track.observation_noise, obs_noise_bool
                    )
                    observations.append(dpos_peg_left)
                elif observation_group == ObservationGroup.DELTA_POS_QUAD_PEG_FRONT_LEFT_CBRT:
                    dpos_peg_front_left = np.cbrt(self.get_delta_peg_pos("front_left"))
                    dpos_peg_front_left = self._apply_pos_noise(
                        dpos_peg_front_left, track.observation_noise, obs_noise_bool
                    )
                    observations.append(dpos_peg_front_left)
                elif observation_group == ObservationGroup.DELTA_POS_QUAD_PEG_LEFT_POLAR:
                    dpos_peg_left = self.get_delta_peg_pos("left")
                    dpos_peg_left = self._apply_pos_noise(
                        dpos_peg_left, track.observation_noise, obs_noise_bool
                    )
                    dpos_peg_left_polar = cart2polar(dpos_peg_left)
                    observations.append(dpos_peg_left_polar)
                elif observation_group == ObservationGroup.DELTA_POS_QUAD_PEG_RIGHT:
                    dpos_peg_right = self.get_delta_peg_pos("right")
                    dpos_peg_right = self._apply_pos_noise(
                        dpos_peg_right, track.observation_noise, obs_noise_bool
                    )
                    observations.append(dpos_peg_right)
                elif observation_group == ObservationGroup.DELTA_POS_QUAD_PEG_RIGHT_CBRT:
                    dpos_peg_right = np.cbrt(self.get_delta_peg_pos("right"))
                    dpos_peg_right = self._apply_pos_noise(
                        dpos_peg_right, track.observation_noise, obs_noise_bool
                    )
                    observations.append(dpos_peg_right)
                elif observation_group == ObservationGroup.DELTA_POS_QUAD_PEG_FRONT_RIGHT_CBRT:
                    dpos_peg_front_right = np.cbrt(self.get_delta_peg_pos("front_right"))
                    dpos_peg_front_right = self._apply_pos_noise(
                        dpos_peg_front_right, track.observation_noise, obs_noise_bool
                    )
                    observations.append(dpos_peg_front_right)
                elif observation_group == ObservationGroup.SIX_DOF:
                    quat = np.asarray(state[track.device_name][DeviceState.EE_QUAT])
                    quat = self._apply_quat_noise(quat, track.observation_noise, obs_noise_bool)
                    sd = quat2sd(quat)
                    observations.append(sd)
                elif observation_group == ObservationGroup.QUAT:
                    quat = np.asarray(state[track.device_name][DeviceState.EE_QUAT])
                    quat = self._apply_quat_noise(quat, track.observation_noise, obs_noise_bool)
                    observations.append(quat)
                elif observation_group == ObservationGroup.EULER:
                    quat = np.asarray(state[track.device_name][DeviceState.EE_QUAT])
                    quat = self._apply_quat_noise(quat, track.observation_noise, obs_noise_bool)
                    euler = np.array(quat2euler(quat))
                    observations.append(euler)
                elif observation_group == ObservationGroup.TARGET_SIX_DOF:
                    quat = track.course.quat[self.last_course_idx]
                    quat = self._apply_quat_noise(quat, track.observation_noise, obs_noise_bool)
                    target_sd = quat2sd(quat)
                    observations.append(target_sd)
                elif observation_group == ObservationGroup.DELTA_TARGET_SIX_DOF:
                    quat = np.asarray(state[track.device_name][DeviceState.EE_QUAT])
                    quat = self._apply_quat_noise(quat, track.observation_noise, obs_noise_bool)
                    target_quat = track.course.quat[self.last_course_idx]
                    dquat = qmult(target_quat, qinverse(quat))
                    dsd = quat2sd(dquat)
                    observations.append(dsd)
                elif observation_group == ObservationGroup.TARGET_QUAT:
                    target_quat = track.course.quat[self.last_course_idx]
                    target_quat = self._apply_quat_noise(
                        target_quat, track.observation_noise, obs_noise_bool
                    )
                    observations.append(target_quat)
                elif observation_group == ObservationGroup.DELTA_TARGET_QUAT:
                    quat = np.asarray(state[track.device_name][DeviceState.EE_QUAT])
                    quat = self._apply_quat_noise(quat, track.observation_noise, obs_noise_bool)
                    target_quat = track.course.quat[self.last_course_idx]
                    dquat = qmult(target_quat, qinverse(quat))
                    observations.append(dquat)
                elif observation_group == ObservationGroup.GRIP_FORCE:
                    gforce_xyz = state[track.device_name][DeviceState.FORCE]
                    # Force/Torques are already very noisy! No need to add more noise
                    observations.append(gforce_xyz)
                elif observation_group == ObservationGroup.GRIP_FORCE_EWMA:
                    gforce_xyz = state[track.device_name][DeviceState.FORCE]
                    gforce_xyz = self._apply_pos_noise(
                        gforce_xyz, track.observation_noise, obs_noise_bool, prefix="force"
                    )
                    if self.__grip_forces_ewma[track.device_name] is None:
                        gforce_ewma = gforce_xyz
                    else:
                        gforce_ewma = track.ewma_alpha * gforce_xyz + (1.0 - track.ewma_alpha) * (
                            self.__grip_forces_ewma[track.device_name]
                        )

                    self.__grip_forces_ewma[track.device_name] = gforce_ewma
                    observations.append(gforce_ewma)
                elif observation_group == ObservationGroup.GRIP_TORQUE:
                    gtorque_xyz = state[track.device_name][DeviceState.TORQUE]
                    # Force/Torques are already very noisy! No need to add more noise
                    observations.append(gtorque_xyz)
                elif observation_group == ObservationGroup.GRIP_TORQUE_EWMA:
                    gtorque_xyz = state[track.device_name][DeviceState.TORQUE]
                    gtorque_xyz = self._apply_pos_noise(
                        gtorque_xyz, track.observation_noise, obs_noise_bool, prefix="torque"
                    )
                    if self.__grip_torques_ewma[track.device_name] is None:
                        gtorque_ewma = gtorque_xyz
                    else:
                        gtorque_ewma = track.ewma_alpha * gtorque_xyz + (1.0 - track.ewma_alpha) * (
                            self.__grip_torques_ewma[track.device_name]
                        )
                    self.__grip_torques_ewma[track.device_name] = gtorque_ewma
                    observations.append(gtorque_ewma)
                else:
                    raise NotImplementedError

                assert len(observations[-1]) == get_observation_group_dim(observation_group)

                if self.debug_mode:
                    print(f"Assigning {observation_group} for {key} : {observations[-1]}")

        return np.concatenate(observations)

    def exec_reward(self, mj_reward_func):
        time_penalty = self.get_path_env_param("reward_time_penalty")
        pos_scale = self.get_path_env_param("reward_pos_scale")
        quat_scale = self.get_path_env_param("reward_quat_scale")
        total_err = 0
        for key in self.keys:
            track = self.tracks[key]
            errs = []
            for action_group in track.action_space:
                target_idx = min(self.expert_target_idx, self.expert_num_samples - 1)
                if action_group == ActionGroup.DELTA_POSITION:
                    diff = track.state.pos - track.course.pos[target_idx]
                    pos_err = np.exp(-pos_scale * (diff**2))
                    errs.append(pos_err)
                elif action_group in track.orientation_groups:
                    dquat = qmult(track.course.quat[target_idx], qinverse(track.state.quat))
                    quat_err = np.exp(-quat_scale * (dquat[1:]) ** 2)
                    errs.append(quat_err)
                else:
                    raise NotImplementedError

            err = np.concatenate(errs).mean() + time_penalty if len(errs) > 0 else 0
            total_err += err

        return total_err

    def exec_done(self, time, mj_done_func):
        pos_tol = self.get_path_env_param("done_pos_tol")
        q_tol = self.get_path_env_param("done_quat_tol")
        done_vec = []
        for key in self.keys:
            track = self.tracks[key]
            for action_group in track.action_space:
                if action_group == ActionGroup.DELTA_POSITION:
                    pos_diff = track.state.calc_distance(*track.course.pos[self.last_course_idx])
                    done_vec.append(pos_diff < pos_tol)
                elif action_group in track.orientation_groups:
                    q_diff = np.array(
                        qmult(track.state.quat, qinverse(track.course.quat[self.last_course_idx]))
                    )
                    done_vec.append(np.linalg.norm(q_diff[1:]) < q_tol)
                else:
                    raise NotImplementedError
        done = np.all(done_vec) and time > self.__min_T
        return done

    def simple_pursuit_expert(self):
        ind = self.expert_target_idx
        ctrl = []
        action_noise_bool = self.get_action_noise_bool()
        for key in self.keys:
            track = self.tracks[key]
            trajectory = track.course
            state = track.state

            if ind < self.last_course_idx:
                target = trajectory.pos[ind]
            else:
                target = trajectory.pos[self.last_course_idx]
                ind = self.last_course_idx

            for action_group in track.action_space:
                if action_group == ActionGroup.DELTA_POSITION:
                    error = target - state.pos
                    self.__integral_state[key] += error

                    gt_idxs = self.__integral_state[key] > self.__i_max
                    if np.any(gt_idxs):
                        self.__integral_state[key][gt_idxs] = [self.__i_max] * np.argwhere(
                            gt_idxs
                        ).size

                    lt_idxs = self.__integral_state[key] < self.__i_min
                    if np.any(lt_idxs):
                        self.__integral_state[key][lt_idxs] = [self.__i_min] * np.argwhere(
                            lt_idxs
                        ).size

                    dpos = (
                        error * self.__p_gain
                        + self.__integral_state[key] * self.__i_gain
                        + ((error - self.__differential_state[key]) * self.__d_gain)
                    )
                    self.__differential_state[key] = error
                    # dpos = normed(dpos)*track.expert_target_speed
                    dpos = self._apply_pos_noise(dpos, track.action_noise, action_noise_bool)
                    ctrl.append(dpos)
                elif action_group == ActionGroup.DELTA_SIX_DOF:
                    dquat = qmult(trajectory.quat[ind], qinverse(state.quat))
                    dquat = self._apply_quat_noise(dquat, track.action_noise, action_noise_bool)
                    dsd = quat2sd(dquat)
                    ctrl.append(dsd)
                elif action_group == ActionGroup.DELTA_QUAT:
                    dquat = qmult(trajectory.quat[ind], qinverse(state.quat))
                    dquat = self._apply_quat_noise(dquat, track.action_noise, action_noise_bool)
                    ctrl.append(dquat)
                elif action_group == ActionGroup.DELTA_EULER:
                    dquat = qmult(trajectory.quat[ind], qinverse(state.quat))
                    dquat = self._apply_quat_noise(dquat, track.action_noise, action_noise_bool)
                    deuler = np.array(quat2euler(dquat))
                    ctrl.append(deuler)
                else:
                    raise NotImplementedError

        ctrl = np.concatenate(ctrl)
        return ctrl

    def _apply_pos_noise(self, dpos, noise_config, noise_bool, prefix=None):
        assert len(dpos) == 3
        x, y, z = dpos

        if prefix is None:
            prefix = ""
        else:
            prefix = f"{prefix}_"

        assert isinstance(noise_bool, bool)
        if noise_bool and noise_config is not None:
            assert isinstance(noise_config, dict)
            if noise_config[f"{prefix}x_scale"] is not None:
                x *= np.random.uniform(*noise_config[f"{prefix}x_scale"])
            if noise_config[f"{prefix}x_bias_m"] is not None:
                x += np.random.uniform(*noise_config[f"{prefix}x_bias_m"])

            if noise_config[f"{prefix}y_scale"] is not None:
                y *= np.random.uniform(*noise_config[f"{prefix}y_scale"])
            if noise_config[f"{prefix}y_bias_m"] is not None:
                y += np.random.uniform(*noise_config[f"{prefix}y_bias_m"])

            if noise_config[f"{prefix}z_scale"] is not None:
                z *= np.random.uniform(*noise_config[f"{prefix}z_scale"])
            if noise_config[f"{prefix}z_bias_m"] is not None:
                z += np.random.uniform(*noise_config[f"{prefix}z_bias_m"])

        dpos_noisy = np.array([x, y, z])
        return dpos_noisy

    def _apply_quat_noise(self, dquat, noise_config, noise_bool):
        assert len(dquat) == 4
        r_roll = 0
        r_pitch = 0
        r_yaw = 0

        assert isinstance(noise_bool, bool)
        if noise_bool and noise_config is not None:
            assert isinstance(noise_config, dict)

            if noise_config["yaw_bias_deg"] is not None:
                r_yaw = np.deg2rad(np.random.uniform(*noise_config["yaw_bias_deg"]))

            if noise_config["pitch_bias_deg"] is not None:
                r_pitch = np.deg2rad(np.random.uniform(*noise_config["pitch_bias_deg"]))

            if noise_config["roll_bias_deg"] is not None:
                r_roll = np.deg2rad(np.random.uniform(*noise_config["roll_bias_deg"]))

            random_quat = mat2quat(euler2mat(r_roll, r_pitch, r_yaw))
            dquat_noisy = qmult(random_quat, dquat)
        else:
            dquat_noisy = dquat

        return dquat_noisy

    def pure_pursuit(self):
        raise NotImplementedError

    def run_pursuit(self):
        if self.pursuit_type == PursuitType.PURE_PURSUIT:
            raise ValueError("Pure Pursuit is unsupported!")
        elif self.pursuit_type == PursuitType.SIMPLE:
            ctrl = self.simple_pursuit_expert()
        else:
            raise ValueError
        return ctrl
