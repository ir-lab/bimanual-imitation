from abc import abstractmethod

import numpy as np

from irl_control.mujoco_gym_app import MujocoGymApp, MujocoGymAppHighFidelity
from irl_environments.path_envs.bimanual_path_env import BimanualPathEnv

FRAME_SKIP = 5 # hardcoded for now (change for better framerate video)

class BimanualMjPathEnv(BimanualPathEnv, MujocoGymApp):
    def __init__(
        self,
        expert_proto,
        scene_file,
        observation_space,
        action_space,
        robot_config_file,
        render_mode="rgb_array",
        randomize_track_length=False,
        gym_app="normal",
        hide_mjpy_warnings=False,
    ):
        BimanualPathEnv.__init__(self, expert_proto, randomize_track_length)
        if gym_app == "normal":
            MujocoGymApp.__init__(
                self,
                robot_config_file,
                scene_file,
                observation_space,
                action_space,
                render_mode=render_mode,
                hide_mjpy_warnings=hide_mjpy_warnings,
            )
        elif gym_app == "high_fidelity":
            MujocoGymAppHighFidelity.__init__(
                self,
                robot_config_file,
                scene_file,
                observation_space,
                action_space,
                render_mode=render_mode,
                hide_mjpy_warnings=hide_mjpy_warnings,
            )
        else:
            raise ValueError(f"Invalid gym_app: {gym_app}")

        self.__frame_idx = 0
        self._rendered_frames = []

    @property
    @abstractmethod
    def runtime_record_gif(self) -> bool:
        raise NotImplementedError

    @property
    def mj_dt(self):
        return self.dt

    def mj_obs_func(self):
        robot_state = self.robot.get_device_states()
        return robot_state

    def mj_update_states_func(self, targets, gripper_ctrl=None):
        ctrlr_output = self.controller.generate(targets)
        ctrl = np.zeros(self.ctrl_action_space.shape)
        for force_idx, force in zip(*ctrlr_output):
            ctrl[force_idx] = force

        if gripper_ctrl is not None:
            for g_idx, g_force in gripper_ctrl:
                ctrl[g_idx] = g_force

        sim_err = False
        try:
            self.do_simulation(ctrl, self.frame_skip)
        except Exception as mj_ex:
            print(f"Sim Error. Mujoco Exception: {mj_ex}")
            sim_err = True

        if not sim_err:
            if self.runtime_record_gif and (self.__frame_idx % FRAME_SKIP == 0):
                self._rendered_frames.append(self.render())
            elif self.render_mode == "human":
                self.render()
            self.__frame_idx += 1

        robot_state = self.robot.get_device_states()
        return sim_err, robot_state

    def dagger_expert_policy_fn(self):
        ctrl = self.simple_pursuit_expert()
        return ctrl
