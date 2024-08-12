import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from irl_environments.constants import IRL_ENVIRONMENTS_BASE_DIR
from irl_environments.data_recorders.path_3D_data_recorder import Path3DDataRecorder
from irl_environments.path_envs.path_3D_env import Path3DEnv


class BasePath3DEnv(Path3DDataRecorder, Path3DEnv, gym.Env):
    def __init__(self, render):
        self.__render = render
        self.__yaml_param_file = IRL_ENVIRONMENTS_BASE_DIR / f"param/{self.env_name}.yaml"
        self.__debug_mode = False

        # TODO: Calculate action/obs sizes
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)
        self.action_space = Box(low=-100, high=100, shape=(0,), dtype=np.float32)

    # This is required by the DataRecorder to animate the path 2D Environment
    def get_course_idx_and_state(self):
        return self.path_env_get_course_idx_and_state()

    def load_new_env(self, seed=None):
        if seed is not None:
            raise NotImplementedError

        Path3DEnv.__init__(self, expert_proto=None)
        gym.Env.__init__(self)
        Path3DDataRecorder.__init__(self, self.__render)

    @property
    def env_name(self):
        return "Path_3D"

    @property
    def yaml_param_file(self):
        return self.__yaml_param_file

    @property
    def debug_mode(self):
        return self.__debug_mode


if __name__ == "__main__":
    env = gym.make("path_3d_v1", render=True)
    env.reset()
    env.run_sequence()
