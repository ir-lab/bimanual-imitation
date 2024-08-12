import logging

import numpy as np
from gymnasium import envs, spaces

from bimanual_imitation.algorithms.core.shared import MDP, ContinuousSpace, FiniteSpace, Simulation

logging.getLogger("gym.core").addHandler(logging.NullHandler())


class RLGymSim(Simulation):
    def __init__(self, env_name):
        # TODO: make environment with kwargs
        # For now, GAIL/DAGGER don't need action chunking kwargs
        self.env = envs.make(env_name)
        self.action_space = self.env.action_space
        self._curr_obs = None
        self._is_done = False

    def step(self, action):
        if isinstance(self.action_space, spaces.Discrete):
            # We encode actions in finite spaces as an integer inside a length-1 array
            # but Gym wants the integer itself
            assert action.ndim == 1 and action.size == 1 and action.dtype in (np.int32, np.int64)
            action = action[0]
        else:
            assert action.ndim == 1 and action.dtype == np.float64

        self._curr_obs, reward, self._is_done, truncated, info = self.env.step(action)
        return reward

    @property
    def obs(self):
        if self._curr_obs is None:
            self._curr_obs, _reset_info = self.env.reset()
        return self._curr_obs.copy()

    @property
    def done(self):
        return self._is_done

    def reset(self):
        self._curr_obs, _reset_info = self.env.reset()
        self._is_done = False
        return self._curr_obs.copy()


def _convert_space(space):
    """Converts a rl-gym space to our own space representation"""
    if isinstance(space, spaces.Box):
        assert space.low.ndim == 1 and space.low.shape[0] >= 1
        return ContinuousSpace(dim=space.low.shape[0])
    elif isinstance(space, spaces.Discrete):
        return FiniteSpace(size=space.n)
    raise NotImplementedError(space)


class RLGymMDP(MDP):
    def __init__(self, env_name):
        # print("Gym version:", gym.version.VERSION)
        self.env_name = env_name

        tmpsim = self.new_sim()
        self._obs_space = _convert_space(tmpsim.env.observation_space)
        self._action_space = _convert_space(tmpsim.env.action_space)
        self.env_spec = tmpsim.env.spec
        self.gym_env = tmpsim.env

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def new_sim(self, init_state=None):
        assert init_state is None
        return RLGymSim(self.env_name)
