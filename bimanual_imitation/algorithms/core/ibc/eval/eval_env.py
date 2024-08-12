# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loads an eval environment based on task name."""
import functools
from typing import Optional

import gym
import gymnasium
from absl import flags
from gym.utils import seeding
from tf_agents.environments import parallel_py_environment  # pylint: disable=g-import-not-at-top
from tf_agents.environments import suite_gym, wrappers
from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper

from bimanual_imitation.algorithms.core.ibc import tasks

flags.DEFINE_bool(
    "eval_env_2dboard_use_normalized_env",
    False,
    "If true, load the normalized version of the environment "
    "(requires different demonstration tfrecords).",
)

FLAGS = flags.FLAGS


class GymnasiumGymEnvWrapper(PyEnvironmentBaseWrapper):
    _np_random: Optional[seeding.RandomNumberGenerator] = None

    def __init__(self, env: gymnasium.Env):
        super(GymnasiumGymEnvWrapper, self).__init__(env)

    @property
    def np_random(self) -> seeding.RandomNumberGenerator:
        """Returns the environment's internal :attr:`_np_random` that
        if not set will initialise with a random seed."""
        if self._np_random is None:
            self._np_random, seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: seeding.RandomNumberGenerator):
        self._np_random = value

    def _reset(self):
        obs, _reset_info = self._env.reset()
        return obs

    @property
    def observation_space(self):
        gymnasium_space = self._env.observation_space
        if isinstance(gymnasium_space, gymnasium.spaces.Box):
            gym_space = gym.spaces.Box(
                low=gymnasium_space.low.flatten(),
                high=gymnasium_space.high.flatten(),
                dtype=gymnasium_space.dtype,
            )
        else:
            raise NotImplementedError

        return gym_space

    @property
    def action_space(self):
        gymnasium_space = self._env.action_space
        if isinstance(gymnasium_space, gymnasium.spaces.Box):
            gym_space = gym.spaces.Box(
                low=gymnasium_space.low.flatten(),
                high=gymnasium_space.high.flatten(),
                dtype=gymnasium_space.dtype,
            )
        else:
            raise NotImplementedError

        return gym_space

    def seed(self, seed):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]


def get_env_name(task, shared_memory_eval, use_image_obs=False):
    """Returns environment name for a given task."""
    if task in ["REACH", "PUSH", "INSERT", "REACH_NORMALIZED", "PUSH_NORMALIZED"]:
        # from ibc.environments.block_pushing import block_pushing

        # env_name = block_pushing.build_env_name(
        #     task, shared_memory_eval, use_image_obs=use_image_obs
        # )
        raise NotImplementedError
    elif task in ["PUSH_DISCONTINUOUS"]:
        # from ibc.environments.block_pushing import block_pushing_discontinuous

        # env_name = block_pushing_discontinuous.build_env_name(
        #     task, shared_memory_eval, use_image_obs=use_image_obs
        # )
        raise NotImplementedError
    elif task in ["PUSH_MULTIMODAL"]:
        # from ibc.environments.block_pushing import block_pushing_multimodal

        # env_name = block_pushing_multimodal.build_env_name(
        #     task, shared_memory_eval, use_image_obs=use_image_obs
        # )
        raise NotImplementedError
    elif task == "PARTICLE":
        env_name = "Particle-v0"
        assert not shared_memory_eval  # Not supported.
        assert not use_image_obs  # Not supported.
    elif task in tasks.D4RL_TASKS or task in tasks.GYM_TASKS:
        env_name = task
        assert not use_image_obs  # Not supported.
    else:
        raise ValueError("unknown task %s" % task)
    return env_name


def get_eval_env(env_name, sequence_length, goal_tolerance, num_envs, gym_kwargs={}):
    """Returns an eval environment for the given task."""
    if env_name in tasks.D4RL_TASKS:
        # try:
        #     from ibc.ibc.eval import d4rl_utils
        # except:
        #     print("WARNING: Could not import d4rl.")
        # load_env_fn = d4rl_utils.load_d4rl
        raise NotImplementedError
    else:
        load_env_fn = lambda env_name: suite_gym.load(
            env_name, gym_kwargs=gym_kwargs, gym_env_wrappers=[GymnasiumGymEnvWrapper]
        )

    if num_envs > 1:

        def load_env_and_wrap(env_name):
            eval_env = load_env_fn(env_name)
            eval_env = wrappers.HistoryWrapper(
                eval_env, history_length=sequence_length, tile_first_step_obs=True
            )
            return eval_env

        env_ctor = functools.partial(load_env_and_wrap, env_name)
        eval_env = parallel_py_environment.ParallelPyEnvironment(
            [env_ctor] * num_envs, start_serially=False
        )
    else:
        eval_env = load_env_fn(env_name)
        if env_name not in tasks.D4RL_TASKS and "Block" in env_name:
            eval_env.set_goal_dist_tolerance(goal_tolerance)
        eval_env = wrappers.HistoryWrapper(
            eval_env, history_length=sequence_length, tile_first_step_obs=True
        )

    return eval_env
