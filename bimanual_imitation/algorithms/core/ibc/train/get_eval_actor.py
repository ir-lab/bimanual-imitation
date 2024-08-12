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

"""Defines an eval actor."""
import os

import numpy as np
from tf_agents.train import actor
from tf_agents.trajectories import StepType
from tf_agents.trajectories.trajectory import Trajectory as TfAgentTrajectory

from bimanual_imitation.algorithms.core.ibc import tasks
from bimanual_imitation.algorithms.core.ibc.utils import strategy_policy
from irl_data.trajectory import TrajBatch
from irl_data.trajectory import Trajectory as ExportTrajectory


class EvalActor:
    def __init__(self):
        self._traj_obs_hist, self._traj_act_hist, self._traj_reward_hist = [], [], []
        self._trajs = []

    def log_episode_complete(self, tf_traj: TfAgentTrajectory):
        if tf_traj.step_type == StepType.LAST:
            self.write_episode()
        else:
            if isinstance(tf_traj.observation, np.ndarray):
                # Use the last observation due to the gym env history wrapper
                self._traj_obs_hist.append(tf_traj.observation[-1:])
            else:
                raise NotImplementedError(f"{type(tf_traj.observation)} not implemented.")

            self._traj_act_hist.append(tf_traj.action[None, :])
            self._traj_reward_hist.append(tf_traj.reward)

    def write_episode(self):
        export_traj = ExportTrajectory(
            obs_T_Do=np.concatenate(self._traj_obs_hist),
            obsfeat_T_Df=np.concatenate(self._traj_obs_hist),
            a_T_Da=np.concatenate(self._traj_act_hist),
            adist_T_Pa=np.ones((len(self._traj_act_hist), 2)) * np.nan,
            r_T=np.asarray(self._traj_reward_hist),
        )
        self._trajs.append(export_traj)
        self._clear_traj_buffer()

    def _clear_traj_buffer(self):
        self._traj_obs_hist, self._traj_act_hist, self._traj_reward_hist = [], [], []

    def get_most_recent_traj(self):
        return self._trajs[-1]

    def get_trajbatch(self, clear_existing=True):
        trajbatch = TrajBatch.FromTrajs(self._trajs)
        if clear_existing:
            self._trajs = []
        return trajbatch

    def get_eval_actor(
        self,
        agent,
        env_name,
        eval_env,
        train_step,
        eval_episodes,
        root_dir,
        viz_img,
        num_envs,
        strategy,
        summary_dir_suffix="",
    ):
        """Defines eval actor."""
        if num_envs > 1:
            eval_greedy_policy = agent.policy
        else:
            eval_greedy_policy = strategy_policy.StrategyPyTFEagerPolicy(
                agent.policy, strategy=strategy
            )

        metrics = actor.eval_metrics(eval_episodes)
        if env_name in tasks.D4RL_TASKS or env_name in tasks.GYM_TASKS:

            success_metric = metrics[0]
            if env_name in tasks.ADROIT_TASKS:
                from ibc.environments.d4rl import metrics as d4rl_metrics

                # Define custom eval success metric for Adroit tasks, since the rewards
                # include reward shaping terms.
                metrics += [d4rl_metrics.D4RLSuccessMetric(env=eval_env, buffer_size=eval_episodes)]
        else:
            env_metrics, success_metric = eval_env.get_metrics(eval_episodes)
            metrics += env_metrics

        if root_dir is None:
            summary_dir = None
        else:
            summary_dir = os.path.join(root_dir, "eval", summary_dir_suffix)

        observers = []
        # Adds a log when an episode is done, allows seeing eval time in the logs.
        observers += [self.log_episode_complete]

        if viz_img and "Particle" in env_name:
            eval_env.set_img_save_dir(summary_dir)
            observers += [eval_env.save_image]

        eval_actor = actor.Actor(
            eval_env,
            eval_greedy_policy,
            train_step,
            observers=observers,
            metrics=metrics,
            summary_dir=summary_dir,
            episodes_per_run=1,  # we are doing seeding, need to handle ourselves.
            summary_interval=-1,
        )  # -1 will make so never automatically writes.
        return eval_actor, success_metric

    def get_checkpoint_eval_actor(
        self,
        policy,
        env_name,
        eval_env,
        train_step,
        eval_episodes,
        root_dir,
        viz_img,
        num_envs,
        strategy,
        summary_dir_suffix="",
    ):
        """Defines eval actor."""

        eval_greedy_policy = policy

        metrics = actor.eval_metrics(eval_episodes)
        if env_name in tasks.D4RL_TASKS or env_name in tasks.GYM_TASKS:
            success_metric = metrics[0]
            if env_name in tasks.ADROIT_TASKS:
                from ibc.environments.d4rl import metrics as d4rl_metrics

                # Define custom eval success metric for Adroit tasks, since the rewards
                # include reward shaping terms.
                metrics += [d4rl_metrics.D4RLSuccessMetric(env=eval_env, buffer_size=eval_episodes)]
        else:
            env_metrics, success_metric = eval_env.get_metrics(eval_episodes)
            metrics += env_metrics

        summary_dir = os.path.join(root_dir, "eval", summary_dir_suffix)

        observers = []
        # Adds a log when an episode is done, allows seeing eval time in the logs.
        observers += [self.log_episode_complete]

        if viz_img and "Particle" in env_name:
            eval_env.set_img_save_dir(summary_dir)
            observers += [eval_env.save_image]

        eval_actor = actor.Actor(
            eval_env,
            eval_greedy_policy,
            train_step,
            observers=observers,
            metrics=metrics,
            summary_dir=summary_dir,
            episodes_per_run=1,  # we are doing seeding, need to handle ourselves.
            summary_interval=-1,
        )  # -1 will make so never automatically writes.
        return eval_actor, success_metric
