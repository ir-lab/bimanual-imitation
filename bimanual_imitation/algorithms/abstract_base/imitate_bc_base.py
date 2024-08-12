from abc import abstractmethod

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from bimanual_imitation.algorithms.abstract_base.imitate_base import ImitateBase
from irl_data.data_chunking import normalize_data, unnormalize_data
from irl_data.trajectory import TrajBatch, Trajectory


class ImitateBcBase(ImitateBase):

    @property
    @abstractmethod
    def train_dataset(self):
        raise NotImplementedError

    @property
    def bc_policy(self):
        raise NotImplementedError

    def bc_policy_fn(self):
        raise NotImplementedError

    @property
    def device(self):
        # NOTE: torch parent classes should override this
        raise NotImplementedError

    def pre_eval(self):
        self.bc_policy.eval()

    def post_eval(self):
        self.bc_policy.train()

    def get_bc_policy_weights(self):
        # NOTE: Currently implemented for torch
        # Override this function for other frameworks
        return self.bc_policy.state_dict()

    def write_snapshot(self, policy_log, iteration):
        # NOTE: Currently implemented for torch
        # Override this function for other frameworks
        import torch

        ckpt_dir = self.export_dir / "checkpoints"
        if not ckpt_dir.exists():
            ckpt_dir.mkdir()

        ckpt_path = ckpt_dir / f"iteration_{str(iteration).zfill(3)}.ckpt"
        torch.save(self.get_bc_policy_weights(), ckpt_path)

    def create_env(self, verbose=False):
        single_horizon = bool(
            np.all(
                [
                    self.train_dataset.pred_horizon == 1,
                    self.train_dataset.obs_horizon == 1,
                    self.train_dataset.action_horizon == 1,
                ]
            )
        )

        if single_horizon:
            gym_spec = gym.spec(self.env_name + "eval")
        else:
            gym_spec = gym.spec(self.env_name + "eval_chunking")

        env = gym_spec.make(
            pred_horizon=self.train_dataset.pred_horizon,
            obs_horizon=self.train_dataset.obs_horizon,
            action_horizon=self.train_dataset.action_horizon,
            verbose=verbose,
        )
        return env

    def obs_to_chunk(self, raw_obs):
        # NOTE: Currently implemented for torch
        # Override this function for other frameworks
        import torch

        obs_horizon = self.train_dataset.obs_horizon
        obs_stats = self.train_dataset.stats["obs"]
        obs_cond = raw_obs.reshape(obs_horizon, -1)
        obs_cond = normalize_data(obs_cond, obs_stats).flatten()
        obs_cond = torch.from_numpy(obs_cond).to(self.device, dtype=torch.float32)
        obs_cond = obs_cond.unsqueeze(0).flatten(start_dim=1)
        return obs_cond

    def flat_naction(self, naction):
        # NOTE: Currently implemented for torch
        # Override this function for other frameworks
        pred_horizon = self.train_dataset.pred_horizon
        action_stats = self.train_dataset.stats["action"]
        naction = naction.detach().to("cpu").numpy()
        naction = naction[0].reshape(pred_horizon, -1)
        action = unnormalize_data(naction, stats=action_stats)
        action = action.flatten()
        return action

    def eval_rollouts(self):

        def _eval_rollout():
            env = self.create_env()

            all_obs, all_rewards, all_actions = [], [], []

            raw_obs, _reset_info = env.reset()
            all_obs.append(raw_obs)
            obs_cond = self.obs_to_chunk(raw_obs)

            self.pre_eval()

            done = False
            max_steps = env.spec.max_episode_steps
            for step in range(max_steps):
                naction = self.bc_policy_fn(obs_cond)
                action = self.flat_naction(naction)
                raw_obs, reward, done, truncated, info = env.step(action)
                obs_cond = self.obs_to_chunk(raw_obs)

                all_actions.append(action)
                all_rewards.append(reward)

                if done or (step == max_steps - 1):
                    break

                all_obs.append(raw_obs)

            self.post_eval()

            traj = Trajectory(
                obs_T_Do=np.vstack(all_obs),
                obsfeat_T_Df=np.ones((len(all_obs), 1)) * np.nan,
                a_T_Da=np.vstack(all_actions),
                adist_T_Pa=np.ones((len(all_actions), 2)) * np.nan,
                r_T=np.array(all_rewards),
            )
            return traj

        trajs, success, returns = [], [], []
        progress_bar = tqdm(range(self.num_rollouts), desc="Rollouts")
        for idx in progress_bar:
            traj = _eval_rollout()
            trajs.append(traj)
            success.append(int(traj.r_T[-1] > 0))
            returns.append(traj.r_T.sum())
            progress_bar.set_postfix(
                AvgReturn=np.mean(returns),
                PctSuccess=np.mean(success) * 100,
                refresh=False,
            )

        trajbatch = TrajBatch.FromTrajs(trajs)
        return trajbatch
