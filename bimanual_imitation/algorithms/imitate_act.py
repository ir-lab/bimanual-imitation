import binascii
import os
import random

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from bimanual_imitation.algorithms.abstract_base.imitate_bc_base import ImitateBcBase
from bimanual_imitation.algorithms.configs import ALG, ActParamConfig
from bimanual_imitation.algorithms.core.act.policy import ACTPolicy
from bimanual_imitation.algorithms.core.shared.util import Timer
from bimanual_imitation.utils import compute_dict_mean, detach_dict
from irl_data.data_chunking import create_chunking_dataset


class ImitateAct(ImitateBcBase):

    @property
    def alg(self):
        return ALG.ACT

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def bc_policy(self):
        return self._policy

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def bc_policy_fn(self, obs):
        action = self._policy(obs)
        return action

    def init_params(self, cfg: ActParamConfig):
        random.seed(os.urandom(4))
        seed = int(binascii.hexlify(os.urandom(4)), 16)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self._train_dataset = create_chunking_dataset(
            environment=self.env_name,
            stage="train",
            pred_horizon=cfg.pred_horizon,
            obs_horizon=cfg.obs_horizon,
            action_horizon=cfg.action_horizon,
            limit_trajs=self.limit_trajs,
            normalize=True,
        )

        self._train_dataloader = DataLoader(
            dataset=self._train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            # num_workers=1,
            # prefetch_factor=1,
        )

        tmp_env = self.create_env(verbose=True)
        self._action_dim_single = tmp_env.action_space.shape[0] // cfg.pred_horizon
        self._obs_dim = tmp_env.observation_space.shape[0]
        del tmp_env

        self._policy = ACTPolicy(
            cfg, action_dim_single=self._action_dim_single, obs_dim=self._obs_dim
        )
        self._policy.to(self.device)
        self._policy.train()

        self._optimizer = self._policy.configure_optimizers()

        self._total_time = 0
        self._global_step = 0
        self._epoch = 0
        self._train_history = []
        self.cfg = cfg

    def suggest_hyperparams(self, trial: optuna.Trial):
        param_cfg = ActParamConfig(
            batch_size=trial.suggest_categorical("batch_size", [256, 512]),
            enc_layers=trial.suggest_categorical("enc_layers", [1, 2, 3]),
            dec_layers=trial.suggest_categorical("dec_layers", [1, 2, 3]),
            latent_dim=trial.suggest_categorical("latent_dim", [8, 16]),
            n_heads=trial.suggest_categorical("n_heads", [4, 8]),
            act_lr=trial.suggest_categorical("act_lr", [5e-4, 1e-4, 5e-5]),
            dropout=trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
            hidden_dim=trial.suggest_categorical("hidden_dim", [128, 256]),
            dim_feedforward=trial.suggest_categorical("dim_feedforward", [256, 512]),
            activation=trial.suggest_categorical("activation", ["relu", "gelu"]),
            kl_weight=trial.suggest_categorical("kl_weight", [1, 10, 100]),
        )

        return param_cfg

    def step_optimizer(self):
        self._optimizer.zero_grad()

        with Timer() as t_train:
            for batch_idx, nbatch in enumerate(self._train_dataloader):
                nobs = nbatch["obs"].to(self.device)
                naction = nbatch["action"].to(self.device)
                B = nobs.shape[0]

                obs_cond = nobs[:, : self.train_dataset.obs_horizon, :]
                obs_cond = obs_cond.flatten(start_dim=1)
                is_pad = torch.zeros((B, naction.shape[1])).bool().to(self.device)

                # get loss info
                forward_dict = self._policy(obs_cond, naction, is_pad)

                # backward
                loss = forward_dict["loss"]
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
                self._train_history.append(detach_dict(forward_dict))

                self._global_step += 1

        epoch_summary = compute_dict_mean(
            self._train_history[(batch_idx + 1) * self._epoch : (batch_idx + 1) * (self._epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]

        self._total_time += t_train.dt
        self._epoch += 1

        iter_info = [
            ("iter", self._epoch, int),
            ("loss", epoch_train_loss, float),
            ("ttrain", t_train.dt, float),
            ("ttotal", self._total_time, float),
        ]

        return iter_info

    def get_default_run_options(self):
        act_defaults = {
            "--mode": ("train_policy", str),
            "--max_iter": (100, int),
            "--num_evals": (10, int),
            "--num_rollouts_per_eval": (10, int),
            "--snapshot_save_freq": (10, int),
            "--print_freq": (1, int),
            "--limit_trajs": (200, int),
            "--export_data": (False, bool),
        }

        return act_defaults


if __name__ == "__main__":
    ImitateAct().run()
