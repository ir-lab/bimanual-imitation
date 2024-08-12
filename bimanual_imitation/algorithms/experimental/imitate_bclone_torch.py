import binascii
import os
import random

import numpy as np
import optuna
import torch
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

from bimanual_imitation.algorithms.abstract_base.imitate_bc_base import ImitateBcBase
from bimanual_imitation.algorithms.configs import ALG, BcloneParamConfig
from bimanual_imitation.algorithms.core.shared.util import Timer
from bimanual_imitation.algorithms.experimental.bclone_torch import GaussianPolicy
from irl_data.data_chunking import create_chunking_dataset


class ImitateBclone(ImitateBcBase):

    @property
    def alg(self):
        return ALG.BCLONE

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
        action, action_std = self._policy(obs)

        if not self.cfg.deterministic_eval:
            action = torch.normal(action, action_std)

        return action

    def init_params(self, cfg: BcloneParamConfig):
        random.seed(os.urandom(4))
        seed = int(binascii.hexlify(os.urandom(4)), 16)
        np.random.seed(seed)
        torch.manual_seed(seed)

        pred_horizon = 1
        obs_horizon = 1
        action_horizon = 1

        self._train_dataset = create_chunking_dataset(
            environment=self.env_name,
            stage="train",
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            limit_trajs=self.limit_trajs,
            normalize=bool(cfg.obsnorm_mode == "expertdata"),
        )

        self._train_dataloader = DataLoader(
            dataset=self._train_dataset,
            batch_size=cfg.bclone_batch_size,
            shuffle=True,
            pin_memory=True,
            # num_workers=1,
            # prefetch_factor=1,
        )

        tmp_env = self.create_env()
        obs_dim = tmp_env.observation_space.shape[0]
        action_dim = tmp_env.action_space.shape[0]
        del tmp_env

        # configure model
        self._policy = GaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg.policy_n_units,
            n_layers=cfg.policy_n_layers,
            activation=cfg.policy_layer_type,
        )

        # device transfer
        _ = self._policy.to(self.device)
        self._policy.train()

        self._optimizer = torch.optim.AdamW(params=self._policy.parameters(), lr=cfg.bclone_lr)

        self._lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self._optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self._train_dataloader) * self.max_iter,
        )

        self._total_time = 0
        self._global_step = 0
        self._epoch = 0
        self.cfg = cfg

    def suggest_hyperparams(self, trial: optuna.Trial):
        param_cfg = BcloneParamConfig(
            bclone_lr=trial.suggest_categorical("bclone_lr", [5e-5, 1e-4, 5e-4]),
            policy_n_layers=trial.suggest_int("policy_n_layers", 2, 3),
            policy_n_units=trial.suggest_categorical("policy_n_units", [256, 512]),
            policy_layer_type=trial.suggest_categorical("policy_layer_type", ["relu", "tanh"]),
            bclone_l1_lambda=trial.suggest_categorical("bclone_l1_lambda", [0, 1e-6, 1e-4]),
            bclone_l2_lambda=trial.suggest_categorical("bclone_l2_lambda", [0, 1e-6, 1e-4]),
            bclone_batch_size=trial.suggest_categorical("bclone_batch_size", [128, 256, 512]),
        )

        return param_cfg

    def step_optimizer(self):
        epoch_loss = []
        with Timer() as t_train:
            for batch_idx, nbatch in enumerate(self._train_dataloader):
                nobs = nbatch["obs"].to(self.device)
                naction = nbatch["action"].to(self.device)
                action_cat = naction.flatten(start_dim=1)

                obs_cond = nobs[:, : self.train_dataset.obs_horizon, :]
                obs_cond = obs_cond.flatten(start_dim=1)

                action_pred, action_std = self._policy(obs_cond)

                # compute log likelihood loss
                action_dist = torch.distributions.Normal(action_pred, action_std)
                loss = -action_dist.log_prob(action_cat).mean()

                parameters = []

                for parameter in self._policy.net.parameters():
                    parameters.append(parameter.view(-1))

                l1 = self.cfg.bclone_l1_lambda * torch.abs(torch.cat(parameters)).mean()
                l2 = self.cfg.bclone_l2_lambda * torch.square(torch.cat(parameters)).mean()

                loss += l1 + l2

                # optimize
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
                # step lr scheduler every batch
                self._lr_scheduler.step()

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                self._global_step += 1

        avg_loss = np.mean(epoch_loss)
        self._total_time += t_train.dt
        self._epoch += 1

        iter_info = [
            ("iter", self._epoch, int),
            ("gstep", int(self._global_step), int),
            ("loss", avg_loss, float),
            ("lr", self._lr_scheduler.get_last_lr()[0], float),
            ("ttrain", t_train.dt, float),
            ("ttotal", self._total_time, float),
        ]

        return iter_info

    def get_default_run_options(self):
        bclone_defaults = {
            "--mode": ("train_policy", str),
            "--max_iter": (10, int),
            "--num_evals": (5, int),
            "--num_rollouts_per_eval": (10, int),
            "--snapshot_save_freq": (2, int),
            "--print_freq": (1, int),
            "--limit_trajs": (200, int),
            "--export_data": (False, bool),
        }

        return bclone_defaults


if __name__ == "__main__":
    ImitateBclone().run()
