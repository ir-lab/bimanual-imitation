import binascii
import os
import random
from copy import deepcopy

import numpy as np
import optuna
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader

from bimanual_imitation.algorithms.abstract_base.imitate_bc_base import ImitateBcBase
from bimanual_imitation.algorithms.configs import ALG, DiffusionParamConfig
from bimanual_imitation.algorithms.core.diffusion.diffusion import ConditionalUnet1D
from bimanual_imitation.algorithms.core.shared.util import Timer
from irl_data.data_chunking import create_chunking_dataset


class ImitateDiffusion(ImitateBcBase):

    @property
    def alg(self):
        return ALG.DIFFUSION

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def bc_policy(self):
        return self._ema_noise_pred_net

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_bc_policy_weights(self):
        # Get a copy of the current ema model
        ema_noise_pred_net = deepcopy(self._noise_pred_net)
        self._ema.copy_to(ema_noise_pred_net.parameters())
        return ema_noise_pred_net.state_dict()

    def pre_eval(self):
        # Note: these models are equal by reference. This may result in more EMA
        # smoothing compared to only performing pre_eval once (after training)
        # Alternatively, use deepcopy
        self._ema_noise_pred_net = self._noise_pred_net
        self._ema.copy_to(self._ema_noise_pred_net.parameters())
        super().pre_eval()

    def bc_policy_fn(self, obs_cond):
        with torch.no_grad():
            B = 1
            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, self.train_dataset.pred_horizon, self._action_dim_single), device=self.device
            )
            naction = noisy_action

            # init scheduler
            self._noise_scheduler.set_timesteps(self.cfg.num_diffusion_iters)

            for k in self._noise_scheduler.timesteps:
                # predict noise
                noise_pred = self._ema_noise_pred_net(
                    sample=naction, timestep=k, global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self._noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

        return naction

    def init_params(self, cfg: DiffusionParamConfig):
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
        self._obs_dim = tmp_env.observation_space.shape[0]
        self._action_dim_single = tmp_env.action_space.shape[0] // cfg.pred_horizon
        del tmp_env

        # configure model
        self._noise_pred_net = ConditionalUnet1D(
            input_dim=self._action_dim_single, global_cond_dim=self._obs_dim
        )

        self._noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        # device transfer
        _ = self._noise_pred_net.to(self.device)

        self._ema = EMAModel(parameters=self._noise_pred_net.parameters())

        self._optimizer = torch.optim.AdamW(
            params=self._noise_pred_net.parameters(),
            lr=cfg.opt_learning_rate,
            weight_decay=cfg.opt_weight_decay,
        )

        self._lr_scheduler = get_scheduler(
            name=cfg.lr_scheduler,
            optimizer=self._optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=len(self._train_dataloader) * self.max_iter,
        )

        self._total_time = 0
        self._global_step = 0
        self._epoch = 0
        self.cfg = cfg

    def suggest_hyperparams(self, trial: optuna.Trial):
        param_cfg = DiffusionParamConfig(
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            num_diffusion_iters=trial.suggest_categorical("num_diffusion_iters", [50, 100]),
            opt_learning_rate=trial.suggest_categorical("opt_learning_rate", [1e-4, 5e-5, 1e-5]),
            opt_weight_decay=trial.suggest_categorical("opt_weight_decay", [1e-3, 1e-6]),
            lr_warmup_steps=trial.suggest_categorical("lr_warmup_steps", [500, 1000]),
        )

        return param_cfg

    def step_optimizer(self):
        epoch_loss = []
        with Timer() as t_train:
            for batch_idx, nbatch in enumerate(self._train_dataloader):
                # data normalized in dataset
                # device transfer
                nobs = nbatch["obs"].to(self.device)
                naction = nbatch["action"].to(self.device)
                B = nobs.shape[0]

                # observation as FiLM conditioning
                # (B, obs_horizon, obs_dim)
                obs_cond = nobs[:, : self.train_dataset.obs_horizon, :]
                # (B, obs_horizon * obs_dim)
                obs_cond = obs_cond.flatten(start_dim=1)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=self.device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, self._noise_scheduler.config.num_train_timesteps, (B,), device=self.device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = self._noise_scheduler.add_noise(naction, noise, timesteps)

                # predict the noise residual
                noise_pred = self._noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                self._lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                self._ema.step(self._noise_pred_net.parameters())

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
        diffusion_defaults = {
            "--mode": ("train_policy", str),
            "--max_iter": (30, int),
            "--num_evals": (10, int),
            "--num_rollouts_per_eval": (10, int),
            "--snapshot_save_freq": (3, int),
            "--print_freq": (1, int),
            "--limit_trajs": (200, int),
            "--export_data": (False, bool),
        }

        return diffusion_defaults


if __name__ == "__main__":
    ImitateDiffusion().run()
