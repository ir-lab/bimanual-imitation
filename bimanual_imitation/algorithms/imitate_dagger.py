import binascii
import os
import random

import numpy as np
import optuna

from bimanual_imitation.algorithms.abstract_base.imitate_theano_base import (
    ImitateTheanoBase,
    get_nn_spec,
    get_theano_policy,
)
from bimanual_imitation.algorithms.configs import ALG, DaggerParamConfig
from bimanual_imitation.algorithms.core.dagger import DAggerOptimizer
from bimanual_imitation.algorithms.core.shared import SimConfig
from bimanual_imitation.utils import get_cpus


class ImitateDagger(ImitateTheanoBase):

    @property
    def alg(self):
        return ALG.DAGGER

    @property
    def opt(self):
        return self._opt

    @property
    def policy(self):
        return self._policy

    @property
    def policy_fn(self):
        return self._policy_fn

    def init_params(self, cfg: DaggerParamConfig):
        random.seed(os.urandom(4))
        seed = int(binascii.hexlify(os.urandom(4)), 16)
        np.random.seed(seed)

        policy_hidden_spec = get_nn_spec(
            cfg.policy_n_layers, cfg.policy_n_units, cfg.policy_layer_type
        )

        self._policy = get_theano_policy(
            self.mdp,
            self.alg,
            policy_hidden_spec,
            cfg.continuous_policy_type,
            cfg.obsnorm_mode,
            cfg.bclone_l1_lambda,
            cfg.bclone_l2_lambda,
        )

        (
            exobs_Bstacked_Do,
            exa_Bstacked_Da,
            ext_Bstacked,
            val_exobs_Bstacked_Do,
            val_exa_Bstacked_Da,
            val_ext_Bstacked,
        ) = self.get_datasets()

        self._opt = DAggerOptimizer(
            mdp=self.mdp,
            policy=self._policy,
            lr=cfg.dagger_lr,
            sim_cfg=SimConfig(
                min_num_trajs=-1,
                min_total_sa=cfg.min_total_sa,
                batch_size=self.cpu_batch_size,
                max_traj_len=self.mdp.env_spec.max_episode_steps,
            ),
            ex_obs=exobs_Bstacked_Do,
            ex_a=exa_Bstacked_Da,
            ex_t=ext_Bstacked,
            val_ex_obs=val_exobs_Bstacked_Do,
            val_ex_a=val_exa_Bstacked_Da,
            val_ex_t=val_ext_Bstacked,
            eval_freq=self.validation_eval_freq,
            num_epochs=cfg.dagger_num_epochs,
            minibatch_size=cfg.dagger_minibatch_size,
            beta_start=cfg.dagger_beta_start,
            beta_decay=cfg.dagger_beta_decay,
            init_bclone=False,
            subsample_rate=1,
        )

        # Set observation normalization
        if cfg.obsnorm_mode == "expertdata":
            self._policy.update_obsnorm(exobs_Bstacked_Do)

        self._policy_fn = lambda obs_B_Do, env: self._opt.policy_fn(
            obs_B_Do, env, cfg.deterministic_eval
        )

    def suggest_hyperparams(self, trial: optuna.Trial):
        # policy_type_combo = trial.suggest_categorical(
        #     "policy_type_combo", ["Deterministic,1", "Gaussian,0", "Gaussian,1"]
        # )
        # continuous_policy_type, deterministic_eval = policy_type_combo.split(",")
        # deterministic_eval = bool(int(deterministic_eval))

        param_cfg = DaggerParamConfig(
            min_total_sa=1,  # because we are multiprocessing (otherwise on 1 core, use ~16,000)
            dagger_lr=trial.suggest_categorical("dagger_lr", [5e-5, 1e-4, 5e-4]),
            policy_n_layers=trial.suggest_categorical("policy_n_layers", [2, 3]),
            policy_n_units=trial.suggest_categorical("policy_n_units", [256, 512]),
            policy_layer_type=trial.suggest_categorical("policy_layer_type", ["relu", "tanh"]),
            obsnorm_mode=trial.suggest_categorical("obsnorm_mode", ["none", "expertdata"]),
            dagger_num_epochs=trial.suggest_categorical("dagger_num_epochs", [64, 128]),
            dagger_beta_decay=trial.suggest_categorical("dagger_beta_start", [0.9, 0.95]),
            bclone_l1_lambda=trial.suggest_categorical("bclone_l1_lambda", [0, 1e-6, 1e-4]),
            bclone_l2_lambda=trial.suggest_categorical("bclone_l2_lambda", [0, 1e-6, 1e-4]),
            dagger_minibatch_size=trial.suggest_categorical("dagger_minibatch_size", [128, 256]),
            # not used for final hp search:
            # dagger_beta_start=trial.suggest_categorical("dagger_beta_start", [1.0, 0.75, 0.5]),
            # continuous_policy_type=continuous_policy_type,
            # deterministic_eval=deterministic_eval,
        )

        return param_cfg

    def get_default_run_options(self):
        dagger_defaults = {
            "--mode": ("train_policy", str),
            "--max_iter": (40, int),
            "--num_evals": (10, int),
            "--num_rollouts_per_eval": (10, int),
            "--snapshot_save_freq": (4, int),
            "--print_freq": (1, int),
            "--limit_trajs": (200, int),
            "--export_data": (False, bool),
            "--data_subsamp_freq": (1, int),
            "--cpu_batch_size": (get_cpus(16), int),
        }

        return dagger_defaults


if __name__ == "__main__":
    ImitateDagger().run()
