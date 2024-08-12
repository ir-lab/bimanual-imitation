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
from bimanual_imitation.algorithms.configs import ALG, BcloneParamConfig
from bimanual_imitation.algorithms.core.bclone import BehavioralCloningOptimizer
from bimanual_imitation.algorithms.core.shared import SimConfig
from bimanual_imitation.utils import get_cpus


class ImitateBClone(ImitateTheanoBase):

    @property
    def alg(self):
        return ALG.BCLONE

    @property
    def opt(self):
        return self._opt

    @property
    def policy(self):
        return self._policy

    @property
    def policy_fn(self):
        return self._policy_fn

    def init_params(self, cfg: BcloneParamConfig):
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

        self._opt = BehavioralCloningOptimizer(
            self.mdp,
            self._policy,
            lr=cfg.bclone_lr,
            batch_size=cfg.bclone_batch_size,
            obsfeat_fn=lambda obs: obs,
            ex_obs=exobs_Bstacked_Do,
            ex_a=exa_Bstacked_Da,
            val_ex_obs=val_exobs_Bstacked_Do,
            val_ex_a=val_exa_Bstacked_Da,
            eval_sim_cfg=SimConfig(  # NOT USED
                min_num_trajs=-1, min_total_sa=-1, batch_size=-1, max_traj_len=-1
            ),
            eval_freq=self.validation_eval_freq,
        )

        # Set observation normalization
        if cfg.obsnorm_mode == "expertdata":
            self._policy.update_obsnorm(exobs_Bstacked_Do)

        self._policy_fn = lambda obs_B_Do: self._policy.sample_actions(
            obs_B_Do, cfg.deterministic_eval
        )

    def suggest_hyperparams(self, trial: optuna.Trial):
        # policy_type_combo = trial.suggest_categorical(
        #     "policy_type_combo", ["Deterministic,1", "Gaussian,0", "Gaussian,1"]
        # )
        # continuous_policy_type, deterministic_eval = policy_type_combo.split(",")
        # deterministic_eval = bool(int(deterministic_eval))

        param_cfg = BcloneParamConfig(
            bclone_lr=trial.suggest_categorical("bclone_lr", [5e-5, 1e-4, 5e-4]),
            obsnorm_mode=trial.suggest_categorical("obsnorm_mode", ["none", "expertdata"]),
            policy_n_layers=trial.suggest_int("policy_n_layers", 2, 3),
            policy_n_units=trial.suggest_categorical("policy_n_units", [256, 512]),
            policy_layer_type=trial.suggest_categorical("policy_layer_type", ["relu", "tanh"]),
            bclone_l1_lambda=trial.suggest_categorical("bclone_l1_lambda", [0, 1e-6, 1e-4]),
            bclone_l2_lambda=trial.suggest_categorical("bclone_l2_lambda", [0, 1e-6, 1e-4]),
            bclone_batch_size=trial.suggest_categorical("bclone_batch_size", [128, 256, 512]),
            # not used for final hp search:
            # continuous_policy_type=continuous_policy_type,
            # deterministic_eval=deterministic_eval,
        )

        return param_cfg

    def get_default_run_options(self):
        bclone_defaults = {
            "--mode": ("train_policy", str),
            "--max_iter": (100_000, int),
            "--num_evals": (10, int),
            "--num_rollouts_per_eval": (10, int),
            "--snapshot_save_freq": (10_000, int),
            "--print_freq": (1000, int),
            "--validation_eval_freq": (10_000, int),
            "--limit_trajs": (200, int),
            "--export_data": (False, bool),
            "--cpu_batch_size": (get_cpus(10), int),
        }

        return bclone_defaults


if __name__ == "__main__":
    ImitateBClone().run()
