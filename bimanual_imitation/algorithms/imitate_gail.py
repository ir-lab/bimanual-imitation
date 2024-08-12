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
from bimanual_imitation.algorithms.configs import ALG, GailParamConfig
from bimanual_imitation.algorithms.core import gail
from bimanual_imitation.algorithms.core.shared import SimConfig, rl
from bimanual_imitation.utils import get_cpus
from bimanual_imitation.utils.multiprocessing import processify


class ImitateGail(ImitateTheanoBase):

    @property
    def alg(self):
        return ALG.GAIL

    @property
    def opt(self):
        return self._opt

    @property
    def policy(self):
        return self._policy

    @property
    def policy_fn(self):
        return self._policy_fn

    def init_params(self, cfg: GailParamConfig):
        random.seed(os.urandom(4))
        seed = int(binascii.hexlify(os.urandom(4)), 16)
        np.random.seed(seed)

        policy_hidden_spec = get_nn_spec(
            cfg.policy_n_layers, cfg.policy_n_units, cfg.policy_layer_type
        )
        reward_hidden_spec = get_nn_spec(
            cfg.reward_n_layers, cfg.reward_n_units, cfg.reward_layer_type
        )
        vf_hidden_spec = get_nn_spec(cfg.vf_n_layers, cfg.vf_n_units, cfg.vf_layer_type)

        self._policy = get_theano_policy(
            self.mdp, self.alg, policy_hidden_spec, cfg.continuous_policy_type, cfg.obsnorm_mode
        )

        (
            exobs_Bstacked_Do,
            exa_Bstacked_Da,
            ext_Bstacked,
            val_exobs_Bstacked_Do,
            val_exa_Bstacked_Da,
            val_ext_Bstacked,
        ) = self.get_datasets()

        if cfg.reward_type == "nn":
            reward = gail.TransitionClassifier(
                hidden_spec=reward_hidden_spec,
                obsfeat_space=self.mdp.obs_space,
                action_space=self.mdp.action_space,
                max_kl=None,
                adam_lr=cfg.reward_lr,
                adam_steps=cfg.reward_steps,
                ent_reg_weight=cfg.reward_ent_reg_weight,
                enable_inputnorm=True,
                include_time=bool(cfg.reward_include_time),
                time_scale=1.0 / self.mdp.env_spec.max_episode_steps,
                favor_zero_expert_reward=bool(cfg.favor_zero_expert_reward),
                varscope_name="TransitionClassifier",
            )
        elif cfg.reward_type in ["l2ball", "simplex"]:
            reward = gail.LinearReward(
                obsfeat_space=self.mdp.obs_space,
                action_space=self.mdp.action_space,
                mode=cfg.reward_type,
                enable_inputnorm=True,
                favor_zero_expert_reward=bool(cfg.favor_zero_expert_reward),
                include_time=bool(cfg.reward_include_time),
                time_scale=1.0 / self.mdp.env_spec.max_episode_steps,
                exobs_Bex_Do=exobs_Bstacked_Do,
                exa_Bex_Da=exa_Bstacked_Da,
                ext_Bex=ext_Bstacked,
            )
        else:
            raise NotImplementedError(cfg.reward_type)

        vf = (
            None
            if bool(cfg.no_vf)
            else rl.ValueFunc(
                hidden_spec=vf_hidden_spec,
                obsfeat_space=self.mdp.obs_space,
                enable_obsnorm=cfg.obsnorm_mode != "none",
                enable_vnorm=True,
                max_kl=cfg.vf_max_kl,
                damping=cfg.vf_cg_damping,
                time_scale=1.0 / self.mdp.env_spec.max_episode_steps,
                varscope_name="ValueFunc",
            )
        )

        self._opt = gail.ImitationOptimizer(
            mdp=self.mdp,
            discount=cfg.discount,
            lam=cfg.lam,
            policy=self._policy,
            sim_cfg=SimConfig(
                min_num_trajs=-1,
                min_total_sa=cfg.min_total_sa,
                batch_size=self.cpu_batch_size,
                max_traj_len=self.mdp.env_spec.max_episode_steps,
            ),
            step_func=rl.TRPO(max_kl=cfg.policy_max_kl, damping=cfg.policy_cg_damping),
            reward_func=reward,
            value_func=vf,
            policy_obsfeat_fn=lambda obs: obs,
            reward_obsfeat_fn=lambda obs: obs,
            policy_ent_reg=cfg.policy_ent_reg,
            ex_obs=exobs_Bstacked_Do,
            ex_a=exa_Bstacked_Da,
            ex_t=ext_Bstacked,
            val_ex_obs=val_exobs_Bstacked_Do,
            val_ex_a=val_exa_Bstacked_Da,
            val_ex_t=val_ext_Bstacked,
            eval_freq=self.validation_eval_freq,
        )

        # Set observation normalization
        if cfg.obsnorm_mode == "expertdata":
            self._policy.update_obsnorm(exobs_Bstacked_Do)
            if reward is not None:
                reward.update_inputnorm(
                    self._opt.reward_obsfeat_fn(exobs_Bstacked_Do), exa_Bstacked_Da
                )
            if vf is not None:
                vf.update_obsnorm(self._opt.policy_obsfeat_fn(exobs_Bstacked_Do))

        self._policy_fn = lambda obs_B_Do: self._policy.sample_actions(
            obs_B_Do, cfg.deterministic_eval
        )

    @processify
    # We need to wrap this function with processify to allow for multiple/repeated instantiantiations
    def hp_objective(self, trial: optuna.Trial):
        return super().hp_objective(trial)

    def suggest_hyperparams(self, trial: optuna.Trial):
        param_cfg = GailParamConfig(
            policy_n_layers=trial.suggest_int("policy_n_layers", 2, 3),
            policy_n_units=trial.suggest_categorical("policy_n_units", [256, 512]),
            policy_layer_type=trial.suggest_categorical("policy_layer_type", ["relu", "tanh"]),
            policy_max_kl=trial.suggest_categorical("policy_max_kl", [0.01, 0.03]),
            policy_cg_damping=trial.suggest_categorical("policy_cg_damping", [0.1, 0.3]),
            policy_ent_reg=trial.suggest_categorical("policy_ent_reg", [0.0, 0.001, 0.01]),
            obsnorm_mode=trial.suggest_categorical("obsnorm_mode", ["none", "expertdata"]),
            reward_lr=trial.suggest_categorical("reward_lr", [1e-5, 5e-5, 1e-4]),
            reward_n_layers=trial.suggest_int("reward_n_layers", 1, 2),
            reward_n_units=trial.suggest_categorical("reward_n_units", [128, 256]),
            reward_layer_type=trial.suggest_categorical("reward_layer_type", ["relu", "tanh"]),
            reward_ent_reg_weight=trial.suggest_categorical(
                "reward_ent_reg_weight", [0.0, 0.001, 0.01]
            ),
            lam=trial.suggest_categorical("lam", [0.97, 0.99]),
            vf_n_layers=trial.suggest_int("vf_n_layers", 1, 2),
            vf_n_units=trial.suggest_categorical("vf_n_units", [128, 256]),
            vf_layer_type=trial.suggest_categorical("vf_layer_type", ["relu", "tanh"]),
            vf_max_kl=trial.suggest_categorical("vf_max_kl", [0.01, 0.03]),
            vf_cg_damping=trial.suggest_categorical("vf_cg_damping", [0.1, 0.3]),
            # not used for final hp search:
            # deterministic_eval=trial.suggest_categorical("deterministic_eval", [False, True]),
        )

        return param_cfg

    def get_default_run_options(self):
        gail_defaults = {
            "--mode": ("train_policy", str),
            "--max_iter": (8000, int),
            "--num_evals": (10, int),
            "--num_rollouts_per_eval": (10, int),
            "--snapshot_save_freq": (400, int),
            "--print_freq": (1, int),
            "--limit_trajs": (200, int),
            "--export_data": (False, bool),
            "--data_subsamp_freq": (1, int),
            "--cpu_batch_size": (get_cpus(16), int),
        }

        return gail_defaults


if __name__ == "__main__":
    ImitateGail().run()
