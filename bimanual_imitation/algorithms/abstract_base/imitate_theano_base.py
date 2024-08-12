from abc import abstractmethod

import numpy as np

from bimanual_imitation.algorithms.abstract_base.imitate_base import ImitateBase
from bimanual_imitation.algorithms.configs import ALG
from bimanual_imitation.algorithms.core.shared import SimConfig, rlgymenv, util
from bimanual_imitation.utils.slurm import SBATCH_CFGS


class ImitateTheanoBase(ImitateBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mdp = rlgymenv.RLGymMDP(self.env_name)
        self.eval_mdp = rlgymenv.RLGymMDP(self.env_name + "eval")

        self.cpu_batch_size = self._args.cpu_batch_size = (
            int(self._args.cpu_batch_size)
            if self._args.cpu_batch_size is not None
            else SBATCH_CFGS[self.alg].num_cpus
        )

    @property
    @abstractmethod
    def opt(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def policy(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def policy_fn(self):
        raise NotImplementedError

    def get_datasets(self):
        # Load expert data
        exobs_Bstacked_Do, exa_Bstacked_Da, ext_Bstacked = load_dataset(
            self.expert_trajs, self.limit_trajs, self.data_subsamp_freq, "Expert"
        )
        assert exobs_Bstacked_Do.shape[1] == self.mdp.obs_space.storage_size
        assert exa_Bstacked_Da.shape[1] == self.mdp.action_space.storage_size
        assert ext_Bstacked.ndim == 1

        val_exobs_Bstacked_Do, val_exa_Bstacked_Da, val_ext_Bstacked = load_dataset(
            self.expert_val_trajs, None, self.data_subsamp_freq, "Validation"
        )
        assert val_exobs_Bstacked_Do.shape[1] == self.mdp.obs_space.storage_size
        assert val_exa_Bstacked_Da.shape[1] == self.mdp.action_space.storage_size
        assert val_ext_Bstacked.ndim == 1
        return (
            exobs_Bstacked_Do,
            exa_Bstacked_Da,
            ext_Bstacked,
            val_exobs_Bstacked_Do,
            val_exa_Bstacked_Da,
            val_ext_Bstacked,
        )

    def eval_rollouts(self):
        trajbatch = self.eval_mdp.sim_mp(
            policy_fn=self.policy_fn,
            obsfeat_fn=lambda obs: obs,
            cfg=SimConfig(
                min_num_trajs=self.num_rollouts,
                min_total_sa=-1,
                batch_size=min(self.cpu_batch_size, self.num_rollouts),
                max_traj_len=self.eval_mdp.env_spec.max_episode_steps,
            ),
            alg=self.alg,
            record_gif=False,
            dagger_eval=bool(self.alg == ALG.DAGGER),
            exact_num_trajs=True,
        )

        return trajbatch

    def step_optimizer(self):
        iter_info = self.opt.step()
        return iter_info

    def write_snapshot(self, policy_log, iteration):
        policy_log.write_snapshot(self.policy, iteration)


def load_dataset(filename, limit_trajs, data_subsamp_freq, dataset_type="Expert"):
    # Import proto_logger here to avoid loading theano when importing utils
    from irl_data import proto_logger

    exobs_B_T_Do, exa_B_T_Da, exr_B_T, exlen_B = proto_logger.load_theano_dataset(filename)
    full_dset_size = exobs_B_T_Do.shape[0]
    dset_size = min(full_dset_size, limit_trajs) if limit_trajs is not None else full_dset_size

    exobs_B_T_Do = exobs_B_T_Do[:dset_size, ...][...]
    exa_B_T_Da = exa_B_T_Da[:dset_size, ...][...]
    exr_B_T = exr_B_T[:dset_size, ...][...]
    exlen_B = exlen_B[:dset_size, ...][...]

    print(
        f"\n{dataset_type} dataset size: {exlen_B.sum()} transitions ({len(exlen_B)}) trajectories"
    )
    print(f"{dataset_type} average return: {exr_B_T.sum(axis=1).mean()}")

    # Stack everything together
    start_times_B = np.random.RandomState(0).randint(0, data_subsamp_freq, size=exlen_B.shape[0])

    exobs_Bstacked_Do = np.concatenate(
        [
            exobs_B_T_Do[i, start_times_B[i] : l : data_subsamp_freq, :]
            for i, l in enumerate(exlen_B)
        ],
        axis=0,
    )
    exa_Bstacked_Da = np.concatenate(
        [exa_B_T_Da[i, start_times_B[i] : l : data_subsamp_freq, :] for i, l in enumerate(exlen_B)],
        axis=0,
    )
    ext_Bstacked = np.concatenate(
        [np.arange(start_times_B[i], l, step=data_subsamp_freq) for i, l in enumerate(exlen_B)]
    ).astype(float)

    assert exobs_Bstacked_Do.shape[0] == exa_Bstacked_Da.shape[0] == ext_Bstacked.shape[0]
    # == np.ceil(exlen_B.astype(float)/data_subsamp_freq).astype(int).sum() > 0

    print(f"Subsampled data every {data_subsamp_freq} timestep(s)")
    print(
        f"Final dataset size: {exobs_Bstacked_Do.shape[0]} transitions"
        + f" (average {float(exobs_Bstacked_Do.shape[0]) / dset_size} per traj)\n"
    )

    return exobs_Bstacked_Do, exa_Bstacked_Da, ext_Bstacked


def get_nn_spec(n_layers, n_units, layer_type):
    layer_template = (
        '{{"type": "fc", "n": {num_units}}}, {{"type": "nonlin", "func": "{layer_type}"}}'
    )
    build_spec = "["
    for layer_idx in range(n_layers):
        build_spec += layer_template.format(num_units=n_units, layer_type=layer_type)
        if layer_idx < n_layers - 1:
            build_spec += ", "
        else:
            build_spec += "]"

    return build_spec


def get_theano_policy(
    mdp: rlgymenv.RLGymMDP,
    alg: ALG,
    policy_hidden_spec,
    continuous_policy_type,
    obsnorm_mode,
    bclone_l1_lambda=0.0,
    bclone_l2_lambda=0.0,
):
    from bimanual_imitation.algorithms.core.shared import ContinuousSpace, rl

    util.header(
        f"\nMDP observation space, action space sizes: {mdp.obs_space.dim}, {mdp.action_space.storage_size}\n"
    )

    # Initialize the policy
    enable_obsnorm = obsnorm_mode != "none"
    if isinstance(mdp.action_space, ContinuousSpace):
        assert bclone_l1_lambda >= 0.0 and bclone_l2_lambda >= 0.0
        if continuous_policy_type == "Gaussian":
            policy_cfg = rl.GaussianPolicyConfig(
                hidden_spec=policy_hidden_spec,
                min_stdev=0.0,
                init_logstdev=0.0,
                enable_obsnorm=enable_obsnorm,
            )
            policy = rl.GaussianPolicy(
                policy_cfg,
                mdp.obs_space,
                mdp.action_space,
                "GaussianPolicy",
                bclone_l1_lambda=bclone_l1_lambda,
                bclone_l2_lambda=bclone_l2_lambda,
            )
        elif continuous_policy_type == "Deterministic":
            assert alg != ALG.GAIL
            policy_cfg = rl.DeterministicPolicyConfig(
                hidden_spec=policy_hidden_spec, enable_obsnorm=enable_obsnorm
            )
            policy = rl.DeterministicPolicy(
                policy_cfg,
                mdp.obs_space,
                mdp.action_space,
                "DeterministicPolicy",
                bclone_l1_lambda=bclone_l1_lambda,
                bclone_l2_lambda=bclone_l2_lambda,
            )
    else:
        policy_cfg = rl.GibbsPolicyConfig(
            hidden_spec=policy_hidden_spec, enable_obsnorm=enable_obsnorm
        )
        policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, "GibbsPolicy")

    util.header("Policy architecture")
    for v in policy.get_trainable_variables():
        util.header("- %s (%d parameters)" % (v.name, v.get_value().size))
    util.header("Total: %d parameters" % (policy.get_num_params(),))
    return policy
