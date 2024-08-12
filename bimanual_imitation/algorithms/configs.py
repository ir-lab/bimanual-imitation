from dataclasses import dataclass
from enum import Enum


class ALG(Enum):
    BCLONE = "bclone"
    IBC = "ibc"
    DAGGER = "dagger"
    GAIL = "gail"
    EXPERT = "expert"
    DIFFUSION = "diffusion"
    ACT = "act"


@dataclass(frozen=True)
class BcloneParamConfig:
    bclone_lr: float = 5e-4
    obsnorm_mode: str = "expertdata"
    policy_n_layers: int = 3
    policy_n_units: int = 512
    policy_layer_type: str = "tanh"
    continuous_policy_type: str = "Gaussian"
    deterministic_eval: bool = False
    bclone_l1_lambda: float = 1e-4
    bclone_l2_lambda: float = 1e-4
    bclone_batch_size: int = 128


@dataclass(frozen=True)
class DaggerParamConfig:
    dagger_lr: float = 5e-05
    obsnorm_mode: str = "none"
    policy_n_layers: int = 3
    policy_n_units: int = 512
    policy_layer_type: str = "tanh"
    continuous_policy_type: str = "Gaussian"
    deterministic_eval: bool = False
    dagger_minibatch_size: int = 256
    dagger_num_epochs: int = 64
    dagger_beta_start: float = 1.0
    dagger_beta_decay: float = 0.95
    min_total_sa: int = 1
    bclone_l1_lambda: float = 1e-6
    bclone_l2_lambda: float = 1e-4


@dataclass(frozen=True)
class GailParamConfig:
    reward_lr: float = 5e-05
    obsnorm_mode: str = "expertdata"
    policy_n_layers: int = 2
    policy_n_units: int = 256
    policy_layer_type: str = "tanh"
    continuous_policy_type: str = "Gaussian"
    deterministic_eval: bool = True
    vf_n_layers: int = 2
    vf_n_units: int = 128
    vf_layer_type: str = "relu"
    reward_n_layers: int = 2
    reward_n_units: int = 256
    reward_layer_type: str = "relu"
    lam: float = 0.99
    policy_max_kl: float = 0.01
    policy_cg_damping: float = 0.3
    vf_max_kl: float = 0.01
    vf_cg_damping: float = 0.1
    reward_ent_reg_weight: float = 0.001
    policy_ent_reg: float = 0.001
    min_total_sa: int = 1
    discount: float = 0.995
    reward_include_time: int = 0
    no_vf: int = 0
    reward_type: str = "nn"
    reward_steps: int = 1
    favor_zero_expert_reward: int = 0


@dataclass(frozen=True)
class IbcParamConfig:
    normalizers_num_batches: int = 100
    normalizers_num_samples: int = 5000
    num_action_samples: int = 512
    use_langevin: bool = True
    fraction_langevin_samples: float = 1.0
    run_full_chain_under_gradient: bool = True
    langevin_num_iterations: int = 100
    dfo_num_iterations: int = 3
    use_dfo: bool = False
    num_counter_examples: int = 32
    activation: str = "relu"
    depth: int = 4
    dropout_rate: float = 0.0
    width: int = 256
    batch_size: int = 512
    fused_train_steps: int = 100
    learning_rate: float = 1e-4
    replay_capacity: int = 10_000


@dataclass(frozen=True)
class DiffusionParamConfig:
    batch_size: int = 512
    num_diffusion_iters: int = 100
    opt_learning_rate: float = 1.0e-4
    opt_weight_decay: float = 1.0e-6
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    obs_horizon: int = 4
    action_horizon: int = 4
    pred_horizon: int = 8


@dataclass(frozen=True)
class ActParamConfig:
    batch_size: int = 256
    enc_layers: int = 1
    dec_layers: int = 1
    latent_dim: int = 8
    n_heads: int = 4
    act_lr: float = 1.0e-4
    dropout: float = 0.1
    weight_decay: float = 1.0e-4
    hidden_dim: int = 128
    dim_feedforward: int = 512
    activation: str = "relu"
    kl_weight: int = 100
    pre_norm: bool = False
    obs_horizon: int = 4
    action_horizon: int = 4
    pred_horizon: int = 8


def get_param_config(alg: ALG):
    param_cfgs = {
        ALG.GAIL: GailParamConfig,
        ALG.IBC: IbcParamConfig,
        ALG.BCLONE: BcloneParamConfig,
        ALG.DAGGER: DaggerParamConfig,
        ALG.DIFFUSION: DiffusionParamConfig,
        ALG.ACT: ActParamConfig,
    }
    return param_cfgs[alg]
