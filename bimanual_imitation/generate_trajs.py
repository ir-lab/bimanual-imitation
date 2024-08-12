import argparse
import json
import os
import re
from abc import ABC, abstractmethod

import gymnasium as gym
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from bimanual_imitation.algorithms.configs import ALG, ActParamConfig
from bimanual_imitation.algorithms.core.shared import util
from bimanual_imitation.pipeline import get_export_dir
from bimanual_imitation.utils import get_enum_value
from irl_data import proto_logger
from irl_data.data_chunking import create_chunking_dataset, normalize_data, unnormalize_data
from irl_data.trajectory import TrajBatch, Trajectory

try:
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    from bimanual_imitation.algorithms.core.act.policy import ACTPolicy
    from bimanual_imitation.algorithms.core.diffusion.diffusion import ConditionalUnet1D

    print("Torch dependencies loaded")
except ImportError:
    pass  # Optionally, print error message here


try:
    import tensorflow as tf
    from tf_agents.policies import py_tf_eager_policy
    from tf_agents.train.utils import train_utils

    from bimanual_imitation.algorithms.core.ibc.eval import eval_env as eval_env_module
    from bimanual_imitation.algorithms.core.ibc.train.get_eval_actor import EvalActor

    print("Tensorflow dependencies loaded")
except ImportError:
    pass  # Optionally, print error message here


try:
    from bimanual_imitation.algorithms.abstract_base.imitate_theano_base import (
        get_nn_spec,
        get_theano_policy,
    )
    from bimanual_imitation.algorithms.core.dagger import DAggerOptimizer
    from bimanual_imitation.algorithms.core.shared import SimConfig, nn, rlgymenv

    print("Theano dependencies loaded")
except ImportError:
    pass  # Optionally, print error message here


BATCH_SIZE = 1  # Num Cores to Allocate in Parallel


class TrajGenerator(ABC):
    def __init__(self, alg: ALG, args):
        self.alg = alg

        head_dir = get_export_dir(args.date_id, "phase3_train")

        strid = (
            f"alg={self.alg.value},env={args.env_name},num_trajs={args.policy_num_trajs:03d}"
            + f",run={args.policy_run:03d},tag={args.tag:02d}"
        )

        self.policy_dir = head_dir / strid
        assert self.policy_dir.exists(), f"Policy directory {self.policy_dir} does not exist"

        self.export_dir = head_dir.parent / "evaluations"
        if not self.export_dir.exists():
            self.export_dir.mkdir(parents=True)

        prefix = f"alg={self.alg.value},env={args.env_name},num_trajs={args.policy_num_trajs:03d},run={args.policy_run:03},"
        prefix_template = prefix + "pct={pct:03d},"
        self.gif_prefix = prefix_template + "rollout"
        self.gif_suffix = "{rollout_idx:03}"

        self.export_trajs = bool(args.export_trajs)
        traj_dir = self.export_dir / "trajs"
        if self.export_trajs:
            if not traj_dir.exists():
                traj_dir.mkdir(parents=True)
            self.traj_path_template = str(traj_dir) + f"/{prefix_template}_rollouts.proto"

        self.run_id = args.policy_run
        self.env_name = str(args.env_name)
        self.snapshot_pct = int(args.snapshot_pct)
        self.num_rollouts = int(args.num_rollouts)

    @abstractmethod
    def generate(self):
        raise NotImplementedError

    @abstractmethod
    def get_snapshot_idxs(self):
        raise NotImplementedError


def get_snapshot_tag_dir(export_dir, snapshot_pct):
    snapshot_dir = export_dir / f"snapshot_{snapshot_pct:03d}"

    if snapshot_dir.exists():
        # print(f"Already generated rollout data in {snapshot_dir}")
        # continue_input = input("Continue? (y/n)")
        # if continue_input.lower() != "y":
        #     print("Exiting...")
        #     exit()
        pass
    else:
        snapshot_dir.mkdir(parents=True)

    return snapshot_dir


def get_pct_snapshot_idxs(all_snapshot_idxs, snapshot_pct):

    def _get_idx_pct(pct):
        assert (pct >= 0) and (pct <= 100)
        snapshot_idx = int(pct / 100.0 * len(all_snapshot_idxs)) - 1
        snapshot_idx = np.clip(0, len(all_snapshot_idxs) - 1, snapshot_idx)
        return all_snapshot_idxs[snapshot_idx]

    if snapshot_pct == -1:
        snapshot_pcts = [10, 50, 100]
        snapshot_idxs = []
        for pct in snapshot_pcts:
            snapshot_idxs.append(_get_idx_pct(pct))
    else:
        snapshot_idx = _get_idx_pct(snapshot_pct)
        snapshot_idxs = [snapshot_idx]
        snapshot_pcts = [snapshot_pct]

    return snapshot_idxs, snapshot_pcts


class TheanoTrajGenerator(TrajGenerator):

    def generate(self):
        # Load the policy parameters
        eval_env = f"{self.env_name}eval"
        mdp = rlgymenv.RLGymMDP(eval_env)

        policy_file = self.policy_dir / "policy_log.h5"
        with h5py.File(policy_file, "r") as f:
            train_args = json.loads(f.attrs["param_args"])

        snapshot_idxs, snapshot_pcts = self.get_snapshot_idxs()
        policystr = "{}/snapshots/iter{:07d}".format(policy_file, snapshot_idxs[0])
        policy_file, policy_key = util.split_h5_name(policystr)

        nn.reset_global_scope()

        policy_hidden_spec = get_nn_spec(
            train_args["policy_n_layers"],
            train_args["policy_n_units"],
            train_args["policy_layer_type"],
        )

        policy = get_theano_policy(
            mdp, self.alg, policy_hidden_spec, "Gaussian", train_args["obsnorm_mode"]
        )

        print(f"Loading policy parameters from {policy_key} in {policy_file}")

        policy.load_h5(policy_file, policy_key)

        if self.alg == ALG.DAGGER:
            dummy_opt = DAggerOptimizer(
                mdp=mdp,
                policy=policy,
                lr=None,
                sim_cfg=None,
                ex_obs=None,
                ex_a=None,
                ex_t=None,
                val_ex_obs=None,
                val_ex_a=None,
                val_ex_t=None,
                eval_freq=None,
            )

        for snapshot_idx, snapshot_pct in zip(snapshot_idxs, snapshot_pcts):
            policystr = f"{policy_file}/snapshots/iter{snapshot_idx:07d}"
            policy_file, policy_key = util.split_h5_name(policystr)

            util.header(f"Generating rollouts for {snapshot_pct}% snapshot")
            print(f"Sampling {self.num_rollouts} trajs from policy {policystr} in {eval_env}")

            if self.alg == ALG.DAGGER:
                dummy_opt.policy.load_h5(policy_file, policy_key)
                policy_fn = lambda obsfeat_B_Df, env: dummy_opt.policy_fn(
                    obsfeat_B_Df, env, deterministic=train_args["deterministic_eval"]
                )
            else:
                policy.load_h5(policy_file, policy_key)
                policy_fn = lambda obs_B_Do: policy.sample_actions(
                    obs_B_Do, train_args["deterministic_eval"]
                )

            trajbatch = mdp.sim_mp(
                policy_fn=policy_fn,
                obsfeat_fn=lambda obs: obs,
                cfg=SimConfig(
                    min_num_trajs=self.num_rollouts,
                    min_total_sa=1,
                    batch_size=BATCH_SIZE,
                    max_traj_len=mdp.env_spec.max_episode_steps,
                ),
                alg=self.alg,
                record_gif=True,
                gif_export_dir=self.export_dir,
                gif_prefix=self.gif_prefix.format(pct=snapshot_pct),
                dagger_eval=bool(self.alg == ALG.DAGGER),
                exact_num_trajs=True,
            )

            if self.export_trajs:
                proto_logger.export_trajs(trajbatch, self.traj_path_template.format(pct=snapshot_pct))

    def get_snapshot_idxs(self):
        with pd.HDFStore(self.policy_dir / "policy_log.h5", "r") as f:
            log_df = f["log"]
            log_df.set_index("iter", inplace=True)
            snapshot_names = f.root.snapshots._v_children.keys()
            assert all(name.startswith("iter") for name in snapshot_names)
            all_snapshot_idxs = np.asarray(
                sorted([int(name[len("iter") :]) for name in snapshot_names])
            )

        snapshot_idxs = get_pct_snapshot_idxs(all_snapshot_idxs, self.snapshot_pct)
        return snapshot_idxs


class TensorflowTrajGenerator(TrajGenerator):

    def generate(self):
        eval_env = f"{self.env_name}eval"

        env_name = eval_env_module.get_env_name(
            eval_env, shared_memory_eval=False, use_image_obs=False
        )

        env = eval_env_module.get_eval_env(
            env_name,
            sequence_length=2,
            goal_tolerance=None,
            num_envs=1,
            gym_kwargs={"disable_env_checker": True},
        )

        saved_model_path = self.policy_dir / "policies/greedy_policy"

        snapshot_idxs, snapshot_pcts = self.get_snapshot_idxs()
        for snapshot_idx, snapshot_pct in zip(snapshot_idxs, snapshot_pcts):
            checkpoint_path = (
                self.policy_dir / f"policies/checkpoints/policy_checkpoint_{snapshot_idx:010d}"
            )
            gif_prefix = self.gif_prefix.format(pct=snapshot_pct)

            policy_cls = py_tf_eager_policy.SavedModelPyTFEagerPolicy

            policy = policy_cls(
                str(saved_model_path),
                env.time_step_spec(),
                env.action_spec(),
                load_specs_from_pbtxt=False,
            )

            policy.update_from_checkpoint(str(checkpoint_path))

            print(
                f"Sampling {self.num_rollouts} trajs from policy {checkpoint_path.name} in {eval_env}"
            )

            strategy = tf.distribute.get_strategy()
            with strategy.scope():
                train_step = train_utils.create_train_step()
                env_name_clean = env_name.replace("/", "_")
                eval_actor_class = EvalActor()
                eval_actor, success_metric = eval_actor_class.get_checkpoint_eval_actor(
                    policy=policy,
                    env_name=env_name,
                    eval_env=env,
                    train_step=train_step,
                    eval_episodes=self.num_rollouts,
                    root_dir=str(self.policy_dir),
                    viz_img=False,
                    num_envs=1,
                    strategy=strategy,
                    summary_dir_suffix=env_name_clean,
                )

                trajs, success, returns = [], [], []
                with tf.name_scope(f"eval_{env_name}"):
                    progress_bar = tqdm(range(self.num_rollouts), desc="Rollouts")
                    for run_idx in progress_bar:
                        gif_suffix = self.gif_suffix.format(rollout_idx=run_idx)
                        gif_filename = self.export_dir / f"{gif_prefix}_{gif_suffix}.mp4"
                        os.environ["IBC_EXPORT_GIF"] = str(gif_filename)

                        eval_actor.reset()
                        eval_actor.run()

                        traj = eval_actor_class.get_most_recent_traj()
                        trajs.append(traj)
                        success.append(int(traj.r_T[-1] > 0))
                        returns.append(traj.r_T.sum())
                        progress_bar.set_postfix(
                            AvgReturn=np.mean(returns),
                            PctSuccess=np.mean(success) * 100,
                            refresh=False,
                        )

                trajbatch = eval_actor_class.get_trajbatch(clear_existing=True)

                if self.export_trajs:
                    proto_logger.export_trajs(trajbatch, self.traj_path_template.format(pct=snapshot_pct))

    def get_snapshot_idxs(self):
        head_checkptdir = self.policy_dir / "policies/checkpoints"
        checkptdirs = [x for x in head_checkptdir.iterdir() if x.is_dir()]
        all_snapshot_idxs = []
        for checkptdir in checkptdirs:
            snapshot_idx = re.match("policy_checkpoint_(\d+)", str(checkptdir.name)).group(1)
            all_snapshot_idxs.append(int(snapshot_idx))

        all_snapshot_idxs = np.asarray(sorted(all_snapshot_idxs))
        snapshot_idxs = get_pct_snapshot_idxs(all_snapshot_idxs, self.snapshot_pct)
        return snapshot_idxs


class TorchTrajGenerator(TrajGenerator):
    def __init__(self, alg: ALG, args):
        super().__init__(alg, args)

        policy_file = self.policy_dir / "policy_log.h5"
        with h5py.File(policy_file, "r") as f:
            general_args = json.loads(f.attrs["args"])
            param_args = json.loads(f.attrs["param_args"])

        pred_horizon = param_args["pred_horizon"]
        obs_horizon = param_args["obs_horizon"]
        action_horizon = param_args["action_horizon"]
        max_trajs = general_args["limit_trajs"]

        self.train_dataset = create_chunking_dataset(
            environment=self.env_name,
            stage="train",
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            limit_trajs=max_trajs,
            normalize=True,
        )

        self.create_policy(param_args)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def bc_policy_fn(self):
        raise NotImplementedError

    @abstractmethod
    def load_policy(self, snapshot_idx_file):
        raise NotImplementedError

    @abstractmethod
    def create_policy(self, param_args):
        raise NotImplementedError

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

    def generate(self):
        def _eval_rollout(gif_export_dir, gif_prefix, rollout_idx):
            env = self.create_env()

            all_obs, all_rewards, all_actions = [], [], []

            raw_obs, _reset_info = env.reset()
            all_obs.append(raw_obs)
            obs_cond = self.obs_to_chunk(raw_obs)

            # must call this after performing env.reset()
            gif_suffix = self.gif_suffix.format(rollout_idx=rollout_idx)
            env.unwrapped.set_gif_recording(gif_export_dir, gif_prefix, gif_suffix)

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

            traj = Trajectory(
                obs_T_Do=np.vstack(all_obs),
                obsfeat_T_Df=np.ones((len(all_obs), 1)) * np.nan,
                a_T_Da=np.vstack(all_actions),
                adist_T_Pa=np.ones((len(all_actions), 2)) * np.nan,
                r_T=np.array(all_rewards),
            )

            env.unwrapped.export_gif_recording()

            return traj

        snapshot_idx_files, snapshot_pcts = self.get_snapshot_idxs()
        for snapshot_idx_file, snapshot_pct in zip(snapshot_idx_files, snapshot_pcts):
            gif_prefix = self.gif_prefix.format(pct=snapshot_pct)

            self.load_policy(snapshot_idx_file)

            trajs, success, returns = [], [], []
            progress_bar = tqdm(range(self.num_rollouts), desc="Rollouts")
            for rollout_idx in progress_bar:
                traj = _eval_rollout(self.export_dir, gif_prefix, rollout_idx)
                trajs.append(traj)
                success.append(int(traj.r_T[-1] > 0))
                returns.append(traj.r_T.sum())
                progress_bar.set_postfix(
                    AvgReturn=np.mean(returns), PctSuccess=np.mean(success) * 100, refresh=False
                )

            trajbatch = TrajBatch.FromTrajs(trajs)

            if self.export_trajs:
                proto_logger.export_trajs(trajbatch, self.traj_path_template.format(pct=snapshot_pct))

    def get_snapshot_idxs(self):
        checkpoints = np.array(list(self.policy_dir.glob("checkpoints/*")))
        checkpoint_idxs = np.array(
            [int(checkpoint.stem.split("_")[-1]) for checkpoint in checkpoints]
        )
        sorted_checkpoints = checkpoints[np.argsort(checkpoint_idxs)]
        snapshot_idx_files, snapshot_pcts = get_pct_snapshot_idxs(
            sorted_checkpoints, self.snapshot_pct
        )
        return snapshot_idx_files, snapshot_pcts


class DiffusionTrajGenerator(TorchTrajGenerator):

    def bc_policy_fn(self, obs_cond):
        with torch.no_grad():
            B = 1
            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, self.train_dataset.pred_horizon, self._action_dim_single), device=self.device
            )
            naction = noisy_action

            # init scheduler
            self._noise_scheduler.set_timesteps(self._num_diffusion_iters)

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

    def load_policy(self, snapshot_idx_file):
        state_dict = torch.load(snapshot_idx_file, map_location="cuda")
        self._ema_noise_pred_net.load_state_dict(state_dict)
        print(f"Pretrained weights loaded from {snapshot_idx_file}.")

    def create_policy(self, param_args):
        tmp_env = self.create_env(verbose=True)
        self._obs_dim = tmp_env.observation_space.shape[0]
        self._action_dim_single = tmp_env.action_space.shape[0] // param_args["pred_horizon"]
        del tmp_env

        # configure model
        self._ema_noise_pred_net = ConditionalUnet1D(
            input_dim=self._action_dim_single, global_cond_dim=self._obs_dim
        )

        self._ema_noise_pred_net.to(self.device)
        self._ema_noise_pred_net.eval()

        self._num_diffusion_iters = param_args["num_diffusion_iters"]

        self._noise_scheduler = DDPMScheduler(
            num_train_timesteps=self._num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )


class ActTrajGenerator(TorchTrajGenerator):

    def bc_policy_fn(self, obs):
        action = self._policy(obs)
        return action

    def load_policy(self, snapshot_idx_file):
        state_dict = torch.load(snapshot_idx_file, map_location="cuda")
        self._policy.load_state_dict(state_dict)
        print(f"Pretrained weights loaded from {snapshot_idx_file}.")

    def create_policy(self, param_args):
        tmp_env = self.create_env(verbose=True)
        self._obs_dim = tmp_env.observation_space.shape[0]
        self._action_dim_single = tmp_env.action_space.shape[0] // param_args["pred_horizon"]
        del tmp_env

        cfg = ActParamConfig(**param_args)
        self._policy = ACTPolicy(
            cfg, action_dim_single=self._action_dim_single, obs_dim=self._obs_dim
        )

        self._policy.to(self.device)
        self._policy.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date_id", type=str, default=None)
    parser.add_argument("--tag", type=int, default=0)
    parser.add_argument("--snapshot_pct", type=int, default=-1)
    parser.add_argument("--alg", type=str, default="act")
    parser.add_argument("--policy_run", type=int, default=0)
    parser.add_argument("--policy_num_trajs", type=int, default=100)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--env_name", type=str, default="quad_insert_a0o0")
    parser.add_argument("--export_trajs", type=bool, default=False)
    args = parser.parse_args()

    alg = get_enum_value(args.alg, ALG)

    traj_generator_map = {
        ALG.BCLONE: TheanoTrajGenerator,
        ALG.DAGGER: TheanoTrajGenerator,
        ALG.GAIL: TheanoTrajGenerator,
        ALG.IBC: TensorflowTrajGenerator,
        ALG.DIFFUSION: DiffusionTrajGenerator,
        ALG.ACT: ActTrajGenerator,
    }

