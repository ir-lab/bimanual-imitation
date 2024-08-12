import atexit
import os
import tempfile
from pathlib import Path

import gin
import numpy as np
import optuna
import tensorflow as tf
from absl import logging
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils
from tqdm import tqdm

from bimanual_imitation.algorithms.abstract_base.imitate_bc_base import ImitateBcBase
from bimanual_imitation.algorithms.configs import ALG, IbcParamConfig
from bimanual_imitation.algorithms.core.ibc.eval import eval_env as eval_env_module
from bimanual_imitation.algorithms.core.ibc.train import get_agent as agent_module
from bimanual_imitation.algorithms.core.ibc.train import (
    get_cloning_network as cloning_network_module,
)
from bimanual_imitation.algorithms.core.ibc.train import get_data as data_module
from bimanual_imitation.algorithms.core.ibc.train import get_learner as learner_module
from bimanual_imitation.algorithms.core.ibc.train import get_normalizers as normalizers_module
from bimanual_imitation.algorithms.core.ibc.train import get_sampling_spec as sampling_spec_module
from bimanual_imitation.algorithms.core.ibc.train.get_eval_actor import EvalActor
from bimanual_imitation.algorithms.core.ibc.utils.tf_recorder import export_to_tfrecord
from bimanual_imitation.algorithms.core.ibc.utils.warnings_filter import filter_warnings
from bimanual_imitation.algorithms.core.shared import util
from bimanual_imitation.constants import BIMANUAL_IMITATION_BASE_DIR


class ImitateIbc(ImitateBcBase):
    def __init__(self):
        self._tmp_files = []
        self._tmp_dirs = []
        self.eval_seed = 0
        atexit.register(self._cleanup_tmp_files)
        super().__init__()

    @property
    def alg(self):
        return ALG.IBC

    @property
    def train_dataset(self):
        return self._train_dataset

    def _create_tmp_path_safe(self, dir, **kwargs):
        if dir:
            _tmp_path = tempfile.TemporaryDirectory()
            self._tmp_dirs.append(_tmp_path)
            tmp_path = Path(_tmp_path.name)
        else:
            tmp_path = tempfile.NamedTemporaryFile(delete=False, **kwargs)
            self._tmp_files.append(tmp_path)

        return tmp_path

    def _cleanup_tmp_files(self):
        for file_obj in self._tmp_files:
            print(f"Unlinking {file_obj.name}")
            file_obj.close()
            if os.path.exists(file_obj.name):
                os.remove(file_obj.name)

        for dir_obj in self._tmp_dirs:
            print(f"Unlinking {dir_obj.name}")
            dir_obj.cleanup()

    # override
    def eval_rollouts(self):
        eval_actor, eval_actor_class, eval_env = self._policy_fn()
        trajs, success, returns = [], [], []

        with tf.name_scope(f"eval_rollouts"):
            progress_bar = tqdm(range(self.num_rollouts), desc="Rollouts")
            for idx in progress_bar:
                eval_env.seed(self.eval_seed)
                eval_actor.reset()
                eval_actor.run()
                self.eval_seed += 1

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
        return trajbatch

    def write_snapshot(self, policy_log, iteration):
        # Skip, since tf-agents already writes checkpoints for us
        pass

    def create_policy_fn(
        self,
        agent,
        env_name,
        eval_env,
        train_step,
        eval_export_dir,
        viz_img,
        num_envs,
        strategy,
        summary_dir_suffix,
    ):

        eval_actor_class = EvalActor()
        eval_actor, success_metric = eval_actor_class.get_eval_actor(
            agent,
            env_name,
            eval_env,
            train_step,
            self.num_rollouts,
            eval_export_dir,
            viz_img,
            num_envs,
            strategy,
            summary_dir_suffix=summary_dir_suffix,
        )

        return eval_actor, eval_actor_class, eval_env

    def init_params(self, cfg: IbcParamConfig):
        export_dir = self.export_dir
        limit_trajs = self.limit_trajs
        dataset_eval_fraction = 0.0
        flatten_action = True
        for_rnn = False
        goal_tolerance = 0.02
        image_obs = False
        loss_type = "ebm"
        network = "MLPEBM"
        num_envs = 1
        sequence_length = 2
        shared_memory_eval = False
        uniform_boundary_buffer = 0.05
        use_warmup = False
        viz_img = False
        summary_interval = 0
        export_eval_events = False
        log_tf = False
        checkpoint_interval = self.snapshot_save_freq
        decay_steps = 100
        export_data = self.export_data

        #############################
        # IBC doesn't use chunking
        pred_horizon = 1
        action_horizon = 1
        obs_horizon = 1

        # task = self.env_name + "eval_chunking"
        # gym_kwargs = {
        #     "pred_horizon": pred_horizon,
        #     "action_horizon": action_horizon,
        #     "obs_horizon": obs_horizon,
        #     "verbose": True,
        #     "disable_env_checker": True,
        # }

        task = self.env_name + "eval"
        gym_kwargs = {"disable_env_checker": True}  # Workaround to use gymnasium instead of gym
        #############################

        param_vars = vars(cfg)
        gin_filename = self._create_tmp_path_safe(dir=False, suffix=".gin")
        tmp_tfrecord_dir = self._create_tmp_path_safe(dir=True)

        gin_path = BIMANUAL_IMITATION_BASE_DIR / "param/ibc_template.gin"
        with open(gin_path) as gin_template, gin_filename as gin_file:
            strs = gin_template.read().format(**param_vars)
            gin_file.write(strs.encode())
            gin_filename = gin_file.name

        gin.add_config_file_search_path(os.getcwd())
        gin.parse_config_files_and_bindings(
            [gin_filename], None, str(self.expert_trajs), skip_unknown=True
        )

        if task is None:
            raise ValueError("task argument must be set.")
        logging.info(("Using task:", task))

        env_name = eval_env_module.get_env_name(task, shared_memory_eval, image_obs)
        logging.info(("Got env name:", env_name))

        self.eval_env = eval_env_module.get_eval_env(
            env_name, sequence_length, goal_tolerance, num_envs, gym_kwargs
        )
        logging.info(("Got eval_env:", self.eval_env))

        obs_tensor_spec, action_tensor_spec, time_step_tensor_spec = spec_utils.get_tensor_specs(
            self.eval_env
        )

        gpu_available = bool(tf.config.list_physical_devices("GPU"))
        strategy = strategy_utils.get_strategy(tpu=None, use_gpu=gpu_available)
        tf.config.run_functions_eagerly(False)

        full_action_size = action_tensor_spec.shape[0]

        assert obs_tensor_spec.shape[0] == sequence_length
        full_obs_size = obs_tensor_spec.shape[1]

        self._train_dataset = export_to_tfrecord(
            self.env_name,
            str(tmp_tfrecord_dir),
            limit_trajs,
            full_action_size=full_action_size,
            full_obs_size=full_obs_size,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            obs_horizon=obs_horizon,
        )

        dataset_path = str(tmp_tfrecord_dir / "*.tfrecord")

        # Compute normalization info from training data.
        create_train_and_eval_fns_unnormalized = data_module.get_data_fns(
            dataset_path,
            sequence_length,
            cfg.replay_capacity,
            cfg.batch_size,
            for_rnn,
            dataset_eval_fraction,
            flatten_action,
        )

        train_data, _ = create_train_and_eval_fns_unnormalized()
        norm_info, norm_train_data_fn = normalizers_module.get_normalizers(
            train_data, cfg.batch_size, env_name
        )

        # Create normalized training data.
        if not strategy:
            strategy = tf.distribute.get_strategy()

        per_replica_batch_size = cfg.batch_size // strategy.num_replicas_in_sync

        create_train_and_eval_fns = data_module.get_data_fns(
            dataset_path,
            sequence_length,
            cfg.replay_capacity,
            per_replica_batch_size,
            for_rnn,
            dataset_eval_fraction,
            flatten_action,
            norm_function=norm_train_data_fn,
        )

        # Create normalization layers for obs and action.
        with strategy.scope():
            # Create train step counter.
            self._train_step = train_utils.create_train_step()

            # Define action sampling spec.
            action_sampling_spec = sampling_spec_module.get_sampling_spec(
                action_tensor_spec,
                min_actions=norm_info.min_actions,
                max_actions=norm_info.max_actions,
                uniform_boundary_buffer=uniform_boundary_buffer,
                act_norm_layer=norm_info.act_norm_layer,
            )

            # This is a common opportunity for a bug, having the wrong sampling min/max so log this.
            logging.info(("Using action_sampling_spec:", action_sampling_spec))

            # Define keras cloning network.
            cloning_network = cloning_network_module.get_cloning_network(
                network,
                obs_tensor_spec,
                action_tensor_spec,
                norm_info.obs_norm_layer,
                norm_info.act_norm_layer,
                sequence_length,
                norm_info.act_denorm_layer,
            )

            # Define tfagent.
            self._agent = agent_module.get_agent(
                loss_type,
                time_step_tensor_spec,
                action_tensor_spec,
                action_sampling_spec,
                norm_info.obs_norm_layer,
                norm_info.act_norm_layer,
                norm_info.act_denorm_layer,
                cfg.learning_rate,
                use_warmup,
                cloning_network,
                self._train_step,
                decay_steps,
            )

        if (export_dir is not None) and export_data:
            export_dir = Path(export_dir)
            if not tf.io.gfile.exists(export_dir):
                tf.io.gfile.makedirs(export_dir)
        else:
            export_dir = self._create_tmp_path_safe(dir=True)

        # Define bc learner.
        self._bc_learner = learner_module.get_learner(
            loss_type,
            export_dir,
            self._agent,
            self._train_step,
            create_train_and_eval_fns,
            summary_interval,
            strategy,
            checkpoint_interval,
            log=log_tf,
        )

        eval_export_dir = export_dir if export_eval_events else None
        env_name_clean = env_name.replace("/", "_")

        self._policy_fn = lambda: self.create_policy_fn(
            self._agent,
            env_name,
            self.eval_env,
            self._train_step,
            eval_export_dir,
            viz_img,
            num_envs,
            strategy,
            env_name_clean,
        )

        logging.info("Saving operative-gin-config.")
        self._param_argstr = gin.operative_config_str()
        with tf.io.gfile.GFile(export_dir / "operative-gin-config.txt", "wb") as f:
            f.write(self._param_argstr)

        self.cfg = cfg
        self._iter = 0
        self._total_time = 0.0

    def step_optimizer(self):
        with util.Timer() as t_train:
            reduced_loss_info = None
            if not hasattr(self._agent, "ebm_loss_type") or self._agent.ebm_loss_type != "cd_kl":
                reduced_loss_info = self._bc_learner.run(iterations=self.cfg.fused_train_steps)
            else:
                for _ in range(self.cfg.fused_train_steps):
                    self._agent.cloning_network_copy.set_weights(
                        self._agent.cloning_network.get_weights()
                    )
                    reduced_loss_info = self._bc_learner.run(iterations=1)
            if reduced_loss_info:
                with self._bc_learner.train_summary_writer.as_default(), tf.summary.record_if(True):
                    tf.summary.scalar("reduced_loss", reduced_loss_info.loss, step=self._train_step)

            reduced_loss = reduced_loss_info.loss.numpy()

        self._iter += 1
        self._total_time += t_train.dt

        iter_info = [
            ("iter", self._iter, int),
            ("gstep", int(self._train_step.numpy()), int),
            ("loss", reduced_loss, float),
            ("ttrain", t_train.dt, float),
            ("ttotal", self._total_time, float),
        ]

        return iter_info

    def suggest_hyperparams(self, trial: optuna.Trial):
        param_cfg = IbcParamConfig(
            use_langevin=True,
            run_full_chain_under_gradient=True,
            use_dfo=False,
            fraction_langevin_samples=trial.suggest_categorical("pct_langevin_samples", [0.8, 1.0]),
            langevin_num_iterations=trial.suggest_categorical("langevin_num_iterations", [50, 100]),
            num_counter_examples=trial.suggest_categorical("num_counter_examples", [16, 32]),
            activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
            dropout_rate=trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2]),
            depth=trial.suggest_categorical("depth", [2, 4]),
            width=trial.suggest_categorical("width", [256, 512]),
            batch_size=trial.suggest_categorical("batch_size", [256, 512]),
            learning_rate=trial.suggest_categorical("learning_rate", [5e-5, 1e-4, 5e-4]),
        )

        return param_cfg

    def get_default_run_options(self):
        ibc_defaults = {
            "--mode": ("train_policy", str),
            "--max_iter": (150, int),
            "--num_evals": (10, int),
            "--num_rollouts_per_eval": (10, int),
            "--snapshot_save_freq": (15, int),
            "--print_freq": (1, int),
            "--limit_trajs": (200, int),
            "--export_data": (False, bool),
        }

        return ibc_defaults


if __name__ == "__main__":
    filter_warnings()  # comment out if desired
    ImitateIbc().run()
