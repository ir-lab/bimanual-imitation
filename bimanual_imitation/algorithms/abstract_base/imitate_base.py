import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import fields
from pathlib import Path

import numpy as np
import optuna
from art import text2art

from bimanual_imitation.algorithms.configs import ALG, get_param_config
from bimanual_imitation.algorithms.core.shared import nn_log, util
from bimanual_imitation.constants import TEST_RESULTS_DIR
from bimanual_imitation.utils import str2bool
from bimanual_imitation.utils.multiprocessing import Locker
from irl_data.constants import EXPERT_TRAJS_DIR, EXPERT_VAL_TRAJS_DIR
from irl_data.trajectory import TrajBatch


class ImitateBase(ABC):
    def __init__(self):
        self.__param_cfg, self._args = self.parse_args()
        args = self._args

        self.export_data = bool(
            (args.export_data == True or args.overwrite) and (args.mode != "hp_search")
        )

        self.export_dir = Path(args.export_dir)
        # if doing hp_search, mkdir for locks / databases
        if self.export_data or args.mode == "hp_search":
            if not self.export_dir.exists():
                self.export_dir.mkdir(parents=True)
        else:
            self.export_dir = None

        self.overwrite_logs = args.overwrite
        self.env_name = args.env_name
        self.max_iter = args.max_iter
        self.num_evals = args.num_evals
        self.num_rollouts = args.num_rollouts_per_eval
        self.snapshot_save_freq = args.snapshot_save_freq
        self.print_freq = args.print_freq
        self.validation_eval_freq = args.validation_eval_freq

        self.data_subsamp_freq = args.data_subsamp_freq
        self.limit_trajs = args.limit_trajs

        self.expert_trajs = EXPERT_TRAJS_DIR / f"{self.env_name}.proto"
        if self.env_name.startswith("quad_insert_"):
            # Only use the non-noisy environment for validation
            self.expert_val_trajs = EXPERT_VAL_TRAJS_DIR / "quad_insert_a0o0.proto"
        else:
            self.expert_val_trajs = EXPERT_VAL_TRAJS_DIR / f"{self.env_name}.proto"

        print("Run Config:")
        self.argstr = json.dumps(vars(args), separators=(",", ":"), indent=2)
        util.header(self.argstr)

    def run(self):
        args = self._args
        if args.mode == "hp_search":
            self.hp_search_num_trials = args.hp_search_num_trials
            self.study_name = f"{self.alg.value}_{str(args.tag).zfill(2)}"
            lock_dir = self.export_dir / "locks"
            if not lock_dir.exists():
                lock_dir.mkdir()
            self._locker = Locker(lock_file=lock_dir / f"{self.study_name}.lock")
            self.run_hp_search()
        elif args.mode == "train_policy":
            self.train_policy(self.__param_cfg)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

    @property
    @abstractmethod
    def alg(self) -> ALG:
        raise NotImplementedError

    @abstractmethod
    def step_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def eval_rollouts(self) -> TrajBatch:
        raise NotImplementedError

    @abstractmethod
    def suggest_hyperparams(self):
        raise NotImplementedError

    @abstractmethod
    def train_policy(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def write_snapshot(self):
        raise NotImplementedError

    @abstractmethod
    def get_default_run_options(self):
        raise NotImplementedError

    @abstractmethod
    def init_params(self):
        raise NotImplementedError

    def parse_args(self):
        def _parse_alg_params():
            parser = argparse.ArgumentParser(
                description=f"{self.alg.value.upper()} Parser", add_help=False
            )
            param_cfg_class = get_param_config(self.alg)
            for field in fields(param_cfg_class):
                field_type = field.type
                if field_type == bool:
                    field_type = str2bool
                parser.add_argument(f"--{field.name}", type=field_type, default=field.default)
            param_args, unknown_args = parser.parse_known_args()
            param_cfg = param_cfg_class(**vars(param_args))
            return param_cfg, parser, unknown_args

        def _parse_other_args():
            defaults = {
                "--mode": ("train_policy", str),
                "--export_dir": (str(TEST_RESULTS_DIR / self.alg.value), str),
                "--env_name": ("quad_insert_a0o0", str),
                "--max_iter": (50, int),
                "--num_evals": (5, int),
                "--num_rollouts_per_eval": (10, int),
                "--snapshot_save_freq": (10, int),
                "--print_freq": (1, int),
                "--validation_eval_freq": (0, int),
                "--tag": (0, int),
                "--data_subsamp_freq": (1, int),
                "--hp_search_num_trials": (10, int),
                "--limit_trajs": (200, int),
                "--export_data": (False, bool),  # can also pass --overwrite to export
            }

            alg_defaults = self.get_default_run_options()
            defaults.update(alg_defaults)

            parser = argparse.ArgumentParser(description="Base Parser", add_help=False)
            for arg, (default, arg_type) in defaults.items():
                parser.add_argument(arg, type=arg_type, default=default)

            parser.add_argument("--overwrite", action="store_true")

            args, unknown_args = parser.parse_known_args()
            return args, parser, unknown_args

        param_cfg, alg_parser, unknown_base_args = _parse_alg_params()
        args, base_parser, unknown_alg_args = _parse_other_args()

        if "--help" in unknown_base_args or "--help" in unknown_alg_args:
            util.header("===== Help menu for training imitation learning policies =====")
            util.warn(
                "NOTE: Both parsers are combined when running this script, so you can pass arguments from both"
            )
            util.header("===== Base Parser =====")
            base_parser.print_help()
            util.header("\n===== Algorithm Parser =====")
            alg_parser.print_help()
            sys.exit()  # Exit gracefully if --help is provided

        return param_cfg, args

    def get_eval_idxs(self, fused_train_steps):
        if self.num_evals != 0:
            eval_idxs = [round(x) for x in np.linspace(0, self.max_iter, self.num_evals + 1)[1:]]
            eval_idxs = np.array([idx - (idx % fused_train_steps) for idx in eval_idxs])
            if eval_idxs[-1] < self.max_iter:
                eval_idxs += fused_train_steps

            if not np.all([idx % self.snapshot_save_freq == 0 for idx in eval_idxs]):
                util.warn(
                    "Warning: you should choose max_iter divisible by checkpoint interval (snapshot_save_freq)"
                )

            assert (
                len(eval_idxs) == self.num_evals and eval_idxs[-1] == self.max_iter
            ), "Must choose max_iter divisible by num_evals"
        else:
            eval_idxs = []

        util.header(f"\nEval Indices: {list(eval_idxs)}\n")
        return eval_idxs

    def iterate_policy(self, param_argstr=""):
        np.set_printoptions(suppress=True, precision=5, linewidth=1000)

        if self.export_data:
            policy_log_file = self.export_dir / "policy_log.h5"
            if policy_log_file.exists():
                if self.overwrite_logs:
                    print("Existing policy log removed")
                    os.remove(policy_log_file)
                else:
                    warn_str = (
                        "Warning! policy log file already exists. "
                        + "NOTE: You can disable this check by passing --overwrite to the script. \nRemove? (y/n)\n"
                    )
                    remove = input(warn_str)
                    if remove == "y":
                        os.remove(policy_log_file)
                    else:
                        print("Exiting gracefully...")
                        exit()
        else:
            policy_log_file = None

        policy_log = nn_log.TrainingLog(
            policy_log_file, [("args", self.argstr), ("param_args", param_argstr)]
        )
        log_kwargs = {"sep": " | ", "width": 8, "precision": 4}

        best_avg_return = -np.inf
        best_trajbatch = None
        best_snapshot = None
        i = 0
        while i < self.max_iter:
            iter_info = self.step_optimizer()
            _, opt_iter, _ = next((info for info in iter_info if info[0] == "iter"), None)
            if i == 0:
                assert opt_iter >= 1
                fused_train_steps = opt_iter
                eval_idxs = self.get_eval_idxs(fused_train_steps)
                header_str = (
                    (("{:^%d}" % log_kwargs["width"]) + log_kwargs["sep"]) * len(iter_info)
                )[: -len(log_kwargs["sep"])].format(*["*" for _ in range(len(iter_info))])
                dash_str = "-" * len(header_str)
                ascii_art = text2art(self.alg.value.upper())
                print(f"{dash_str}\n{ascii_art}", end="")

            i = opt_iter
            print_header = (i % (20 * fused_train_steps * self.print_freq) == 0) or (
                i == fused_train_steps
            )

            display = (i % (fused_train_steps * self.print_freq) == 0) or (i == fused_train_steps)
            if display:
                policy_log.print(iter_info, print_header=print_header, **log_kwargs)

            avg_return = avg_length = np.nan
            if i in eval_idxs:
                trajbatch = self.eval_rollouts()
                avg_return = np.mean([traj.r_T.sum() for traj in trajbatch])
                avg_length = np.mean([len(traj) for traj in trajbatch])
                print(
                    f"{dash_str}\nIteration: {i}/{self.max_iter}, AvgReturn: {avg_return}, AvgLength: {avg_length}\n{dash_str}"
                )
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    best_trajbatch = trajbatch
                    best_snapshot = i

            iter_info += [
                ("rollout_avgr", avg_return, float),
                ("rollout_avglen", avg_length, float),
            ]

            policy_log.write(iter_info)

            write_snapshot = (self.snapshot_save_freq != 0) and (i % self.snapshot_save_freq == 0)
            if write_snapshot and self.export_data:
                self.write_snapshot(policy_log, i)

        if best_trajbatch is not None and self.export_data:
            df = best_trajbatch.to_dataframe()
            df.to_hdf(
                self.export_dir / f"rollouts_snapshot_{str(best_snapshot).zfill(7)}.h5",
                key="rollouts",
                index=None,
            )

        policy_log.close()

        return best_avg_return

    def get_param_argstr(self, param_cfg):
        param_argstr = json.dumps(vars(param_cfg), separators=(",", ":"), indent=2)
        print("Parameter Config:")
        util.header(param_argstr)
        return param_argstr

    def train_policy(self, param_cfg):
        param_argstr = self.get_param_argstr(param_cfg)
        self.init_params(param_cfg)
        self.iterate_policy(param_argstr)

    def hp_objective(self, trial: optuna.Trial):
        if self._locker.lock_file_fd is None:
            self._locker.acquire()

        param_cfg = self.suggest_hyperparams(trial)

        self._locker.release()

        param_argstr = self.get_param_argstr(param_cfg)
        self.init_params(param_cfg)
        best_avg_return = self.iterate_policy(param_argstr)
        cost = -best_avg_return

        return cost

    def run_hp_search(self):
        if self._locker.lock_file_fd is None:
            self._locker.acquire()

        database_dir = self.export_dir / "databases"
        if not database_dir.exists():
            print(f"Creating database directory: {database_dir}")
            database_dir.mkdir()

        study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            storage=f"sqlite:///{database_dir}/{self.study_name}.db",
            load_if_exists=True,
        )

        study.optimize(self.hp_objective, n_trials=self.hp_search_num_trials)

        best_params = study.best_params
        results_str = f"Best Params: {best_params}"
        util.header(f"{'='*len(results_str)}\n{results_str}\n{'='*len(results_str)}")
