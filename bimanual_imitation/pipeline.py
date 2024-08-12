import argparse
import datetime
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bimanual_imitation.algorithms.configs import ALG, get_param_config
from bimanual_imitation.algorithms.core.shared import util
from bimanual_imitation.constants import BIMANUAL_IMITATION_BASE_DIR, RESULTS_DIR
from bimanual_imitation.utils import download_covering_array, get_enum_value
from bimanual_imitation.utils.slurm import run_sbatch
from irl_data.constants import EXPERT_TRAJS_DIR


def get_export_dir(date_id, sub_dir: str) -> Path:
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)

    if date_id is None:
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}$")
        subdirs = [
            d for d in RESULTS_DIR.iterdir() if d.is_dir() and bool(date_pattern.match(d.name))
        ]
        if len(subdirs) > 0:
            sorted_subdirs = sorted(subdirs, key=lambda x: x.name, reverse=True)
            # Use most recent date_id
            export_dir = sorted_subdirs[0] / sub_dir
            print("Using Most recent date_id:", export_dir)
            if not export_dir.exists():
                print(f"Creating directory: {export_dir}")
                export_dir.mkdir(parents=True)
        else:
            date_id = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            export_dir = RESULTS_DIR / date_id / sub_dir
            print(f"Creating directory: {export_dir}")
            export_dir.mkdir(parents=True)

    else:
        export_dir = RESULTS_DIR / str(date_id) / sub_dir
        if not export_dir.exists():
            print(f"Creating directory: {export_dir}")
            export_dir.mkdir(parents=True)
        else:
            print(f"Using date_id: {export_dir}")

    return export_dir


def export_h5(
    args,
    h5_filename,
    prev_sub_dir,
    act_obs_dim=(18, 36),
    chunking_policies=[ALG.ACT, ALG.DIFFUSION],
    act_obs_pred_horizons=(4, 4, 8),
    unique_num_trajs=(50, 100, 200),
    export_metadata=True,
    export_rollouts=True,
):
    from irl_data import proto_logger
    from irl_data.trajectory import TrajBatch
    from irl_environments.constants import IRL_ENVIRONMENTS_BASE_DIR
    from irl_environments.core.utils import (
        ActionGroup,
        ObservationGroup,
        get_action_group_dim,
        get_observation_group_dim,
    )

    def _get_rename_map(env_name):
        env_yaml = IRL_ENVIRONMENTS_BASE_DIR / f"param/{env_name}.yaml"
        assert env_yaml.exists()
        with open(env_yaml, "r") as env_config_file:
            env_config = yaml.safe_load(env_config_file)

        obs_idx = 0
        act_idx = 0
        rename_map = dict()
        for device_name, device_config in env_config["devices"].items():
            dev_space_name = device_config["space"]
            dev_space = env_config["spaces"][dev_space_name]
            obs_space_strs = dev_space["observation_space"]
            act_space_strs = dev_space["action_space"]

            for obs_str in obs_space_strs:
                obs_enum = get_enum_value(obs_str, ObservationGroup)
                obs_dim = get_observation_group_dim(obs_enum)
                if obs_dim == 1:
                    rename_map[f"obs_{obs_idx}"] = f"{device_name} {obs_str}"
                    obs_idx += 1
                else:
                    for dim in range(obs_dim):
                        rename_map[f"obs_{obs_idx}"] = f"{device_name} {obs_str} {dim+1}"
                        obs_idx += 1

            for act_str in act_space_strs:
                act_enum = get_enum_value(act_str, ActionGroup)
                act_dim = get_action_group_dim(act_enum)
                if act_dim == 1:
                    rename_map[f"act_{act_idx}"] = f"{device_name} {act_str}"
                    act_idx += 1
                else:
                    for dim in range(act_dim):
                        rename_map[f"act_{act_idx}"] = f"{device_name} {act_str} {dim+1}"
                        act_idx += 1

        return rename_map

    prev_export_dir = get_export_dir(args.date_id, prev_sub_dir)

    dir_pattern = r"^alg=(\w+),env=(\w+),num_trajs=(\d{3}),run=(\d{3}),tag=(\d{2})$"
    dfs = []
    mdfs = []
    envs = []
    for dir_name in prev_export_dir.iterdir():
        assert dir_name.is_dir()
        dir_match = re.match(dir_pattern, dir_name.name)
        rollout_files = list(dir_name.glob("rollouts_snapshot_*.h5"))
        policy_log = dir_name / "policy_log.h5"
        output_log = dir_name / "output.log"
        if dir_match and rollout_files:
            assert len(rollout_files) == 1
            rollout_file = rollout_files[0]
            assert policy_log.exists()
            assert output_log.exists()
            tag_group = int(dir_match.group(5))
            if tag_group == args.tag:
                alg = get_enum_value(str(dir_match.group(1)), ALG)
                env_group = str(dir_match.group(2))
                if env_group not in envs:
                    envs.append(env_group)
                num_trajs_group = int(dir_match.group(3))
                run_group = int(dir_match.group(4))

                if export_rollouts:
                    df = pd.read_hdf(rollout_file)

                if export_metadata:
                    with pd.HDFStore(policy_log, "r") as f:
                        log_df = f["log"]
                        log_df.set_index("iter", inplace=True)

                    mdf = log_df[
                        log_df["rollout_avgr"].notna() & log_df["rollout_avglen"].notna()
                    ].copy()

                    mdf.loc[:, "iter_id"] = np.arange(mdf.shape[0])

                    mdf.reset_index(inplace=True)

                # Rename to environment obs/act names
                if export_rollouts:
                    if alg in chunking_policies:
                        # Trim the action chunk so that the dataframe is a consistent shape
                        # among all policies (some which may not use chunking)

                        # Extract the *first piece* of action chunk sent to the simulator
                        act_dim, obs_dim = act_obs_dim
                        act_horizon, obs_horizon, pred_horizon = act_obs_pred_horizons
                        start_act_idx = (obs_horizon - 1) * act_dim
                        end_act_idx = start_act_idx + act_dim
                        act_keys = [f"act_{i}" for i in range(start_act_idx, end_act_idx)]
                        act_df = df[act_keys]
                        new_act_keys = [f"act_{i}" for i in range(act_dim)]
                        act_df.columns = new_act_keys
                        df = df[df.columns.drop(list(df.filter(regex="act_*")))]
                        df = pd.concat((df, act_df), axis=1)

                        # Extract the observation aligning with the action
                        start_obs_idx = (obs_horizon - 1) * obs_dim
                        end_obs_idx = start_obs_idx + obs_dim
                        obs_keys = [f"obs_{i}" for i in range(start_obs_idx, end_obs_idx)]
                        obs_df = df[obs_keys]
                        new_obs_keys = [f"obs_{i}" for i in range(obs_dim)]
                        obs_df.columns = new_obs_keys
                        df = df[df.columns.drop(list(df.filter(regex="obs_*")))]
                        df = pd.concat((df, obs_df), axis=1)

                    rename_map = _get_rename_map(env_group)
                    for key, val in rename_map.items():
                        if key in df.columns.values:
                            df = df.rename(columns={key: val})
                # Add metadata
                add_keys = {
                    "alg": alg.value,
                    "num_trajs": num_trajs_group,
                    "env": env_group,
                    "seed": run_group,
                }

                # Dagger doesn't depend on prior expert trajs, so make copy for all num_trajs
                # inefficient, but allows for plotting combinations easily
                if alg == ALG.DAGGER:
                    for nt in unique_num_trajs:
                        if export_rollouts:
                            dfc = df.copy(deep=True)
                        if export_metadata:
                            mdfc = mdf.copy(deep=True)
                        add_keys["num_trajs"] = nt
                        for key, val in add_keys.items():
                            if export_rollouts:
                                dfc[key] = [val] * dfc.shape[0]
                            if export_metadata:
                                mdfc[key] = [val] * mdfc.shape[0]
                        if export_rollouts:
                            dfs.append(dfc)
                        if export_metadata:
                            mdfs.append(mdfc)
                else:
                    for key, val in add_keys.items():
                        if export_rollouts:
                            df[key] = [val] * df.shape[0]
                        if export_metadata:
                            mdf[key] = [val] * mdf.shape[0]
                    if export_rollouts:
                        dfs.append(df)
                    if export_metadata:
                        mdfs.append(mdf)

    if export_rollouts:
        util.warn("This may take a while (do not exit)...")
        for env in envs:
            expert_proto = EXPERT_TRAJS_DIR / f"{env}.proto"
            ex_trajs = proto_logger.load_trajs(expert_proto)
            for nt in unique_num_trajs:
                add_keys = {"alg": ALG.EXPERT.value, "num_trajs": nt, "env": env, "seed": 0}
                ex_df = TrajBatch.FromTrajs(ex_trajs[:nt]).to_dataframe(**add_keys)
                rename_map = _get_rename_map(env)
                for key, val in rename_map.items():
                    if key in ex_df.columns.values:
                        ex_df = ex_df.rename(columns={key: val})
                dfs.append(ex_df)
        util.warn("Done exporting rollouts!")

    def _export_dfs(in_dfs, h5_filename, pivot_cols=None):
        final_df = pd.concat(in_dfs)
        if pivot_cols is not None:
            all_cols = set(final_df.columns.values)
            remain_cols = set.difference(all_cols, set(pivot_cols))
            final_df = final_df.pivot_table(index=pivot_cols, values=remain_cols)

        final_df.to_hdf(h5_filename, key="irl_data", mode="w")
        print(f"Exported {h5_filename}!")
        print(final_df)

    if export_rollouts:
        _export_dfs(
            dfs, h5_filename, pivot_cols=["alg", "env", "num_trajs", "seed", "rollout", "time"]
        )

    if export_metadata:
        _export_dfs(
            mdfs,
            h5_filename.with_name("metadata_" + h5_filename.name),
            pivot_cols=["alg", "env", "num_trajs", "seed", "iter_id"],
        )


def phase1_hp_search(args):
    util.header("=== Running Phase 1: HP Search ===")
    alg = get_enum_value(args.alg, ALG)
    sub_dir = "phase1_hp_search"
    export_dir = get_export_dir(args.date_id, sub_dir)
    tag = str(args.tag).zfill(2)

    sbatch_dir = export_dir / "sbatch"
    if not sbatch_dir.exists():
        sbatch_dir.mkdir()

    log_dir = export_dir / "output_logs"
    target_script = BIMANUAL_IMITATION_BASE_DIR / f"algorithms/imitate_{alg.value}.py"
    cmd_str = (
        f"python3 -u {target_script} --mode hp_search --env_name {args.env_name} --export_dir {export_dir} --tag {args.tag}"
        + f" --hp_search_num_trials {args.hp_search_num_trials}"
    )

    cmd_templates, outputfilenames, argdicts = [], [], []
    for worker in range(args.hp_search_num_workers):
        cmd_templates.append(cmd_str)
        outputfile = log_dir / f"alg={alg.value},tag={tag},worker={str(worker).zfill(3)}.log"
        outputfilenames.append(outputfile)
        argdicts.append({})

    sbatch_script = sbatch_dir / f"{alg.value}_{tag}.sh"
    assert (
        not sbatch_script.exists()
    ), "Tag for this study is/has already been created. Must specify new tag!"

    run_sbatch(
        alg,
        cmd_templates,
        outputfilenames,
        argdicts=argdicts,
        sh_script=sbatch_script,
        max_num_workers=args.max_num_workers,
        run_local=args.run_local,
        sleep_time=60,
    )


def phase2_hp_search_analysis(args):
    import optuna
    from scipy.stats import sem

    util.header("=== Running Phase 2: HP Search Analysis ===")
    study_id = f"{args.alg}_{str(args.tag).zfill(2)}"
    prev_phase_sub_dir = "phase1_hp_search"
    prev_phase_export_dir = get_export_dir(args.date_id, prev_phase_sub_dir)
    assert prev_phase_export_dir.exists(), "Study must already exist!"

    cur_phase_sub_dir = "phase2_hp_search_analysis"
    cur_phase_export_dir = get_export_dir(args.date_id, cur_phase_sub_dir)

    database_dir = prev_phase_export_dir / "databases"
    study = optuna.load_study(study_name=None, storage=f"sqlite:///{database_dir}/{study_id}.db")
    # study = optuna.load_study(study_name=study_id, storage=f"sqlite:///{database_dir}/{study_id}.db")

    class hashabledict(dict):
        def __hash__(self):
            return hash(tuple(sorted(self.items())))

    cmap = OrderedDict()
    for trial in study.trials:
        if trial.values:
            assert len(trial.values) == 1
            key = hashabledict(trial.params)
            # NOTE: Multiply by -1 to convert costs to rewards
            if key not in cmap.keys():
                cmap[key] = -1 * np.array(trial.values)
            else:
                cmap[key] = np.append(cmap[key], -1 * trial.values[0])

    means = np.array([np.mean(values) for values in cmap.values()])
    std_errs = np.array([sem(values) for values in cmap.values()])
    bounds = []
    lens = []
    worsts = []
    for i, (mean, se) in enumerate(zip(means, std_errs)):
        lens.append(len(list(cmap.values())[i]))
        worsts.append(min(list(cmap.values())[i]))
        if np.isnan(se):
            bounds.append(mean)
        else:
            bounds.append(mean - se)

    best_idxs = np.argsort(bounds)[::-1][:30]
    # best_idxs = np.argsort(lens)[::-1][:30]

    print("Sorted Params:")
    for i, idx in enumerate(best_idxs):
        params = list(cmap.keys())[idx]
        print(
            f"[{i}], Num Trials [{lens[idx]}], Avg: {means[idx]},  Worst: {worsts[idx]}, Lower: {bounds[idx]}, Params: {params}"
        )


def phase3_train(args):
    util.header("=== Running Phase 3: Train ===")
    alg = get_enum_value(args.alg, ALG)
    tag = str(args.tag).zfill(2)
    sub_dir = "phase3_train"
    export_dir = get_export_dir(args.date_id, sub_dir)

    sbatch_dir = export_dir / "sbatch"
    pipeline_dir = export_dir / "pipelines"

    if not sbatch_dir.exists():
        sbatch_dir.mkdir()

    if not pipeline_dir.exists():
        pipeline_dir.mkdir()

    next_suffix = -1
    sbatch_pattern = rf"{re.escape(alg.value)}_{re.escape(tag)}" + r"-(\d{2})\.sh"
    for sb in Path(sbatch_dir).iterdir():
        match = re.match(sbatch_pattern, sb.name)
        if match:
            assert sb.is_file()
            suffix = int(match.group(1))
            if suffix > next_suffix:
                next_suffix = suffix

    sbatch_script = sbatch_dir / f"{alg.value}_{tag}-{str(next_suffix+1).zfill(2)}.sh"

    if next_suffix == -1:
        if args.spec is None:
            spec_file = BIMANUAL_IMITATION_BASE_DIR / f"param/{alg.value}.yaml"
        else:
            spec_file = Path(args.spec)

        assert spec_file.exists()
        with open(spec_file, "r") as f:
            spec = yaml.safe_load(f)
    else:
        spec_file = pipeline_dir / f"{alg.value}_{tag}.yaml"
        assert spec_file.exists()
        with open(spec_file, "r") as f:
            spec = yaml.safe_load(f)
        util.header(f"Re-using {spec_file} for this run!")

    dir_pattern = r"^alg=(\w+),env=(\w+),num_trajs=(\d{3}),run=(\d{3}),tag=(\d{2})$"
    max_prev_runs = {}
    for dir_name in Path(export_dir).iterdir():
        assert dir_name.is_dir()
        dir_match = re.match(dir_pattern, dir_name.name)
        if dir_match:
            alg_enum = get_enum_value(str(dir_match.group(1)), ALG)
            tag_group = int(dir_match.group(5))
            if alg_enum == alg and (tag_group == int(args.tag)):
                env_group = str(dir_match.group(2))
                num_trajs_group = int(dir_match.group(3))
                run_group = int(dir_match.group(4))
                key = (env_group, num_trajs_group)
                if key not in max_prev_runs.keys():
                    max_prev_runs[key] = run_group
                else:
                    if run_group > max_prev_runs[key]:
                        max_prev_runs[key] = run_group

    cmd_templates, outputfilenames, argdicts = [], [], []
    for task in spec["tasks"]:
        for num_trajs in spec["training"]["dataset_num_trajs"]:
            assert num_trajs <= spec["training"]["full_dataset_num_trajs"]

            key = (task["env"], num_trajs)
            if key in max_prev_runs.keys():
                start_idx = max_prev_runs[key] + 1
            else:
                start_idx = 0

            for run in range(start_idx, spec["training"]["runs"] + start_idx):
                strid = f"alg={alg.value},env={task['env']},num_trajs={str(num_trajs).zfill(3)},run={str(run).zfill(3)},tag={tag}"
                target_script = BIMANUAL_IMITATION_BASE_DIR / f"algorithms/imitate_{alg.value}.py"
                run_dir = export_dir / strid
                cmd_str = f"python3 -u {target_script} --mode train_policy --env_name {task['env']} --export_dir {run_dir} "
                cmd_str += (
                    f"--data_subsamp_freq {task['data_subsamp_freq']} --limit_trajs {num_trajs} "
                )

                cfg_class = get_param_config(alg)

                if alg == ALG.GAIL:
                    spec["params"]["favor_zero_expert_reward"] = int(task["cuts_off_on_success"])

                cfg = cfg_class(**spec["params"])

                for name, value in vars(cfg).items():
                    cmd_str += f"--{name} {value} "

                for name, value in spec["options"].items():
                    cmd_str += f"--{name} {value} "

                cmd_templates.append(cmd_str)
                outputfilename = run_dir / "output.log"
                outputfilenames.append(outputfilename)
                argdicts.append({})

    try:
        run_sbatch(
            alg,
            cmd_templates,
            outputfilenames,
            argdicts=argdicts,
            sh_script=sbatch_script,
            max_num_workers=args.max_num_workers,
            run_local=args.run_local,
        )

        if next_suffix == -1:
            spec_file_copy = pipeline_dir / f"{alg.value}_{tag}.yaml"
            shutil.copyfile(spec_file, spec_file_copy)

    except Exception as e:
        print(f"Failed with exception: {e}")
        if sbatch_script.exists():
            sbatch_script.unlink()


def phase4_train_analysis(args):
    util.header("=== Running Phase 4: Train Analysis ===")
    tag = str(args.tag).zfill(2)
    cur_sub_dir = "phase4_train_analysis"
    cur_export_dir = get_export_dir(args.date_id, cur_sub_dir)
    h5_filename = cur_export_dir / f"train_analysis_{tag}.h5"

    # if not h5_filename.exists():
    export_h5(
        args,
        h5_filename,
        prev_sub_dir="phase3_train",
        act_obs_dim=(18, 36),
        chunking_policies=[ALG.ACT, ALG.DIFFUSION],
        unique_num_trajs=(50, 100, 200),
        export_metadata=True,
        export_rollouts=True,
    )


def phase5_sensitivity(args):
    import itertools

    util.header("=== Running Phase 5: Sensitivity ===")
    max_sensitivity_workers = (
        args.max_sensitivity_workers if args.max_sensitivity_workers else np.inf
    )

    alg = get_enum_value(args.alg, ALG)
    tag = str(args.tag).zfill(2)
    sub_dir = "phase5_sensitivity"
    export_dir = get_export_dir(args.date_id, sub_dir)

    sbatch_dir = export_dir / "sbatch"
    pipeline_dir = export_dir / "pipelines"
    if not sbatch_dir.exists():
        sbatch_dir.mkdir()
    if not pipeline_dir.exists():
        pipeline_dir.mkdir()

    next_suffix = -1
    sbatch_pattern = rf"{re.escape(alg.value)}_{re.escape(tag)}" + r"-(\d{2})\.sh"
    for sb in sbatch_dir.iterdir():
        match = re.match(sbatch_pattern, sb.name)
        if match:
            assert sb.is_file()
            suffix = int(match.group(1))
            if suffix > next_suffix:
                next_suffix = suffix

    sbatch_script = sbatch_dir / f"{alg.value}_{tag}-{str(next_suffix+1).zfill(2)}.sh"

    if next_suffix == -1:
        if args.spec is None:
            spec_file = BIMANUAL_IMITATION_BASE_DIR / f"param/sensitivity/{alg.value}.yaml"
            with open(spec_file, "r") as f:
                spec = yaml.safe_load(f)
        else:
            spec_file = Path(args.spec)
            assert spec_file.exists()
            with open(spec_file, "r") as f:
                spec = yaml.safe_load(f)
    else:
        spec_file = pipeline_dir / f"{alg.value}_{tag}.yaml"
        assert spec_file.exists()
        with open(spec_file, "r") as f:
            spec = yaml.safe_load(f)
        util.header(f"Re-using {spec_file} for this run!")

    dir_pattern = r"^alg=(\w+),env=(\w+),num_trajs=(\d{3}),run=(\d{3}),tag=(\d{2})$"
    max_prev_runs = {}
    for dir_name in export_dir.iterdir():
        assert dir_name.is_dir()
        dir_match = re.match(dir_pattern, dir_name.name)
        if dir_match:
            alg_group = str(dir_match.group(1))
            tag_group = int(dir_match.group(5))
            env_group = str(dir_match.group(2))
            num_trajs_group = int(dir_match.group(3))
            if get_enum_value(alg_group, ALG) == alg and (tag_group == args.tag):
                run_group = int(dir_match.group(4))
                key = (env_group, num_trajs_group)
                if key not in max_prev_runs.keys():
                    max_prev_runs[key] = run_group
                else:
                    if run_group > max_prev_runs[key]:
                        max_prev_runs[key] = run_group

    cmd_templates, outputfilenames, argdicts = [], [], []

    for task in spec["tasks"]:
        for num_trajs in spec["training"]["dataset_num_trajs"]:
            assert num_trajs <= spec["training"]["full_dataset_num_trajs"]
            key = (task["env"], num_trajs)
            if key in max_prev_runs.keys():
                start_idx = max_prev_runs[key] + 1
            else:
                start_idx = 0

            alg_k = len(spec["param_min_max_vals"])
            if alg_k > 4:
                alg_ca = download_covering_array(k=alg_k)
            else:
                alg_ca = np.array(list(itertools.product([0, 1], repeat=alg_k)))

            hps = {}
            for start_run, ca_row in enumerate(alg_ca[start_idx:]):
                if start_run < max_sensitivity_workers:
                    run = start_run + start_idx
                    for hp_name, ca_val in zip(spec["param_min_max_vals"], ca_row):
                        if np.isnan(ca_val):
                            ca_val = np.random.rand() > 0.5
                        hp_val = spec["param_min_max_vals"][hp_name][int(ca_val)]
                        hps[hp_name] = hp_val

                    target_script = (
                        BIMANUAL_IMITATION_BASE_DIR / f"algorithms/imitate_{alg.value}.py"
                    )
                    strid = f"alg={alg.value},env={task['env']},num_trajs={str(num_trajs).zfill(3)},run={str(run).zfill(3)},tag={str(args.tag).zfill(2)}"
                    cmd_str = f"python3 -u {target_script} --mode train_policy --env_name {task['env']} --export_dir {export_dir / strid} "
                    cmd_str += f"--data_subsamp_freq {task['data_subsamp_freq']} --limit_trajs {num_trajs} "

                    cfg_class = get_param_config(alg)
                    if alg == ALG.GAIL:
                        hps["favor_zero_expert_reward"] = int(task["cuts_off_on_success"])

                    cfg = cfg_class(**hps)

                    for name, value in vars(cfg).items():
                        cmd_str += f"--{name} {value} "

                    for name, value in spec["options"].items():
                        cmd_str += f"--{name} {value} "

                    cmd_templates.append(cmd_str)
                    outputfilenames.append(export_dir / strid / "output.log")
                    argdicts.append({})

    if cmd_templates:
        try:
            run_sbatch(
                alg,
                cmd_templates,
                outputfilenames,
                argdicts=argdicts,
                sh_script=sbatch_script,
                max_num_workers=args.max_num_workers,
                run_local=args.run_local,
            )
            if next_suffix == -1:
                shutil.copyfile(spec_file, pipeline_dir / f"{alg.value}_{tag}.yaml")
        except:
            if sbatch_script.exists():
                sbatch_script.unlink()
    else:
        print("All covering array values already submitted!")


def phase6_sensitivity_analysis(args):
    util.header("=== Running Phase 6: Sensitivity Analysis ===")
    tag = str(args.tag).zfill(2)
    cur_sub_dir = "phase6_sensitivity_analysis"
    cur_export_dir = get_export_dir(args.date_id, cur_sub_dir)
    h5_filename = cur_export_dir / f"sensitivity_analysis_{tag}.h5"

    # if not h5_filename.exists():
    export_h5(
        args,
        h5_filename,
        prev_sub_dir="phase5_sensitivity",
        unique_num_trajs=(200,),
        export_rollouts=True,
        export_metadata=True,
    )

    df = pd.read_hdf(h5_filename)
    print(df)


if __name__ == "__main__":
    phases = {
        "1_hp_search": phase1_hp_search,
        "2_hp_search_analysis": phase2_hp_search_analysis,
        "3_train": phase3_train,
        "4_train_analysis": phase4_train_analysis,
        "5_sensitivity": phase5_sensitivity,
        "6_sensitivity_analysis": phase6_sensitivity_analysis,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str)
    parser.add_argument("--spec", type=str)
    parser.add_argument("--phase", choices=sorted(phases.keys()), required=True)
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--max_num_workers", type=int)
    parser.add_argument("--hp_search_num_workers", type=int, default=4)
    parser.add_argument("--hp_search_num_trials", type=int, default=10)
    parser.add_argument("--max_sensitivity_workers", type=int)
    parser.add_argument("--date_id", type=str)
    parser.add_argument("--tag", type=int, default=0)
    parser.add_argument("--run_local", action="store_true")
    args = parser.parse_args()

    phases[args.phase](args)
