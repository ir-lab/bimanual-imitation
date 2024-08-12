import subprocess
import tempfile
from dataclasses import dataclass
from os import path
from pathlib import Path
from typing import Dict

from bimanual_imitation.algorithms.configs import ALG

GPU_SBATCH = """
#SBATCH --job-name=bimanual_imitation
#SBATCH -p gpu
#SBATCH --gres=gpu:{num_gpus}
#SBATCH -c {num_cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -t {time}
"""


# SBATCH -p {queue}
CPU_SBATCH = """
#SBATCH --job-name=bimanual_imitation
#SBATCH --cpus-per-task {num_cpus}
#SBATCH --mem-per-cpu {mem_per_cpu}
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -t {time}   # time limit: (D-HH:MM)
"""


@dataclass
class SbatchCfg:
    queue: str
    num_cpus: int
    mem_per_cpu: int
    time: str


@dataclass
class CpuCfg(SbatchCfg):
    pass


@dataclass
class GpuCfg(SbatchCfg):
    num_gpus: int


SBATCH_CFGS: Dict[ALG, SbatchCfg] = {
    ALG.BCLONE: CpuCfg(queue="htc", num_cpus=8, mem_per_cpu=2000, time="0-04:00"),
    ALG.IBC: GpuCfg(queue="gpu", num_gpus=2, num_cpus=2, mem_per_cpu=8000, time="0-60:00"),
    ALG.DAGGER: CpuCfg(queue="parallel", num_cpus=16, mem_per_cpu=2000, time="0-60:00"),
    ALG.GAIL: CpuCfg(queue="parallel", num_cpus=16, mem_per_cpu=2000, time="0-72:00"),
    ALG.DIFFUSION: GpuCfg(queue="gpu", num_gpus=1, num_cpus=2, mem_per_cpu=4000, time="0-4:00"),
    ALG.ACT: GpuCfg(queue="gpu", num_gpus=1, num_cpus=1, mem_per_cpu=8000, time="0-60:00"),
}


CONDA_ENVS: Dict[ALG, str] = {
    ALG.BCLONE: "irl_theano",
    ALG.IBC: "irl_tensorflow",
    ALG.DAGGER: "irl_theano",
    ALG.GAIL: "irl_theano",
    ALG.DIFFUSION: "irl_torch",
    ALG.ACT: "irl_torch",
}


def create_sbatch_script(alg, sbatch_str, commands, outputfiles, sleep_time=0):
    assert len(commands) == len(outputfiles)
    template = """#!/bin/bash
{sbatch}

read -r -d '' COMMANDS << END
{cmds_str}
END
cmd=$(echo "$COMMANDS" | awk "NR == $SLURM_ARRAY_TASK_ID")
echo $cmd

read -r -d '' OUTPUTFILES << END
{outputfiles_str}
END
outputfile=$(echo "$OUTPUTFILES" | awk "NR == $SLURM_ARRAY_TASK_ID")
echo $outputfile
# Make sure output directory exists
mkdir -p "`dirname \"$outputfile\"`" 2>/dev/null

echo $cmd >$outputfile

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

eval "$(conda shell.bash hook)"
conda activate {conda_env}

# prevent scripts from all launching at same time for hp search database
length_seconds=$(((SLURM_ARRAY_TASK_ID - 1) * {sleep_time}))
sleep $length_seconds

$cmd 2>&1 | tee $outputfile
"""
    return template.format(
        sbatch=sbatch_str,
        cmds_str="\n".join(commands),
        outputfiles_str="\n".join(outputfiles),
        conda_env=CONDA_ENVS[alg],
        sleep_time=sleep_time,
    )


def create_local_script(alg, commands, outputfiles):
    # Ensure both arrays have the same length
    if len(commands) != len(outputfiles):
        raise ValueError("commands and output_files must have the same length")

    # Create the shell script content
    script_lines = [f"conda activate {CONDA_ENVS[alg]}"]
    for cmd, out_file in zip(commands, outputfiles):
        # Add each command with redirection to the output file
        directory = path.dirname(out_file)
        # Ensure the directory is created
        if directory:
            script_lines.append(f"mkdir -p {directory}")
        script_lines.append(f"{cmd} 2>&1 | tee {out_file}")

    # Join all lines with newlines
    script_content = "\n".join(script_lines)

    return script_content


def run_sbatch(
    alg,
    cmd_templates,
    outputfilenames,
    argdicts,
    sh_script=None,
    max_num_workers=None,
    run_local=False,
    sleep_time=0,
):

    assert len(cmd_templates) == len(outputfilenames) == len(argdicts)
    num_cmds = len(cmd_templates)

    cmds, outputfiles = [], []
    for i in range(num_cmds):
        cmds.append(cmd_templates[i].format(**argdicts[i]))
        output_file = Path(outputfilenames[i])
        outputfiles.append(str(output_file))

    if run_local:
        script = create_local_script(alg, cmds, outputfiles)
        with open(sh_script, "wb") as f:
            f.write(script.encode())
            f.flush()
        print("Created script! Run the following command:")
        print(f"source {f.name}")
    else:
        sbatch_cfg = SBATCH_CFGS[alg]
        if isinstance(sbatch_cfg, GpuCfg):
            sbatch_str = GPU_SBATCH.format(
                queue=sbatch_cfg.queue,
                num_gpus=sbatch_cfg.num_gpus,
                num_cpus=sbatch_cfg.num_cpus,
                mem_per_cpu=sbatch_cfg.mem_per_cpu,
                time=sbatch_cfg.time,
            )
        elif isinstance(sbatch_cfg, CpuCfg):
            sbatch_str = CPU_SBATCH.format(
                queue=sbatch_cfg.queue,
                num_cpus=sbatch_cfg.num_cpus,
                mem_per_cpu=sbatch_cfg.mem_per_cpu,
                time=sbatch_cfg.time,
            )
        else:
            raise ValueError

        script = create_sbatch_script(alg, sbatch_str, cmds, outputfiles, sleep_time)

        with tempfile.NamedTemporaryFile(suffix=".sh") as f:
            f.write(script.encode())
            f.flush()

            if max_num_workers is None:
                cmd = "sbatch --array=[%d-%d] %s" % (1, len(cmds), f.name)
            else:
                cmd = "sbatch --array=[%d-%d]%%%d %s" % (1, len(cmds), int(max_num_workers), f.name)

            print("Running command:", cmd)
            print("ok ({} jobs)? y/n".format(num_cmds))
            if input() == "y":
                # Write a copy of the script
                if sh_script is not None:
                    # assert not os.path.exists(qsub_script_copy)
                    with open(sh_script, "w") as fcopy:
                        fcopy.write(script)
                    print(f"qsub script written to {sh_script}")
                # Run qsub
                subprocess.check_call(cmd, shell=True)

            else:
                raise RuntimeError("Canceled.")
