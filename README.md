# A Comparison of Imitation Learning Algorithms for Bimanual Manipulation

This codebase contains the implementation of the algorithms and environments evaluated in [*A Comparison of Imitation Learning Algorithms for Bimanual Manipulation*](https://bimanual-imitation.github.io/).


### Installation

#### Step 1:
*Install the conda environment*. Depending on your platform, install the correct `<arch> = x86` or `<arch> = arm` environments, located in the [requirements](./requirements) folder.  The installation command for the corresponding algorithms are given below:
- `conda env create -f torch_<arch>.yml` for: [**ACT**, **Diffusion**] (and experimental: **BC**)
- `conda env create -f tensorflow_<arch>.yml` for: [**IBC**]
- `conda env create -f theano_x86.yml` for: [**GAIL**, **DAgger**, **BC**] (x86 only)


#### Step 2:
*Activate the conda environment*. The conda environments for the corresponding algorithms are given below:
- `conda activate irl_torch` for: [**ACT**, **Diffusion**] (and experimental: **BC**)
- `conda activate irl_tensorflow` for: [**IBC**]
- `conda activate irl_theano` for: [**GAIL**, **DAgger**, **BC**]

#### Step 3:

*Install bimanual_imitation*. Change to main repo directory and run: `pip install -e .` to install the bimanual_imitation library.


### Training the Policies

To train the policies, go to the [bimanual_imitation/algorithms](./bimanual_imitation/algorithms/) folder and run: 

```
conda activate <conda_env_name>
python imitate_<algorithm>.py
```
where:
- \<algorithm\> is replaced with the method you intend to use: for example, `diffusion`
- \<conda_env_name\> is replaced according to [Step 2](#step-2) in the section above: for example, `irl_torch`

**Note**: you can pass `--help` to the script to view the available arguments. You can also modify the default [configs](./bimanual_imitation/algorithms/configs.py) accordingly.


### Running the Quad Peg Insertion Task

To visualize the gymnasium environment used for training the policies, go to the [irl_environments](./irl_environments/) folder and run:

```
python bimanual_quad_insert.py
```

You can modify the bottom of the script to run different environments, export gifs, and plot the features.<br>
Environments are named as `quad_insert_<action_noise><observation_noise>`, where:

- <action_noise> is replaced with: `a0` (None), `aL` (Low), `aM` (Medium/High)
- <observation_noise> is replaced with: `o0` (None), `oL` (Low), `oM` (Medium/High)

For example: `quad_insert_a0o0`, `quad_insert_aLoL`, and `quad_insert_aMoM`.

### Running the pipeline on a slurm cluster
To train the  the policies on the cluster, go to the [bimanual_imitation](./bimanual_imitation/) folder and run:

```
python pipeline.py --phase <phase_name> --alg <alg_name> --env_name <gym_env_name>
```

Example Usage:

```
python pipeline.py --phase 3_train --alg act --env_name quad_insert_aLoL
```
**Note:** you can pass `--help` to the script to view the available arguments. To run locally, pass the `--run_local` argument. Modify the [slurm configuration](./bimanual_imitation/utils/slurm.py) according to your available resources.

After running the pipeline, you can generate rollouts and videos of the policy by running: `python generate_trajs.py`

### Loading Expert Datasets

The expert datasets (stored as protobuf files) are located in the [irl_data/expert_trajectories](./irl_data/expert_trajectories/) folder. You can extract these to a trajectory class using the `load_trajs` function inside of the [proto_logger](./irl_data/proto_logger.py).


### Acknowledgements
Please refer to the [algorithms](./bimanual_imitation/algorithms/) folder, where you can find work from the original authors of the compared methods.

