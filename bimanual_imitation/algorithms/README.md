### Source Code / Acknowledgements

The algorithms in this folder are based on the following sources. Please refer to the works below:

- **ACT**: [Code](https://github.com/tonyzhaozh/act) and [Paper](https://arxiv.org/pdf/2304.13705)
- **Diffusion** [Code](https://github.com/real-stanford/diffusion_policy) and [Paper](https://arxiv.org/pdf/2303.04137v4)
- **IBC**: [Code](https://github.com/google-research/ibc) and [Paper](https://arxiv.org/pdf/2109.00137)
- **GAIL**: [Code](https://github.com/openai/imitation) and [Paper](https://arxiv.org/pdf/1606.03476)

### Training the Policies

To train the policies, simply run: 

```
conda activate <conda_env_name>
python imitate_<algorithm>.py
```
where:
- \<algorithm\> should be replaced with the method you intend to use: for example, `diffusion`
- \<conda_env_name\> is replaced according to the main README: for example, `irl_torch`

**NOTE**: you can pass `--help` to the script to view the available arguments.

When running the script, make sure to use the correct conda environment:
- `conda activate irl_torch` for: [**ACT**, **Diffusion**] (and experimental: **BC**)
- `conda activate irl_tensorflow` for: [**IBC**]
- `conda activate irl_theano` for: [**GAIL**, **DAgger**, **BC**]

**NOTE**: you can pass `--help` to the script to view the available arguments.



