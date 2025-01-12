# Curriculum Learning-Based Trajectory Valuation (CLTV)

## Setup

### Option 1: Docker Container
Change to root directory of project.
```bash
cd CLTV
```
Create docker image.
```bash
docker build -t <image name> .
```
Create docker container.
```bash
docker create -it --gpus 'all' --name <container name> -v <path/to/project>:/CLTV <docker image name> bash
```
Start container in interation mode.
```bash
docker start -i <docker container name>
```

### Option 2: GPUs Cluster using Slurm
Create Anaconda environment and activate the environment
```bash
conda create -n <env name> python=3.10
conda activate <env name>
```
Change to root directory of project.
```bash
cd CLTV
```
Run setup.sh
```bash
sh setup.sh
```

## Experiments

### On Local Machine

Reproduce any type of experiments.
```bash
python3 experiment.py --gpu <node number> --seed <seed number> --env <env name> --dataset_types <type1 type2> --dataset_ratios <ratio1 ratio2> --baseline <baseline name> --method <method name>
```
- `--gpu`
  - GPU node number you would like to use. (e.g. `--gpu 0`)
- `--seed`
  - Seed for experiment. (e.g. `--seed 4111`)
- `--env`
  - Mujoco environment for experiment. (e.g. `--env ant`)
- `--dataset_types`
  - Pair of types for source and target datasets. (e.g. `--dataset_types random-v2 medium-v2`)
  - `<type1>` specifies source dataset.
  - `<type2>` specifies target dataset.
- `--dataset_ratios`
  - Pair of ratios for mixing source and target datasets. (e.g. `--dataset_ratios 0.9 0.1`)
  - `<ratio1>` specifies ratio of source dataset.
  - `<ratio2>` specifies ratio of target dataset.
- `--baseline`
  - Offline RL algorithm used. (e.g. `--baseline CQL`)
  - All options are: CQL, and IQL.
- `--method`
  - Method for experiment. (e.g. `--method Vanilla`)
  - All options are: Vanilla, CUORL, Harness, CLTV, TS, reward_TD.

### On GPUs Cluster using Slurm

We provide all bash scripts for running experiments on Slurm, they can be found in `reproduction/scripts/`. <br>
For running bash on Slurm:
```bash
sbatch reproduction/scripts/<path/to/script>
```
Examples: <br>
Vanilla
```bash
sbatch reproduction/scripts/1_Vanilla/vanilla_ant_cql.sh
```
CUORL
```bash
sbatch reproduction/scripts/2_CUORL/cuorl_ant_cql.sh
```
Harness
```bash
sbatch reproduction/scripts/3_Harness/harness_ant_cql.sh
```
CLTV
```bash
sbatch reproduction/scripts/4_CLTV/cltv_ant_cql.sh
```
TS
```bash
sbatch reproduction/scripts/5_TS/ts_withoutCL_ant_cql.sh
```
Reward1
```bash
sbatch reproduction/scripts/Ablation_Study/reward_functions/Reward1/abs_r1_ant_cql.sh
```
Reward2
```bash
sbatch reproduction/scripts/Ablation_Study/reward_functions/Reward2/abs_r2_ant_cql.sh
```
PairImpact
```bash
sbatch reproduction/scripts/Ablation_Study/PairImpact/Seed_500/abs_pairImpact_ant_cql.sh
```

### Tables and Plots

To produce table 1:
```bash
jupyter notebook reproduction/Table_Results.ipynb
```
To produce figure 1, 2, 3:
```bash
jupyter notebook reproduction/Learning_Curves.ipynb
```
To produce figure 4:
```bash
jupyter notebook reproduction/Runtime_Analysis.ipynb
```
To produce figure 5:
```bash
jupyter notebook reproduction/Ablation_Study.ipynb
```

## Acknowledgments

* [harness-offline-rl](https://github.com/Improbable-AI/harness-offline-rl.git)
* [d3rlpy](https://github.com/takuseno/d3rlpy.git)
* [D4RL](https://github.com/Farama-Foundation/D4RL.git)
* [UtilsRL](https://github.com/typoverflow/UtilsRL.git)
