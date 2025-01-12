#!/bin/bash
#SBATCH --job-name=ANT_ABS_R1_CQL_
#SBATCH --output=./logs/ant/TrajV_CQL_output_%A_%a.out
#SBATCH --error=./logs/ant/TrajV_CQL_error_%A_%a.err
#SBATCH --nodelist=gpunode03
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4411 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4412 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4413 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4414 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4415 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4416 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4417 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4418 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4419 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4420 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4421 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4422 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4423 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4424 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline cql --method reward1 --seed 4425 &
wait
