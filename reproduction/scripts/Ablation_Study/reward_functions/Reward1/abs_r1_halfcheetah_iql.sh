#!/bin/bash
#SBATCH --job-name=HCH_ABS_R1_IQ_
#SBATCH --output=./logs/halfcheetah/TrajV_IQ_output_%A_%a.out
#SBATCH --error=./logs/halfcheetah/TrajV_IQ_error_%A_%a.err
#SBATCH --nodelist=gpunode01
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3421 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3422 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3423 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3424 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3425 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3426 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3427 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3428 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3429 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3430 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3431 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3432 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3433 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3434 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 3435 &
wait
