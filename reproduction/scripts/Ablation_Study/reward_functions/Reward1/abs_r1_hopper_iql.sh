#!/bin/bash
#SBATCH --job-name=HOP_ABS_R1_IQ_
#SBATCH --output=./logs/hopper/TrajV_IQ_output_%A_%a.out
#SBATCH --error=./logs/hopper/TrajV_IQ_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1421 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1422 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1423 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1424 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1425 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1426 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1427 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1428 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1429 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1430 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1431 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1432 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1433 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1434 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation --baseline iql --method reward1 --seed 1435 &
wait
