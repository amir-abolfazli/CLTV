#!/bin/bash
#SBATCH --job-name=HOP_VAN_IQ_
#SBATCH --output=./logs/hopper/VAN_IQ_output_%A_%a.out
#SBATCH --error=./logs/hopper/VAN_IQ_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1121 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1122 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1123 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1124 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1125 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1126 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1127 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1128 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1129 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1130 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1131 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1132 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1133 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1134 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 1135 &
wait
