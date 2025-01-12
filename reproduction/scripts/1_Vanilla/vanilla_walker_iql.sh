#!/bin/bash
#SBATCH --job-name=WA_VAN_IQ_
#SBATCH --output=./logs/walker2d/VAN_IQ_output_%A_%a.out
#SBATCH --error=./logs/walker2d/VAN_IQ_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2121 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2122 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2123 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2124 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2125 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2126 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2127 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2128 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2129 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2130 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2131 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2132 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2133 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2134 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 2135 &
wait
