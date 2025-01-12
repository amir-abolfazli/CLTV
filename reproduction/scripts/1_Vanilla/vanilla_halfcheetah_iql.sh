#!/bin/bash
#SBATCH --job-name=HCH_VAN_IQ_
#SBATCH --output=./logs/halfcheetah/VAN_IQ_output_%A_%a.out
#SBATCH --error=./logs/halfcheetah/VAN_IQ_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3121 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3122 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3123 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3124 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3125 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3126 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3127 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3128 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3129 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3130 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3131 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3132 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3133 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3134 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Vanilla --seed 3135 &
wait
