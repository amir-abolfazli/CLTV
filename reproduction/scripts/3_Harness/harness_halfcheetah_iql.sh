#!/bin/bash
#SBATCH --job-name=HCH_HAR_IQ_
#SBATCH --output=./logs/halfcheetah/HAR_IQ_output_%A_%a.out
#SBATCH --error=./logs/halfcheetah/HAR_IQ_error_%A_%a.err
#SBATCH --nodelist=gpunode02
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3321 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3322 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3323 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3324 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3325 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3326 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3327 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3328 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3329 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3330 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3331 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3332 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3333 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3334 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 3335 &
wait
