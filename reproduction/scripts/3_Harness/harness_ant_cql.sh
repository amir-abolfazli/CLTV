#!/bin/bash
#SBATCH --job-name=HCH_HAR_CQL_
#SBATCH --output=./logs/ant/HAR_CQL_output_%A_%a.out
#SBATCH --error=./logs/ant/HAR_CQL_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4311 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4312 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4313 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4314 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4315 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4316 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4317 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4318 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4319 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4320 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4321 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4322 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4323 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4324 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 4325 &
wait
