#!/bin/bash
#SBATCH --job-name=HOP_HAR_CQL_
#SBATCH --output=./logs/hopper/HAR_CQL_output_%A_%a.out
#SBATCH --error=./logs/hopper/HAR_CQL_error_%A_%a.err
#SBATCH --nodelist=gpunode101
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1311 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1312 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1313 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1314 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1315 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1316 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1317 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1318 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1319 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1320 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1321 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1322 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1323 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1324 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Harness --seed 1325 &
wait
