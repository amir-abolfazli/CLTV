#!/bin/bash
#SBATCH --job-name=WA_ABS_WCL_IQ_
#SBATCH --output=./logs/walker2d/TrajV_IQ_output_%A_%a.out
#SBATCH --error=./logs/walker2d/TrajV_IQ_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2421 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2422 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2423 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2424 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2425 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2426 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2427 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2428 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2429 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2430 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2431 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2432 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2433 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2434 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method TS --seed 2435 &
wait
