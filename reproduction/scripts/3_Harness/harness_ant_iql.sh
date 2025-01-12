#!/bin/bash
#SBATCH --job-name=HCH_HAR_IQ_
#SBATCH --output=./logs/ant/HAR_IQ_output_%A_%a.out
#SBATCH --error=./logs/ant/HAR_IQ_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4321 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4322 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4323 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4324 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4325 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4326 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4327 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4328 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4329 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4330 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4331 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4332 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4333 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4334 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method Harness --seed 4335 &
wait
