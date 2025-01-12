#!/bin/bash
#SBATCH --job-name=HCH_CU_CQL_
#SBATCH --output=./logs/halfcheetah/CU_CQL_output_%A_%a.out
#SBATCH --error=./logs/halfcheetah/CU_CQL_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3211 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3212 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3213 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3214 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3215 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3216 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3217 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3218 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3219 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3220 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3221 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3222 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3223 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3224 &
python3 experiment.py --gpu 0 --env halfcheetah --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 3225 &
wait
