#!/bin/bash
#SBATCH --job-name=HCH_CU_CQL_
#SBATCH --output=./logs/ant/CU_CQL_output_%A_%a.out
#SBATCH --error=./logs/ant/CU_CQL_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4211 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4212 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4213 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4214 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4215 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4216 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4217 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4218 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4219 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4220 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4221 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4222 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4223 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4224 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 4225 &
wait
