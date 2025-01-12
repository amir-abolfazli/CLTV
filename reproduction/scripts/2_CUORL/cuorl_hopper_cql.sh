#!/bin/bash
#SBATCH --job-name=HOP_CU_CQL_
#SBATCH --output=./logs/hopper/CU_CQL_output_%A_%a.out
#SBATCH --error=./logs/hopper/CU_CQL_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1211 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1212 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1213 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1214 &
python3 experiment.py --gpu 0 --env hopper --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1215 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1216 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1217 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1218 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1219 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1220 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1221 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1222 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1223 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1224 &
python3 experiment.py --gpu 0 --env hopper --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method CUORL --seed 1225 &
wait
