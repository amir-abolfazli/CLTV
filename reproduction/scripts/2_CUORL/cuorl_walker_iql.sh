#!/bin/bash
#SBATCH --job-name=WA_CU_IQ_
#SBATCH --output=./logs/walker2d/CU_IQ_output_%A_%a.out
#SBATCH --error=./logs/walker2d/CU_IQ_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2221 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2222 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2223 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2224 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2225 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2226 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2227 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2228 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2229 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2230 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2231 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2232 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2233 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2234 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline IQL --method CUORL --seed 2235 &
wait
