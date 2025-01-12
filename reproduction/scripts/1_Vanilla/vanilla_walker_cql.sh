#!/bin/bash
#SBATCH --job-name=WA_VAN_CQL_
#SBATCH --output=./logs/walker2d/VAN_CQL_output_%A_%a.out
#SBATCH --error=./logs/walker2d/VAN_CQL_error_%A_%a.err
#SBATCH --nodelist=gpunode05
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2111 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2112 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2113 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2114 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2115 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2116 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2117 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2118 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2119 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2120 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2121 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2122 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2123 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2124 &
python3 experiment.py --gpu 0 --env walker2d --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project WB_CLTV --baseline CQL --method Vanilla --seed 2125 &
wait
