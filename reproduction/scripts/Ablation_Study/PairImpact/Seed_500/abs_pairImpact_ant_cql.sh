#!/bin/bash
#SBATCH --job-name=ANT_ABS_PAIR_CQL_
#SBATCH --output=./logs/ant/TrajV_CQL_output_%A_%a.out
#SBATCH --error=./logs/ant/TrajV_CQL_error_%A_%a.err
#SBATCH --nodelist=gpunode03
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=40-0

source /opt/conda/etc/profile.d/conda.sh
conda activate cltv



python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.0 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.2 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.4 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.6 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.8 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types medium-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 1.0 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.0 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.2 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.4 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.6 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.8 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 expert-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 1.0 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.0 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.2 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.4 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.6 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 0.8 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
python3 experiment.py --gpu 0 --env ant --dataset_types random-v2 medium-v2 --dataset_ratios 0.9 0.1 --project Ablation_PairImpact --baseline cql --method pairimpact --delta 1.0 --lmd 0.0 0.2 0.4 0.6 0.8 1.0 --seed 500 &
wait
