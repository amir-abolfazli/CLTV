import os
import argparse
import d3rlpy
import d4rl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import wandb
from UtilsRL.exp import parse_args


def harness(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    d3rlpy.seed(args.seed)

    args_ex = parse_args("./cql/params.py")
    for k, v in args_ex.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # logging
    wandb.init(
        project=args.project,
        name=f"CQL(harness)_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
        config=args
    )

    from cql.common import wrap_sampler
    from ts.datasets import mix_mdp_mujoco_datasets
    # d3rlpy dataset
    env, dataset = mix_mdp_mujoco_datasets(args.env,
                      n=args.dataset_size,
                      dataset_types=args.dataset_types,
                      ratios=args.dataset_ratios)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    td3 = wrap_sampler(args.sampler, d3rlpy.algos.CQL)(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0.0001,
        alpha_learning_rate=0.0001,
        batch_size=256,
        scaler="standard",
        use_gpu=args.gpu
    )

    num_episodes = 100
    for e in range(num_episodes):
        results = td3.fit(dataset.episodes,
                eval_episodes=test_episodes,
                n_steps=5000,
                n_steps_per_epoch=1000,
                save_interval=10,
                scorers={
                    'environment': d3rlpy.metrics.evaluate_on_environment(env),
                    'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
                },
                experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}")

        avg_critic_loss = 0
        avg_actor_loss = 0
        avg_environment = 0
        avg_value_scale = 0
        for result in results:
            avg_critic_loss += result[1]['critic_loss'].item()
            avg_actor_loss += result[1]['actor_loss'].item()
            avg_environment += result[1]['environment'] / d4rl.infos.REF_MAX_SCORE[args.env + '-expert-v2'] * 100
            avg_value_scale += result[1]['value_scale']
        metrics = {
            'critic_loss': avg_critic_loss / len(results),
            'actor_loss': avg_actor_loss / len(results),
            'environment': avg_environment / len(results),
            'value_scale': avg_value_scale / len(results)
        }
        wandb.log(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--sampler', type=str, default="uniform")
    parser.add_argument('--dataset_size', type=int, default=int(1e6))
    parser.add_argument('--dataset_types', type=str, nargs="+", required=True)
    parser.add_argument('--dataset_ratios', type=float, nargs="+", required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--project', type=str)
    args = parser.parse_args()

    harness(args)