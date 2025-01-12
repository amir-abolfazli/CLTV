import argparse
import os
import pandas as pd
import d3rlpy
import d4rl
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
from pathlib import Path
from ts.datasets import mix_mdp_mujoco_datasets_new
from ts.replay_buffer import ReplayBuffer
from ts.delta_classifier import DeltaCla
from ts.ts import DVRL
from ts.cltv import modify_rewards
from UtilsRL.exp import parse_args


def baseline(args):
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
        name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
        config=args
    )

    # d3rlpy dataset
    env, dataset_source, dataset_target = mix_mdp_mujoco_datasets_new(args.env,
                                                                      n=args.dataset_size,
                                                                      dataset_types=args.dataset_types,
                                                                      ratios=args.dataset_ratios)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    dataset_source.extend(dataset_target)
    dataset_train = dataset_source

    _, test_episodes = train_test_split(dataset_train, test_size=0.2)

    # Train
    policy = d3rlpy.algos.CQL(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0.0001,
        alpha_learning_rate=0.0001,
        batch_size=256,
        scaler="standard",
        use_gpu=args.gpu
    )

    args.num_epochs = 100
    for epoch in range(args.num_epochs):
        results = policy.fit(
            dataset_train,
            eval_episodes=test_episodes,
            n_steps=5000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}"
        )

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

def without_CL(args):
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
        name=f"CQL(withoutCL)_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
        config=args
    )

    # d3rlpy dataset
    env, dataset_source, dataset_target = mix_mdp_mujoco_datasets_new(args.env,
                                                                      n=args.dataset_size,
                                                                      dataset_types=args.dataset_types,
                                                                      ratios=args.dataset_ratios)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    source_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for episode in tqdm(dataset_source.episodes):
        for transition in episode:
            source_buffer.add(
                transition.observation,
                transition.action,
                transition.next_observation,
                transition.reward,
                transition.terminal
            )
        if not source_buffer.terminal[source_buffer.ptr-1]:
            source_buffer.terminal[source_buffer.ptr-1] = int(1)
    source_buffer.create_trajs()
    target_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for episode in tqdm(dataset_target.episodes):
        for transition in episode:
            target_buffer.add(
                transition.observation,
                transition.action,
                transition.next_observation,
                transition.reward,
                transition.terminal
            )
        if not target_buffer.terminal[target_buffer.ptr-1]:
            target_buffer.terminal[target_buffer.ptr-1] = int(1)
    target_buffer.create_trajs()

    ###  Train Delta Classifier
    delta = DeltaCla(env.observation_space.shape[0], env.action_space.shape[0], args)
    delta_model_path = f"./models/cql/{args.env}/{args.dataset_types[0]}_{args.dataset_types[1]}"
    # Define the model file names
    cf_model_files = ['cla_sa', 'cla_sas']
    # Check if the model files exist
    if all(Path(delta_model_path + '/delta_models/' + f).is_file() for f in cf_model_files):
        # If the model files exist, load them
        print('Loading delta classifiers...')
        delta.load_delta_models(delta_model_path)
        print('Finished loading delta classifiers')
    else:
        # If the model files do not exist, train the models and save them
        print('Start training delta classifiers...')
        delta.train(source_buffer, target_buffer, args)
        # for model_file in cf_model_files:
        delta.save_delta_models(delta_model_path)
        print('Finished training delta classifiers')

    ### Train DVE
    results_dir = f"./models/cql/{args.env}/{args.dataset_types[0]}_{args.dataset_types[1]}/dve"
    dvrl = DVRL(source_buffer, target_buffer, args.dev, env, results_dir, args.ex_configs, args)
    # Define the model file names
    dve_model_files = ['reinforce_final', 'reinforce_optimizer_final']
    # Check if the model files exist
    if all(Path("%s/dvrl_models/Trained_With_Seed_%d_Friction_%f_Mass_%f_Gamma_%f/" % \
                (results_dir, args.source_seed, args.source_env_friction,
                 args.source_env_mass_torso, args.discount) + f).is_file() for f in dve_model_files):
        # If the model files exist, load them
        print('Loading DVE...')
        dvrl.load_dve(dvrl.model_dir, type='final')
    else:
        # If the model files do not exist, train the models and save them
        print('Start training DVE...')
        dvrl.train()
        print('Finished training DVE...')

    dve_out, sel_vec = dvrl.data_valuate(source_buffer, args.batch_size)
    source_buffer = modify_rewards(source_buffer, dve_out, ratio=args.modify_ratio)

    dataset_source_new = d3rlpy.dataset.MDPDataset(
        source_buffer.state[:source_buffer.size],
        source_buffer.action[:source_buffer.size],
        source_buffer.reward[:source_buffer.size],
        source_buffer.terminal[:source_buffer.size]
    )
    dataset_source_new.extend(dataset_target)
    dataset_train = dataset_source_new

    _, test_episodes = train_test_split(dataset_train, test_size=0.2)

    # Train
    policy = d3rlpy.algos.CQL(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0.0001,
        alpha_learning_rate=0.0001,
        batch_size=256,
        scaler="standard",
        use_gpu=args.gpu
    )

    args.num_epochs = 100
    for epoch in range(args.num_epochs):
        results = policy.fit(
            dataset_train,
            eval_episodes=test_episodes,
            n_steps=5000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}"
        )

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

def reward1(args):
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
        name=f"CQL(reward1)_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
        config=args
    )

    # d3rlpy dataset
    env, dataset_source, dataset_target = mix_mdp_mujoco_datasets_new(args.env,
                                                                      n=args.dataset_size,
                                                                      dataset_types=args.dataset_types,
                                                                      ratios=args.dataset_ratios)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    dataset_source.extend(dataset_target)
    dataset_train = dataset_source

    policy = d3rlpy.algos.CQL(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0.0001,
        alpha_learning_rate=0.0001,
        batch_size=256,
        scaler="standard",
        use_gpu=args.gpu
    )

    policy.fit(
        dataset_train,
        n_steps=50000,
        n_steps_per_epoch=1000,
        save_interval=10,
        save_metrics=False
    )

    # apply reward function
    train_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for episode in tqdm(dataset_train.episodes):
        for transition in episode:
            state = transition.observation
            action = transition.action
            next_state = transition.next_observation
            reward = transition.reward
            terminal = transition.terminal

            if transition.next_transition is not None:
                advantage = reward + policy.gamma * policy.predict_value(np.expand_dims(next_state, axis=0),
                                                                         np.expand_dims(
                                                                             transition.next_transition.action,
                                                                             axis=0)) - \
                            policy.predict_value(np.expand_dims(state, axis=0), np.expand_dims(action, axis=0))
            else:
                advantage = reward - policy.predict_value(np.expand_dims(state, axis=0), np.expand_dims(action, axis=0))

            train_buffer.add(
                state,
                action,
                next_state,
                advantage,
                terminal
            )
            if not train_buffer.terminal[train_buffer.ptr - 1]:
                train_buffer.terminal[train_buffer.ptr - 1] = int(1)

    dataset_train = d3rlpy.dataset.MDPDataset(
        train_buffer.state[:train_buffer.size],
        train_buffer.action[:train_buffer.size],
        train_buffer.reward[:train_buffer.size],
        train_buffer.terminal[:train_buffer.size]
    )

    _, test_episodes = train_test_split(dataset_train, test_size=0.2)

    # Train
    policy = d3rlpy.algos.CQL(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0.0001,
        alpha_learning_rate=0.0001,
        batch_size=256,
        scaler="standard",
        use_gpu=args.gpu
    )

    import logging
    avg_rewards = []
    logs = []

    args.num_epochs = 100
    for epoch in range(args.num_epochs):
        results = policy.fit(
            dataset_train,
            eval_episodes=test_episodes,
            n_steps=5000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}"
        )

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
        logs.append(metrics)
        wandb.log(metrics)

        avg_r = avg_environment / len(results)
        avg_rewards.append(avg_r)
        logging.info(f"Epoch: [{epoch}/{args.num_epochs}], Average Reward: {avg_r}")

    if not Path(f"./results/cql/reward1/{args.env}/{'_'.join(args.dataset_types)}").exists():
        os.makedirs(f"./results/cql/reward1/{args.env}/{'_'.join(args.dataset_types)}")
    np.save(f"./results/cql/reward1/{args.env}/{'_'.join(args.dataset_types)}/{args.seed}.npy", avg_rewards)
    df = {
        'critic_loss': [],
        'actor_loss': [],
        'environment': [],
        'value_scale': []
    }
    for log in logs:
        df['critic_loss'].append(log['critic_loss'])
        df['actor_loss'].append(log['actor_loss'])
        df['environment'].append(log['environment'])
        df['value_scale'].append(log['value_scale'])
    df = pd.DataFrame(data=df)
    df.to_csv(f"./results/cql/reward1/{args.env}/{'_'.join(args.dataset_types)}/{args.seed}.csv")

def reward2(args):
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

    # modify torso and friction if applied
    if args.torso is not None:
        import gym
        from xml.etree import ElementTree as ET
        env_name_map = {
            'ant': 'ant',
            'hopper': 'hopper',
            'walker2d': 'walker2d',
            'halfcheetah': 'half_cheetah'
        }
        path = os.path.join(os.path.dirname(gym.__file__), 'envs', 'mujoco', 'assets',
                            f'{env_name_map[args.env]}.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()
        for geom in root.iter('geom'):
            if geom.get('name') == 'torso_geom':
                geom.set('size', str(args.torso))
    if args.friction is not None:
        import gym
        from xml.etree import ElementTree as ET
        env_name_map = {
            'ant': 'ant',
            'hopper': 'hopper',
            'walker2d': 'walker2d',
            'halfcheetah': 'half_cheetah'
        }
        path = os.path.join(os.path.dirname(gym.__file__), 'envs', 'mujoco', 'assets',
                            f'{env_name_map[args.env]}.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()
        for geom in root.iter('geom'):
            if geom.get('name') == 'foot_geom':
                geom.set('friction', str(args.friction))

    # logging
    wandb.init(
        project=args.project,
        name=f"CQL(reward2)_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
        config=args
    )

    # d3rlpy dataset
    env, dataset_source, dataset_target = mix_mdp_mujoco_datasets_new(args.env,
                                                                      n=args.dataset_size,
                                                                      dataset_types=args.dataset_types,
                                                                      ratios=args.dataset_ratios)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    dataset_source.extend(dataset_target)
    dataset_train = dataset_source

    policy = d3rlpy.algos.CQL(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0.0001,
        alpha_learning_rate=0.0001,
        batch_size=256,
        scaler="standard",
        use_gpu=args.gpu
    )

    policy.fit(
        dataset_train,
        n_steps=50000,
        n_steps_per_epoch=1000,
        save_interval=10,
        save_metrics=False
    )

    # apply reward function
    train_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for episode in tqdm(dataset_train.episodes):
        for transition in episode:
            state = transition.observation
            action = transition.action
            next_state = transition.next_observation
            reward = transition.reward
            terminal = transition.terminal

            if transition.next_transition is not None:
                shaping = policy.gamma * policy.predict_value(np.expand_dims(next_state, axis=0),
                                                              np.expand_dims(
                                                                  transition.next_transition.action,
                                                                  axis=0)) - \
                          policy.predict_value(np.expand_dims(state, axis=0), np.expand_dims(action, axis=0))
            else:
                shaping = -policy.predict_value(np.expand_dims(state, axis=0), np.expand_dims(action, axis=0))
            shaping = shaping.item()

            reward = reward + 0.1 * shaping

            train_buffer.add(
                state,
                action,
                next_state,
                reward,
                terminal
            )
        if not train_buffer.terminal[train_buffer.ptr - 1]:
            train_buffer.terminal[train_buffer.ptr - 1] = int(1)

    dataset_train = d3rlpy.dataset.MDPDataset(
        train_buffer.state[:train_buffer.size],
        train_buffer.action[:train_buffer.size],
        train_buffer.reward[:train_buffer.size],
        train_buffer.terminal[:train_buffer.size]
    )

    _, test_episodes = train_test_split(dataset_train, test_size=0.2)

    # Train
    policy = d3rlpy.algos.CQL(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0.0001,
        alpha_learning_rate=0.0001,
        batch_size=256,
        scaler="standard",
        use_gpu=args.gpu
    )

    import logging
    avg_rewards = []
    logs = []

    args.num_epochs = 100
    for epoch in range(args.num_epochs):
        results = policy.fit(
            dataset_train,
            eval_episodes=test_episodes,
            n_steps=5000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}"
        )

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
        logs.append(metrics)
        wandb.log(metrics)

        avg_r = avg_environment / len(results)
        avg_rewards.append(avg_r)
        logging.info(f"Epoch: [{epoch}/{args.num_epochs}], Average Reward: {avg_r}")

    # for metric in logs:
    #     wandb.log(metric)
    if not Path(f"./results/cql/reward2/{args.env}/{'_'.join(args.dataset_types)}").exists():
        os.makedirs(f"./results/cql/reward2/{args.env}/{'_'.join(args.dataset_types)}")
    np.save(f"./results/cql/reward2/{args.env}/{'_'.join(args.dataset_types)}/{args.seed}.npy", avg_rewards)
    df = {
        'critic_loss': [],
        'actor_loss': [],
        'environment': [],
        'value_scale': []
    }
    for log in logs:
        df['critic_loss'].append(log['critic_loss'])
        df['actor_loss'].append(log['actor_loss'])
        df['environment'].append(log['environment'])
        df['value_scale'].append(log['value_scale'])
    df = pd.DataFrame(data=df)
    df.to_csv(f"./results/cql/reward2/{args.env}/{'_'.join(args.dataset_types)}/{args.seed}.csv")

def pair_impact(args):
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

    # modify torso and friction if applied
    if args.torso is not None:
        import gym
        from xml.etree import ElementTree as ET
        env_name_map = {
            'ant': 'ant',
            'hopper': 'hopper',
            'walker2d': 'walker2d',
            'halfcheetah': 'half_cheetah'
        }
        path = os.path.join(os.path.dirname(gym.__file__), 'envs', 'mujoco', 'assets',
                            f'{env_name_map[args.env]}.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()
        for geom in root.iter('geom'):
            if geom.get('name') == 'torso_geom':
                geom.set('size', str(args.torso))
    if args.friction is not None:
        import gym
        from xml.etree import ElementTree as ET
        env_name_map = {
            'ant': 'ant',
            'hopper': 'hopper',
            'walker2d': 'walker2d',
            'halfcheetah': 'half_cheetah'
        }
        path = os.path.join(os.path.dirname(gym.__file__), 'envs', 'mujoco', 'assets',
                            f'{env_name_map[args.env]}.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()
        for geom in root.iter('geom'):
            if geom.get('name') == 'foot_geom':
                geom.set('friction', str(args.friction))

    def run():
        # logging
        wandb.init(
            project=args.project,
            name=f"CQL(pairimpact)_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_delta{args.dcla_ratio}_lambda{args.modify_ratio}_{args.seed}",
            config=args
        )

        # d3rlpy dataset
        env, dataset_source, dataset_target = mix_mdp_mujoco_datasets_new(args.env,
                                                                          n=args.dataset_size,
                                                                          dataset_types=args.dataset_types,
                                                                          ratios=args.dataset_ratios)

        # fix seed
        d3rlpy.seed(args.seed)
        env.seed(args.seed)

        source_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
        for episode in tqdm(dataset_source.episodes):
            for transition in episode:
                source_buffer.add(
                    transition.observation,
                    transition.action,
                    transition.next_observation,
                    transition.reward,
                    transition.terminal
                )
            if not source_buffer.terminal[source_buffer.ptr - 1]:
                source_buffer.terminal[source_buffer.ptr - 1] = int(1)
        source_buffer.create_trajs()
        target_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
        for episode in tqdm(dataset_target.episodes):
            for transition in episode:
                target_buffer.add(
                    transition.observation,
                    transition.action,
                    transition.next_observation,
                    transition.reward,
                    transition.terminal
                )
            if not target_buffer.terminal[target_buffer.ptr - 1]:
                target_buffer.terminal[target_buffer.ptr - 1] = int(1)
        target_buffer.create_trajs()

        ###  Train Delta Classifier
        delta = DeltaCla(env.observation_space.shape[0], env.action_space.shape[0], args)
        delta_model_path = f"./models/cql/{args.env}/{args.dataset_types[0]}_{args.dataset_types[1]}"
        print('Start training delta classifiers...')
        delta.train(source_buffer, target_buffer, args)
        # for model_file in cf_model_files:
        delta.save_delta_models(delta_model_path)
        print('Finished training delta classifiers')

        ### Train DVE
        results_dir = f"./models/cql/{args.env}/{args.dataset_types[0]}_{args.dataset_types[1]}/dve"
        dvrl = DVRL(source_buffer, target_buffer, args.dev, env, results_dir, args.ex_configs, args)
        print('Start training DVE...')
        dvrl.train()
        print('Finished training DVE...')

        dve_out, sel_vec = dvrl.data_valuate(source_buffer, args.batch_size)
        source_buffer = modify_rewards(source_buffer, dve_out, ratio=args.modify_ratio)

        dataset_source_new = d3rlpy.dataset.MDPDataset(
            source_buffer.state[:source_buffer.size],
            source_buffer.action[:source_buffer.size],
            source_buffer.reward[:source_buffer.size],
            source_buffer.terminal[:source_buffer.size]
        )

        dataset_source = dataset_source_new

        ### train agent of target domain over target offline dataset
        policy_target = d3rlpy.algos.CQL(
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=0.0001,
            alpha_learning_rate=0.0001,
            batch_size=256,
            scaler="standard",
            use_gpu=args.gpu
        )
        policy_target.fit(
            dataset_target,
            n_steps=50000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
            save_metrics=False
        )

        ### valaute trajectory
        def valuate_traj(traj, policy_source, policy_target, coeff=0.8):
            obs = []
            acts = []
            rewards = []
            for transition in traj:
                obs.append(transition.observation)
                acts.append(transition.action)
                rewards.append(transition.reward)
            obs = torch.tensor(np.array(obs)).to(args.dev)
            acts = torch.tensor(np.array(acts)).to(args.dev)
            rewards = np.array(rewards)

            act_mean_source = torch.from_numpy(policy_source.predict(obs)).to(args.dev)
            act_mean_target = torch.from_numpy(policy_target.predict(obs)).to(args.dev)
            beta = coeff
            dist = torch.mean((1 - beta) * torch.norm(acts - act_mean_source, p=2, dim=(1,))
                              + beta * torch.norm(acts - act_mean_target, p=2, dim=(1,)))

            value = np.exp(-dist.cpu().numpy()) * np.sum(rewards)

            return value

        def exponential_scheduler(t):
            beta = 0.5
            tem_max = 100

            return beta ** t * tem_max

        def coeff_scheduler(t):
            discount = 0.99
            coeff_min = 0.45

            return max(coeff_min, discount ** t)

        tem_scheduler = exponential_scheduler

        policy_source = d3rlpy.algos.CQL(
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=0.0001,
            alpha_learning_rate=0.0001,
            batch_size=256,
            scaler="standard",
            use_gpu=args.gpu
        )
        policy_source.fit(
            dataset_source,
            n_steps=1000,
            n_steps_per_epoch=1000,
            save_interval=10,
            save_metrics=False
        )

        _, test_episodes = train_test_split(dataset_target, test_size=0.2)

        # Train
        args.num_epochs = 100
        for epoch in range(args.num_epochs):
            # compute the value of each trajectory in source offline dataset
            trajs_value = []
            for traj in dataset_source.episodes:
                v = valuate_traj(traj, policy_source, policy_target, coeff_scheduler(epoch))
                trajs_value.append(v)
            trajs_value = torch.tensor(trajs_value)
            selection_prob = torch.nn.functional.softmax(trajs_value / tem_scheduler(epoch))
            # sample from selection probability
            selected_size = int(len(dataset_source.episodes) / 10)
            indices = torch.multinomial(selection_prob, selected_size)

            episodes_source = dataset_source.episodes
            episodes_target = dataset_target.episodes
            episodes_train = []
            for idx in indices:
                episodes_train.append(episodes_source[idx])
            episodes_train += episodes_target

            results = policy_source.fit(
                episodes_train,
                eval_episodes=test_episodes,
                n_steps=5000,
                n_steps_per_epoch=1000,
                save_interval=10,
                scorers={
                    'environment': d3rlpy.metrics.evaluate_on_environment(env),
                    'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
                },
                experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
                save_metrics=False
            )

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

        wandb.finish()

    for delta in args.delta:
        for lmd in args.lmd:
            args.dcla_ratio = delta
            args.modify_ratio = lmd
            run()

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

    # modify torso and friction if applied
    if args.torso is not None:
        import gym
        from xml.etree import ElementTree as ET
        env_name_map = {
            'ant': 'ant',
            'hopper': 'hopper',
            'walker2d': 'walker2d',
            'halfcheetah': 'half_cheetah'
        }
        path = os.path.join(os.path.dirname(gym.__file__), 'envs', 'mujoco', 'assets',
                            f'{env_name_map[args.env]}.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()
        for geom in root.iter('geom'):
            if geom.get('name') == 'torso_geom':
                geom.set('size', str(args.torso))
    if args.friction is not None:
        import gym
        from xml.etree import ElementTree as ET
        env_name_map = {
            'ant': 'ant',
            'hopper': 'hopper',
            'walker2d': 'walker2d',
            'halfcheetah': 'half_cheetah'
        }
        path = os.path.join(os.path.dirname(gym.__file__), 'envs', 'mujoco', 'assets',
                            f'{env_name_map[args.env]}.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()
        for geom in root.iter('geom'):
            if geom.get('name') == 'foot_geom':
                geom.set('friction', str(args.friction))

    def run():
        # logging
        wandb.init(
            project=args.project,
            name=f"CQL(pairimpact)_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_delta{args.dcla_ratio}_lambda{args.modify_ratio}_{args.seed}",
            config=args
        )

        # d3rlpy dataset
        env, dataset_source, dataset_target = mix_mdp_mujoco_datasets_new(args.env,
                                                                          n=args.dataset_size,
                                                                          dataset_types=args.dataset_types,
                                                                          ratios=args.dataset_ratios)

        # fix seed
        d3rlpy.seed(args.seed)
        env.seed(args.seed)

        source_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
        for episode in tqdm(dataset_source.episodes):
            for transition in episode:
                source_buffer.add(
                    transition.observation,
                    transition.action,
                    transition.next_observation,
                    transition.reward,
                    transition.terminal
                )
            if not source_buffer.terminal[source_buffer.ptr - 1]:
                source_buffer.terminal[source_buffer.ptr - 1] = int(1)
        source_buffer.create_trajs()
        target_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
        for episode in tqdm(dataset_target.episodes):
            for transition in episode:
                target_buffer.add(
                    transition.observation,
                    transition.action,
                    transition.next_observation,
                    transition.reward,
                    transition.terminal
                )
            if not target_buffer.terminal[target_buffer.ptr - 1]:
                target_buffer.terminal[target_buffer.ptr - 1] = int(1)
        target_buffer.create_trajs()

        ###  Train Delta Classifier
        delta = DeltaCla(env.observation_space.shape[0], env.action_space.shape[0], args)
        delta_model_path = f"./models/cql/{args.env}/{args.dataset_types[0]}_{args.dataset_types[1]}"
        print('Start training delta classifiers...')
        delta.train(source_buffer, target_buffer, args)
        # for model_file in cf_model_files:
        delta.save_delta_models(delta_model_path)
        print('Finished training delta classifiers')

        ### Train DVE
        results_dir = f"./models/cql/{args.env}/{args.dataset_types[0]}_{args.dataset_types[1]}/dve"
        dvrl = DVRL(source_buffer, target_buffer, args.dev, env, results_dir, args.ex_configs, args)
        print('Start training DVE...')
        dvrl.train()
        print('Finished training DVE...')

        dve_out, sel_vec = dvrl.data_valuate(source_buffer, args.batch_size)
        source_buffer = modify_rewards(source_buffer, dve_out, ratio=args.modify_ratio)

        dataset_source_new = d3rlpy.dataset.MDPDataset(
            source_buffer.state[:source_buffer.size],
            source_buffer.action[:source_buffer.size],
            source_buffer.reward[:source_buffer.size],
            source_buffer.terminal[:source_buffer.size]
        )

        dataset_source = dataset_source_new

        ### train agent of target domain over target offline dataset
        policy_target = d3rlpy.algos.CQL(
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=0.0001,
            alpha_learning_rate=0.0001,
            batch_size=256,
            scaler="standard",
            use_gpu=args.gpu
        )
        policy_target.fit(
            dataset_target,
            n_steps=50000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
            save_metrics=False
        )

        ### valaute trajectory
        def valuate_traj(traj, policy_source, policy_target, coeff=0.8):
            obs = []
            acts = []
            rewards = []
            for transition in traj:
                obs.append(transition.observation)
                acts.append(transition.action)
                rewards.append(transition.reward)
            obs = torch.tensor(np.array(obs)).to(args.dev)
            acts = torch.tensor(np.array(acts)).to(args.dev)
            rewards = np.array(rewards)

            act_mean_source = torch.from_numpy(policy_source.predict(obs)).to(args.dev)
            act_mean_target = torch.from_numpy(policy_target.predict(obs)).to(args.dev)
            beta = coeff
            dist = torch.mean((1 - beta) * torch.norm(acts - act_mean_source, p=2, dim=(1,))
                              + beta * torch.norm(acts - act_mean_target, p=2, dim=(1,)))

            value = np.exp(-dist.cpu().numpy()) * np.sum(rewards)

            return value

        def exponential_scheduler(t):
            beta = 0.5
            tem_max = 100

            return beta ** t * tem_max

        def coeff_scheduler(t):
            discount = 0.99
            coeff_min = 0.45

            return max(coeff_min, discount ** t)

        tem_scheduler = exponential_scheduler

        policy_source = d3rlpy.algos.CQL(
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=0.0001,
            alpha_learning_rate=0.0001,
            batch_size=256,
            scaler="standard",
            use_gpu=args.gpu
        )
        policy_source.fit(
            dataset_source,
            n_steps=1000,
            n_steps_per_epoch=1000,
            save_interval=10,
            save_metrics=False
        )

        _, test_episodes = train_test_split(dataset_target, test_size=0.2)

        # Train
        args.num_epochs = 100
        for epoch in range(args.num_epochs):
            # compute the value of each trajectory in source offline dataset
            trajs_value = []
            for traj in dataset_source.episodes:
                v = valuate_traj(traj, policy_source, policy_target, coeff_scheduler(epoch))
                trajs_value.append(v)
            trajs_value = torch.tensor(trajs_value)
            selection_prob = torch.nn.functional.softmax(trajs_value / tem_scheduler(epoch))
            # sample from selection probability
            selected_size = int(len(dataset_source.episodes) / 10)
            indices = torch.multinomial(selection_prob, selected_size)

            episodes_source = dataset_source.episodes
            episodes_target = dataset_target.episodes
            episodes_train = []
            for idx in indices:
                episodes_train.append(episodes_source[idx])
            episodes_train += episodes_target

            results = policy_source.fit(
                episodes_train,
                eval_episodes=test_episodes,
                n_steps=5000,
                n_steps_per_epoch=1000,
                save_interval=10,
                scorers={
                    'environment': d3rlpy.metrics.evaluate_on_environment(env),
                    'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
                },
                experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
                save_metrics=False
            )

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

        wandb.finish()

    for sdratio in args.sdratios_list:
        args.dataset_ratios = [sdratio, 1.0 - sdratio]
        run()

def without_TS(args):
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

    # modify torso and friction if applied
    if args.torso is not None:
        import gym
        from xml.etree import ElementTree as ET
        env_name_map = {
            'ant': 'ant',
            'hopper': 'hopper',
            'walker2d': 'walker2d',
            'halfcheetah': 'half_cheetah'
        }
        path = os.path.join(os.path.dirname(gym.__file__), 'envs', 'mujoco', 'assets',
                            f'{env_name_map[args.env]}.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()
        for geom in root.iter('geom'):
            if geom.get('name') == 'torso_geom':
                geom.set('size', str(args.torso))
    if args.friction is not None:
        import gym
        from xml.etree import ElementTree as ET
        env_name_map = {
            'ant': 'ant',
            'hopper': 'hopper',
            'walker2d': 'walker2d',
            'halfcheetah': 'half_cheetah'
        }
        path = os.path.join(os.path.dirname(gym.__file__), 'envs', 'mujoco', 'assets',
                            f'{env_name_map[args.env]}.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()
        for geom in root.iter('geom'):
            if geom.get('name') == 'foot_geom':
                geom.set('friction', str(args.friction))

    # logging
    wandb.init(
        project=args.project,
        name=f"CQL(trajValuation)_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
        config=args
    )

    # d3rlpy dataset
    env, dataset_source, dataset_target = mix_mdp_mujoco_datasets_new(args.env,
                                                                      n=args.dataset_size,
                                                                      dataset_types=args.dataset_types,
                                                                      ratios=args.dataset_ratios)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    ### train agent of target domain over target offline dataset
    policy_target = d3rlpy.algos.CQL(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0.0001,
        alpha_learning_rate=0.0001,
        batch_size=256,
        scaler="standard",
        use_gpu=args.gpu
    )
    policy_target.fit(
        dataset_target,
        n_steps=50000,
        n_steps_per_epoch=1000,
        save_interval=10,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
        },
        experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
        save_metrics=False
    )

    ### valaute trajectory
    def valuate_traj(traj, policy_source, policy_target, coeff=0.8):
        obs = []
        acts = []
        rewards = []
        for transition in traj:
            obs.append(transition.observation)
            acts.append(transition.action)
            rewards.append(transition.reward)
        obs = torch.tensor(np.array(obs)).to(args.dev)
        acts = torch.tensor(np.array(acts)).to(args.dev)
        rewards = np.array(rewards)

        act_mean_source = torch.from_numpy(policy_source.predict(obs)).to(args.dev)
        act_mean_target = torch.from_numpy(policy_target.predict(obs)).to(args.dev)
        beta = coeff
        dist = torch.mean((1 - beta) * torch.norm(acts - act_mean_source, p=2, dim=(1,))
                          + beta * torch.norm(acts - act_mean_target, p=2, dim=(1,)))

        value = np.exp(-dist.cpu().numpy()) * np.sum(rewards)

        return value

    def exponential_scheduler(t):
        beta = 0.5
        tem_max = 100

        return beta ** t * tem_max

    def coeff_scheduler(t):
        discount = 0.99
        coeff_min = 0.45

        return max(coeff_min, discount ** t)

    tem_scheduler = exponential_scheduler

    policy_source = d3rlpy.algos.CQL(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0.0001,
        alpha_learning_rate=0.0001,
        batch_size=256,
        scaler="standard",
        use_gpu=args.gpu
    )
    policy_source.fit(
        dataset_source,
        n_steps=1000,
        n_steps_per_epoch=1000,
        save_interval=10,
        save_metrics=False
    )

    _, test_episodes = train_test_split(dataset_target, test_size=0.2)

    # Train
    args.num_epochs = 100
    for epoch in range(args.num_epochs):
        # compute the value of each trajectory in source offline dataset
        trajs_value = []
        for traj in dataset_source.episodes:
            v = valuate_traj(traj, policy_source, policy_target, coeff_scheduler(epoch))
            trajs_value.append(v)
        trajs_value = torch.tensor(trajs_value)
        selection_prob = torch.nn.functional.softmax(trajs_value / tem_scheduler(epoch))
        # sample from selection probability
        selected_size = int(len(dataset_source.episodes) / 10)
        indices = torch.multinomial(selection_prob, selected_size)

        episodes_source = dataset_source.episodes
        episodes_target = dataset_target.episodes
        episodes_train = []
        for idx in indices:
            episodes_train.append(episodes_source[idx])
        episodes_train += episodes_target

        results = policy_source.fit(
            episodes_train,
            eval_episodes=test_episodes,
            n_steps=5000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
            save_metrics=False
        )

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


def traj_valuation(args):
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
        name=f"CQL(trajValuation)_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
        config=args
    )

    # d3rlpy dataset
    env, dataset_source, dataset_target = mix_mdp_mujoco_datasets_new(args.env,
                                                                      n=args.dataset_size,
                                                                      dataset_types=args.dataset_types,
                                                                      ratios=args.dataset_ratios)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    source_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for episode in tqdm(dataset_source.episodes):
        for transition in episode:
            source_buffer.add(
                transition.observation,
                transition.action,
                transition.next_observation,
                transition.reward,
                transition.terminal
            )
        if not source_buffer.terminal[source_buffer.ptr-1]:
            source_buffer.terminal[source_buffer.ptr-1] = int(1)
    source_buffer.create_trajs()
    target_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for episode in tqdm(dataset_target.episodes):
        for transition in episode:
            target_buffer.add(
                transition.observation,
                transition.action,
                transition.next_observation,
                transition.reward,
                transition.terminal
            )
        if not target_buffer.terminal[target_buffer.ptr-1]:
            target_buffer.terminal[target_buffer.ptr-1] = int(1)
    target_buffer.create_trajs()

    ###  Train Delta Classifier
    delta = DeltaCla(env.observation_space.shape[0], env.action_space.shape[0], args)
    delta_model_path = f"./models/cql/{args.env}/{args.dataset_types[0]}_{args.dataset_types[1]}"
    # Define the model file names
    cf_model_files = ['cla_sa', 'cla_sas']
    # Check if the model files exist
    if all(Path(delta_model_path + '/delta_models/' + f).is_file() for f in cf_model_files):
        # If the model files exist, load them
        print('Loading delta classifiers...')
        delta.load_delta_models(delta_model_path)
        print('Finished loading delta classifiers')
    else:
        # If the model files do not exist, train the models and save them
        print('Start training delta classifiers...')
        delta.train(source_buffer, target_buffer, args)
        # for model_file in cf_model_files:
        delta.save_delta_models(delta_model_path)
        print('Finished training delta classifiers')

    ### Train DVE
    results_dir = f"./models/cql/{args.env}/{args.dataset_types[0]}_{args.dataset_types[1]}/dve"
    dvrl = DVRL(source_buffer, target_buffer, args.dev, env, results_dir, args.ex_configs, args)
    # Define the model file names
    dve_model_files = ['reinforce_final', 'reinforce_optimizer_final']
    # Check if the model files exist
    if all(Path("%s/dvrl_models/Trained_With_Seed_%d_Friction_%f_Mass_%f_Gamma_%f/" % \
                (results_dir, args.source_seed, args.source_env_friction,
                 args.source_env_mass_torso, args.discount) + f).is_file() for f in dve_model_files):
        # If the model files exist, load them
        print('Loading DVE...')
        dvrl.load_dve(dvrl.model_dir, type='final')
    else:
        # If the model files do not exist, train the models and save them
        print('Start training DVE...')
        dvrl.train()
        print('Finished training DVE...')

    dve_out, sel_vec = dvrl.data_valuate(source_buffer, args.batch_size)
    source_buffer = modify_rewards(source_buffer, dve_out, ratio=args.modify_ratio)

    dataset_source_new = d3rlpy.dataset.MDPDataset(
        source_buffer.state[:source_buffer.size],
        source_buffer.action[:source_buffer.size],
        source_buffer.reward[:source_buffer.size],
        source_buffer.terminal[:source_buffer.size]
    )

    dataset_source = dataset_source_new

    ### train agent of target domain over target offline dataset
    policy_target = d3rlpy.algos.CQL(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0.0001,
        alpha_learning_rate=0.0001,
        batch_size=256,
        scaler="standard",
        use_gpu=args.gpu
    )
    policy_target.fit(
        dataset_target,
        n_steps=50000,
        n_steps_per_epoch=1000,
        save_interval=10,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
        },
        experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
        save_metrics=False
    )

    ### valaute trajectory
    def valuate_traj(traj, policy_source, policy_target, coeff=0.8):
        obs = []
        acts = []
        rewards = []
        for transition in traj:
            obs.append(transition.observation)
            acts.append(transition.action)
            rewards.append(transition.reward)
        obs = torch.tensor(np.array(obs)).to(args.dev)
        acts = torch.tensor(np.array(acts)).to(args.dev)
        rewards = np.array(rewards)

        act_mean_source = torch.from_numpy(policy_source.predict(obs)).to(args.dev)
        act_mean_target = torch.from_numpy(policy_target.predict(obs)).to(args.dev)
        beta = coeff
        dist = torch.mean((1 - beta) * torch.norm(acts - act_mean_source, p=2, dim=(1,))
                          + beta * torch.norm(acts - act_mean_target, p=2, dim=(1,)))

        value = np.exp(-dist.cpu().numpy()) * np.sum(rewards)

        return value

    def exponential_scheduler(t):
        beta = 0.5
        tem_max = 100

        return beta ** t * tem_max

    def coeff_scheduler(t):
        discount = 0.99
        coeff_min = 0.45

        return max(coeff_min, discount ** t)

    tem_scheduler = exponential_scheduler

    policy_source = d3rlpy.algos.CQL(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0.0001,
        alpha_learning_rate=0.0001,
        batch_size=256,
        scaler="standard",
        use_gpu=args.gpu
    )
    policy_source.fit(
        dataset_source,
        n_steps=1000,
        n_steps_per_epoch=1000,
        save_interval=10,
        save_metrics=False
    )

    _, test_episodes = train_test_split(dataset_target, test_size=0.2)

    # Train
    args.num_epochs = 100
    for epoch in range(args.num_epochs):
        # compute the value of each trajectory in source offline dataset
        trajs_value = []
        for traj in dataset_source.episodes:
            v = valuate_traj(traj, policy_source, policy_target, coeff_scheduler(epoch))
            trajs_value.append(v)
        trajs_value = torch.tensor(trajs_value)
        selection_prob = torch.nn.functional.softmax(trajs_value / tem_scheduler(epoch))
        # sample from selection probability
        selected_size = int(len(dataset_source.episodes) / 10)
        indices = torch.multinomial(selection_prob, selected_size)

        episodes_source = dataset_source.episodes
        episodes_target = dataset_target.episodes
        episodes_train = []
        for idx in indices:
            episodes_train.append(episodes_source[idx])
        episodes_train += episodes_target

        results = policy_source.fit(
            episodes_train,
            eval_episodes=test_episodes,
            n_steps=5000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"CQL_{args.dataset_types}_{args.dataset_ratios}_{args.dataset_size}_{args.seed}",
            save_metrics=False
        )

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

    # without_CL(args)
    traj_valuation(args)
