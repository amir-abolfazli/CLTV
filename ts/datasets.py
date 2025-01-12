import random
import gym
import numpy as np

import d3rlpy
from d3rlpy.datasets import MDPDataset

def mix_mdp_mujoco_datasets(env_id, n, dataset_types, ratios):
  assert len(dataset_types) == len(ratios)
  assert sum(ratios) == 1

  print(f"Dataset size: {n}")
  if np.all(np.asarray(ratios) != 1.0):
    episodes = []
    for dataset_type, ratio in zip(dataset_types, ratios):
      dataset_id = f"{env_id}-{dataset_type}"

      print("Env:", env_id)
      if env_id in ["pen", "hammer", "door", "relocate"]:
        dataset, env = d3rlpy.datasets.get_d4rl(f"{env_id}-{dataset_type}")
      else:
        dataset, env = d3rlpy.datasets.get_dataset(f"{env_id}-{dataset_type}")

      num_transitions = 0
      print(f"Mix {ratio} ({n * ratio}) data from {dataset_type}.")
      while num_transitions < int(n * ratio):
        episode = random.choice(dataset.episodes)
        episodes.append(episode)
        num_transitions += len(episode.transitions)

    observations = []
    actions = []
    rewards = []
    terminals = []
    episode_terminals = []

    ep_rets = []
    for episode in episodes:
      ep_rets.append(episode.rewards.sum())
      for idx, transition in enumerate(episode):
        observations.append(transition.observation)
        if isinstance(env.action_space, gym.spaces.Box):
          actions.append(np.reshape(transition.action, env.action_space.shape))
        else:
          actions.append(transition.action)
        rewards.append(transition.reward)
        terminals.append(transition.terminal)
        episode_terminals.append(idx == len(episode) - 1)
    print(f"Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")
    dataset = MDPDataset(
      observations=np.stack(observations),
      actions=np.stack(actions),
      rewards=np.stack(rewards),
      terminals=np.stack(terminals).astype(float),
      episode_terminals=np.stack(episode_terminals).astype(float))
  else:
    for dataset_type, ratio in zip(dataset_types, ratios):
      if ratio == 1.0:
        if env_id in ["pen", "hammer", "door", "relocate"]:
          dataset, env = d3rlpy.datasets.get_d4rl(f"{env_id}-{dataset_type}")
        else:
          dataset, env = d3rlpy.datasets.get_dataset(f"{env_id}-{dataset_type}")
        break
    ep_rets = []
    for episode in dataset.episodes:
      ep_rets.append(episode.rewards.sum())
    print(f"Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")
  return env, dataset

def mix_mdp_mujoco_datasets_new(env_id, n, dataset_types, ratios):
  assert len(dataset_types) == len(ratios)
  assert sum(ratios) == 1

  print(f"Dataset size: {n}")
  if np.all(np.asarray(ratios) != 1.0):
    episodes_source, episodes_target = [], []

    dataset_type_source, dataset_type_target = dataset_types[0], dataset_types[1]
    ratio_source, ratio_target = ratios[0], ratios[1]

    print("Env:", env_id)
    if env_id in ["pen", "hammer", "door", "relocate"]:
      dataset_source, env = d3rlpy.datasets.get_d4rl(f"{env_id}-{dataset_type_source}")
      dataset_target, env = d3rlpy.datasets.get_d4rl(f"{env_id}-{dataset_type_target}")
    else:
      dataset_source, env = d3rlpy.datasets.get_dataset(f"{env_id}-{dataset_type_source}")
      dataset_target, env = d3rlpy.datasets.get_dataset(f"{env_id}-{dataset_type_target}")

    num_transitions_source, num_transitions_target = 0, 0
    print(f"Mix {ratio_source} ({n * ratio_source}) data from {dataset_type_source}.")
    while num_transitions_source < int(n * ratio_source):
      episode = random.choice(dataset_source.episodes)
      episodes_source.append(episode)
      num_transitions_source += len(episode.transitions)
    print(f"Mix {ratio_target} ({n * ratio_target}) data from {dataset_type_target}.")
    while num_transitions_target < int(n * ratio_target):
      episode = random.choice(dataset_target.episodes)
      episodes_target.append(episode)
      num_transitions_target += len(episode.transitions)

    # prepare MDPDataset of source dataset
    observations = []
    actions = []
    rewards = []
    terminals = []
    episode_terminals = []

    ep_rets = []
    for episode in episodes_source:
      ep_rets.append(episode.rewards.sum())
      for idx, transition in enumerate(episode):
        observations.append(transition.observation)
        if isinstance(env.action_space, gym.spaces.Box):
          actions.append(np.reshape(transition.action, env.action_space.shape))
        else:
          actions.append(transition.action)
        rewards.append(transition.reward)
        terminals.append(transition.terminal)
        episode_terminals.append(idx == len(episode) - 1)
    print(f"Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")
    dataset_source = MDPDataset(
      observations=np.stack(observations),
      actions=np.stack(actions),
      rewards=np.stack(rewards),
      terminals=np.stack(terminals).astype(float),
      episode_terminals=np.stack(episode_terminals).astype(float))

    # prepare MDPDataset of target dataset
    observations = []
    actions = []
    rewards = []
    terminals = []
    episode_terminals = []

    ep_rets = []
    for episode in episodes_target:
      ep_rets.append(episode.rewards.sum())
      for idx, transition in enumerate(episode):
        observations.append(transition.observation)
        if isinstance(env.action_space, gym.spaces.Box):
          actions.append(np.reshape(transition.action, env.action_space.shape))
        else:
          actions.append(transition.action)
        rewards.append(transition.reward)
        terminals.append(transition.terminal)
        episode_terminals.append(idx == len(episode) - 1)
    print(f"Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")
    dataset_target = MDPDataset(
      observations=np.stack(observations),
      actions=np.stack(actions),
      rewards=np.stack(rewards),
      terminals=np.stack(terminals).astype(float),
      episode_terminals=np.stack(episode_terminals).astype(float))
  else:
    dataset_type_source, dataset_type_target = dataset_types[0], dataset_types[1]
    ratio_source, ratio_target = ratios[0], ratios[1]

    if ratio_source == 1.0:
      if env_id in ["pen", "hammer", "door", "relocate"]:
        dataset_source, env = d3rlpy.datasets.get_d4rl(f"{env_id}-{dataset_type_source}")
      else:
        dataset_source, env = d3rlpy.datasets.get_dataset(f"{env_id}-{dataset_type_source}")
      ep_rets = []
      for episode in dataset_source.episodes:
        ep_rets.append(episode.rewards.sum())
      print(f"Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")
    elif ratio_target == 1.0:
      if env_id in ["pen", "hammer", "door", "relocate"]:
        dataset_target, env = d3rlpy.datasets.get_d4rl(f"{env_id}-{dataset_type_target}")
      else:
        dataset_target, env = d3rlpy.datasets.get_dataset(f"{env_id}-{dataset_type_target}")
      ep_rets = []
      for episode in dataset_target.episodes:
        ep_rets.append(episode.rewards.sum())
      print(f"Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")

  return env, dataset_source, dataset_target

