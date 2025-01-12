import random
from pathlib import Path
import gym
import d4rl
from ts.replay_buffer import ReplayBuffer
from tqdm import tqdm
import numpy as np


def get_gym(env_name):
    """
    Create gym instance and modify mass and friction.
    env_name: name of gym env
    friction: desired friction
    mass: desired torso mass
    returns: gym env
    """
    env = gym.make(env_name)
    return env

def make_results_dir(env_name):
    """
    Create results folder and return path.
    env_name: name of gym env
    returns: results folder path
    """
    if env_name == "Pendulum-v0":
        results_dir = "./results/%s" % (env_name)
        Path(results_dir).mkdir(parents=True, exist_ok=True)
    else:
        date = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
        results_dir = "./results/%s-%s" % (env_name, date)
        Path(results_dir).mkdir(parents=True, exist_ok=True)
    return results_dir

def make_dvrl_dir(results_dir, args):
    """
    Create results folder and return path.
    results_dir: base dir
    returns: results folder path
    """
    if args.env == "Pendulum-v0":
        directory = "%s/dvrl_models/Trained_With_Seed_%d_Gamma_%f/" %\
                (results_dir, args.source_seed, args.discount)
    else:
        directory = "%s/dvrl_models/Trained_With_Seed_%d_Friction_%f_Mass_%f_Gamma_%f/" %\
                (results_dir, args.source_seed, args.source_env_friction,
                 args.source_env_mass_torso, args.discount)
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def make_final_results_dir(res_path, rl_model, run_id):
    """
    Create results folder and return path.
    env_name: name of gym env
    returns: results folder path
    """
    results_dir = f"./results/AAMAS/{res_path}/DV{rl_model}/{run_id}/"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return results_dir


def detach(x):
    """
    Detach tensors and return numpy array.
    """
    return x.cpu().detach().numpy()


def get_d4rl_buffer(env, device):

    # read D4RL dataset
    dataset = d4rl.qlearning_dataset(env)
    data_num = len(dataset['observations'])
    states = dataset['observations']
    actions = dataset['actions']

    n_states = dataset['next_observations']
    rewards = dataset['rewards']
    dones = dataset['terminals']
    source_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device)
    for i in tqdm(range(data_num)):
        source_buffer.add(states[i], actions[i], n_states[i], rewards[i], dones[i])
    source_buffer.create_trajs()
    return source_buffer


def get_replay_buffer(replay_buffer, trajectories):
    # Add each individual step from all trajectories to the replay buffer
    for traj in tqdm(trajectories):
        states = traj["state"]
        actions = traj["action"]
        next_states = traj["next_state"]
        rewards = traj["reward"]
        not_dones = traj["not_done"]

        # Add each individual step to the buffer
        for state, action, next_state, reward, not_done in zip(states, actions, next_states, rewards, not_dones):
            replay_buffer.add(state, action, next_state, reward, 1.0 - not_done)

    # Create trajectories in the combined buffer
    replay_buffer.create_trajs()

    return replay_buffer



def get_combined_buffer(env, source_buffer, target_buffer, ratio, device):
    new_source_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device)
    small_target_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device)

    # Create trajectories from both buffers
    source_trajs = source_buffer.trajs
    target_trajs = target_buffer.trajs

    # Calculate the number of trajectories to sample from each buffer
    n_source = int(np.ceil(len(source_trajs) * ratio))
    n_target = int(np.ceil(len(target_trajs) * (1.0 - ratio)))

    # Sample trajectories
    random.seed(0)  # Set the seed
    source_sample = random.sample(source_trajs, n_source)
    target_sample = random.sample(target_trajs, n_target)

    # Combine trajectories from both source and target samples
    combined_sample = source_sample + target_sample

    new_source_buffer = get_replay_buffer(new_source_buffer, combined_sample)
    small_target_buffer = get_replay_buffer(small_target_buffer, target_sample)


    return new_source_buffer, small_target_buffer


def get_subset_target_buffer(env, target_buffer, device):
    target_buffer_subset = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=device)
    tbs = target_buffer.sample(100000, to_device=False)
    for i in range(100000):
        target_buffer_subset.add(tbs[0][i], tbs[1][i], tbs[2][i], tbs[3][i], 1.0 - tbs[4][i])