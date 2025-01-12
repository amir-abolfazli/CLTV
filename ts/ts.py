""" Data Valuation based Batch-Constrained Reinforcement Learning """
import ts.utils
from ts.utils import detach
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import torch
import json
from numpy.random import Generator, PCG64
from ts.delta_classifier import DeltaCla
import matplotlib.pyplot as plt
from matplotlib import cm

from ts.replay_buffer import ReplayBuffer
from ts.reinforce import REINFORCE
from ts.utils import get_gym


class DVRL(object):
    def __init__(self, source_dataset, target_dataset, device, target_env, results_dir, ex_configs, args={}):

        self.dict = vars(args)
        self.args = args
        assert len(self.dict) > 0

        # data
        self.model_dir = ts.utils.make_dvrl_dir(results_dir, args)
        self.state_dim = target_env.observation_space.shape[0]
        self.action_dim = target_env.action_space.shape[0]
        self.action_space = target_env.action_space
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.results_dir = results_dir
        self.device = device
        self.env_name = args.env

        self.dist_epsilon = self.dict['dist_epsilon']
        self.epsilon_minimum = 1.2
        self.epsilon_decay = 0.99

        self.g_min = 10000.
        self.g_max = 0.

        self.g_dvbcq_rew_history = []
        self.g_bcq_rew_history = []

        self.expert_actions = None

        # Used to store samples selected by the value estimator in each outer iteration.
        self.sampling_replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, device)

        self.ev_buf = ReplayBuffer(self.state_dim, self.action_dim, device)

        reinforce_state_dim = self.state_dim * 2 + self.action_dim

        reinforce_action_dim = 1  # Outputs probability for each sample
        self.dve_model = REINFORCE(reinforce_state_dim,
                                   reinforce_action_dim,
                                   layers=self.dict['reinforce_layers'],
                                   args=self.dict).to(device)
        # optimizers
        self.dve_optimizer = optim.Adam(self.dve_model.parameters(), self.dict['dve_lr'])

        args.dcla_cuda = True
        self.delta = DeltaCla(target_env.observation_space.shape[0], target_env.action_space.shape[0], args)

        self.model = None
        # record average rewards
        self.avg_rewards = []

        self.delta_ratio = args.dcla_ratio

    def make_envs(self):
        self.target_env = get_gym(self.dict['d4rl_target_env'])

    def to_device(self, x):
        """
        Copy x to GPU
        """
        return torch.FloatTensor(x).to(self.device)

    def train_dve(self, dve_model, dve_optimizer, x, s_input, reward):
        """
         Training data value estimator
         s_input: selection probability
         x_input: Sample tuple
         reward_input: reward signal for training the reinforce agent
        """
        est_data_value = dve_model(x)
        dve_optimizer.zero_grad()

        one = torch.ones_like(est_data_value, dtype=est_data_value.dtype)
        prob = torch.sum(s_input * torch.log(est_data_value + self.dict['epsilon']) + \
                         (one - s_input) * \
                         torch.log(one - est_data_value + self.dict['epsilon']))

        zero = torch.Tensor([0.0])
        zero = zero.to(est_data_value.device)

        # 1e3= multiplier for the regularizer
        loss = (-reward * prob) + \
               1e3 * torch.maximum(torch.mean(est_data_value) - self.dict['threshold'], zero) + \
               1e3 * torch.maximum(1 - self.dict['threshold'] - torch.mean(est_data_value), zero)

        loss.backward()
        dve_optimizer.step()

    def train(self):
        """
        RL model and DVE iterative training
        """

        reinforce_rewards = np.array([])

        for iteration in tqdm(range(self.dict['outer_iterations'])):

            states, actions, next_states, rewards, terminals = self.source_dataset.sample(self.dict['batch_size'],
                                                                                          to_device=False)  # O

            dvrl_input = torch.FloatTensor(np.hstack((states,
                                                      next_states,
                                                      actions))).to(self.device)

            est_dv_curr = self.dve_model(dvrl_input)

            ratio = self.delta_ratio
            delta_avg = ratio * -np.sum(est_dv_curr.squeeze(1).detach().cpu().numpy() * np.abs(
                self.delta.delta_dynamic(torch.cuda.FloatTensor(states), torch.cuda.FloatTensor(actions),
                                         torch.cuda.FloatTensor(next_states)))) + (1 - ratio) * np.sum(
                est_dv_curr.squeeze(1).detach().cpu().numpy())

            reinforce_reward = delta_avg
            reinforce_rewards = np.append(reinforce_rewards, reinforce_reward)
            if np.max(reinforce_rewards - np.min(reinforce_rewards)) > 1e-5:
                normalized_reinforce_rewards = 2 * (reinforce_rewards - np.min(reinforce_rewards)) / np.max(
                    reinforce_rewards - np.min(reinforce_rewards)) - 1
            else:
                normalized_reinforce_rewards = reinforce_rewards
            reinforce_reward = normalized_reinforce_rewards[-1]

            print("----------------------------------------")
            print("Reinforce reward: ", "%.10f" % (reinforce_reward))
            print("----------------------------------------")

            self.train_dve(self.dve_model, self.dve_optimizer, dvrl_input, est_dv_curr, reinforce_reward)

        self.save_dve(self.model_dir, type="final")  # Save data value estimator

    def save_dve(self, path, type):
        """
        Save reinforce model
        """
        torch.save(self.dve_model.state_dict(), path + f"reinforce_{type}")
        torch.save(self.dve_optimizer.state_dict(), path + f"reinforce_optimizer_{type}")

    def load_dve(self, path, type):
        """
        Load reinforce model
        """
        self.dve_model.load_state_dict(torch.load(path + f"reinforce_{type}", map_location=torch.device(self.device)))
        self.dve_optimizer.load_state_dict(
            torch.load(path + f"reinforce_optimizer_{type}", map_location=torch.device(self.device)))

    def eval_policy(self, policy, env, eval_episodes=10, plot=False):
        avg_reward = 0.
        plt.clf()
        start_states = []
        color_list = cm.rainbow(np.linspace(0, 1, eval_episodes + 2))

        for i in range(eval_episodes):
            state, done = env.reset(), False
            states_list = []
            start_states.append(state)
            while not done:
                action = policy.select_action(np.array(state))
                state, reward, done, _ = env.step(action)
                avg_reward += reward
                states_list.append(state)
            states_list = np.array(states_list)

            if plot:
                plt.scatter(states_list[:, 0], states_list[:, 1], color=color_list[i], alpha=0.1)
                plt.scatter(8, 10, color='white', alpha=0.1)
                plt.scatter(2, 0, color='white', alpha=0.1)
        if plot:
            start_states = np.array(start_states)
            plt.scatter(start_states[:, 0], start_states[:, 1], color='red')
            plt.savefig('./eval_fig')

        avg_reward /= eval_episodes
        normalized_score = env.get_normalized_score(avg_reward)

        info = {'AverageReturn': avg_reward, 'NormReturn': normalized_score}
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, {normalized_score:.3f}")
        print("---------------------------------------")
        return normalized_score

    ###### for D4RL
    def eval_policy_norm_score(self, model, model_name="", eval_episodes=10, seed_num=None, disable_tqdm=True,
                               t_env="target_1"):
        """
        Runs policy for X episodes and returns average normalized score
        A fixed seed is used for the eval environment
        """
        env = self.target_env
        env.seed(100)

        sum_reward = 0.

        for _ in tqdm(range(eval_episodes), disable=disable_tqdm):
            state, done = env.reset(), False
            while not done:
                action = model.select_action(np.array(state), eval=True)
                state, reward, done, _ = env.step(action)
                sum_reward += reward
            score = env.get_normalized_score(sum_reward)

        avg_score = score / eval_episodes * 100
        print("---------------------------------------")
        print(f"{self.env_name} | {model_name}: Evaluation over {eval_episodes} episodes of {t_env}: {avg_score:.3f}")
        print("---------------------------------------")
        return avg_score

    def data_valuate(self, dataset, batch_size):
        """
        Estimate the value of each sample in the specified data set
        """
        print('save data values')
        file_path = '%s/dvrl_%s_train_%d.json' % (self.results_dir, self.dict["env"], len(dataset))
        data_values = []
        not_dones = []

        rng = Generator(PCG64(12345))

        for i in tqdm(range(0, dataset.size, batch_size)):
            s, a, s_, r, nd = dataset.sample(batch_size=batch_size, to_device=False)  # O

            with torch.no_grad():
                batch_values = self.dve_model(torch.FloatTensor(np.hstack((s, s_, a))).to(self.device))
            data_values.extend(detach(batch_values))
            not_dones.extend(nd)
        data_values = np.array(data_values)
        sel_vec = rng.binomial(1, data_values, data_values.shape)

        not_dones = np.array(not_dones)

        print(data_values)
        print(np.mean(data_values))

        np.save(f'{self.results_dir}/plot_data_values', data_values, allow_pickle=True)
        np.save(f'{self.results_dir}/plot_not_dones', not_dones, allow_pickle=True)

        dvrl_out = [str(data_values[i]) for i in range(data_values.shape[0])]
        json.dump(dvrl_out, open(file_path, 'w'), indent=4)

        return data_values, sel_vec