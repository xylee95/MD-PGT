import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import pfrl
import pdb
import envs
from envs import rastrigin

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
					help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
args = parser.parse_args()

#env = gym.make('CartPole-v1')
#env.seed(args.seed)
dimension = 3
env = rastrigin.Rastrigin(dimension=dimension)
torch.manual_seed(args.seed)

class Policy(nn.Module):
	def __init__(self, action_dim):
		super(Policy, self).__init__()
		self.dense1 = nn.Linear(action_dim, 128)
		self.dense2 = nn.Linear(128, 64)
		self.dense3 = nn.Linear(64, action_dim)
		self.distribution = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
			action_size=dimension,
			var_type="diagonal",
			var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
			var_param_init=0,  # log std = 0 => std = 1
			)

		self.saved_log_probs = []
		self.rewards = []

	def forward(self, x):
		x = torch.tanh(self.dense1(x))
		x = torch.tanh(self.dense2(x))
		x = self.dense3(x)
		dist = self.distribution(x)
		return dist

policy = Policy(action_dim=dimension)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
	state = torch.from_numpy(state).float().unsqueeze(0)
	dist = policy(state)
	#m = Categorical(probs)
	action = dist.sample()
	policy.saved_log_probs.append(dist.log_prob(action))
	return action

def finish_episode():
	R = 0
	policy_loss = []
	returns = []
	for r in policy.rewards[::-1]:
		R = r + args.gamma * R
		returns.insert(0, R)
	returns = torch.tensor(returns)
	returns = (returns - returns.mean()) / (returns.std() + eps)
	for log_prob, R in zip(policy.saved_log_probs, returns):
		policy_loss.append(-log_prob * R)
	optimizer.zero_grad()
	policy_loss = torch.cat(policy_loss).sum()
	policy_loss.backward()
	optimizer.step()
	del policy.rewards[:]
	del policy.saved_log_probs[:]

def main():
	num_episodes = 2000
	done = False
	max_eps_len = 100
	R = 0 
	R_hist = []
	for i in range(num_episodes):
		state = env.reset()
		for t in range(1, max_eps_len):  # Don't infinite loop while learning
			action = select_action(state)
			action = np.clip(action, env.min_action, env.max_action)
			state, reward, done, y = env.step(action)
			# if i % 100 == 0:
			# 	print('Action: {} State: {} F(y):{} Reward: {} Done: {}'.format(action, state, y, reward, done))
			policy.rewards.append(reward)
			R += reward
			reset = t == max_eps_len-1
			if done or reset:
				R_hist.append(R)
				state = env.reset()
				R = 0
				break

		finish_episode()
		if i % args.log_interval == 0:
			running_reward = np.mean(R_hist)
			print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
				  i, R_hist[-1], running_reward))
		if i % 100 == 0:
			print('Last Action: {} State: {} F(y):{} Reward: {} Done: {}'.format(action, state, y, reward, done))
			R_hist = []


if __name__ == '__main__':
	main()