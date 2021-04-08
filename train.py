import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib.pyplot as plt

import pfrl
import pdb
import envs
from envs import rastrigin, quadratic, sphere, griewangk, styblinski_tang
import plot_surface

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99,
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--dim', type=int, default=1, help='Number of dimension')
parser.add_argument('--num_agents', type=int, default=2, help='Number of agents')
parser.add_argument('--max_eps_len', type=int, default=100, help='Number of steps per episode')
parser.add_argument('--num_episodes', type=int, default=5000, help='Number training episodes')
parser.add_argument('--env', type=str, default='quad2d', help='Training env',
					choices=('rastrigin','quad2d','quad3d','sphere','griewangk','tang'))
parser.add_argument('--gpu', type=bool, default=False, help='Enable GPU')
args = parser.parse_args()

torch.manual_seed(args.seed)

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()
		self.dense1 = nn.Linear(state_dim, 128)
		self.dense2 = nn.Linear(128, 64)
		self.dense3 = nn.Linear(64, 1)
		self.distribution = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
			action_size=action_dim,
			var_type="diagonal",
			var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
			var_param_init=0,  # log std = 0 => std = 1
			)

		self.saved_log_probs = []
		self.rewards = []

	def forward(self, x):
		x1 = torch.tanh(self.dense1(x))
		x2 = torch.tanh(self.dense2(x1))
		x3 = self.dense3(x2)
		dist = self.distribution(x3)
		return dist

def select_action(state, policy):
	state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
	dist = policy(state)
	action = dist.sample()
	policy.saved_log_probs.append(dist.log_prob(action))
	return action

def global_average(agents, num_agents):
	layer_1_w = []
	layer_1_b = []

	layer_2_w = []
	layer_2_b = []

	layer_3_w = []
	layer_3_b = []

	for agent in agents:
		layer_1_w.append(agent.dense1.weight.data)
		layer_1_b.append(agent.dense1.bias.data)

		layer_2_w.append(agent.dense2.weight.data)
		layer_2_b.append(agent.dense2.bias.data)

		layer_3_w.append(agent.dense3.weight.data)
		layer_3_b.append(agent.dense3.bias.data)

	layer_1_w = torch.sum(torch.stack(layer_1_w),0) / num_agents
	layer_1_b = torch.sum(torch.stack(layer_1_b),0) / num_agents

	layer_2_w = torch.sum(torch.stack(layer_2_w),0) / num_agents
	layer_2_b = torch.sum(torch.stack(layer_2_b),0) / num_agents

	layer_3_w = torch.sum(torch.stack(layer_3_w),0) / num_agents
	layer_3_b = torch.sum(torch.stack(layer_3_b),0) / num_agents

	for agent in agents:
		agent.dense1.weight.data = layer_1_w
		agent.dense1.bias.data = layer_1_b

		agent.dense2.weight.data = layer_2_w
		agent.dense2.bias.data = layer_2_b

		agent.dense3.weight.data = layer_3_w
		agent.dense3.bias.data = layer_3_b

	return agents

def compute_grads(policy, optimizer):
	eps = np.finfo(np.float32).eps.item()
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
	policy_loss = torch.stack(policy_loss).sum()
	policy_loss.backward()

def update_weights(policy, optimizer):
	optimizer.step()
	del policy.rewards[:]
	del policy.saved_log_probs[:]

def main():
	# initialize env
	num_agents = args.num_agents
	dimension = args.dim
	if num_agents == 1:
		action_dim = dimension
		setup = 'centralized'
	else:
		action_dim = 1
		setup = 'decentralized'

	if args.env == 'rastrigin':
		env = rastrigin.Rastrigin(dimension=dimension, seed=args.seed)
	elif args.env == 'quad2d':
		env = quadratic.Quadratic(dimension=2, seed=args.seed)
	elif args.env == 'quad3d':
		env = quadratic.Quadratic3D(dimension=3, seed=args.seed)
	elif args.env == 'sphere':
		env = sphere.Sphere(dimension=dimension, seed=args.seed)
	elif args.env == 'griewangk':
		env = griewangk.Griewangk(dimension=dimension, seed=args.seed)
	elif args.env == 'tang':
		env = styblinski_tang.Styblinski_Tang(dimension=dimension, seed=args.seed)
	else:
		print('wrong spelling')
		exit()

	# initliaze multiple agents and optimizer
	if args.gpu:
		device = 'cuda:0'
	else:
		device = 'cpu'

	agents = []
	optimizers = []
	for i in range(num_agents):
		agents.append(Policy(state_dim=dimension, action_dim=action_dim).to(device))
		optimizers.append(optim.Adam(agents[i].parameters(), lr=3e-4))

	# RL setup
	num_episodes = args.num_episodes
	done = False
	max_eps_len = args.max_eps_len
	R = 0 
	R_hist = []
	y_hist = []
	R_hist_plot = []
	y_hist_plot = []

	for episode in range(num_episodes):
		state = env.reset()
		if episode == num_episodes - 1:
			path = [state]
		for t in range(1, max_eps_len):  # Don't infinite loop while learning
			actions = []
			for policy in agents:
				action = select_action(state, policy)
				actions.append(action)
			
			if action_dim == 1:
				actions = torch.clip(torch.as_tensor([actions]), env.min_action, env.max_action)
			else:
				actions = np.clip(actions[0], env.min_action, env.max_action)

			#step through enviroment with set of actions. rewards is list of reward
			state, rewards, done, y = env.step(actions)
			if episode == num_episodes - 1:
				path.append(state)
			for agent in agents:
				agent.rewards.append(rewards)
				
			R += rewards
			reset = t == max_eps_len-1
			if done or reset:
				R_hist.append(R)
				y_hist.append(y)
				state = env.reset()
				R = 0
				done = False
				break

		for policy, optimizer in zip(agents, optimizers):
			compute_grads(policy, optimizer)
		if action_dim != 1:
			agents = global_average(agents, num_agents)
		for policy, optimizer in zip(agents, optimizers):
			update_weights(policy, optimizer)

		if episode == num_episodes - 1 and args.dim == 2:
			plot_surface.visualize(env, path, setup + ' ' + args.env)

		if episode % args.log_interval == 0:
			avg_reward = np.sum(R_hist)/len(R_hist)
			avg_y = np.sum(y_hist)/len(y_hist)
			y_hist_plot.append(avg_y)
			R_hist_plot.append(avg_reward)
			y_hist = []
			R_hist = []
			print(f'Episode:{episode} Average reward:{avg_reward:.2f}')
							
		if episode % 100 == 0:
			print(f'Last Action: {actions} State: {state} F(y):{y} Reward: {rewards} Done: {done}')

	plt.figure()
	plt.plot(R_hist_plot)
	plt.ylabel('Reward')
	plt.xlabel('Episodes')
	plt.title(str(dimension) + '-d ' + setup + ' ' + args.env)
	plt.savefig(str(dimension) + '-d ' + setup + ' ' + args.env + '_R.jpg')
	plt.close()

	plt.figure()
	plt.plot(y_hist_plot)
	plt.ylabel('F(y)')
	plt.xlabel('Episodes')
	plt.title(str(dimension) + '-d ' + setup + ' ' + args.env)
	plt.savefig(str(dimension) + '-d ' + setup + ' ' + args.env + '_Y.jpg')

if __name__ == '__main__':
	main()