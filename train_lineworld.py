import argparse
import gym
import os
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
from envs import lineworld

import time, datetime

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99,
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--dim', type=int, default=2, help='Number of dimension')
parser.add_argument('--num_agents', type=int, default=2, help='Number of agents')
parser.add_argument('--max_eps_len', type=int, default=500, help='Number of steps per episode')
parser.add_argument('--num_episodes', type=int, default=5000, help='Number training episodes')
parser.add_argument('--env', type=str, default='lineworld', help='Training env')
parser.add_argument('--gpu', type=bool, default=False, help='Enable GPU')
parser.add_argument('--opt', type=str, default='sgd', help='Optimizer',
					choices=('adam', 'sgd', 'rmsprop'))
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum term for SGD')

args = parser.parse_args()

torch.manual_seed(args.seed)

class Policy(nn.Module):
	def __init__(self, state_dim):
		super(Policy, self).__init__()
		self.dense1 = nn.Linear(state_dim, 64)
		self.dense2 = nn.Linear(64, 64)
		self.dense3 = nn.Linear(64, 3)
		self.saved_log_probs = []
		self.rewards = []

	def forward(self, x):
		x1 = torch.tanh(self.dense1(x))
		x2 = torch.tanh(self.dense2(x1))
		x3 = self.dense3(x2)
		dist = Categorical(logits=x3)
		return dist

def select_action(state, policy):
	try:
		state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
	except:
		pass
	dist = policy(state)
	action = dist.sample() #action will be sampled from 3 categories
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
	#print(policy_loss)
	policy_loss.backward()

def update_weights(policy, optimizer):
	optimizer.step()
	del policy.rewards[:]
	del policy.saved_log_probs[:]

def main():
	# initialize env
	num_agents = args.num_agents
	dimension = args.dim
	fpath = os.path.join('dpg_results', args.env, str(dimension) + 'D', args.opt)

	if not os.path.isdir(fpath):
		os.makedirs(fpath)

	env = lineworld.LineWorld(dimension=dimension, seed=args.seed)

	# initliaze multiple agents and optimizer
	if args.gpu:
		device = 'cuda:0'
	else:
		device = 'cpu'

	agents = []
	optimizers = []
	if args.opt == 'adam':
		for i in range(num_agents):
			agents.append(Policy(state_dim=dimension).to(device))
			optimizers.append(optim.Adam(agents[i].parameters(), lr=3e-4))
	elif args.opt == 'sgd':
		for i in range(num_agents):
			agents.append(Policy(state_dim=dimension).to(device))
			optimizers.append(optim.SGD(agents[i].parameters(), lr=3e-4, momentum=args.momentum))
	elif args.opt == 'rmsprop':
		for i in range(num_agents):
			agents.append(Policy(state_dim=dimension).to(device))
			optimizers.append(optim.RMSprop(agents[i].parameters(), lr=3e-4))

	# RL setup
	num_episodes = args.num_episodes
	done = False
	max_eps_len = args.max_eps_len
	R = 0 
	R_hist = []
	R_hist_plot = []

	train_start = time.time()
	for episode in range(num_episodes):
		ep_start = time.time()
		state = env.reset()
		state = torch.FloatTensor(state).to(device)
		if episode == num_episodes - 1:
			path = [state]
		for t in range(1, max_eps_len):  # Don't infinite loop while learning
			actions = []
			for policy in agents:
				action = select_action(state, policy)
				actions.append(action)
			actions = torch.as_tensor([actions])
	
			#step through enviroment with set of actions. rewards is list of reward
			state, rewards, done = env.step(actions)
			for i in range(len(agents)):
				#print('r:', rewards[i])
				agents[i].rewards.append(rewards[i])
				
			state = torch.FloatTensor(state).to(device)
			R += np.sum(rewards)
			reset = t == max_eps_len-1
			if done or reset:
				print(f'Done: {done} Reset:{reset} State:{state} reward:{rewards}')
				R_hist.append(R)
				R = 0
				break

		for policy, optimizer in zip(agents, optimizers):
			compute_grads(policy, optimizer)
		if num_agents > 1:
			agents = global_average(agents, num_agents)
		for policy, optimizer in zip(agents, optimizers):
			update_weights(policy, optimizer)

		if episode % args.log_interval == 0:
			avg_reward = np.sum(R_hist)/len(R_hist)
			R_hist_plot.append(avg_reward)
			R_hist = []
			print(f'Episode:{episode} Average reward:{avg_reward:.2f}')
							
		if episode % 100 == 0:
			ep_time = datetime.timedelta(seconds=time.time() - ep_start)
			print(f'Episode {episode} elapsed time: %s s' % (ep_time))
			print(f'Last Action: {actions} State: {state} Reward: {rewards} Done: {done}')

	total_time = datetime.timedelta(seconds=time.time() - train_start)
	print(f'Total elapsed time: %s s' % (total_time))
	
	plt.figure()
	plt.plot(R_hist_plot)
	plt.ylabel('Reward')
	plt.xlabel('Episodes')
	plt.title(str(dimension) + '-d ' + ' ' + args.env + args.opt + ' ' + str(args.seed))
	plt.savefig(os.path.join(fpath, str(args.seed) + '_R.jpg'))
	plt.close()

	np.save(os.path.join(os.path.join(fpath, 'R_array_' + str(args.seed) + '.npy')), R_hist_plot)

if __name__ == '__main__':
	main()