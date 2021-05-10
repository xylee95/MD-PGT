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
from envs import rastrigin, quadratic, sphere, griewangk, styblinski_tang
import plot_surface

import copy

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
					choices=('rastrigin','quad2d','quad3d', 'quad5d','quad10d', 'sphere','griewangk','tang'))
parser.add_argument('--gpu', type=bool, default=False, help='Enable GPU')
parser.add_argument('--opt', type=str, default='sgd', help='Optimizer',
					choices=('adam', 'sgd', 'rmsprop'))
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum term for SGD')
#parser.add_argument('--entropy_coef', type=float, default=0.0, help='Entropy coefficient term')
args = parser.parse_args()

torch.manual_seed(args.seed)

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()
		self.dense1 = nn.Linear(state_dim, 64)
		self.dense2 = nn.Linear(64, 64)
		self.dense3 = nn.Linear(64, 3)
		self.saved_log_probs = []
		#self.entropy = []
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
	dist = policy(state) # --> Gaussian ( mu | sigma )
	action = dist.sample() # action ~ Gaussian(mu | sigma)
	policy.saved_log_probs.append(dist.log_prob(action)) ## log prob( a | s)
	#policy.entropy.append(dist.entropy())
	return action

def grad_traj_prev_weights(state_list, action_list, policy, old_policy):

	old_policy_log_probs = []

	for i in range(len(state_list)):
		dist = old_policy(state_list[i])
		log_prob = dist.log_prob(action_list[i])
		old_policy_log_probs.append(log_prob)

	eps = np.finfo(np.float32).eps.item()
	R = 0
	policy_loss = []
	returns = []
	for r in policy.rewards[::-1]:
		R = r + args.gamma * R
		returns.insert(0, R)
	returns = torch.tensor(returns)
	returns = (returns - returns.mean()) / (returns.std() + eps)
	for old_log_prob, R in zip(old_policy_log_probs, returns):
		policy_loss.append(-old_log_prob * R)
	
	optimizer.zero_grad() 

	policy_loss = torch.stack(policy_loss).sum() 
	print('Loss:',policy_loss)
	policy_loss.backward()

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

	#policy_entropy = torch.stack(policy.entropy).sum()
	policy_loss = torch.stack(policy_loss).sum() 
	#policy_loss = torch.clip(policy_loss, -1, 1)
	#policy_loss = policy_loss + entropy_coef*policy_entropy
	print('Loss:',policy_loss)
	policy_loss.backward()

def update_weights(policy, optimizer):
	optimizer.step()
	del policy.rewards[:]
	del policy.saved_log_probs[:]

# unoptimized version
def compute_IS_weight(action_list, state_list, cur_policy, old_policy):
	num_list = []
	denom_list = []
	weight_list = []
	# for each policy
	for i in range(len(old_policy)):
		prob_curr_traj = [] # list of probability of every action taken under curreny policy
		# for each step taken
		for j in range(len(action_list)):
			# ### accurate method ####
			# # obtain distribution for given state
			# cur_dist = cur_policy[i](torch.FloatTensor(state_list[j]))
			# # compute log prob of action
			# # action_list is in the shape of (episode_len, 1, num_agents)
			# log_prob = cur_dist.log_prob(action_list[j][0][i])
			# ##

			### faster method ###
			# use save log probability attached to agent
			log_prob = cur_policy[i].saved_log_probs[j][0]
			###
			prob = np.exp(log_prob.detach().numpy())
			prob_curr_traj.append(prob)

		# multiply along all
		prob_tau = np.prod(prob_curr_traj)

		prob_old_traj = []
		# for each step taken
		for j in range(len(action_list)):
			# obtain distribution for given state
			old_dist = old_policy[i](torch.FloatTensor(state_list[j]))
			# compute log prob of action
			# action_list is in the shape of (episode_len, 1, num_agents)
			log_prob = old_dist.log_prob(action_list[j][0][i])
			prob = np.exp(log_prob.detach().numpy())
			prob_old_traj.append(prob)

		# multiply along all
		prob_old_tau = np.prod(prob_old_traj)

		weight = prob_old_tau / (prob_tau + 1e-8)
		weight_list.append(weight)
		num_list.append(prob_old_tau)
		denom_list.append(prob_tau)
	
	return weight_list, num_list, denom_list


def main():
	# initialize env
	num_agents = args.num_agents
	dimension = args.dim
	if num_agents == 1:
		action_dim = dimension
		setup = 'centralized'
		fpath = os.path.join('results', setup, args.env, str(dimension) + 'D', args.opt)
	else:
		assert num_agents > 1
		action_dim = 1
		setup = 'decentralized'
		fpath = os.path.join('results', setup, args.env, str(dimension) + 'D', args.opt)

	if not os.path.isdir(fpath):
		os.makedirs(fpath)

	if args.env == 'rastrigin':
		env = rastrigin.Rastrigin(dimension=dimension, seed=args.seed)
	elif args.env == 'quad2d':
		env = quadratic.Quadratic(dimension=2, seed=args.seed)
	elif args.env == 'quad3d':
		env = quadratic.Quadratic3D(dimension=3, seed=args.seed)
	elif args.env == 'quad5d':
		env = quadratic.Quadratic5D(dimension=5, seed=args.seed)
	elif args.env == 'quad10d':
		env = quadratic.Quadratic10D(dimension=10, seed=args.seed)
	elif args.env == 'sphere':
		env = sphere.Sphere(dimension=dimension, seed=args.seed)
	elif args.env == 'griewangk':
		env = griewangk.Griewangk(dimension=dimension, seed=args.seed)
	elif args.env == 'tang':
		env = styblinski_tang.Styblinski_Tang(dimension=dimension, seed=args.seed)
	else:
		print('Wrong spelling')
		exit()

	# initliaze multiple agents and optimizer
	if args.gpu:
		device = 'cuda:0'
	else:
		device = 'cpu'

	agents = []	
	optimizers = []
	if args.opt == 'adam':
		for i in range(num_agents):
			agents.append(Policy(state_dim=dimension, action_dim=action_dim).to(device))
			optimizers.append(optim.Adam(agents[i].parameters(), lr=3e-4))
	elif args.opt == 'sgd':
		for i in range(num_agents):
			agents.append(Policy(state_dim=dimension, action_dim=action_dim).to(device))
			optimizers.append(optim.SGD(agents[i].parameters(), lr=3e-4, momentum=args.momentum))
	elif args.opt == 'rmsprop':
		for i in range(num_agents):
			agents.append(Policy(state_dim=dimension, action_dim=action_dim).to(device))
			optimizers.append(optim.RMSprop(agents[i].parameters(), lr=3e-4))

	#create copy of old agents
	old_agents = copy.deepcopy(agents)
	action_list = []
	state_list = []

	# RL setup
	num_episodes = args.num_episodes
	max_eps_len = args.max_eps_len
	done = False
	R = 0 
	R_hist = []
	y_hist = []
	R_hist_plot = []
	y_hist_plot = []
	isw_plot = []
	num_plot = []
	denom_plot = []

	for episode in range(num_episodes):
		state = env.reset()
		state_list.append(state)
		state = torch.FloatTensor(state).to(device)

		phi = copy.deepcopy(old_agents)
		old_agent = copy.deepcopy(agents)

		if episode == num_episodes - 1:
			path = [state]

		for t in range(1, max_eps_len):  # Don't infinite loop while learning
			actions = []
			for policy in agents:
				action = select_action(state, policy)
				actions.append(action)
			
			if action_dim == 1:
				actions = torch.as_tensor([actions])

			#step through enviroment with set of actions. rewards is list of reward
			state, rewards, done, y = env.step(actions)
			#print('State:', state, 'Action:', actions, 'Rewards:', rewards)
			action_list.append(actions)
			state_list.append(state)
			if episode == num_episodes - 1:
				path.append(state)
			for agent in agents:
				agent.rewards.append(rewards)
				
			state = torch.FloatTensor(state).to(device)
			R += rewards
			reset = t == max_eps_len-1
			if done or reset:
				print('Episode:',episode, 'Reward', R, 'Done', done)
				R_hist.append(R)
				y_hist.append(y)
				R = 0
				break

		isw, num, denom = compute_IS_weight(action_list, state_list, agents, phi)
		print(isw)
		isw_plot.append(isw)
		num_plot.append(num)
		denom_plot.append(denom)

		for policy, optimizer in zip(agents, optimizers):
			compute_grads(policy, optimizer)

		if num_agents > 1:
			agents = global_average(agents, num_agents)

		for policy, optimizer in zip(agents, optimizers):
			update_weights(policy, optimizer)

		#update old_agents to current agent
		action_list = []
		state_list = []

		if episode == num_episodes - 1 and dimension == 2:
			path.pop(0)
			plot_surface.visualize(env, path, fpath, setup + ' ' + args.env + '' + args.opt)

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
	plt.title(str(dimension) + '-d ' + setup + ' ' + args.env + args.opt + ' ' + str(args.seed))
	plt.savefig(os.path.join(fpath, str(args.seed) + '_R.jpg'))
	plt.close()

	plt.figure()
	plt.plot(y_hist_plot)
	plt.ylabel('F(y)')
	plt.xlabel('Episodes')
	plt.title(str(dimension) + '-d ' + setup + ' ' + args.env + args.opt + ' ' + str(args.seed))
	plt.savefig(os.path.join(fpath, str(args.seed) + '_Y.jpg'))

	isw_plot = np.array(isw_plot)
	num_plot = np.array(num_plot)
	denom_plot = np.array(denom_plot)

	plt.figure()
	for i in range(num_agents):
		plt.plot(isw_plot[:,i], label='Agent' + str(i))
	plt.legend()
	plt.ylabel('Importance Sampling Weight')
	plt.xlabel('Episodes')
	plt.title(str(dimension) + '-d ' + setup + ' ' + args.env + args.opt + ' ' + str(args.seed))
	plt.savefig(os.path.join(fpath, str(args.seed) + '_ISW.jpg'))

	plt.figure()
	for i in range(num_agents):
		plt.plot(num_plot[:,i], label='Agent' + str(i))
	plt.legend()
	plt.ylabel('Importance Sampling Weight (Numerator)')
	plt.xlabel('Episodes')
	plt.title(str(dimension) + '-d ' + setup + ' ' + args.env + args.opt + ' ' + str(args.seed))
	plt.savefig(os.path.join(fpath, str(args.seed) + '_ISW_num.jpg'))

	plt.figure()
	for i in range(num_agents):
		plt.plot(denom_plot[:,i], label='Agent' + str(i))
	plt.legend()
	plt.ylabel('Importance Sampling Weight (Denominator)')
	plt.xlabel('Episodes')
	plt.title(str(dimension) + '-d ' + setup + ' ' + args.env + args.opt + ' ' + str(args.seed))
	plt.savefig(os.path.join(fpath, str(args.seed) + '_ISW_denom.jpg'))

	np.save(os.path.join(os.path.join(fpath, 'R_array_' + str(args.seed) + '.npy')), R_hist_plot)
	np.save(os.path.join(os.path.join(fpath, 'Y_array_' + str(args.seed) + '.npy')), y_hist_plot)

if __name__ == '__main__':
	main()