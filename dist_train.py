import os
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

import torch.distributed as dist
import time, datetime

import warnings
warnings.filterwarnings("ignore")

# os.environ['OMP_NUM_THREADS'] = '5'


opt_list = {'adam': optim.Adam, 'sgd': optim.SGD, 'rmsprop': optim.RMSprop}

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99,
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--dim', type=int, default=1, help='Number of dimension')
parser.add_argument('--max_eps_len', type=int, default=100, help='Number of steps per episode')
parser.add_argument('--num_episodes', type=int, default=5000, help='Number training episodes')
parser.add_argument('--env', type=str, default='quad2d', help='Training env',
					choices=('rastrigin','quad2d','quad3d', 'quad10d', 'sphere','griewangk','tang'))
parser.add_argument('--opt', type=str, default='sgd', help='Optimizer',
					choices=('adam', 'sgd', 'rmsprop'))
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum term for SGD')

parser.add_argument('--gpu', action='store_true', default=False, help='Enable GPU')

# ================ For torch.distributed.launch multi-process argument ================
parser.add_argument('--local_rank', type=int, help='Required argument for torch.distributed.launch, similar as rank')

args = parser.parse_args()

# torch.manual_seed(args.seed)

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()
		self.dense1 = nn.Linear(state_dim, 128)
		self.dense2 = nn.Linear(128, 64)
		self.dense3 = nn.Linear(64, action_dim)
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
	try:
		state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
	except:
		pass
	dist = policy(state)
	action = dist.sample()
	policy.saved_log_probs.append(dist.log_prob(action))
	return action

def global_average_dist(dist, opt, pi):
	for param in opt.param_groups[0]['params']:
		#### All_gather parameters ####
		# Initialize agent parmeters placeholder across agents
		ag_params = [torch.zeros(param.size(),dtype=param.dtype) for _ in range(dist.get_world_size())]
		dist.all_gather(ag_params, param)
		
		param.data.mul_(0.0) # zero out curent param
		dist.barrier()

		# Multiply agent params with Pi and sum
		for ag_param, pival in zip(ag_params, pi):
			param.data.add_(other=ag_param.data, alpha=pival) # assign back avg params

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

	# Initialize dist. settings
	dist.init_process_group(backend='gloo', init_method='env://')

	# Dist. "Hello World" variables
	wsize = dist.get_world_size()
	rank = dist.get_rank()

	torch.manual_seed(rank)

	# Fully connected pi
	if wsize > 1:
		pi = [1/wsize for _ in range(wsize)]
	print(rank,': Hello World!')

	# num_agents = args.num_agents
	dimension = args.dim
	if wsize == 1:
		action_dim = dimension
		setup = 'centralized'
	else:
		action_dim = 1
		setup = 'decentralized'

	if rank == 0: # Single/First agent only
		fpath = os.path.join('results', setup, args.env, str(dimension) + 'D', args.opt)

		if not os.path.isdir(fpath):
			os.makedirs(fpath)

		# initialize env
		if args.env == 'rastrigin':
			env = rastrigin.Rastrigin(dimension=dimension, seed=args.seed)
		elif args.env == 'quad2d':
			env = quadratic.Quadratic(dimension=2, seed=args.seed)
		elif args.env == 'quad3d':
			env = quadratic.Quadratic3D(dimension=3, seed=args.seed)
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

		#### Scatter env to other agents? ####

	device = 'cuda:0' if args.gpu else 'cpu'

	dist.barrier()

	# initliaze multiple agents(policy) and optimizer
	policy = Policy(state_dim=dimension, action_dim=action_dim).to(device)
	# optimizer = optim.Adam(policy.parameters(), lr=3e-4)
	if args.opt=='sgd':
		optimizer = opt_list[args.opt](policy.parameters(), lr=3e-4, momentum=args.momentum)
	else:
		optimizer = opt_list[args.opt](policy.parameters(), lr=3e-4)

	# RL setup
	num_episodes = args.num_episodes
	done = False # not used
	max_eps_len = args.max_eps_len
	R = 0 
	R_hist = []
	y_hist = []
	R_hist_plot = []
	y_hist_plot = []

	train_start = time.time()
	for episode in range(num_episodes):
		ep_start = time.time()
		state = torch.tensor([0.0 for _ in range(dimension)], dtype=torch.float64) # Initialize placeholder across agents
		if rank == 0:
			state = env.reset()
			#### Scatter state to other agents ####
			state = torch.tensor(state)
			tmp = [state for _ in range(wsize)] # tensors to be scattered
			dist.scatter(state, tmp)
		else:
			dist.scatter(state)

		if episode == num_episodes - 1:
			path = [state.numpy()]

		for t in range(1, max_eps_len):  # Don't infinite loop while learning
			action = select_action(state, policy) # first action is all the same due to seed (double check updates!)
			#### Gather actions to agent 0 ####
			if rank == 0:
				actions = [action*0 for _ in range(wsize)]
				dist.gather(action, actions)
			else:
				dist.gather(action)
			
			# Initialize placeholders across agents
			rewards = torch.tensor(0.0, dtype=torch.float64)
			done_or_reset = torch.tensor(0.0, dtype=torch.float64)

			if rank == 0: # Only rank 0 have gathered actions
				if action_dim == 1:
					actions = torch.clip(torch.as_tensor([actions]), env.min_action, env.max_action)
				else:
					actions = np.clip(actions[0], env.min_action, env.max_action)

				#step through enviroment with set of actions. rewards is list of reward
				state, rewards, done, y = env.step(actions)
				if episode == num_episodes - 1:
					path.append(state)
					
				R += rewards
				reset = t == max_eps_len-1

				#### Scatter "done" and "reset" to other agents or they stuck in eps loop! ####
				# Boolean to floats
				reset_ = 0.0 if reset==False else 1.0
				done_ = 0.0 if done==False else 2.0

				# Combine to single var.
				done_or_reset = torch.tensor(done_+reset_, dtype=torch.float64)
				tmp1 = [done_or_reset for _ in range(wsize)] # tensors to be scattered
				dist.scatter(done_or_reset, tmp1)

			else:
				dist.scatter(done_or_reset)

			dist.barrier()

			#### Scatter reward to other agents ####
			if rank == 0:
				rewards = torch.tensor(rewards, dtype=torch.float64)
				tmp = [rewards for _ in range(wsize)] # tensors to be scattered
				dist.scatter(rewards, tmp)
			else:
				dist.scatter(rewards)

			if done_or_reset >= 1.0:
				if rank == 0:
					R_hist.append(R)
					y_hist.append(y)
					state = env.reset()
					R = 0
				else:
					pass
				break

			dist.barrier()

			policy.rewards.append(rewards) # Now append the rewards

		compute_grads(policy, optimizer)

		#### All_gather params (all_gather > gather because can account for pi later) ####
		if wsize > 1:
			global_average_dist(dist, optimizer, pi)

		update_weights(policy, optimizer)

		# Logging and misc. only on agent 0
		if rank == 0:
			if episode == num_episodes - 1 and args.dim == 2:
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
				ep_time = datetime.timedelta(seconds=time.time() - ep_start)
				print(f'Episode {episode} elapsed time: %s s' % (ep_time))
				print(f'Last Action: {actions} State: {state} F(y):{y} Reward: {rewards} Done: {done}')
		else:
			pass

	if rank == 0:
		total_time = datetime.timedelta(seconds=time.time() - train_start)
		print(f'Total elapsed time: %s s' % (total_time))

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

		np.save(os.path.join(os.path.join(fpath, 'R_array_' + str(args.seed) + '.npy')), R_hist_plot)
		np.save(os.path.join(os.path.join(fpath, 'Y_array_' + str(args.seed) + '.npy')), y_hist_plot)

if __name__ == '__main__':
	main()
