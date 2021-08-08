import torch
import numpy as np
import json

###### Function related to topology and pi
def load_pi(num_agents, topology):
	wsize = num_agents
	if topology == 'dense':
		topo = 1
	elif topology == 'ring':
		topo = 2
	elif topology == 'bipartite':
		topo = 3

	with open('generate_topology/connectivity/%s_%s.json'%(wsize,topo), 'r') as f:
		cdict = json.load(f) # connectivity dict.
	return cdict['pi']

###### Update functions shared by DPG, MDPG and MDPGT

# this should act the same as global average but generalize to various topology
def take_param_consensus(agents, pi):
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

	# layer_1_w = torch.sum(torch.stack(tuple(layer_1_w)),0)
	# layer_1_b = torch.sum(torch.stack(tuple(layer_1_b)),0)

	# layer_2_w = torch.sum(torch.stack(tuple(layer_2_w)),0)
	# layer_2_b = torch.sum(torch.stack(tuple(layer_2_b)),0)

	# layer_3_w = torch.sum(torch.stack(tuple(layer_3_w)),0)
	# layer_3_b = torch.sum(torch.stack(tuple(layer_3_b)),0)


	for agent_idx, agent in enumerate(agents):
		agent.dense1.weight.data = torch.sum(torch.stack(tuple(layer_1_w))*torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1),0).clone()
		agent.dense1.bias.data = torch.sum(torch.stack(tuple(layer_1_b))*torch.tensor(pi[agent_idx]).unsqueeze(-1),0).clone()

		agent.dense2.weight.data = torch.sum(torch.stack(tuple(layer_2_w))*torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1),0).clone()
		agent.dense2.bias.data = torch.sum(torch.stack(tuple(layer_2_b))*torch.tensor(pi[agent_idx]).unsqueeze(-1),0).clone()

		agent.dense3.weight.data = torch.sum(torch.stack(tuple(layer_3_w))*torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1),0).clone()
		agent.dense3.bias.data = torch.sum(torch.stack(tuple(layer_3_b))*torch.tensor(pi[agent_idx]).unsqueeze(-1),0).clone()

	return agents

	# layer_1_w = dict()
	# layer_1_b = dict()

	# layer_2_w = dict()
	# layer_2_b = dict()

	# layer_3_w = dict()
	# layer_3_b = dict()

	# for i in range(len(agents)):
	# 	layer_1_w[i] = agents[i].dense1.weight.data
	# 	layer_1_b[i] = agents[i].dense1.bias.data

	# 	layer_2_w[i] = agents[i].dense2.weight.data
	# 	layer_2_b[i] = agents[i].dense2.bias.data

	# 	layer_3_w[i] = agents[i].dense3.weight.data
	# 	layer_3_b[i] = agents[i].dense3.bias.data

	# # for each agent
	# for j in range(len(agents)):
	# 	consensus_layer_1_w = 0
	# 	consensus_layer_1_b = 0 

	# 	consensus_layer_2_w = 0
	# 	consensus_layer_2_b = 0 

	# 	consensus_layer_3_w = 0
	# 	consensus_layer_3_b = 0 

	# 	# for each row of pi, loop over columns (other agents)
	# 	for k in range(len(agents)):
	# 		consensus_layer_1_w += pi[j][k]*layer_1_w[k]
	# 		consensus_layer_1_b += pi[j][k]*layer_1_b[k] 

	# 		consensus_layer_2_w += pi[j][k]*layer_2_w[k]
	# 		consensus_layer_2_b += pi[j][k]*layer_2_b[k] 

	# 		consensus_layer_3_w += pi[j][k]*layer_3_w[k]
	# 		consensus_layer_3_b += pi[j][k]*layer_3_b[k] 
 

	# 	agents[j].dense1.weight.data = consensus_layer_1_w
	# 	agents[j].dense1.bias.data = consensus_layer_1_b

	# 	agents[j].dense2.weight.data = consensus_layer_2_w
	# 	agents[j].dense2.bias.data = consensus_layer_2_b

	# 	agents[j].dense3.weight.data = consensus_layer_3_w
	# 	agents[j].dense3.bias.data = consensus_layer_3_b

	# return agents

# unused, remove later
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

	layer_1_w = torch.sum(torch.stack(layer_1_w)/num_agents,0) 
	layer_1_b = torch.sum(torch.stack(layer_1_b)/num_agents,0)

	layer_2_w = torch.sum(torch.stack(layer_2_w)/num_agents,0)
	layer_2_b = torch.sum(torch.stack(layer_2_b)/num_agents,0)

	layer_3_w = torch.sum(torch.stack(layer_3_w)/num_agents,0)
	layer_3_b = torch.sum(torch.stack(layer_3_b)/num_agents,0)

	for agent in agents:
		agent.dense1.weight.data = layer_1_w.clone()
		agent.dense1.bias.data = layer_1_b.clone()

		agent.dense2.weight.data = layer_2_w.clone()
		agent.dense2.bias.data = layer_2_b.clone()

		agent.dense3.weight.data = layer_3_w.clone()
		agent.dense3.bias.data = layer_3_b.clone()
	return agents

def compute_IS_weight(action_list, state_list, cur_policy, old_policy, min_isw):
	num_list = []
	denom_list = []
	weight_list = []
	# for each policy
	for i in range(len(old_policy)):
		prob_curr_traj = [] # list of probability of every action taken under curreny policy
		# for each step taken
		for j in range(len(action_list)):
			# use save log probability attached to agent
			log_prob = cur_policy[i].saved_log_probs[j][0]
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
		weight_list.append(np.max((min_isw, weight)))
		num_list.append(prob_old_tau)
		denom_list.append(prob_tau)
	
	return weight_list, num_list, denom_list

def compute_grad_traj_prev_weights(args, state_list, action_list, policy, old_policy, optimizer):

	old_policy_log_probs = []

	for i in range(len(action_list)):
		dist = old_policy(torch.FloatTensor(state_list[i]))
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
	policy_loss.backward()
	# list of tensors gradients, each tensor has shape
	grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
			else None for group in optimizer.param_groups for p in group['params']]
	return grad

def compute_grads(args, policy, optimizer, minibatch_init=False):
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

	# list of tensors gradients, each tensor has shape
	if minibatch_init == True:
		grad = [np.array(p.grad.detach().clone().flatten()) if (p.requires_grad is True and p.grad is not None)
			else None for group in optimizer.param_groups for p in group['params']]
	elif minibatch_init == False:
		grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
			else None for group in optimizer.param_groups for p in group['params']]
	return grad

def update_weights(policy, optimizer, grads=None):
	if grads is not None:
		optimizer.step(grads=grads)
	else:
		optimizer.step()
	del policy.rewards[:]
	del policy.saved_log_probs[:]


def compute_u(args, policy, optimizer, prev_u, isw, prev_g, beta):

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

	# list of tensors gradients, each tensor has shape
	grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
			else None for group in optimizer.param_groups for p in group['params']]
	
	# grad, prev_u and prev_g are all flatten grads
	assert grad[0].shape == prev_u[0].shape == prev_g[0].shape
	# list of flatten surrogate grads [128, 64, 4096, ...]
	grad_surrogate = [1]*len(grad)
	#u = beta*grad + (1-beta)*(prev_u + grad - isw*prev_g)
	for i in range(len(grad_surrogate)):
		grad_surrogate[i] = beta*grad[i] + (1-beta)*(prev_u[i] + grad[i] - isw*prev_g[i])
	return grad_surrogate


###### Update Functions Specific to MDPGT

def update_v(v_k, u_k, prev_u_k):
	assert v_k[0].shape == u_k[0].shape == prev_u_k[0].shape
	next_v_k = [1]*len(v_k)
	#next_v_k = v_k + u_k - prev_u_k
	for i in range(len(v_k)):
		next_v_k[i] = v_k[i] + u_k[i] - prev_u_k[i]
	return next_v_k

def take_grad_consensus(v_k_list, pi):

	# list of n agents, each agent a list of 6 layers
	grads_0 = []
	grads_1 = []
	grads_2 = []
	grads_3 = []
	grads_4 = []
	grads_5 = []

	for i in range(len(v_k_list)):
		grads_0.append(v_k_list[i][0])
		grads_1.append(v_k_list[i][1])
		grads_2.append(v_k_list[i][2])
		grads_3.append(v_k_list[i][3])
		grads_4.append(v_k_list[i][4])
		grads_5.append(v_k_list[i][5])

	consensus_v_k = []
	for j in range(len(v_k_list)):
		grads_0_j = torch.sum(torch.stack(tuple(grads_0))*torch.tensor(pi[j]).unsqueeze(-1),0).clone()
		grads_1_j = torch.sum(torch.stack(tuple(grads_1))*torch.tensor(pi[j]).unsqueeze(-1),0).clone()
		grads_2_j = torch.sum(torch.stack(tuple(grads_2))*torch.tensor(pi[j]).unsqueeze(-1),0).clone()
		grads_3_j = torch.sum(torch.stack(tuple(grads_3))*torch.tensor(pi[j]).unsqueeze(-1),0).clone()
		grads_4_j = torch.sum(torch.stack(tuple(grads_4))*torch.tensor(pi[j]).unsqueeze(-1),0).clone()
		grads_5_j = torch.sum(torch.stack(tuple(grads_5))*torch.tensor(pi[j]).unsqueeze(-1),0).clone()
		v_k_list = [grads_0_j, grads_1_j, grads_2_j, grads_3_j, grads_4_j, grads_5_j]
		consensus_v_k.append(v_k_list)

	return consensus_v_k


	# # v_k_list is a list of n agents, each agent a list of 6 layers
	# grads_0 = dict()
	# grads_1 = dict()
	# grads_2 = dict()
	# grads_3 = dict()
	# grads_4 = dict()
	# grads_5 = dict()

	# # grads_0 denotes dict of gradient of layer 0 with keys of n agents
	# for i in range(len(v_k_list)):
	# 	grads_0[i] = v_k_list[i][0]
	# 	grads_1[i] = v_k_list[i][1]
	# 	grads_2[i] = v_k_list[i][2]
	# 	grads_3[i] = v_k_list[i][3]
	# 	grads_4[i] = v_k_list[i][4]
	# 	grads_5[i] = v_k_list[i][5]

	# consensus_v_k_list = []

	# # for each agent
	# for j in range(len(v_k_list)):

	# 	consensus_grad_0 = 0
	# 	consensus_grad_1 = 0
	# 	consensus_grad_2 = 0
	# 	consensus_grad_3 = 0
	# 	consensus_grad_4 = 0
	# 	consensus_grad_5 = 0

	# 	# for each agent, loop over other agent
	# 	for k in range(len(v_k_list)):

	# 		consensus_grad_0 += pi[j][k]*grads_0[k]
	# 		consensus_grad_1 += pi[j][k]*grads_1[k]
	# 		consensus_grad_2 += pi[j][k]*grads_2[k]
	# 		consensus_grad_3 += pi[j][k]*grads_3[k]
	# 		consensus_grad_4 += pi[j][k]*grads_4[k]
	# 		consensus_grad_5 += pi[j][k]*grads_5[k]

	# 	consensus_v_k_list.append([consensus_grad_0, consensus_grad_1, consensus_grad_2, \
	# 		consensus_grad_3, consensus_grad_4, consensus_grad_5])

	# return consensus_v_k_list

#unused, remove later
def take_consensus(v_k_list, num_agents):
	# list of n agents, each agent a list of 6 layers
	grads_0 = []
	grads_1 = []
	grads_2 = []
	grads_3 = []
	grads_4 = []
	grads_5 = []

	for i in range(len(v_k_list)):
		grads_0.append(v_k_list[i][0])
		grads_1.append(v_k_list[i][1])
		grads_2.append(v_k_list[i][2])
		grads_3.append(v_k_list[i][3])
		grads_4.append(v_k_list[i][4])
		grads_5.append(v_k_list[i][5])

	grads_0 = torch.sum(torch.stack(grads_0)/len(grads_0),0).clone()
	grads_1 = torch.sum(torch.stack(grads_1)/len(grads_1),0).clone()
	grads_2 = torch.sum(torch.stack(grads_2)/len(grads_2),0).clone()
	grads_3 = torch.sum(torch.stack(grads_3)/len(grads_3),0).clone()
	grads_4 = torch.sum(torch.stack(grads_4)/len(grads_4),0).clone()
	grads_5 = torch.sum(torch.stack(grads_5)/len(grads_5),0).clone()

	v_k_list = [grads_0, grads_1, grads_2, grads_3, grads_4, grads_5]

	consensus_v_k = [v_k_list]*num_agents
	return consensus_v_k