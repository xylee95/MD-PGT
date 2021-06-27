import torch
import numpy as np

###### Update functions shared by DPG, MDPG and MDPGT

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

	grads_0 = torch.sum(torch.stack(grads_0),0)/len(grads_0)
	grads_1 = torch.sum(torch.stack(grads_1),0)/len(grads_1)
	grads_2 = torch.sum(torch.stack(grads_2),0)/len(grads_2)
	grads_3 = torch.sum(torch.stack(grads_3),0)/len(grads_3)
	grads_4 = torch.sum(torch.stack(grads_4),0)/len(grads_4)
	grads_5 = torch.sum(torch.stack(grads_5),0)/len(grads_5)

	v_k_list = [grads_0, grads_1, grads_2, grads_3, grads_4, grads_5]

	consensus_v_k = [v_k_list]*num_agents
	return consensus_v_k