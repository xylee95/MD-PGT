# MD-PGT
Repository for implementing decentralized PGT

# Available Environments
- Quadratic 2D, 3D and 10D
- Rastrigin
- Griewangk
- Styblinski-Tang

# Available Agents
- (De)centralized policy gradient (PG)

# Running Codes
- Centralized
~~~
python train.py --num_agents 2 --dim 2 --env rastrigin --num_episodes 20
~~~
- Decentralized
~~~
python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 dist_train.py --dim 2 --env rastrigin --num_episodes 20
~~~

# To do:
- Try same agent initialization
- Run one complete set experiments to compare decentralized vs centralized with different optimizers and multiple seeds for all environmens
- Implement MD-PGT (decentralized version with gradient tracking and variance reduction)
- Implement simple multi agent grid world?
- Enforce different topology for decentralized agents

# Decentralized To do:
- Try other env
- Verify performance
- Fix seed?
- Verify path=[state]

# Bugs to fix:



# Notes:
- For convex functions (quadratic and sphere), reward of -F(x) works well
- For non-convex functions (Rastrigin), reward of -||x|| works slightly better than -F(x)
- After limiting action bounds of Rastrgin to [-0.05, 0.05], both forms of reward function works and not much differences. Keeping it to -F(x) for simplicity. 
