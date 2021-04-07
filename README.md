# MD-PGT
Repository for implementing decentralized PGT

# Available Environments
- Quadratic 2D and 3D
- Rastrigin

# Available Agents
- (De)centralized policy gradient (PG)

# To do:
- Implement MD-PGT (decentralized version with gradient tracking and variance reduction)
- Implement discrete version for PG in Rastrigin to see if better performance
- Test better reward functions for Rastrigin centralized PG version
- Implement more test functions (Complex convex functions, Griewank, Etc)
- Enforce different topology for decentralized agents

# Bugs to fix:
> python -m pdb train.py --env rastrigin --num_agents 2 --dim 2

This triggers an error, where the state obs input passes through the first dense layer and produces a NaN.

# Notes:
- For convex functions (quadratic and sphere), reward of -F(x) works well
- For non-convex functions (Rastrigin), reward of -||x|| works slightly better than -F(x)
- After limiting action bounds of Rastrgin to [-0.05, 0.05], both forms of reward function works and not much differences. Keeping it to -F(x) for simplicity. 
