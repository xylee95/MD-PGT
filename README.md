# MD-PGT
Repository for implementing decentralized PGT

# Available Environments
- Centralized Quadratic 2D and 3D
- Centralized Rastrigin

# Available Agents
- Centralized policy gradient (PG)

# To do:
- ~~Implement decentralized version of Quadratic and Rastrigin environments (each agent takes one dimension)~~
- ~~Implement MA-PG (simple decentralized parameter-wise consensus)~~
- Implement MD-PGT (decentralized version with gradient tracking and variance reduction)
- Implement discrete version for PG in Rastrigin to see if better performance
- Test better reward functions for Rastrigin centralized PG version
- Implement more test functions (Complex convex functions, Griewank, Etc)
- Enforce different topology for decentralized agents

# Bugs to fix:
> python -m pdb train.py --env rastrigin --num_agents 2 --dim 2

This triggers an error, where the state obs input passes through the first dense layer and produces a NaN.
