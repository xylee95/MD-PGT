# MD-PGT
Repository for implementing decentralized PGT

# Available Environments
- Quadratic, Rastrigin, Griewangk, Styblinski-Tang for 2D, 3D, 5D and 10D (Unused, removing later)
- Lineworld
- Particle-world (available in another repo forked from Loew et al)
- Gridworld (Check with Russell if still used)

# Available Agents
- DPG
- MDPG
- MDPGT

# Main files used are:
- train_lineworld_dpg.py
- train_lineworld_mdpg.py
- train_lineworld_mdpgt.py
- train_particleworld_dpg.py
- train_particleworld_mdpg.py
- train_particleworld_mdpgt.py
- model.py : code for policy network and related functions
- update_functions.py : all functions related to update rules and consensus for MDPG and MDPGT

Both MDPG and MDPGT has the option of using Minibatch Initialization to compute batch gradient surrogate.
