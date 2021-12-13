# MD-PGT
Repository for implementing and reproducing the results for the paper MDPGT: Momentum-based Decentralized Policy Gradient Tracking: https://arxiv.org/abs/2112.02813

# Available Environments
- Lineworld
- Particle-world (installation instructions available at https://github.com/xylee95/MDPGT_particleworld

# Available Agents
- DPG: Decentralized Policy Gradients
- MDPG : Momentum Decentralized Policy Gradients
- MDPGT : Momentum-based Decentralized Policy Gradient Tracking

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

# Reproducing the results:
To reproduce the results shown in the paper, please check `run_exp.sh` for the relevant python commands. 

