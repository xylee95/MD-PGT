# MD-PGT
Repository for implementing decentralized PGT

# Available Environments
- Quadratic 2D, 3D and 10D
- Rastrigin
- Griewangk
- Styblinski-Tang

# Available Agents
- (De)centralized policy gradient (PG)

# To do:
- Debug sphere
- Run one complete set experiments to compare decentralized vs centralized with different optimizers and multiple seeds for all environmens
- Implement MD-PGT (decentralized version with gradient tracking and variance reduction)
- Implement simple multi agent grid world?
- Enforce different topology for decentralized agents

# Bugs to fix:
- Debug sphere centralized PG
- Need to modify contour plotting code to account for different envs below:
- Griewangk domain ranges from -600 to 600
- Styblinski-Tang minima is at f(x) = -39.16599 x d and x = (-2.903534,...) and domain ranges from -5 to 5


# Notes:
- For convex functions (quadratic and sphere), reward of -F(x) works well
- For non-convex functions (Rastrigin), reward of -||x|| works slightly better than -F(x)
- After limiting action bounds of Rastrgin to [-0.05, 0.05], both forms of reward function works and not much differences. Keeping it to -F(x) for simplicity. 
