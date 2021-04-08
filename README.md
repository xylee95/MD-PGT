# MD-PGT
Repository for implementing decentralized PGT

# Available Environments
- Quadratic 2D and 3D
- Rastrigin
- Griewangk
- Styblinski-Tang

# Available Agents
- (De)centralized policy gradient (PG)

# To do:
- Run one complete set experiments to compare decentralized vs centralized for all enviroment once before moving to MD-PGT
- Implement MD-PGT (decentralized version with gradient tracking and variance reduction)
- Implement more test functions (Complex convex functions, Griewank, Etc)
- Enforce different topology for decentralized agents

# Bugs to fix:
- Need to modify contour plotting code to account for different envs below:
- Griewangk domain ranges from -600 to 600
- Styblinski-Tang minima is at -39.16599xd and domain ranges from -5 to 5


# Notes:
- For convex functions (quadratic and sphere), reward of -F(x) works well
- For non-convex functions (Rastrigin), reward of -||x|| works slightly better than -F(x)
- After limiting action bounds of Rastrgin to [-0.05, 0.05], both forms of reward function works and not much differences. Keeping it to -F(x) for simplicity. 
