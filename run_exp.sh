# Experiments to generate results to compare DPG, MDPG and MDPGT for 5 agents in gridworld
python train_particleworld_dpg.py --num_agents 5 --dim 5 --seed 0 --topology dense
python train_particleworld_dpg.py --num_agents 5 --dim 5 --seed 1 --topology dense
python train_particleworld_dpg.py --num_agents 5 --dim 5 --seed 2 --topology dense
python train_particleworld_dpg.py --num_agents 5 --dim 5 --seed 3 --topology dense
python train_particleworld_dpg.py --num_agents 5 --dim 5 --seed 4 --topology dense

python train_particleworld_mdpg.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 0 --topology dense
python train_particleworld_mdpg.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 1 --topology dense
python train_particleworld_mdpg.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 2 --topology dense
python train_particleworld_mdpg.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 3 --topology dense
python train_particleworld_mdpg.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 4 --topology dense

python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 0 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 1 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 2 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 3 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 4 --topology dense

# experiments to compare different Beta values
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.2 --min_isw 0 --seed 0 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.3 --min_isw 0 --seed 1 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.4 --min_isw 0 --seed 2 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 3 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.6 --min_isw 0 --seed 4 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.7 --min_isw 0 --seed 5 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.8 --min_isw 0 --seed 6 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.9 --min_isw 0 --seed 7 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.99 --min_isw 0 --seed 8 --topology dense

# experiments to compare different topologies
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 0 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 1 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 2 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 3 --topology dense
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 4 --topology dense

python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 0 --topology ring
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 1 --topology ring
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 2 --topology ring
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 3 --topology ring
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 4 --topology ring

python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 0 --topology bipartite
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 1 --topology bipartite
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 2 --topology bipartite
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 3 --topology bipartite
python train_particleworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 4 --topology bipartite

# example command to reproduce supplementary results
python train_lineworld_dpg.py --num_agents 5 --dim 5 --seed 0 --topology dense
python train_lineworld_dpg.py --num_agents 5 --dim 5 --seed 1 --topology dense
python train_lineworld_dpg.py --num_agents 5 --dim 5 --seed 2 --topology dense
python train_lineworld_dpg.py --num_agents 5 --dim 5 --seed 3 --topology dense
python train_lineworld_dpg.py --num_agents 5 --dim 5 --seed 4 --topology dense

python train_lineworld_mdpg.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 0 --topology dense
python train_lineworld_mdpg.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 1 --topology dense
python train_lineworld_mdpg.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 2 --topology dense
python train_lineworld_mdpg.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 3 --topology dense
python train_lineworld_mdpg.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 4 --topology dense

python train_lineworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 0 --topology dense
python train_lineworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 1 --topology dense
python train_lineworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 2 --topology dense
python train_lineworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 3 --topology dense
python train_lineworld_mdpgt.py --num_agents 5 --dim 5 --beta 0.5 --min_isw 0 --seed 4 --topology dense