import numpy as np
import gym
from gym import spaces
from typing import List, Tuple

TRAP = -1
EMPTY = 0
AGENT = 1
WALL = 2
GOAL = 100

class agent:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.pos = tuple(np.random.randint((self.row, self.col)))
        self.goal = tuple(np.random.randint((self.row, self.col)))

    def reset(self):
        self.pos = tuple(np.random.randint((self.row, self.col)))
        self.goal = tuple(np.random.randint((self.row, self.col)))


class grid:
    def __init__(self, size: Tuple[int,int], agents: int, seed: int):
        assert len(size) == 2, "grid size must be 2D"
        self.grid_row = size[0]
        self.grid_col = size[1]
        self.agents = agents
        np.random.seed(seed)

        self.list_of_agents = [agent(self.grid_row, self.grid_col) for i in range(self.agents)]

        # Setup action mapper
        self.action_mapper = {0: 'up', 1: 'down', 2: 'left', 3: 'right', 4: 'no-op'}
        self.store_position = {}
        self.env_params()

    def env_params(self):
        # reset gridworld state
        self.grid = np.zeros([self.grid_row, self.grid_col])

        # reset individual goal pos-goal pairs and register in grid
        for i in range(self.agents):
            agent_n = self.list_of_agents[i]
            agent_n.reset()
            self.list_of_agents[i] = agent_n
            self.grid[agent_n.pos] = AGENT
            self.grid[agent_n.goal] = GOAL
            self.check_spawn(i)

        self.update_state()

    def update_state(self):
        # compile per-agent states
        self.state = []
        for i in range(self.agents):
            # per-agent state is [pos, goal, other-pos, other-goal]
            temp_state = []
            mod_list = list(range(self.agents))
            mod_list.remove(i)
            temp_state.append(self.list_of_agents[i].pos)
            temp_state.append(self.list_of_agents[i].goal)
            for others in mod_list:
                temp_state.append(self.list_of_agents[others].pos)
            for others in mod_list:
                temp_state.append(self.list_of_agents[others].goal)
            self.state.append(np.array(temp_state).reshape(-1))


    def check_spawn(self, idx):
        condition = 1
        while condition:
            up = (self.list_of_agents[idx].goal[0] - 1, self.list_of_agents[idx].goal[1])
            down = (self.list_of_agents[idx].goal[0] + 1, self.list_of_agents[idx].goal[1])
            left = (self.list_of_agents[idx].goal[0], self.list_of_agents[idx].goal[1] - 1)
            right = (self.list_of_agents[idx].goal[0], self.list_of_agents[idx].goal[1] + 1)
            if np.array_equal(self.list_of_agents[idx].pos, self.list_of_agents[idx].goal):
                self.list_of_agents[idx].pos = tuple(np.random.randint((self.grid_row, self.grid_col)))
                continue
            elif np.array_equal(self.list_of_agents[idx].pos, up):
                self.list_of_agents[idx].pos = tuple(np.random.randint((self.grid_row, self.grid_col)))
                continue
            elif np.array_equal(self.list_of_agents[idx].pos, down):
                self.list_of_agents[idx].pos = tuple(np.random.randint((self.grid_row, self.grid_col)))
                continue
            elif np.array_equal(self.list_of_agents[idx].pos, left):
                self.list_of_agents[idx].pos = tuple(np.random.randint((self.grid_row, self.grid_col)))
                continue
            elif np.array_equal(self.list_of_agents[idx].pos, right):
                self.list_of_agents[idx].pos = tuple(np.random.randint((self.grid_row, self.grid_col)))
                continue
            else:
                condition = 0

    def reset(self):
        self.env_params()
        return self.state

    def getreward(self):
        reward = np.full(self.agents, 0)
        done_status = []

        for agent in range(self.agents):
             # and self.sparse[agent] == 0
            if np.array_equal(self.list_of_agents[agent].goal, self.list_of_agents[agent].pos):
                reward[agent] += 1
                # self.sparse[agent] = 1
            else:
                reward[agent] += 0

        for agent in range(self.agents):
            if np.array_equal(self.list_of_agents[agent].goal, self.list_of_agents[agent].pos):
                done_status.append(1)
            else:
                done_status.append(0)

        if np.sum(done_status) == self.agents:
            for agent in range(self.agents):
                reward[agent] += 100*self.agents

        return reward

    def getdone(self):
        done = []
        for i in range(self.agents):
            if np.array_equal(self.list_of_agents[i].goal, self.list_of_agents[i].pos):
                done.append(1)
            else:
                done.append(0)
        
        if np.sum(done) == self.agents:
            done = 1
        else:
            done = 0

        return done

    def step(self, raw_action):
        raw_action = raw_action[0]
        assert len(raw_action) == self.agents
        # self.state = self.state.reshape(self.agents, -1)

        """
        Reference
        0 | 1 | 2
        --
        1
        --
        2
        --

        (row, col)

        ------------
        Actions
        UP    -> row -1
        DOWN  -> row +1
        LEFT  -> col -1
        RIGHT -> col +1
        STAY  -> no-op
        ------------
        """

        for i in range(self.agents):
            self.grid[self.list_of_agents[i].pos] = EMPTY
            self.grid[self.list_of_agents[i].goal] = GOAL
            action = self.action_mapper[raw_action[i].numpy().item()]

            if action == 'up':
                nextstep =  tuple(np.array([self.list_of_agents[i].pos[0] - 1, self.list_of_agents[i].pos[1]]))
            elif action == 'down':
                nextstep =  tuple(np.array([self.list_of_agents[i].pos[0] + 1, self.list_of_agents[i].pos[1]]))
            elif action == 'left':
                nextstep =  tuple(np.array([self.list_of_agents[i].pos[0], self.list_of_agents[i].pos[1] - 1]))
            elif action == 'right':
                nextstep =  tuple(np.array([self.list_of_agents[i].pos[0], self.list_of_agents[i].pos[1] + 1]))
            else:
                nextstep = tuple(np.array([self.list_of_agents[i].pos[0], self.list_of_agents[i].pos[1]]))

            if nextstep[0] >= 0 and nextstep[0] <= self.grid_row - 1:
                if nextstep[1] >= 0 and nextstep[1] <= self.grid_col - 1:
                    self.list_of_agents[i].pos = nextstep

        for i in range(self.agents):
            self.grid[self.list_of_agents[i].pos] = AGENT

        reward = self.getreward()
        done = self.getdone()
        self.update_state()

        return self.state, reward, done, 0

    def render(self):
        # agent_states = self.state.reshape(self.agents, -1)

        for i in range(self.grid_row):
            rowstrings = '--'
            for rs in range(self.grid_col):
                rowstrings += '-'*4
            print(rowstrings)
            out = '| '
            for j in range(self.grid_col):
                if self.grid[i, j] == AGENT:
                    token = '('
                    for idx in range(self.agents):
                        if np.array_equal(self.list_of_agents[idx].pos, np.array([i,j])):
                            token += f'{idx}'
                    token += ')'
                elif self.grid[i, j] == TRAP:
                    token = 'T'
                elif self.grid[i, j] == EMPTY:
                    token = '0'
                elif self.grid[i, j] == GOAL:
                    token = 'G'
                elif self.grid[i, j] == WALL:
                    token = '^'
                else:
                    print(f"Invalid value inside grid: {self.grid[i, j]}")
                    print(self.grid[i, j] == AGENT)
                    print(self.grid)
                out += token + ' | '
            print(out)
        print(rowstrings)

