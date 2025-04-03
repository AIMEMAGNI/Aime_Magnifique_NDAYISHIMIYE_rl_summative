import gymnasium as gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class MissionEnv(gym.Env):
    def __init__(self):
        super(MissionEnv, self).__init__()

        # Define grid size and initial position
        self.grid_size = 5
        self.agent_position = (0, 0)

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Discrete(
            self.grid_size * self.grid_size)

        # Define reward grid
        self.reward_grid = np.zeros((self.grid_size, self.grid_size))
        self.reward_grid[1, 1] = 1
        self.reward_grid[3, 3] = 1
        self.reward_grid[2, 2] = 1
        self.reward_grid[0, 3] = -1
        self.reward_grid[3, 1] = -1
        self.reward_grid[4, 3] = -1

        # Max steps and render mode
        self.max_steps = 50
        self.current_step = 0
        self.render_mode = 'rgb_array'

        self.fig = None  # Initialize to None for rendering

    def reset(self, seed=None, options=None):
        self.agent_position = (0, 0)
        self.current_step = 0
        return np.array([self._encode_state()], dtype=np.int32), {}

    def step(self, action):
        x, y = self.agent_position

        # Action movement logic
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.grid_size - 1, y + 1)

        self.agent_position = (x, y)
        reward = float(self.reward_grid[x, y])

        # Termination check
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # No truncation for now

        return np.array([self._encode_state()], dtype=np.int32), reward, terminated, truncated, {}

    def _encode_state(self):
        return self.agent_position[0] * self.grid_size + self.agent_position[1]

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self._render_rgb_array()

        if mode == 'human':
            return self._render_human()

    def _render_rgb_array(self):
        # Initialize the figure and axis if needed
        if not self.fig:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.ax.set_xticks(np.arange(0, self.grid_size, 1))
            self.ax.set_yticks(np.arange(0, self.grid_size, 1))
            self.ax.set_xticklabels([])  # Remove x-axis labels
            self.ax.set_yticklabels([])  # Remove y-axis labels
            self.ax.set_xlim([0, self.grid_size])
            self.ax.set_ylim([0, self.grid_size])
            self.ax.grid(which='both')

        # Clear previous frame
        self.ax.cla()

        # Redraw grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.reward_grid[i, j] == 1:
                    self.ax.add_patch(patches.Rectangle(
                        (j, self.grid_size - 1 - i), 1, 1, edgecolor='black', facecolor='green'))
                elif self.reward_grid[i, j] == -1:
                    self.ax.add_patch(patches.Rectangle(
                        (j, self.grid_size - 1 - i), 1, 1, edgecolor='black', facecolor='red'))

        # Draw the agent
        agent_x, agent_y = self.agent_position[1] + \
            0.5, self.grid_size - 1 - self.agent_position[0] + 0.5
        agent_radius = 0.3
        self.ax.add_patch(plt.Circle((agent_x, agent_y),
                          agent_radius, color='yellow', ec='black', lw=1))

        # Invert y-axis to match grid coordinates
        self.ax.invert_yaxis()

        # Draw everything
        self.fig.canvas.draw()

        # Return the image as RGBA array
        return np.array(self.fig.canvas.renderer.buffer_rgba())

    def _render_human(self):
        plt.show(block=True)  # This will display the plot in a blocking way
