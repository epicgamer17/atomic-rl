import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SlipperyGridEnv(gym.Env):
    """
    A Standard Gym Environment for the Slippery Grid World.

    Designed for Debugging Stochastic MuZero.

    The Map:
    - 0: Empty (Safe)
    - 1: Agent Start
    - 2: Wall (Bounce back)
    - 3: Cliff (Terminates with negative reward)
    - 4: Goal (Terminates with positive reward)

    Stochasticity (The "Chance Codes"):
    - 0: Intended Move (Prob = 1 - slip_prob)
    - 1: Slip 90° Clockwise (Prob = slip_prob / 2)
    - 2: Slip 90° Counter-Clockwise (Prob = slip_prob / 2)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, slip_probability=0.3, max_episode_steps=100):
        super().__init__()

        self.slip_probability = slip_probability
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Default "Cliff" Map Layout
        # 0: Empty, 1: Start, 2: Wall, 3: Cliff, 4: Goal
        self.map_layout = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],  # Safe path (longer)
                [0, 3, 3, 3, 3, 4],  # Cliff path (shorter but risky)
                [1, 0, 0, 0, 0, 0],  # Start
            ],
            dtype=np.int32,
        )

        self.h, self.w = self.map_layout.shape
        self.window = None
        self.clock = None

        # Action Space: 0:Up, 1:Right, 2:Down, 3:Left
        self.action_space = spaces.Discrete(4)

        # Observation Space: 3 Channels x Height x Width
        # Channel 0: Agent Position (1.0 where agent is, 0.0 elsewhere)
        # Channel 1: Hazards/Cliffs (1.0 where cliff is)
        # Channel 2: Goal (1.0 where goal is)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.h, self.w), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_step = 0

        # Locate start position
        start_indices = np.where(self.map_layout == 1)
        self.agent_pos = np.array([start_indices[0][0], start_indices[1][0]])

        # Prepare static feature maps
        self._cliff_map = (self.map_layout == 3).astype(np.float32)
        self._goal_map = (self.map_layout == 4).astype(np.float32)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {
            "chance": 0,
            "legal_moves": [0, 1, 2, 3],  # All moves always valid input
        }

    def step(self, action):
        self.episode_step += 1

        # --- Stochastic Logic ---
        # 0: Intended, 1: Slip CW, 2: Slip CCW
        rand_val = self.np_random.random()
        chance_outcome = 0
        actual_move = action

        if rand_val < self.slip_probability:
            if rand_val < self.slip_probability / 2:
                chance_outcome = 1  # Slip CW
                actual_move = (action + 1) % 4
            else:
                chance_outcome = 2  # Slip CCW
                actual_move = (action - 1) % 4

        # --- Dynamics ---
        # 0:Up, 1:Right, 2:Down, 3:Left
        dy, dx = 0, 0
        if actual_move == 0:
            dy = -1
        elif actual_move == 1:
            dx = 1
        elif actual_move == 2:
            dy = 1
        elif actual_move == 3:
            dx = -1

        target_y = self.agent_pos[0] + dy
        target_x = self.agent_pos[1] + dx

        # Boundary Check
        if target_y < 0 or target_y >= self.h or target_x < 0 or target_x >= self.w:
            target_y, target_x = self.agent_pos  # Hit boundary, stay put

        cell_type = self.map_layout[target_y, target_x]

        # Interaction Logic
        reward = -0.1  # Step penalty
        terminated = False

        if cell_type == 2:  # Wall
            # Bounce back to original position
            target_y, target_x = self.agent_pos
        elif cell_type == 3:  # Cliff
            reward = -10.0
            terminated = True
        elif cell_type == 4:  # Goal
            reward = 10.0
            terminated = True

        self.agent_pos = np.array([target_y, target_x])

        truncated = self.episode_step >= self.max_episode_steps
        if truncated:
            reward = -10.0

        if self.render_mode == "human":
            self._render_frame()

        # --- Info Dict (Crucial for Debugging) ---
        info = {
            "chance": chance_outcome,  # The Ground Truth Code (0, 1, or 2)
            "actual_action": actual_move,
            "legal_moves": [0, 1, 2, 3],  # All moves always valid input
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs = np.zeros((3, self.h, self.w), dtype=np.float32)

        # Channel 0: Agent
        obs[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        # Channel 1: Cliff
        obs[1] = self._cliff_map
        # Channel 2: Goal
        obs[2] = self._goal_map

        return obs

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # Optional: Simple PyGame render for human viewing
        if self.window is None and self.render_mode == "human":
            import pygame

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.w * 64, self.h * 64))

        if self.clock is None and self.render_mode == "human":
            import pygame

            self.clock = pygame.time.Clock()

        # Canvas drawing
        import pygame

        canvas = pygame.Surface((self.w * 64, self.h * 64))
        canvas.fill((255, 255, 255))

        colors = {
            0: (240, 240, 240),  # Floor
            1: (240, 240, 240),  # Start (looks like floor)
            2: (50, 50, 50),  # Wall
            3: (200, 50, 50),  # Cliff
            4: (50, 200, 50),  # Goal
        }

        for y in range(self.h):
            for x in range(self.w):
                color = colors.get(self.map_layout[y, x])
                pygame.draw.rect(canvas, color, pygame.Rect(x * 64, y * 64, 64, 64))
                pygame.draw.rect(
                    canvas, (200, 200, 200), pygame.Rect(x * 64, y * 64, 64, 64), 1
                )

        # Draw Agent
        pygame.draw.circle(
            canvas,
            (50, 50, 200),
            (int((self.agent_pos[1] + 0.5) * 64), int((self.agent_pos[0] + 0.5) * 64)),
            20,
        )

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
