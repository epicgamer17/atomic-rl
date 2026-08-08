import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
import functools


class MatchingPenniesEnv(AECEnv):
    """
    Turn-based Matching Pennies environment following PettingZoo AEC standards.

    In this turn-based version:
    - Players take turns choosing heads (0) or tails (1)
    - After both players have chosen, rewards are calculated
    - If choices match, Player 0 wins (+1, -1)
    - If choices don't match, Player 1 wins (-1, +1)
    """

    metadata = {
        "render_modes": ["human"],
        "name": "matching_pennies_v0",
    }

    def __init__(self, render_mode=None, max_cycles=100):
        super().__init__()

        self.possible_agents = ["player_0", "player_1"]
        self.max_cycles = max_cycles
        self.render_mode = render_mode

        # Action space: 0 = heads, 1 = tails
        self._action_spaces = {
            agent: spaces.Discrete(2) for agent in self.possible_agents
        }

        # Observation space: [round_number, my_last_action, opponent_last_action, my_score, opponent_score, phase]
        # phase: 0 = waiting for my action, 1 = waiting for opponent action, 2 = round complete
        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Initialize state
        self.agents = []
        self.agent_selection = None
        self.agent_selector = None
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }

        # Game state
        self.round_number = 0
        self.current_actions = {}
        self.last_actions = {"player_0": -1, "player_1": -1}
        self.scores = {"player_0": 0, "player_1": 0}
        self.phase = 0  # 0: new round, 1: waiting for second player, 2: round complete

        # AEC required attributes
        self.terminations = {}
        self.truncations = {}
        self.rewards = {}
        self.infos = {}
        self._cumulative_rewards = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def observe(self, agent):
        """Get observation for the current agent."""
        opponent = "player_1" if agent == "player_0" else "player_0"

        # Determine phase for this agent
        if self.phase == 0:  # New round
            agent_phase = 0 if agent == self.agent_selection else 1
        elif self.phase == 1:  # One action taken
            if agent in self.current_actions:
                agent_phase = 1  # Waiting for opponent
            else:
                agent_phase = 0  # My turn
        else:  # Round complete
            agent_phase = 2

        observation = np.array(
            [
                float(self.round_number),
                float(self.last_actions[agent]),
                float(self.last_actions[opponent]),
                float(self.scores[agent]),
                float(self.scores[opponent]),
                float(agent_phase),
            ],
            dtype=np.float32,
        )

        return observation

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()

        # Reset game state
        self.round_number = 0
        self.current_actions = {}
        self.last_actions = {"player_0": -1, "player_1": -1}
        self.scores = {"player_0": 0, "player_1": 0}
        self.phase = 0

        # Reset AEC attributes
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {
            agent: {"player": self.agents.index(agent)} for agent in self.agents
        }
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

    def step(self, action):
        """Execute one step for the current agent."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # Agent is done, just return with no-op
            return self._no_op_step(action)

        # Store the current agent's action
        current_agent = self.agent_selection
        self.current_actions[current_agent] = action

        # Clear rewards for this step
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {
            agent: {
                "round_number": self.round_number,
                "last_action": self.last_actions[agent],
                "player": self.agents.index(agent),
            }
            for agent in self.agents
        }

        if len(self.current_actions) == 1:
            # First player has acted, wait for second player
            self.phase = 1
            self.agent_selection = self.agent_selector.next()
            # Update infos
        elif len(self.current_actions) == 2:
            # Both players have acted, calculate rewards
            self.phase = 2
            self._calculate_round_rewards()
            self._complete_round()

        # Update cumulative rewards
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

    def _calculate_round_rewards(self):
        """Calculate rewards for the completed round."""
        action_0 = self.current_actions["player_0"]
        action_1 = self.current_actions["player_1"]

        # Matching pennies rules
        if action_0 == action_1:  # Actions match
            self.rewards = {"player_0": 1.0, "player_1": -1.0}
            # print("winner: player_0")
        else:  # Actions don't match
            self.rewards = {"player_0": -1.0, "player_1": 1.0}
            # print("winner: player_1")

        # Update scores
        for agent in self.agents:
            self.scores[agent] += self.rewards[agent]

    def _complete_round(self):
        """Complete the current round and prepare for next."""
        # Store last actions
        self.last_actions = self.current_actions.copy()

        # Clear current actions for next round
        self.current_actions = {}

        # Increment round
        self.round_number += 1

        # Check if game should end
        if self.round_number >= self.max_cycles:
            self.truncations = {agent: True for agent in self.agents}
        else:
            # Start new round
            self.phase = 0
            self.agent_selection = self.agent_selector.next()

    def _no_op_step(self, action):
        """Handle steps for agents that are already done."""
        # Don't change anything, agent is done
        pass

    def last(self):
        """Return observation for the last agent to act."""
        agent = self.agent_selection
        observation = self.observe(agent)
        return (
            observation,
            self.rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"\n=== Matching Pennies - Round {self.round_number} ===")
            print(
                f"Scores: Player 0: {self.scores['player_0']}, Player 1: {self.scores['player_1']}"
            )

            if self.phase == 0:
                print(f"New round - {self.agent_selection} to act")
            elif self.phase == 1:
                acting_player = [
                    p for p in self.possible_agents if p not in self.current_actions
                ][0]
                print(f"Waiting for {acting_player} to act")
            elif self.phase == 2:
                action_names = {0: "Heads", 1: "Tails"}
                p0_action = self.current_actions.get("player_0", -1)
                p1_action = self.current_actions.get("player_1", -1)
                if p0_action != -1 and p1_action != -1:
                    print(
                        f"Round complete: Player 0 chose {action_names[p0_action]}, Player 1 chose {action_names[p1_action]}"
                    )
                    print(
                        f"Rewards: Player 0: {self.rewards['player_0']:+.0f}, Player 1: {self.rewards['player_1']:+.0f}"
                    )

    def close(self):
        """Close the environment."""
        pass


def env(**kwargs):
    """Factory function for creating the environment."""
    env_instance = MatchingPenniesEnv(**kwargs)
    # Wrap with standard PettingZoo wrappers
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


# Parallel version (converts AEC to parallel)
def parallel_env(**kwargs):
    """Factory function for creating the parallel version."""
    from pettingzoo.utils import aec_to_parallel

    aec_env_instance = env(**kwargs)
    parallel_env_instance = aec_to_parallel(aec_env_instance)
    return parallel_env_instance


# Gymnasium wrapper for single-agent training (turn-based against AI)
class MatchingPenniesGymEnv(gym.Env):
    """
    Turn-based Gymnasium wrapper for Matching Pennies environment.
    Player alternates with an AI opponent.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, opponent_policy="random", max_rounds=50):
        super().__init__()

        self.render_mode = render_mode
        self.opponent_policy = opponent_policy
        self.max_rounds = max_rounds

        # Action space: 0 = heads, 1 = tails
        self.action_space = spaces.Discrete(2)

        # Observation space: [round, my_last, opponent_last, my_score, opponent_score, turn_phase]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Game state
        self.round_number = 0
        self.player_score = 0
        self.opponent_score = 0
        self.last_player_action = -1
        self.last_opponent_action = -1
        self.is_player_turn = True

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        self.round_number = 0
        self.player_score = 0
        self.opponent_score = 0
        self.last_player_action = -1
        self.last_opponent_action = -1
        self.is_player_turn = True

        observation = self._get_observation()
        info = {"phase": "player_turn"}

        return observation, info

    def step(self, action):
        """Execute one step in the environment."""
        if action not in [0, 1]:
            raise ValueError("Action must be 0 (heads) or 1 (tails)")

        if not self.is_player_turn:
            raise ValueError("Not player's turn")

        player_action = action

        # Get opponent action
        opponent_action = self._get_opponent_action()

        # Calculate rewards
        if player_action == opponent_action:  # Actions match
            player_reward = 1.0
            opponent_reward = -1.0
        else:  # Actions don't match
            player_reward = -1.0
            opponent_reward = 1.0

        # Update scores
        self.player_score += player_reward
        self.opponent_score += opponent_reward

        # Store last actions
        self.last_player_action = player_action
        self.last_opponent_action = opponent_action

        # Update round
        self.round_number += 1

        # Check if game is done
        terminated = False
        truncated = self.round_number >= self.max_rounds

        observation = self._get_observation()

        info = {
            "player_action": player_action,
            "opponent_action": opponent_action,
            "round": self.round_number,
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
            "phase": "round_complete",
        }

        if self.render_mode == "human":
            self._render_round(player_action, opponent_action, player_reward)

        return observation, player_reward, terminated, truncated, info

    def _get_opponent_action(self):
        """Get opponent's action based on policy."""
        if self.opponent_policy == "random":
            return self.np_random.integers(0, 2)
        elif self.opponent_policy == "always_heads":
            return 0
        elif self.opponent_policy == "always_tails":
            return 1
        elif self.opponent_policy == "copy":
            return self.last_player_action if self.last_player_action != -1 else 0
        elif self.opponent_policy == "opposite":
            return 1 - self.last_player_action if self.last_player_action != -1 else 1
        elif self.opponent_policy == "tit_for_tat":
            # Copy what worked for opponent last time
            if self.last_opponent_action != -1 and self.last_player_action != -1:
                if (
                    self.last_opponent_action == self.last_player_action
                ):  # Opponent won last time
                    return self.last_opponent_action
                else:  # Opponent lost, try opposite
                    return 1 - self.last_opponent_action
            return self.np_random.integers(0, 2)
        else:
            return self.np_random.integers(0, 2)

    def _get_observation(self):
        """Get current observation."""
        return np.array(
            [
                float(self.round_number),
                float(self.last_player_action),
                float(self.last_opponent_action),
                float(self.player_score),
                float(self.opponent_score),
                float(1 if self.is_player_turn else 0),
            ],
            dtype=np.float32,
        )

    def _render_round(self, player_action, opponent_action, reward):
        """Render the completed round."""
        action_names = {0: "Heads", 1: "Tails"}
        print(
            f"Round {self.round_number}: Player={action_names[player_action]}, "
            f"Opponent={action_names[opponent_action]}, Reward={reward:+.0f}"
        )
        print(f"Scores: Player={self.player_score}, Opponent={self.opponent_score}")

    def render(self):
        """Render the environment."""
        print(f"Matching Pennies - Round {self.round_number}")
        print(f"Scores: Player={self.player_score}, Opponent={self.opponent_score}")

    def close(self):
        """Close the environment."""
        pass
