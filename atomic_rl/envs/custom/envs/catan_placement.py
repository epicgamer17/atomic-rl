import functools
from typing import List, Optional, Tuple, Dict
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import agent_selector

from catanatron.game import Game
from catanatron.models.enums import ActionType, Action, SETTLEMENT, CITY
from catanatron.models.map import NUM_NODES
from catanatron.models.board import get_edges
from catanatron.players.minimax import AlphaBetaPlayer

from .catan import CatanAECEnv, normalize_action

from pettingzoo.utils import wrappers

# --- Placement-Specific Action Space ---
PLACEMENT_SETTLEMENT_ACTIONS = [
    (ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(NUM_NODES)
]
PLACEMENT_ROAD_ACTIONS = [
    (ActionType.BUILD_ROAD, tuple(sorted(edge))) for edge in get_edges()
]


class CatanPlacementAECEnv(CatanAECEnv):
    """
    A specialized Catan environment for initial placement only.
    Agents only control settlement (and optionally road) placements.
    The rest of the game is played by AlphaBeta bots.
    """

    def __init__(
        self,
        num_players: int = 4,
        include_roads_in_action_space: bool = False,
        auto_play_roads: bool = True,
        *args,
        **kwargs,
    ):
        self.include_roads_in_action_space = include_roads_in_action_space
        self.auto_play_roads = auto_play_roads

        # Override action array for this specific environment
        if self.include_roads_in_action_space:
            self.placement_actions = (
                PLACEMENT_SETTLEMENT_ACTIONS + PLACEMENT_ROAD_ACTIONS
            )
        else:
            self.placement_actions = PLACEMENT_SETTLEMENT_ACTIONS

        super().__init__(num_players=num_players, *args, **kwargs)

        # Override the action spaces with the reduced size
        self.placement_action_space_size = len(self.placement_actions)
        self._action_spaces = {
            agent: spaces.Discrete(self.placement_action_space_size)
            for agent in self.possible_agents
        }

        # Observation spaces also need update since action_mask size changed
        for agent in self.possible_agents:
            self._observation_spaces[agent]["action_mask"] = spaces.Box(
                low=0, high=1, shape=(self.placement_action_space_size,), dtype=np.int8
            )

        # Initialize the bot that will play the rest of the game
        self.bot = AlphaBetaPlayer(None)  # Color is set dynamically

    def _get_placement_action_index(self, action: Action) -> Optional[int]:
        norm = normalize_action(action)
        try:
            return self.placement_actions.index((norm.action_type, norm.value))
        except ValueError:
            return None

    def _get_action_from_placement_index(
        self, index: int, playable_actions: List[Action]
    ) -> Action:
        action_type, value = self.placement_actions[index]
        for action in playable_actions:
            norm = normalize_action(action)
            if norm.action_type == action_type and norm.value == value:
                return action
        raise ValueError(
            f"Action index {index} ({action_type}, {value}) not found in playable_actions"
        )

    def _get_action_mask(self, legal_moves: List[Action]) -> np.ndarray:
        mask = np.zeros(self.placement_action_space_size, dtype=np.int8)
        for action in legal_moves:
            idx = self._get_placement_action_index(action)
            if idx is not None:
                mask[idx] = 1
        return mask

    def observe(self, agent):
        """Returns the observation for the specified agent with corrected action mask."""
        obs_dict = super().observe(agent)

        # The parent observe() calls its own _get_action_mask which uses the full ACTIONS_ARRAY.
        # We need to re-generate it for our reduced action space.
        legal_moves = (
            self.game.playable_actions
            if self.agent_map.get(self.game.state.current_color()) == agent
            else []
        )
        obs_dict["action_mask"] = self._get_action_mask(legal_moves)
        return obs_dict

    def _is_placement_phase(self) -> bool:
        """Returns True if the game is still in the initial placement phase."""
        # In Catan, the placement phase ends when all players have placed exactly 2 initial settlements.
        # Since settlements can be upgraded to cities later in the game, we must count BOTH
        # settlements and cities to prevent the environment from thinking we reverted to the placement phase.
        for player in self.game.state.players:
            settlements = len(
                self.game.state.buildings_by_color[player.color].get(SETTLEMENT, [])
            )
            cities = len(self.game.state.buildings_by_color[player.color].get(CITY, []))

            if (settlements + cities) < 2:
                return True
        return False

    def _is_agent_turn(self) -> bool:
        """Determines if the current action should be taken by an RL agent."""
        if not self._is_placement_phase():
            return False

        current_color = self.game.state.current_color()
        playable = self.game.playable_actions
        if not playable:
            return False

        # If it's a settlement action, it's always the agent's turn during placement phase
        # unless we want the bot to do it (but this env is for placement).
        if any(a.action_type == ActionType.BUILD_SETTLEMENT for a in playable):
            return True

        # If it's a road action:
        if any(a.action_type == ActionType.BUILD_ROAD for a in playable):
            return self.include_roads_in_action_space or not self.auto_play_roads

        return False

    def _step_bot(self):
        """Executes actions using the AlphaBeta bot until it's an agent's turn or game ends."""
        while self.game.winning_color() is None and not self._is_agent_turn():
            current_color = self.game.state.current_color()
            self.bot.color = current_color

            playable = self.game.playable_actions
            if not playable:
                break

            action = self.bot.decide(self.game, playable)
            self.game.execute(action)

            # Check for game over
            winning_color = self.game.winning_color()
            if winning_color is not None:
                self.game.state.is_game_over = True
                break

        # After bot finishes, update rewards and terminations
        self._update_state_post_step()

    def _update_state_post_step(self):
        """Syncs the PettingZoo state with the internal Catanatron game state."""
        winning_color = self.game.winning_color()
        # Force check for VPs if catanatron is lazy
        if winning_color is None:
            from catanatron.state_functions import get_actual_victory_points

            for color in self.game.state.colors:
                if get_actual_victory_points(self.game.state, color) >= self.vps_to_win:
                    winning_color = color
                    self.game.state.is_game_over = True
                    break

        is_terminated = winning_color is not None
        is_truncated = self.game.state.num_turns >= 1000  # TURNS_LIMIT

        if is_terminated:
            winner_agent = self.agent_map[winning_color]
            for agent in self.possible_agents:
                self.terminations[agent] = True
                self.rewards[agent] = 1 if agent == winner_agent else -1
        elif is_truncated:
            for agent in self.possible_agents:
                self.truncations[agent] = True
                self.rewards[agent] = 0

        if not (is_terminated or is_truncated):
            curr_color = self.game.state.current_color()
            if curr_color in self.agent_map:
                self.agent_selection = self.agent_map[curr_color]
            else:
                # This might happen if it's some intermediate state, but AlphaBeta should handle it
                pass
        else:
            # Game is over
            self.agents = []  # PettingZoo way to signal end for all

        for agent in self.possible_agents:
            self._cumulative_rewards[agent] += self.rewards.get(agent, 0)

    def step(self, action_index):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action_index)
            return

        current_agent = self.agent_selection
        playable = self.game.playable_actions

        # Execute agent action
        catan_action = self._get_action_from_placement_index(action_index, playable)
        self.game.execute(catan_action)

        self.rewards = {agent: 0 for agent in self.possible_agents}

        # Let bot play until it's an agent turn again
        self._step_bot()

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # If the first turn is NOT an agent turn, let the bot play
        if not self._is_agent_turn():
            self._step_bot()

        return self.observe(self.agent_selection), {}


def env(**kwargs):
    """Factory function for creating the AEC environment."""
    env_instance = CatanPlacementAECEnv(**kwargs)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


def parallel_env(**kwargs):
    """Factory function for creating the parallel API version of the environment."""
    from pettingzoo.utils import aec_to_parallel

    aec_env_instance = env(**kwargs)
    parallel_env_instance = aec_to_parallel(aec_env_instance)
    return parallel_env_instance
