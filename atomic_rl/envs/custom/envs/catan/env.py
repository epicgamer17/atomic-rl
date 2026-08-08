import functools
from typing import Dict, List, Optional, Union, Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player
from catanatron.models.map import build_map
from catanatron.models.enums import RESOURCES
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.board_tensor_features import (
    create_board_tensor, 
    get_channels, 
    is_graph_feature, 
    get_spatial_dims, 
    SpatialEncoding
)
from catanatron.state_functions import (
    player_num_resource_cards, 
    get_actual_victory_points
)

from .geometry import (
    NODES_COORDINATES, 
    EDGES_COORDINATES, 
    TILES_COORDINATES, 
    NUMBERS_COORDINATES, 
    compute_port_coordinates
)
from .actions import ACTION_SPACE_SIZE, to_action_space, from_action_space
from .rendering import CatanRenderMixin

HIGH = 19 * 5

class CatanAECEnv(AECEnv, CatanRenderMixin):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "catanatron_v1",
        "is_parallelizable": True,
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode=None,
        num_players=2,
        map_type="BASE",
        vps_to_win=10,
        representation="image",
        invalid_action_reward=-1,
        auto_play_single_action: bool = False,
        bandit_mode: bool = False,
        spatial_encoding: SpatialEncoding = "axial",
        include_validity_mask: bool = True,
        include_last_roll: bool = True,
        include_game_phase: bool = True,
        include_bank_state: bool = True,
        include_road_distance: bool = True,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.screen_width = 900
        self.screen_height = 700
        self._pygame_initialized = False
        self._pygame_clock = None
        self.screen = None
        self.font = None
        self.font_bold = None
        self.pygame_colors = {}
        self.tile_colors = {}
        self.node_coords = NODES_COORDINATES
        self.edge_coords = EDGES_COORDINATES
        self.tile_coords = TILES_COORDINATES
        self.number_coords = NUMBERS_COORDINATES
        self.board_pos = (0, 0)
        self.info_panel_x_start = 630
        self.map_type = map_type
        self.vps_to_win = vps_to_win
        self.representation = representation
        self.invalid_action_reward = invalid_action_reward
        self.spatial_encoding = spatial_encoding
        self.include_validity_mask = include_validity_mask
        self.include_last_roll = include_last_roll
        self.include_game_phase = include_game_phase
        self.include_bank_state = include_bank_state
        self.include_road_distance = include_road_distance
        assert self.representation in ["mixed", "image", "vector"]
        assert 2 <= num_players <= 4, "Catan must be played with 2 to 4 players"
        self.auto_play_single_action = bool(auto_play_single_action)
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = self.possible_agents[:]
        self.agent_selection = self.possible_agents[0]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.color_map = {agent: list(Color)[i] for i, agent in enumerate(self.possible_agents)}
        self.agent_map = {color: agent for agent, color in self.color_map.items()}
        self.catan_players = [Player(color) for agent, color in self.color_map.items()]
        self._action_spaces = {agent: spaces.Discrete(ACTION_SPACE_SIZE) for agent in self.possible_agents}
        self.features = get_feature_ordering(len(self.possible_agents), self.map_type)
        self.numeric_features = [f for f in self.features if not is_graph_feature(f)]
        spatial_width, spatial_height = get_spatial_dims(self.spatial_encoding)
        if self.representation == "mixed":
            channels = get_channels(len(self.possible_agents), include_validity_mask=self.include_validity_mask, include_last_roll=self.include_last_roll, include_game_phase=self.include_game_phase, include_bank_state=self.include_bank_state, include_road_distance=self.include_road_distance)
            board_tensor_space = spaces.Box(low=0, high=1, shape=(channels, spatial_width, spatial_height), dtype=np.float32)
            numeric_space = spaces.Box(low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=np.float32)
            core_obs_space = spaces.Dict({"board": board_tensor_space, "numeric": numeric_space})
        elif self.representation == "image":
            board_channels = get_channels(len(self.possible_agents), include_validity_mask=self.include_validity_mask, include_last_roll=self.include_last_roll, include_game_phase=self.include_game_phase, include_bank_state=self.include_bank_state, include_road_distance=self.include_road_distance)
            total_channels = board_channels + len(self.numeric_features)
            core_obs_space = spaces.Box(low=0, high=HIGH, shape=(total_channels, spatial_width, spatial_height), dtype=np.float32)
        else:
            core_obs_space = spaces.Box(low=0, high=HIGH, shape=(len(self.features),), dtype=np.float32)
        self._observation_spaces = {agent: spaces.Dict({"observation": core_obs_space, "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8)}) for agent in self.possible_agents}
        self.game: Game = None
        self.invalid_actions_count = {}
        self.max_invalid_actions = 10
        self.bandit_mode = bandit_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent): return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): return self._action_spaces[agent]

    def observe(self, agent):
        agent_color = self.color_map[agent]
        core_observation = self._get_core_observation(agent_color)
        legal_moves = self.game.playable_actions if self.agent_map.get(self.game.state.current_color()) == agent else []
        action_mask = self._get_action_mask(legal_moves)
        return {"observation": core_observation, "action_mask": action_mask}

    def _auto_advance(self):
        if not self.auto_play_single_action or self.game is None: return
        while True:
            winning_color = self._check_winner()
            if winning_color is not None: break
            legal_moves = list(self.game.playable_actions)
            if len(legal_moves) != 1: break
            self.game.execute(legal_moves[0])
            winning_color = self._check_winner()
            is_terminated = winning_color is not None
            current_agent = self.agent_map.get(self.game.state.current_color())
            is_truncated = self.game.state.num_turns >= TURNS_LIMIT or (current_agent and self.invalid_actions_count.get(current_agent, 0) > self.max_invalid_actions)
            if is_terminated:
                winner_agent = self.agent_map[winning_color]
                for agent in self.agents:
                    self.rewards[agent] = 1 if agent == winner_agent else -1
                    self.terminations[agent] = True
                self.agent_selection = self._agent_selector.next()
                break
            elif is_truncated:
                for agent in self.agents: self.truncations[agent] = True
                self.agent_selection = self._agent_selector.next()
                break
            else:
                self.agent_selection = self.agent_map[self.game.state.current_color()]

    def _check_winner(self):
        if self.bandit_mode:
            curr = self.game.state.current_color()
            if curr and player_num_resource_cards(self.game.state, curr) >= self.vps_to_win:
                return curr
        winning_color = self.game.winning_color()
        if winning_color is None:
            for color in self.game.state.colors:
                if get_actual_victory_points(self.game.state, color) >= self.vps_to_win:
                    winning_color = color
                    self.game.state.is_game_over = True
                    self.game.playable_actions = []
                    break
        return winning_color

    def reset(self, seed=None, options=None):
        catan_map = build_map(self.map_type)
        for player in self.catan_players: player.reset_state()
        self.game = Game(players=self.catan_players, seed=seed, catan_map=catan_map, vps_to_win=self.vps_to_win, restrict_dice_to_board=(self.map_type == "MINI"))
        self.port_coords = compute_port_coordinates(self.game.state.board.map, self.node_coords, offset_dist=0)
        self.invalid_actions_count = {agent: 0 for agent in self.possible_agents}
        shuffled_colors = self.game.state.colors
        self.color_map = {agent: shuffled_colors[i] for i, agent in enumerate(self.possible_agents)}
        self.agent_map = {color: agent for agent, color in self.color_map.items()}
        self.catan_players = [next(p for p in self.game.state.players if p.color == self.color_map[agent]) for agent in self.possible_agents]
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_map[self.game.state.current_color()]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        if self.auto_play_single_action: self._auto_advance()
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards.get(agent, 0)
            self.infos[agent] = {"turn": self.game.state.num_turns, "player": self.possible_agents.index(agent), "legal_moves": (list(self._get_valid_action_indices()) if agent == self.agent_selection else [])}
        if self.render_mode == "human": self.render()

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        current_agent = self.agent_selection
        is_action_valid = action in self._get_valid_action_indices()
        if not is_action_valid:
            self.invalid_actions_count[current_agent] += 1
            self.rewards = {agent: 0 for agent in self.agents}
            self.rewards[current_agent] = self.invalid_action_reward
        else:
            catan_action = from_action_space(action, self.game.playable_actions)
            self.game.execute(catan_action)
            self.rewards = {agent: 0 for agent in self.agents}
        winning_color = self._check_winner()
        is_terminated = winning_color is not None
        is_truncated = self.game.state.num_turns >= TURNS_LIMIT or self.invalid_actions_count[current_agent] > self.max_invalid_actions
        if is_terminated:
            winner_agent = self.agent_map[winning_color]
            for agent in self.agents:
                self.rewards[agent] = 1 if agent == winner_agent else -1
                self.terminations[agent] = True
        elif is_truncated:
            for agent in self.agents:
                self.truncations[agent] = True
                self.rewards[agent] = 0
        if not (is_terminated or is_truncated):
            self.agent_selection = self.agent_map[self.game.state.current_color()]
            if self.auto_play_single_action: self._auto_advance()
        else:
            self.agent_selection = self._agent_selector.next()
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
            self.infos[agent] = {"turn": self.game.state.num_turns, "player": self.possible_agents.index(agent), "legal_moves": (list(self._get_valid_action_indices()) if agent == self.agent_selection else [])}
        if self.render_mode == "human": self.render()

    def _get_action_mask(self, legal_moves):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        if not legal_moves: return mask
        legal_indices = [to_action_space(action) for action in legal_moves]
        mask[legal_indices] = 1
        return mask

    def _get_valid_action_indices(self):
        return {to_action_space(action) for action in self.game.playable_actions}

    def _get_core_observation(self, agent_color: Color) -> Union[np.ndarray, dict]:
        sample = create_sample(self.game, agent_color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(self.game, agent_color, channels_first=True, spatial_encoding=self.spatial_encoding, include_validity_mask=self.include_validity_mask, include_last_roll=self.include_last_roll, include_game_phase=self.include_game_phase, include_bank_state=self.include_bank_state, include_road_distance=self.include_road_distance).astype(np.float32)
            numeric = np.array([float(sample[i]) for i in self.numeric_features], dtype=np.float32)
            return {"board": board_tensor, "numeric": numeric}
        if self.representation == "image":
            board_tensor = create_board_tensor(self.game, agent_color, channels_first=True, spatial_encoding=self.spatial_encoding, include_validity_mask=self.include_validity_mask, include_last_roll=self.include_last_roll, include_game_phase=self.include_game_phase, include_bank_state=self.include_bank_state, include_road_distance=self.include_road_distance).astype(np.float32)
            numeric = np.array([float(sample[i]) for i in self.numeric_features], dtype=np.float32)
            _, h, w = board_tensor.shape
            numeric_planes = np.tile(numeric[:, None, None], (1, h, w))
            return np.concatenate((board_tensor, numeric_planes), axis=0)
        else:
            return np.array([float(sample[i]) for i in self.features], dtype=np.float32)
