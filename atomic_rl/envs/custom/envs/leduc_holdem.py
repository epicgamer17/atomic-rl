# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import pygame
# import copy
# import rlcard


# class LeducHoldemEnv(gym.Env):
#     metadata = {"render_modes": [], "render_fps": 1}

#     def __init__(self, render_mode=None, players=2, encode_player_turn=False):
#         self.game = rlcard.make("leduc-holdem")
#         self.encode_player_turn = encode_player_turn
#         self.observation_space = spaces.Box(
#             low=-10,
#             high=10,
#             shape=(37,) if self.encode_player_turn else (36,),
#             dtype=np.int8,
#         )
#         self.players = players

#         # We have 9 actions, corresponding to each cell
#         self.action_space = spaces.Discrete(4)

#         assert render_mode is None or render_mode in self.metadata["render_modes"]
#         self.render_mode = render_mode

#         """
#         If human-rendering is used, `self.window` will be a reference
#         to the window that we draw to. `self.clock` will be a clock that is used
#         to ensure that the environment is rendered at the correct framerate in
#         human-mode. They will remain `None` until human-mode is used for the
#         first time.
#         """
#         self.window = None
#         self.clock = None

#     # def _get_obs(self):
#     #     return self.game.get_state()

#     def _get_info(self):
#         return {
#             "legal_moves": copy.deepcopy(self._legal_moves),
#             "player": self._current_player,
#         }

#     def reset(self, seed=None, options=None):
#         # We need the following line to seed self.np_random
#         super().reset(seed=seed)

#         dict, self._rlcard_current_player = self.game.reset()
#         self._current_player = 0
#         self._player_diff = self._rlcard_current_player
#         # print(self._current_player)
#         self._rlcard_legal_moves = copy.deepcopy(dict["legal_actions"]).keys()
#         # print(legal_moves)
#         if 3 in self._rlcard_legal_moves:
#             del dict["legal_actions"][3]
#             legal_moves = dict["legal_actions"].keys()
#             self._legal_moves = list(legal_moves)
#             self._legal_moves.append(0)
#         else:
#             self._legal_moves = list(self._rlcard_legal_moves)

#         observation = dict["obs"]
#         if self.encode_player_turn:
#             observation = np.append(observation, [self._current_player])
#             observation = np.reshape(observation, (37,))
#         move_history = dict["action_record"]

#         info = self._get_info()

#         # if self.render_mode == "human":
#         #     self._render_frame()

#         return observation, info

#     def step(self, action):
#         if action < 0 or action > self.action_space.n - 1:
#             raise ValueError(
#                 "Action must be between 0 and {}".format(self.action_space.n - 1)
#             )
#         # if action not in self._legal_moves:
#         # raise ValueError(
#         #     "Illegal move {} Legal Moves {}".format(action, self._legal_moves)
#         # )

#         if action == 0:
#             call = False
#             for i in self._rlcard_legal_moves:
#                 call = i == 0 or call
#             if not call:
#                 action = 3
#         dict, self._rlcard_current_player = self.game.step(action)
#         self._current_player = (
#             self._rlcard_current_player - self._player_diff % self.players
#         )
#         # print(self._current_player)
#         self._rlcard_legal_moves = copy.deepcopy(dict["legal_actions"]).keys()

#         if 3 in self._rlcard_legal_moves:
#             del dict["legal_actions"][3]
#             legal_moves = dict["legal_actions"].keys()
#             self._legal_moves = list(legal_moves)
#             self._legal_moves.append(0)
#         else:
#             self._legal_moves = list(self._rlcard_legal_moves)
#         # print(self._legal_moves)
#         # print(self._rlcard_legal_moves)
#         # print(self._current_player)
#         # self._legal_moves = list(dict["legal_actions"].keys())
#         observation = dict["obs"]  # copy.deepcopy(dict["obs"])?
#         if self.encode_player_turn:
#             observation = np.append(observation, [self._current_player])
#             observation = np.reshape(observation, (37,))
#         move_history = dict["action_record"]

#         terminated = self.game.is_over()  # copy.deepcopy(self.game.is_over())?
#         rewards = (
#             self.game.get_payoffs() if terminated else [0] * self.players
#         )  # copy.deepcopy(self.game.get_payoffs()) if terminated else [0] * self.players
#         # print(rewards)
#         rewards = [
#             rewards[(i + self._player_diff) % self.players] for i in range(self.players)
#         ]
#         info = self._get_info()

#         if self.render_mode == "human":
#             self._render_frame()

#         return observation, rewards, terminated, False, info


from re import I
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import copy
import rlcard


class LeducHoldemEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(self, render_mode=None, players=2, encode_player_turn=False):
        self.game = rlcard.make("leduc-holdem")
        self.encode_player_turn = encode_player_turn
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(31,) if self.encode_player_turn else (30,),
            dtype=np.int8,
        )
        self.players = players

        # We have 9 actions, corresponding to each cell
        self.action_space = spaces.Discrete(3)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    # def _get_obs(self):
    #     return self.game.get_state()

    def _get_info(self):
        return {
            "legal_moves": copy.deepcopy(self._legal_moves),
            "player": self._current_player,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # print("Reset!")
        dict, self._rlcard_current_player = self.game.reset()
        # print(self.game.game.round_counter)
        self._current_player = 0
        self._player_diff = self._rlcard_current_player
        # print(self._current_player)
        self._rlcard_legal_moves = copy.deepcopy(dict["legal_actions"]).keys()

        if 3 in self._rlcard_legal_moves:
            del dict["legal_actions"][3]
            legal_moves = dict["legal_actions"].keys()
            self._legal_moves = list(legal_moves)
            self._legal_moves.append(0)
        else:
            self._legal_moves = list(self._rlcard_legal_moves)
        cards = dict["obs"]
        self._cards = cards[0:6]

        self._betting_history = np.zeros((2, 2, 3, 2))
        self._num_raises = 0
        self._prev_round_counter = 0
        observation = np.append(self._cards, self._betting_history)
        observation = np.reshape(observation, (30,))
        # print(observation)
        # observation = cards
        if self.encode_player_turn:
            observation = np.append(observation, [self._current_player])
            observation = np.reshape(observation, (31,))
        move_history = dict["action_record"]

        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):
        if action < 0 or action > self.action_space.n - 1:
            raise ValueError(
                "Action must be between 0 and {}".format(self.action_space.n - 1)
            )
        # if action not in self._legal_moves:
        # raise ValueError(
        #     "Illegal move {} Legal Moves {}".format(action, self._legal_moves)
        # )
        # print("action", action)
        if action == 0:
            call = False
            for i in self._rlcard_legal_moves:
                call = i == 0 or call
            if not call:
                action = 3
        if self.game.game.round_counter != self._prev_round_counter:
            self._num_raises = 0

        if action != 2:
            # print("current player", self._current_player)
            # print("round", self.game.game.round_counter)
            # print("number of raises", self._num_raises)
            self._betting_history[self._current_player][self.game.game.round_counter][
                self._num_raises
            ][action % 3] = 1

        dict, self._rlcard_current_player = self.game.step(action)

        if action == 1:
            self._num_raises += 1

        self._current_player = (
            self._rlcard_current_player - self._player_diff
        ) % self.players
        # print(self._current_player)
        self._rlcard_legal_moves = copy.deepcopy(dict["legal_actions"]).keys()

        if 3 in self._rlcard_legal_moves:
            del dict["legal_actions"][3]
            legal_moves = dict["legal_actions"].keys()
            self._legal_moves = list(legal_moves)
            self._legal_moves.append(0)
        else:
            self._legal_moves = list(self._rlcard_legal_moves)
        # print(self._legal_moves)

        cards = dict["obs"]
        self._cards = cards[0:6]

        observation = np.append(self._cards, self._betting_history)
        temp_betting_history = copy.deepcopy(self._betting_history)
        self._betting_history[0] = self._betting_history[1]
        self._betting_history[1] = temp_betting_history[0]
        observation = np.reshape(observation, (30,))
        # print(observation)
        # observation = cards
        if self.encode_player_turn:
            observation = np.append(observation, [self._current_player])
            observation = np.reshape(observation, (31,))
        move_history = dict["action_record"]

        terminated = self.game.is_over()  # copy.deepcopy(self.game.is_over())?
        # print(terminated)
        rewards = (
            (self.game.get_payoffs() * 2) if terminated else [0] * self.players
        )  # copy.deepcopy(self.game.get_payoffs()) if terminated else [0] * self.players
        # print(rewards)
        rewards = [
            rewards[(i + self._player_diff) % self.players] for i in range(self.players)
        ]
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, rewards, terminated, False, info
