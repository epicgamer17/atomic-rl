import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import copy


class MississippiMarblesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, players=6):
        self.players = players
        self.max_players = 16  # some number so that the state space can stay the same regardless of player count (can we find a way to remove this? maybe only showing top 3 players score or neighbours scores etc?)
        self.window_size = 512  # The size of the PyGame window

        # Observation is a list.
        # The first 6 elements are the dice values rolled (0 for not rolled)
        # The next element is the number of dice remaining (for when dice are passed for piggybacking)
        # The next element is the score collected this turn by the player
        # the next element is the score recieved by the player from the previous player (piggybacking)
        # Last max_players elements are the scores of the players (starting from current players turn)
        self.observation_space = spaces.Discrete(6 + 1 + 1 + 1 + self.max_players)
        # self.observation_space = spaces.Box(
        #     [
        #         1,
        #         1,
        #         1,
        #         1,
        #         1,
        #         1,
        #         0,
        #         -20000,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #     ],
        #     [
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #         20000,
        #     ],
        #     (25,),
        #     np.int8,
        # )

        # OPTION 1 FOR ACTION SPACE (ROLLING HAPPENS AUTOMATICALLY, ALL POSSIBLE OPTIONS FOR SCORING + 1 FOR PASSING DICE)
        # We have 94 Actions
        # 1 and 5 Combos
        # Keep One 1, Keep Two 1s
        # Keep One 5, Keep Two 5s
        # Keep One 1 and One 5, Keep Two 1s and One 5
        # Keep One 1 and One 5, Keep One 1 and Two 5s

        # Three of a Kind
        # Keep Three 1s, Keep Three 2s, Keep Three 3s, Keep Three 4s, Keep Three 5s, Keep Three 6s

        # Three of a Kind Combos
        # Keep Three 1s and One 5, Keep Three 1s and Two 5s, Keep Three 1s and Three 5s # FORCED COULD REMOVE
        # Keep Three 2s and One 5, Keep Three 2s and Two 5s, Keep Three 2s and Three 5s # FORCED COULD REMOVE
        # Keep Three 3s and One 5, Keep Three 3s and Two 5s, Keep Three 3s and Three 5s # FORCED COULD REMOVE
        # Keep Three 4s and One 5, Keep Three 4s and Two 5s, Keep Three 4s and Three 5s # FORCED COULD REMOVE
        # Keep Three 6s and One 5, Keep Three 6s and Two 5s, Keep Three 6s and Three 5s # FORCED COULD REMOVE

        # Keep Three 2s and One 1, Keep Three 2s and Two 1s, Keep Three 2s and Three 1s # FORCED COULD REMOVE
        # Keep Three 3s and One 1, Keep Three 3s and Two 1s, Keep Three 3s and Three 1s # FORCED COULD REMOVE
        # Keep Three 4s and One 1, Keep Three 4s and Two 1s, Keep Three 4s and Three 1s # FORCED COULD REMOVE
        # Keep Three 5s and One 1, Keep Three 5s and Two 1s, Keep Three 5s and Three 1s # FORCED COULD REMOVE
        # Keep Three 6s and One 1, Keep Three 6s and Two 1s, Keep Three 6s and Three 1s # FORCED COULD REMOVE

        # Four of a Kind
        # Keep Four 1s, Keep Four 2s, Keep Four 3s, Keep Four 4s, Keep Four 5s, Keep Four 6s

        # Four of a Kind Combos
        # Keep Four 1s and One 5, Keep Four 1s and Two 5s # FORCED COULD REMOVE
        # Keep Four 2s and One 5, Keep Four 2s and Two 5s # FORCED COULD REMOVE
        # Keep Four 3s and One 5, Keep Four 3s and Two 5s # FORCED COULD REMOVE
        # Keep Four 4s and One 5, Keep Four 4s and Two 5s # FORCED COULD REMOVE
        # Keep Four 6s and One 5, Keep Four 6s and Two 5s # FORCED COULD REMOVE

        # Keep Four 2s and One 1, Keep Four 2s and Two 1s # FORCED COULD REMOVE
        # Keep Four 3s and One 1, Keep Four 3s and Two 1s # FORCED COULD REMOVE
        # Keep Four 4s and One 1, Keep Four 4s and Two 1s # FORCED COULD REMOVE
        # Keep Four 5s and One 1, Keep Four 5s and Two 1s # FORCED COULD REMOVE
        # Keep Four 6s and One 1, Keep Four 6s and Two 1s # FORCED COULD REMOVE

        # Five of a Kind
        # Keep Five 1s, Keep Five 2s, Keep Five 3s, Keep Five 4s, Keep Five 5s, Keep Five 6s

        # Five of a Kind Combos
        # Keep Five 1s and One 5 # FORCED COULD REMOVE
        # Keep Five 2s and One 5 # FORCED COULD REMOVE
        # Keep Five 3s and One 5 # FORCED COULD REMOVE
        # Keep Five 4s and One 5 # FORCED COULD REMOVE
        # Keep Five 6s and One 5 # FORCED COULD REMOVE

        # Keep Five 2s and One 1 # FORCED COULD REMOVE
        # Keep Five 3s and One 1 # FORCED COULD REMOVE
        # Keep Five 4s and One 1 # FORCED COULD REMOVE
        # Keep Five 5s and One 1 # FORCED COULD REMOVE
        # Keep Five 6s and One 1 # FORCED COULD REMOVE

        # Six of a Kind
        # Keep Six 1s, Keep Six 2s, Keep Six 3s, Keep Six 4s, Keep Six 5s, Keep Six 6s # FORCED COULD REMOVE

        # Straight
        # Keep the Straight # FORCED COULD REMOVE

        # Pass Dice # MUST HAVE ROLLED FIRST
        # Don't Piggyback

        # OPTION 2 FOR ACTION SPACE (ROLLING AUTOMATIC MINIMUM SCORING COMBOS ONLY)
        # We have 10 Actions
        # Keep a 1 # Can't do 3 times for 1 roll
        # Keep a 5 # Can't do 3 times for 1 roll
        # Keep three of a kind
        # Keep four of a kind
        # Keep five of a kind
        # Keep six of a kind # FORCED COULD REMOVE
        # Keep a straight # FORCED COULD REMOVE
        # Roll
        # Pass Dice # MUST HAVE ROLLED FIRST
        # Don't Piggyback

        # OPTION 3 (option 1 remove forced 6 die combos)
        # We have 58 Actions

        # OPTION 4 (option 2 remove forced 6 die combos)
        # We have 8 Actions
        self.action_space = spaces.Discrete(10)

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

    def _get_obs(self):
        return np.concatenate(
            (
                self._dice,
                np.array(
                    [
                        self._dice_remaining,
                        self._score_this_turn / 50,
                        self._score_piggyback / 50,
                    ]
                ),
                self._score[self._current_player - 1 : self.max_players] / 50,
                self._score[0 : self._current_player - 1] / 50,
            )
        )

    def _get_info(self):
        return {"legal_moves": self._legal_moves, "player": self._current_player}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set a blank board
        self._dice = np.zeros(6, dtype=np.int8)
        self._dice_remaining = 6
        self._score_this_turn = 0
        self._score_piggyback = 0
        self._score = np.zeros(self.max_players, dtype=np.int16)
        self._current_player = 0
        self._can_pass = False

        self._dice = np.random.randint(1, 7, 6)
        self._can_pass = True
        # rolled = True
        # Check for four 2s
        if np.sum(self._dice == 2) == 4:
            self._score_this_turn -= self._score[0]
            self._score[0] = 0

        # Reset legal moves
        self._legal_moves = [1]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        if action < 0 or action > self.action_space.n:
            raise ValueError(
                "Action must be between 0 and {}".format(self.action_space.n)
            )
        if action not in self._legal_moves:
            # could return a negative reward
            raise ValueError(
                "Illegal move {} Legal Moves {}".format(action, self._legal_moves)
            )

        # USING OPTION 2
        # reward = 0
        # rolled = False
        collected = False
        busted = False
        if action == 0:
            print(
                "Player {} Passed and Scored {} Points".format(
                    self._current_player, self._score_this_turn
                )
            )
            self._score[self._current_player] += self._score_this_turn
            self._current_player += 1
            self._current_player = self._current_player % self.players
            self._can_pass = False
            self._score_piggyback = self._score_this_turn
            # reward = self._score_this_turn
        if action == 1:
            if self._legal_moves == [1, 2]:
                print("Piggybacked off {} Points".format(self._score_piggyback))
            # Roll Dice
            self._dice = np.random.randint(1, 7, 6)
            self._dice[self._dice_remaining :] = 0
            self._can_pass = True
            # rolled = True
            # Check for four 2s
            if np.sum(self._dice == 2) == 4:
                self._score_this_turn -= self._score[0]
                self._score[0] = 0
        if action == 2:  # Don't Piggyback
            self._dice = np.random.randint(1, 7, 6)
            self._can_pass = True
            # rolled = True
            # Check for four 2s
            if np.sum(self._dice == 2) == 4:
                self._score_this_turn -= self._score[0]
                self._score[0] = 0
        if action == 3:
            # Keep a 1
            self._score_this_turn += 100
            self._dice_remaining -= 1
            self._dice[np.where(self._dice == 1)[0][0]] = 0
            collected = True
        if action == 4:
            # Keep a 5
            self._score_this_turn += 50
            self._dice_remaining -= 1
            self._dice[np.where(self._dice == 5)[0][0]] = 0
            collected = True
        if action == 5:
            # Keep three of a kind
            for i in range(1, 7):
                if np.sum(self._dice == i) == 3:
                    if i == 1:
                        self._score_this_turn += 500
                    else:
                        self._score_this_turn += i * 100
                    self._dice_remaining -= 3
                    self._dice[self._dice == i] = 0
                    collected = True
                    break
        if action == 6:
            # Keep four of a kind
            for i in range(1, 7):
                if np.sum(self._dice == i) == 4:
                    if i == 2:
                        raise ValueError("Four 2s is a bust! This is illegal!")
                    self._score_this_turn += 1000
                    self._dice_remaining -= 4
                    self._dice[self._dice == i] = 0
                    collected = True
                    break
        if action == 7:
            # Keep five of a kind
            for i in range(1, 7):
                if np.sum(self._dice == i) == 5:
                    self._score_this_turn += 3000
                    self._dice_remaining -= 5
                    self._dice[self._dice == i] = 0
                    collected = True
                    break
        if action == 8:
            # Keep six of a kind
            for i in range(1, 7):
                if np.sum(self._dice == i) == 6:
                    self._score_this_turn += 6000
                    self._dice_remaining -= 6
                    self._dice[self._dice == i] = 0
                    collected = True
                    break
        if action == 9:
            # Keep a straight
            self._score_this_turn += 2000
            collected = True
            self._dice_remaining = 0

        # Update Legal Moves
        if action != 0:
            if collected:
                if self._score[self._current_player] + self._score_this_turn >= 700:
                    self._legal_moves = [0, 1]
                else:
                    self._legal_moves = [1]
                if self._dice_remaining == 0:
                    print("Collected all dice!")
                    self._dice_remaining = 6
            else:
                self._legal_moves = []

            if 1 in self._dice and np.sum(self._dice == 1) < 3:
                self._legal_moves.append(3)
            if 5 in self._dice and np.sum(self._dice == 5) < 3:
                self._legal_moves.append(4)
            for i in range(1, 7):
                if np.sum(self._dice == i) == 3:
                    self._legal_moves.append(5)
                if np.sum(self._dice == i) == 4:
                    if i != 2:
                        self._legal_moves.append(6)
                    else:
                        busted = True
                if np.sum(self._dice == i) == 5:
                    self._legal_moves.append(7)
                if np.sum(self._dice == i) == 6:
                    self._legal_moves.append(8)
            if np.all(np.sort(self._dice) == np.array([1, 2, 3, 4, 5, 6])):
                self._legal_moves.append(9)

            if len(self._legal_moves) == 0:
                busted = True

            if busted:
                print("Busted!")
                self._current_player += 1
                self._current_player = self._current_player % self.players
                self._dice_remaining = 6
                self._score_this_turn = 0
                self._score_piggyback = 0
                self._can_pass = False
                self._legal_moves = [1]
                # reward = self._score_this_turn

        else:
            print("Passed!")
            if self._score[self._current_player] + self._score_this_turn >= 700:
                self._legal_moves = [1, 2]  # Piggyback or Don't Piggyback
            else:
                self._legal_moves = [2]  # Don't Piggyback

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame(action)

        if self._score[self._current_player] >= 11000:
            if (
                np.max(self._score) == self._score[self._current_player]
                and self._current_player == 0
            ):
                return (
                    observation,
                    1,
                    True,
                    False,
                    info,
                )  # some way of telling agent which model won so it can properly do rewards in replay buffer when storing games
            elif np.max(self._score) == self._score[self._current_player]:
                return (
                    observation,
                    1,
                    False,
                    False,
                    info,
                )
            else:
                return (
                    observation,
                    0,
                    False,
                    False,
                    info,
                )  # some way of telling agent which model won so it can properly do rewards in replay buffer when storing games

        if np.max(self._score) >= 11000 and self._current_player == 0:
            return (
                observation,
                0,
                True,
                False,
                info,
            )

        return observation, 0, False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self, action=None):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # First we draw the dice
        pix_square_size = self.window_size / 6

        for i in range(6):
            if self._dice[i] == 1:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 2,
                        pix_square_size // 2,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
            if self._dice[i] == 2:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 4,
                        pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + 3 * pix_square_size // 4,
                        3 * pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
            if self._dice[i] == 3:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 4,
                        pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 2,
                        pix_square_size // 2,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + 3 * pix_square_size // 4,
                        3 * pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
            if self._dice[i] == 4:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 4,
                        pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + 3 * pix_square_size // 4,
                        pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 4,
                        3 * pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + 3 * pix_square_size // 4,
                        3 * pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
            if self._dice[i] == 5:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 4,
                        pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + 3 * pix_square_size // 4,
                        pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 2,
                        pix_square_size // 2,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 4,
                        3 * pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + 3 * pix_square_size // 4,
                        3 * pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
            if self._dice[i] == 6:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 4,
                        pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + 3 * pix_square_size // 4,
                        pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 4,
                        pix_square_size // 2,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + 3 * pix_square_size // 4,
                        pix_square_size // 2,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + pix_square_size // 4,
                        3 * pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (
                        i * pix_square_size + 3 * pix_square_size // 4,
                        3 * pix_square_size // 4,
                    ),
                    pix_square_size // 10,
                    width=3,
                )

        # Next we draw the action
        action_meaning = {
            0: "Pass",
            1: "Roll or Piggyback {}".format(self._score_piggyback),
            2: "Don't Piggyback",
            3: "Keep a 1",
            4: "Keep a 5",
            5: "Keep Three of a Kind",
            6: "Keep Four of a Kind",
            7: "Keep Five of a Kind",
            8: "Keep Six of a Kind",
            9: "Keep a Straight",
        }
        font = pygame.font.Font(None, 36)
        text = font.render(
            "Action {}".format(
                action_meaning[action] if action is not None else "Start"
            ),
            True,
            (0, 0, 0),
        )
        canvas.blit(text, (0, self.window_size - 108))

        # Next we draw the current player
        font = pygame.font.Font(None, 36)
        text = font.render(
            "Player: {} | Score This Turn: {}".format(
                self._current_player, self._score_this_turn
            ),
            True,
            (0, 0, 0),
        )
        canvas.blit(text, (0, self.window_size - 72))

        # Finally we draw the score
        font = pygame.font.Font(None, 36)
        text = font.render("Score: {}".format(self._score), True, (0, 0, 0))
        canvas.blit(text, (0, self.window_size - 36))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
