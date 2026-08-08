import os
import math
import pygame
from catanatron.models.enums import (
    CITY, 
    RESOURCES, 
    SHEEP, 
    WOOD, 
    WHEAT, 
    ORE, 
    BRICK
)
from catanatron.models.player import Color
from catanatron.state_functions import (
    get_visible_victory_points,
    get_actual_victory_points,
    get_player_freqdeck,
    get_played_dev_cards,
    get_dev_cards_in_hand,
    get_longest_road_color,
    get_largest_army,
)
from catanatron.models.decks import freqdeck_count
from .constants import HEX_SIZE, RECTANGLE_WIDTH

class CatanRenderMixin:
    def _init_pygame(self):
        if self._pygame_initialized:
            return
        if self.render_mode == "rgb_array" and "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = os.environ.get("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.display.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Catanatron")
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self._pygame_clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 18)
        self.font_bold = pygame.font.Font(None, 20)
        self.pygame_colors = {
            Color.RED: pygame.Color(228, 6, 6),
            Color.BLUE: pygame.Color(0, 128, 255),
            Color.ORANGE: pygame.Color(255, 128, 0),
            Color.WHITE: pygame.Color(230, 230, 230),
            "background": pygame.Color(245, 245, 245),
            "text": pygame.Color(30, 30, 30),
            "text_light": pygame.Color(100, 100, 100),
            "text_red": pygame.Color(228, 6, 6),
        }
        self.tile_colors = {
            SHEEP: pygame.Color(144, 238, 144),
            WOOD: pygame.Color(34, 139, 34),
            WHEAT: pygame.Color(255, 255, 0),
            ORE: pygame.Color(169, 169, 169),
            BRICK: pygame.Color(255, 140, 0),
            None: pygame.Color(245, 222, 179),
        }
        self._pygame_initialized = True

    def _draw_text(self, text, pos, color, center=False, bold=False):
        if isinstance(color, tuple) and not isinstance(color, pygame.Color):
            color_val = pygame.Color(*color)
        else:
            color_val = color
        font = self.font_bold if bold else self.font
        if font is None:
            font = pygame.font.SysFont(None, 18)
        text_surface = font.render(str(text), True, color_val)
        if center:
            text_rect = text_surface.get_rect(center=pos)
            self.screen.blit(text_surface, text_rect)
        else:
            self.screen.blit(text_surface, pos)

    def render(self):
        if self.render_mode is None: return
        if not self._pygame_initialized: self._init_pygame()
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
        self._render_frame()
        if self.render_mode == "human":
            pygame.display.flip()
            self._pygame_clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def _render_frame(self):
        if self.game is None or not self._pygame_initialized: return
        state = self.game.state
        board = state.board
        self.screen.fill(self.pygame_colors["background"])
        for tile_id, (x, y) in self.tile_coords.items():
            tile = board.map.tiles_by_id.get(tile_id)
            if tile is None:
                for coord, map_tile in getattr(board.map, "land_tiles", {}).items():
                    if map_tile is not None and map_tile.id == tile_id:
                        tile = map_tile
                        break
            if tile is None: continue
            color = self.tile_colors.get(getattr(tile, "resource", None), pygame.Color(200, 200, 200))
            self._draw_hexagon(x, y, HEX_SIZE - RECTANGLE_WIDTH, color)
        for tile_id, (x, y) in self.number_coords.items():
            tile = board.map.tiles_by_id.get(tile_id)
            if tile is None:
                for coord, map_tile in board.map.land_tiles.items():
                    if map_tile is not None and map_tile.id == tile_id:
                        tile = map_tile
                        break
            if tile is not None and tile.resource is not None:
                number = tile.number
                text = str(number)
                color = self.pygame_colors["text_red"] if number in [6, 8] else self.pygame_colors["text"]
                self._draw_text(text, (x, y), color, center=True, bold=True)
        for port_id, (x, y) in self.port_coords.items():
            pygame.draw.circle(self.screen, (200, 200, 255), (int(x), int(y)), 10)
            port = board.map.ports_by_id.get(port_id)
            if port is None: continue
            text = port.resource if port.resource else "3:1"
            self._draw_text(text, (x, y), self.pygame_colors["text"], center=True, bold=True)
        for edge_tuple, color in board.roads.items():
            pygame_color = self.pygame_colors.get(color, pygame.Color(0, 0, 0))
            edge_key = tuple(sorted(edge_tuple))
            if edge_key in self.edge_coords:
                p1_x, p1_y = self.edge_coords[edge_key][0]
                p2_x, p2_y = self.edge_coords[edge_key][1]
                dx, dy = p2_x - p1_x, p2_y - p1_y
                length = math.hypot(dx, dy)
                if length == 0: continue
                half_width = RECTANGLE_WIDTH / 2
                px, py = (dy / length) * half_width, (-dx / length) * half_width
                rectangle_points = [(p1_x + px, p1_y + py), (p2_x + px, p2_y + py), (p2_x - px, p2_y - py), (p1_x - px, p1_y - py)]
                pygame.draw.polygon(self.screen, pygame_color, rectangle_points)
                border_half_width = (RECTANGLE_WIDTH + 4) / 2
                bpx, bpy = (dy / length) * border_half_width, (-dx / length) * border_half_width
                border_points = [(p1_x + bpx, p1_y + bpy), (p2_x + bpx, p2_y + bpy), (p2_x - bpx, p2_y - bpy), (p1_x - bpx, p1_y - bpy)]
                pygame.draw.polygon(self.screen, pygame.Color(0, 0, 0), border_points, 2)
        for node_id, (color, building_type) in board.buildings.items():
            pygame_color = self.pygame_colors.get(color, pygame.Color(0, 0, 0))
            if node_id not in self.node_coords: continue
            x, y = self.node_coords[node_id]
            if building_type == CITY:
                width, height = 24, 16
                rect = (x - width // 2, y - height // 2, width, height)
                pygame.draw.rect(self.screen, (0, 0, 0), rect)
                pygame.draw.rect(self.screen, pygame_color, rect, width=3)
            else:
                size = 24
                rect = (x - size // 2, y - size // 2, size, size)
                pygame.draw.rect(self.screen, pygame_color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, width=2)
        robber_coord = board.robber_coordinate
        robber_tile = None
        for t_id, tile in board.map.tiles_by_id.items():
            found = False
            for coord, map_tile in board.map.land_tiles.items():
                if map_tile.id == tile.id and coord == robber_coord:
                    robber_tile = map_tile
                    found = True
                    break
            if found: break
        if robber_tile and robber_tile.id in self.tile_coords:
            x, y = self.tile_coords[robber_tile.id]
            size = 30
            pygame.draw.rect(self.screen, pygame.Color(128, 128, 128), (x - size // 2, y - size // 2, size, size))
            pygame.draw.rect(self.screen, pygame.Color(0, 0, 0), (x - size // 2, y - size // 2, size, size), width=2)
        x_info = self.info_panel_x_start + 15
        y_info = 20
        self._draw_text(f"Turn: {state.num_turns}", (x_info, y_info), self.pygame_colors["text"], bold=True)
        y_info += 25
        current_color = state.current_color()
        if current_color:
            current_agent = self.agent_map.get(current_color, "N/A")
            self._draw_text(f"Current Player:", (x_info, y_info), self.pygame_colors["text_light"])
            y_info += 20
            self._draw_text(f"{current_agent}", (x_info + 10, y_info), self.pygame_colors.get(current_color, self.pygame_colors["text"]), bold=True)
            y_info += 30
        last_roll = state.last_roll
        roll_text = f"Roll: {sum(last_roll)} ({last_roll[0]}+{last_roll[1]})" if last_roll else "Roll: -"
        self._draw_text(roll_text, (x_info, y_info), self.pygame_colors["text"])
        y_info += 35
        for agent in self.agents:
            color = self.color_map.get(agent)
            p_x, p_y = x_info, y_info
            pygame.draw.rect(self.screen, self.pygame_colors[color], (p_x, p_y + 3, 15, 15))
            self._draw_text(f"{agent}", (p_x + 22, p_y), self.pygame_colors["text"], bold=True)
            p_y += 25
            self._draw_text(f"VISIBLE VP: {get_visible_victory_points(state, color)} | ACTUAL VP: {get_actual_victory_points(state, color)}", (p_x, p_y), self.pygame_colors["text"], bold=True)
            p_y += 25
            res_freq = get_player_freqdeck(state, color)
            for i, count in enumerate(res_freq):
                col, row = i % 3, i // 3
                self._draw_text(f"{RESOURCES[i][:3]}: {count}", (p_x + col * 60, p_y + row * 18), self.pygame_colors["text_light"])
            p_y += (len(RESOURCES) // 3 + 1) * 18
            dev_count = get_dev_cards_in_hand(state, color)
            knights = get_played_dev_cards(state, color, "KNIGHT")
            self._draw_text(f"Dev Cards: {dev_count}", (p_x, p_y), self.pygame_colors["text_light"])
            self._draw_text(f"Knights Played: {knights}", (p_x + 120, p_y), self.pygame_colors["text_light"])
            p_y += 18
            stats = []
            if get_longest_road_color(state) == color: stats.append("Longest Road")
            if get_largest_army(state)[0] == color: stats.append("Largest Army")
            if stats:
                self._draw_text(f"{', '.join(stats)}", (p_x, p_y), self.pygame_colors["text_light"], bold=True)
                p_y += 18
            y_info = p_y + 15
            pygame.draw.line(self.screen, (220, 220, 220), (x_info, y_info - 8), (self.screen_width - 15, y_info - 8), 1)
        y_info = max(y_info, self.screen_height - 150)
        self._draw_text("Bank:", (x_info, y_info), self.pygame_colors["text"], bold=True)
        y_info += 22
        for i, res in enumerate(RESOURCES):
            self._draw_text(f"{res}: {freqdeck_count(state.resource_freqdeck, res)}", (x_info + (i % 2) * 100, y_info + (i // 2) * 20), self.pygame_colors["text_light"])
        y_info += (len(RESOURCES) // 2 + 1) * 20 + 5
        self._draw_text(f"Dev Cards Left: {len(state.development_listdeck)}", (x_info, y_info), self.pygame_colors["text_light"])

    def _draw_hexagon(self, x, y, size, color):
        angles = [-90 + 60 * i for i in range(6)]
        points = [(int(x + size * math.cos(math.radians(a))), int(y + size * math.sin(math.radians(a)))) for a in angles]
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (0, 0, 0), points, 2)

    def _draw_rainbow_square(self, x, y, size):
        colors = [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (75, 0, 130), (148, 0, 211)]
        stripe_h = size // len(colors)
        for i, col in enumerate(colors):
            pygame.draw.rect(self.screen, col, (x, y + i * stripe_h, size, stripe_h))
        pygame.draw.rect(self.screen, (0, 0, 0), (x, y, size, size), 2)

    def close(self):
        if self._pygame_initialized:
            pygame.display.quit()
            pygame.quit()
            self._pygame_initialized = False
