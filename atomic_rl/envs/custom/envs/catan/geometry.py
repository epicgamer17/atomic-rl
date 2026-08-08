import math
from collections import OrderedDict
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    NUM_NODES,
    LandTile,
    PORT_DIRECTION_TO_NODEREFS,
)
from catanatron.models.board import get_edges
from .constants import HEX_SIZE, ORIGIN, BOARD_RADIUS, sqrt3

def axial_to_pixel(q, r, size=HEX_SIZE, origin=ORIGIN):
    x = size * sqrt3 * (q + r / 2.0)
    y = size * 3.0 / 2.0 * r
    ox, oy = origin
    return (ox + x, oy + y)

def hex_corners(cx, cy, size=HEX_SIZE):
    pts = []
    for i in range(6):
        angle_deg = -90 + 60 * i
        rad = math.radians(angle_deg)
        pts.append((cx + size * math.cos(rad), cy + size * math.sin(rad)))
    return pts

def _clockwise_corners_from_top(cx, cy, corners):
    corner_angles = []
    for p in corners:
        dx = p[0] - cx
        dy = p[1] - cy
        angle_deg = math.degrees(math.atan2(dy, dx))
        angle_cw = (angle_deg + 90) % 360
        corner_angles.append((p, angle_cw))
    corner_angles.sort(key=lambda t: t[1])
    best_idx = min(
        range(6),
        key=lambda i: min(
            abs(corner_angles[i][1] - 0.0), 360.0 - abs(corner_angles[i][1] - 0.0)
        ),
    )
    rotated = corner_angles[best_idx:] + corner_angles[:best_idx]
    return [p for p, _ in rotated]

tiles_axial = []
for q in range(-BOARD_RADIUS, BOARD_RADIUS + 1):
    for r in range(-BOARD_RADIUS, BOARD_RADIUS + 1):
        s = -q - r
        if -BOARD_RADIUS <= s <= BOARD_RADIUS:
            tiles_axial.append((q, r))

tiles_with_meta = []
for q, r in tiles_axial:
    px, py = axial_to_pixel(q, r)
    s = -q - r
    distance = max(abs(q), abs(r), abs(s))
    angle_deg = math.degrees(math.atan2(py - ORIGIN[1], px - ORIGIN[0]))
    angle_cw = (angle_deg + 360) % 360.0
    tiles_with_meta.append(((q, r), (px, py), distance, angle_cw))

tiles_with_meta.sort(key=lambda t: (t[2], t[3]))
tiles_centers = [((q, r), (px, py)) for (q, r), (px, py), _, _ in tiles_with_meta]

TILES_COORDINATES_ORDERED = OrderedDict()
for i, ((q, r), (x, y)) in enumerate(tiles_centers):
    TILES_COORDINATES_ORDERED[i] = (int(round(x)), int(round(y)))

node_map = {}
nodes = []
edges = set()
nid = 0

for tile_index in TILES_COORDINATES_ORDERED.keys():
    (q, r), (cx_f, cy_f) = tiles_centers[tile_index]
    raw_corners = hex_corners(cx_f, cy_f)
    corners = _clockwise_corners_from_top(cx_f, cy_f, raw_corners)
    corner_ids = []
    for x_f, y_f in corners:
        key = (int(round(x_f)), int(round(y_f)))
        if key not in node_map:
            node_map[key] = nid
            nodes.append((nid, key))
            nid += 1
        corner_ids.append(node_map[key])
    for i in range(6):
        a = corner_ids[i]
        b = corner_ids[(i + 1) % 6]
        edges.add(tuple(sorted((a, b))))

NODES_COORDINATES = dict(OrderedDict((nid, coord) for nid, coord in nodes))
sorted_edges = sorted(edges)
EDGES_COORDINATES = dict(OrderedDict(
    ((a, b), [NODES_COORDINATES[a], NODES_COORDINATES[b]]) for a, b in sorted_edges
))
NUMBERS_COORDINATES = {
    i: (int(round(x)), int(round(y + HEX_SIZE * 0.15)))
    for i, (x, y) in TILES_COORDINATES_ORDERED.items()
}
TILES_COORDINATES = dict(TILES_COORDINATES_ORDERED)

def compute_port_coordinates(catan_map, node_coordinates, offset_dist=25):
    port_coords = {}
    for port_id, port in catan_map.ports_by_id.items():
        a_ref, b_ref = PORT_DIRECTION_TO_NODEREFS[port.direction]
        a = port.nodes[a_ref]
        b = port.nodes[b_ref]
        if a not in node_coordinates or b not in node_coordinates:
            continue
        x1, y1 = node_coordinates[a]
        x2, y2 = node_coordinates[b]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = mx - ORIGIN[0], my - ORIGIN[1]
        dist = math.sqrt(dx**2 + dy**2)
        if dist == 0: dist = 1
        ux, uy = dx / dist, dy / dist
        px = mx + ux * offset_dist
        py = my + uy * offset_dist
        port_coords[port_id] = (px, py)
    return port_coords

BASE_TOPOLOGY = BASE_MAP_TEMPLATE.topology
TILE_COORDINATE_IDS = [x for x, y in BASE_TOPOLOGY.items() if y == LandTile]
