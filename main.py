from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable
import pygame
import sys
import time
import random
from queue import PriorityQueue, Queue, LifoQueue
import math


WIDTH, HEIGHT = 1500, 750
ROWS = 40
MARGIN_RIGHT = 260
GRID_W = WIDTH - MARGIN_RIGHT
CELL = GRID_W // ROWS
GRID_H = CELL * ROWS
FPS = 120

WHITE = (245, 245, 245)
BLACK = (25, 25, 25)
BG = (18, 18, 20)
GRID_LINE = (48, 48, 52)
START = (255, 195, 0)
END = (255, 77, 77)
VISITED = (90, 154, 255)
FRONTIER = (140, 200, 255)
PATH = (0, 230, 118)
WALL = (30, 30, 30)
WEIGHT = (110, 70, 160)
HUD_TEXT = (230, 230, 235)
HUD_DIM = (150, 150, 158)
BAD = (255, 120, 120)
OK = (160, 255, 160)

DIR4 = [(1,0), (-1,0), (0,1), (0,-1)]
DIAG = [(1,1), (1,-1), (-1,1), (-1,-1)]


@dataclass
class Cell:
    r: int
    c: int
    wall: bool = False
    weight = 1
    visited: bool = False
    in_frontier: bool = False
    in_path: bool = False
    is_start: bool = False
    is_end: bool = False

    def base_color(self):
        if self.wall:
            return WALL
        if self.weight > 1:
            return WEIGHT
        return WHITE

class Grid:
    def __init__(self, rows):
        self.rows = rows
        self.cells: List[List[Cell]] = [[Cell(r, c) for c in range(rows)] for r in range(rows)]
        self.start: Optional[Tuple[int,int]] = None
        self.end: Optional[Tuple[int,int]] = None
        self.allow_diagonals = False

    def reset(self):
        for row in self.cells:
            for cell in row:
                cell.wall = False
                cell.weight = 1
                cell.visited = False
                cell.in_frontier = False
                cell.in_path = False
                cell.is_start = False
                cell.is_end = False
        self.start = None
        self.end = None

    def clear_grid(self):
        for row in self.cells:
            for cell in row:
                cell.visited = False
                cell.in_frontier = False
                cell.in_path = False

    def within_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.rows

    def neighbors(self, r, c):
        steps = DIR4 + (DIAG if self.allow_diagonals else [])
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if not self.within_bounds(nr, nc):
                continue
            ncell = self.cells[nr][nc]
            if ncell.wall:
                continue
            base = math.sqrt(2) if (dr != 0 and dc != 0) else 1.0
            yield nr, nc, base * ncell.weight


def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def octile(a, b):
    dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
    F = math.sqrt(2) - 1
    return (F * min(dx, dy) + max(dx, dy))


@dataclass
class RunStats:
    name: str
    found: bool
    expanded: int
    path_len: int
    path_cost: float
    runtime_ms: float
    peak_frontier: int
    eff_branching: Optional[float]

def reconstruct(came_from, grid, end):
    path = []
    cur = end
    cost = 0.0
    while cur in came_from:
        prev = came_from[cur]
        pr, pc = prev
        for nr, nc, move_cost in grid.neighbors(pr, pc):
            if (nr, nc) == cur:
                cost += move_cost
                break
        path.append(cur)
        cur = prev
    path.reverse()
    return path, cost

def approx_effective_branching(expanded, depth):
    if depth <= 1 or expanded <= 1:
        return None
    low, high = 1.01, 10.0
    for _ in range(40):
        mid = (low + high) / 2
        denom = (mid - 1.0)
        if denom <= 1e-9:
            low = mid
            continue
        approxN = (mid**(depth+1) - 1.0) / denom
        if approxN < expanded:
            low = mid
        else:
            high = mid
    return round((low + high)/2, 3)

def algo_bfs(grid):
    name = "BFS (unweighted optimal)"
    if not grid.start or not grid.end:
        return
    start, goal = grid.start, grid.end
    q = Queue()
    q.put(start)
    came = {}
    visited = set([start])
    expanded = 0
    peak_frontier = 1

    t0 = time.perf_counter()
    while not q.empty():
        peak_frontier = max(peak_frontier, q.qsize())
        r, c = q.get()
        expanded += 1
        cell = grid.cells[r][c]
        if not cell.is_start:
            cell.visited = True

        if (r, c) == goal:
            path, cost = reconstruct(came, grid, goal)
            t1 = time.perf_counter()
            for pr, pc in path:
                if not grid.cells[pr][pc].is_end:
                    grid.cells[pr][pc].in_path = True
                yield
            stats = RunStats(
                name=name, found=True, expanded=expanded, path_len=len(path),
                path_cost=cost, runtime_ms=(t1 - t0) * 1000, peak_frontier=peak_frontier,
                eff_branching=approx_effective_branching(expanded, len(path))
            )
            yield stats
            return

        for nr, nc, _w in grid.neighbors(r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                came[(nr, nc)] = (r, c)
                grid.cells[nr][nc].in_frontier = True
                q.put((nr, nc))
        yield

    t1 = time.perf_counter()
    stats = RunStats(name=name, found=False, expanded=expanded, path_len=0,
                    path_cost=0.0, runtime_ms=(t1 - t0)*1000, peak_frontier=peak_frontier,
                    eff_branching=None)
    yield stats

def algo_dfs(grid: Grid):
    name = "DFS (uninformed, not optimal)"
    if not grid.start or not grid.end:
        return
    start, goal = grid.start, grid.end
    st = LifoQueue()
    st.put(start)
    came = {}
    visited = set([start])
    expanded = 0
    peak_frontier = 1

    t0 = time.perf_counter()
    while not st.empty():
        peak_frontier = max(peak_frontier, st.qsize())
        r, c = st.get()
        expanded += 1
        cell = grid.cells[r][c]
        if not cell.is_start:
            cell.visited = True

        if (r, c) == goal:
            path, cost = reconstruct(came, grid, goal)
            t1 = time.perf_counter()
            for pr, pc in path:
                if not grid.cells[pr][pc].is_end:
                    grid.cells[pr][pc].in_path = True
                yield
            stats = RunStats(
                name=name, found=True, expanded=expanded, path_len=len(path),
                path_cost=cost, runtime_ms=(t1 - t0) * 1000, peak_frontier=peak_frontier,
                eff_branching=approx_effective_branching(expanded, len(path))
            )
            yield stats
            return

        neighs = list(grid.neighbors(r, c))
        for nr, nc, _w in reversed(neighs):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                came[(nr, nc)] = (r, c)
                grid.cells[nr][nc].in_frontier = True
                st.put((nr, nc))
        yield

    t1 = time.perf_counter()
    stats = RunStats(name=name, found=False, expanded=expanded, path_len=0,
                     path_cost=0.0, runtime_ms=(t1 - t0)*1000, peak_frontier=peak_frontier,
                     eff_branching=None)
    yield stats

def algo_dijkstra(grid: Grid):
    name = "Dijkstra (uniform-cost search)"
    if not grid.start or not grid.end:
        return
    start, goal = grid.start, grid.end
    pq = PriorityQueue()
    pq.put((0.0, start))
    dist = {start: 0.0}
    came = {}
    visited = set()
    expanded = 0
    peak_frontier = 1

    t0 = time.perf_counter()
    while not pq.empty():
        peak_frontier = max(peak_frontier, pq.qsize())
        d, (r, c) = pq.get()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        expanded += 1
        cell = grid.cells[r][c]
        if not cell.is_start:
            cell.visited = True

        if (r, c) == goal:
            path, cost = reconstruct(came, grid, goal)
            t1 = time.perf_counter()
            for pr, pc in path:
                if not grid.cells[pr][pc].is_end:
                    grid.cells[pr][pc].in_path = True
                yield
            stats = RunStats(
                name=name, found=True, expanded=expanded, path_len=len(path),
                path_cost=cost, runtime_ms=(t1 - t0)*1000, peak_frontier=peak_frontier,
                eff_branching=approx_effective_branching(expanded, len(path))
            )
            yield stats
            return

        for nr, nc, w in grid.neighbors(r, c):
            nd = d + w
            if nd < dist.get((nr, nc), float('inf')):
                dist[(nr, nc)] = nd
                came[(nr, nc)] = (r, c)
                grid.cells[nr][nc].in_frontier = True
                pq.put((nd, (nr, nc)))
        yield

    t1 = time.perf_counter()
    stats = RunStats(name=name, found=False, expanded=expanded, path_len=0,
                     path_cost=0.0, runtime_ms=(t1 - t0)*1000, peak_frontier=peak_frontier,
                     eff_branching=None)
    yield stats

def algo_astar(grid: Grid):
    name = "A* (octile heuristic)"
    if not grid.start or not grid.end:
        return
    start, goal = grid.start, grid.end
    pq = PriorityQueue()
    g = {start: 0.0}
    f = {start: octile(start, goal)}
    pq.put((f[start], start))
    came = {}
    closed = set()
    expanded = 0
    peak_frontier = 1

    t0 = time.perf_counter()
    while not pq.empty():
        peak_frontier = max(peak_frontier, pq.qsize())
        _f, cur = pq.get()
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        r, c = cur
        cell = grid.cells[r][c]
        if not cell.is_start:
            cell.visited = True

        if cur == goal:
            path, cost = reconstruct(came, grid, goal)
            t1 = time.perf_counter()
            for pr, pc in path:
                if not grid.cells[pr][pc].is_end:
                    grid.cells[pr][pc].in_path = True
                yield
            stats = RunStats(
                name=name, found=True, expanded=expanded, path_len=len(path),
                path_cost=cost, runtime_ms=(t1 - t0)*1000, peak_frontier=peak_frontier,
                eff_branching=approx_effective_branching(expanded, len(path))
            )
            yield stats
            return

        for nr, nc, w in grid.neighbors(r, c):
            tentative = g[cur] + w
            if tentative < g.get((nr, nc), float('inf')):
                came[(nr, nc)] = cur
                g[(nr, nc)] = tentative
                h = octile((nr, nc), goal)
                f[(nr, nc)] = tentative + h
                grid.cells[nr][nc].in_frontier = True
                pq.put((f[(nr, nc)], (nr, nc)))
        yield

    t1 = time.perf_counter()
    stats = RunStats(name=name, found=False, expanded=expanded, path_len=0,
                     path_cost=0.0, runtime_ms=(t1 - t0)*1000, peak_frontier=peak_frontier,
                     eff_branching=None)
    yield stats

def algo_greedy(grid: Grid):
    name = "Greedy Best-First (octile h)"
    if not grid.start or not grid.end:
        return
    start, goal = grid.start, grid.end
    pq = PriorityQueue()
    pq.put((octile(start, goal), start))
    came = {}
    visited = set()
    expanded = 0
    peak_frontier = 1

    t0 = time.perf_counter()
    while not pq.empty():
        peak_frontier = max(peak_frontier, pq.qsize())
        _, cur = pq.get()
        if cur in visited:
            continue
        visited.add(cur)
        expanded += 1
        r, c = cur
        cell = grid.cells[r][c]
        if not cell.is_start:
            cell.visited = True

        if cur == goal:
            path, cost = reconstruct(came, grid, goal)
            t1 = time.perf_counter()
            for pr, pc in path:
                if not grid.cells[pr][pc].is_end:
                    grid.cells[pr][pc].in_path = True
                yield
            stats = RunStats(
                name=name, found=True, expanded=expanded, path_len=len(path),
                path_cost=cost, runtime_ms=(t1 - t0)*1000, peak_frontier=peak_frontier,
                eff_branching=approx_effective_branching(expanded, len(path))
            )
            yield stats
            return

        for nr, nc, _ in grid.neighbors(r, c):
            if (nr, nc) not in visited:
                came[(nr, nc)] = cur
                grid.cells[nr][nc].in_frontier = True
                pq.put((octile((nr, nc), goal), (nr, nc)))
        yield

    t1 = time.perf_counter()
    stats = RunStats(name=name, found=False, expanded=expanded, path_len=0,
                     path_cost=0.0, runtime_ms=(t1 - t0)*1000, peak_frontier=peak_frontier,
                     eff_branching=None)
    yield stats

ALGORITHMS = {
    pygame.K_1: ("BFS", algo_bfs),
    pygame.K_2: ("DFS", algo_dfs),
    pygame.K_3: ("Dijkstra", algo_dijkstra),
    pygame.K_4: ("A*", algo_astar),
    pygame.K_5: ("Greedy", algo_greedy),
}


def draw_grid(surface, grid, font):
    pygame.draw.rect(surface, BG, (0, 0, GRID_W, GRID_H))
    for r in range(grid.rows):
        for c in range(grid.rows):
            cell = grid.cells[r][c]
            color = cell.base_color()
            if cell.in_frontier:
                color = FRONTIER
            if cell.visited:
                color = VISITED
            if cell.in_path:
                color = PATH
            if cell.is_start:
                color = START
            if cell.is_end:
                color = END
            pygame.draw.rect(surface, color, (c*CELL, r*CELL, CELL, CELL))

    for i in range(grid.rows+1):
        pygame.draw.line(surface, GRID_LINE, (0, i*CELL), (GRID_W, i*CELL), 1)
        pygame.draw.line(surface, GRID_LINE, (i*CELL, 0), (i*CELL, GRID_H), 1)

def draw_hud(surface, font, small, grid, msg, last_stats, speed_ms):
    x0 = GRID_W + 10
    pygame.draw.rect(surface, (28,28,30), (GRID_W, 0, MARGIN_RIGHT, HEIGHT))
    def line(text, y, color=HUD_TEXT):
        surface.blit(font.render(text, True, color), (x0, y))
    def sline(text, y, color=HUD_DIM):
        surface.blit(small.render(text, True, color), (x0, y))

    y = 12
    line("Pathfinding Lab", y); y += 30
    sline("Left: wall  |  Shift+Left: start/end  |  Right: erase", y); y += 24
    sline("W: weight  |  D: diagonals  |  +/-: speed", y); y += 24
    sline("1 BFS  2 DFS  3 Dijkstra  4 A*  5 Greedy", y); y += 24
    sline("M: random maze  |  C: clear  |  R: reset", y); y += 24
    sline("ESC: quit", y); y += 10
    pygame.draw.line(surface, GRID_LINE, (GRID_W+6, y), (WIDTH-6, y), 1); y += 10

    line(f"Diagonals: {'ON' if grid.allow_diagonals else 'OFF'}", y); y += 26
    line(f"Anim speed: {speed_ms} ms/step", y); y += 26
    if grid.start:
        line(f"Start: {grid.start}", y); y += 26
    else:
        line("Start: (not set)", y, BAD); y += 26
    if grid.end:
        line(f"End:   {grid.end}", y); y += 26
    else:
        line("End:   (not set)", y, BAD); y += 26

    pygame.draw.line(surface, GRID_LINE, (GRID_W+6, y), (WIDTH-6, y), 1); y += 10

    if last_stats:
        color = OK if last_stats.found else BAD
        line(last_stats.name, y, color); y += 28
        sline(f"Found: {last_stats.found}", y, color); y += 22
        sline(f"Expanded: {last_stats.expanded}", y); y += 20
        sline(f"Path length: {last_stats.path_len}", y); y += 20
        sline(f"Path cost: {last_stats.path_cost:.3f}", y); y += 20
        sline(f"Runtime: {last_stats.runtime_ms:.2f} ms", y); y += 20
        sline(f"Peak frontier: {last_stats.peak_frontier}", y); y += 20
        if last_stats.eff_branching is not None:
            sline(f"Eff. branching ~ {last_stats.eff_branching}", y); y += 20
        y += 6
    else:
        sline("Run an algorithm to see stats.", y); y += 22

    pygame.draw.line(surface, GRID_LINE, (GRID_W+6, y), (WIDTH-6, y), 1); y += 10

    if msg:
        wrap = wrap_text(msg, small, WIDTH - (GRID_W + 20))
        for line_txt in wrap:
            surface.blit(small.render(line_txt, True, HUD_TEXT), (x0, y))
            y += 18

def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.size(test)[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def gen_maze(grid, density=0.3):
    for r in range(grid.rows):
        for c in range(grid.rows):
            cell = grid.cells[r][c]
            if not cell.is_start and not cell.is_end:
                cell.wall = (random.random() < density)
                cell.weight = 1
                cell.visited = cell.in_frontier = cell.in_path = False


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pathfinding Lab — BFS / DFS / Dijkstra / A* / Greedy")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    small = pygame.font.SysFont("consolas", 16)

    grid = Grid(ROWS)
    last_stats = None
    msg = "Draw walls, set start/end, then press an algorithm key."
    placing_state = 0
    algo_runner = None
    speed_ms = 8

    def cell_at_mouse():
        x, y = pygame.mouse.get_pos()
        if x >= GRID_W or y >= GRID_H:
            return None
        return (y // CELL, x // CELL)

    running = True
    while running:
        dt = clock.tick(FPS)
        if algo_runner is not None:
            try:
                item = next(algo_runner)
                if isinstance(item, RunStats):
                    last_stats = item
                    algo_runner = None
                    msg = f"{item.name}: {'Path found' if item.found else 'No path'} in {item.runtime_ms:.2f} ms."
                else:
                    pygame.time.delay(max(0, speed_ms))
            except StopIteration:
                algo_runner = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]):
                pos = cell_at_mouse()
                if pos:
                    r, c = pos
                    cell = grid.cells[r][c]
                    mods = pygame.key.get_mods()
                    shift_held = bool(mods & pygame.KMOD_SHIFT)
                    if pygame.mouse.get_pressed()[0]:
                        if shift_held:
                            if not grid.start:
                                for row in grid.cells:
                                    for ce in row:
                                        ce.is_start = False
                                grid.start = (r, c)
                                cell.is_start = True
                                cell.wall = False
                                msg = "Start set."
                            elif not grid.end and (r, c) != grid.start:
                                for row in grid.cells:
                                    for ce in row:
                                        ce.is_end = False
                                grid.end = (r, c)
                                cell.is_end = True
                                cell.wall = False
                                msg = "End set. Choose algorithm (1..5)."
                        else:
                            if not cell.is_start and not cell.is_end:
                                cell.wall = True
                                cell.weight = 1
                    elif pygame.mouse.get_pressed()[2]:
                        cell.wall = False
                        cell.weight = 1
                        cell.visited = cell.in_frontier = cell.in_path = False
                        if cell.is_start:
                            cell.is_start = False
                            grid.start = None
                        if cell.is_end:
                            cell.is_end = False
                            grid.end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_w:
                    pos = cell_at_mouse()
                    if pos:
                        r, c = pos
                        cell = grid.cells[r][c]
                        if not cell.wall and not cell.is_start and not cell.is_end:
                            cell.weight = 1 if cell.weight > 1 else 5

                if event.key == pygame.K_d:
                    grid.allow_diagonals = not grid.allow_diagonals
                    msg = f"Diagonals {'ENABLED' if grid.allow_diagonals else 'DISABLED'}."

                if event.key == pygame.K_c:
                    grid.clear_grid()
                    last_stats = None
                    msg = "Cleared visited/path."
                if event.key == pygame.K_r:
                    grid.reset()
                    last_stats = None
                    msg = "Grid reset."

                if event.key == pygame.K_m:
                    gen_maze(grid)
                    grid.clear_grid()
                    last_stats = None
                    msg = "Random maze generated. Set start and end."

                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed_ms = max(0, speed_ms - 2)
                if event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    speed_ms = min(200, speed_ms + 2)

                if event.key in ALGORITHMS:
                    name, fn = ALGORITHMS[event.key]
                    if not grid.start or not grid.end:
                        msg = "Set start and end before running."
                    else:
                        grid.clear_grid()
                        algo_runner = fn(grid)
                        last_stats = None
                        msg = f"Running {name}..."

        screen.fill(BG)
        draw_grid(screen, grid, font)
        draw_hud(screen, font, small, grid, msg, last_stats, speed_ms)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
