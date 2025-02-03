import pygame
import sys
from queue import PriorityQueue

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
ROWS = 50
CELL_SIZE = WIDTH // ROWS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
GREEN = (0, 255, 0)   # Start point
RED = (255, 0, 0)     # End point
BLUE = (0, 0, 255)    # Visited nodes
YELLOW = (255, 255, 0)  # Final path

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pathfinding Algorithm Visualizer")

# Create a grid to store cell states
grid = [[WHITE for _ in range(ROWS)] for _ in range(ROWS)]

# Variables to track start and end points
start_pos = None
end_pos = None

# Function to draw the grid with colors
def draw_grid():
    for row in range(ROWS):
        for col in range(ROWS):
            color = grid[row][col]
            pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, GREY, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

# Convert mouse position to grid coordinates
def get_clicked_pos(pos):
    x, y = pos
    row = y // CELL_SIZE
    col = x // CELL_SIZE
    return row, col

# Get neighbors for a cell (up, down, left, right)
def get_neighbors(pos):
    row, col = pos
    neighbors = []

    if row > 0 and grid[row - 1][col] != BLACK:  # Up
        neighbors.append((row - 1, col))
    if row < ROWS - 1 and grid[row + 1][col] != BLACK:  # Down
        neighbors.append((row + 1, col))
    if col > 0 and grid[row][col - 1] != BLACK:  # Left
        neighbors.append((row, col - 1))
    if col < ROWS - 1 and grid[row][col + 1] != BLACK:  # Right
        neighbors.append((row, col + 1))

    return neighbors

# Dijkstra's Algorithm (using BFS behavior for equal weights)
def dijkstra(start, end):
    queue = PriorityQueue()
    queue.put((0, start))
    came_from = {}
    visited = set()

    while not queue.empty():
        _, current = queue.get()

        if current == end:
            reconstruct_path(came_from, end)
            return True

        if current in visited:
            continue

        visited.add(current)

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                queue.put((1, neighbor))  # All moves have the same cost
                came_from[neighbor] = current
                if grid[neighbor[0]][neighbor[1]] != RED:
                    grid[neighbor[0]][neighbor[1]] = BLUE  # Mark as visited

        draw_grid()
        pygame.display.update()

    return False  # No path found

# Function to reconstruct and highlight the shortest path
def reconstruct_path(came_from, current):
    while current in came_from:
        current = came_from[current]
        if grid[current[0]][current[1]] != GREEN:
            grid[current[0]][current[1]] = YELLOW  # Highlight path
        draw_grid()
        pygame.display.update()

# Main loop
def main():
    global start_pos, end_pos
    running = True

    while running:
        screen.fill(WHITE)
        draw_grid()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle left mouse click
            if pygame.mouse.get_pressed()[0]:  # Left click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos)

                if not start_pos:  # First click sets start
                    start_pos = (row, col)
                    grid[row][col] = GREEN
                elif not end_pos and (row, col) != start_pos:  # Second click sets end
                    end_pos = (row, col)
                    grid[row][col] = RED
                elif (row, col) != start_pos and (row, col) != end_pos:  # Other clicks set walls
                    grid[row][col] = BLACK

            # Handle right mouse click to reset
            if pygame.mouse.get_pressed()[2]:  # Right click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos)
                grid[row][col] = WHITE

                if (row, col) == start_pos:
                    start_pos = None
                elif (row, col) == end_pos:
                    end_pos = None

            # Start Dijkstra's Algorithm when spacebar is pressed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start_pos and end_pos:
                    dijkstra(start_pos, end_pos)

        pygame.display.update()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
