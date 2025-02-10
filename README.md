# Pathfinding Algorithm Visualizer

This is a **Python-based pathfinding visualiser** using ``pygame``. It allows users to visualise how different search algorithms (Dijkstra and A*) solve a grid-based labyrinth.

## Features
- **Interactive Grid**: Left click to set the start, end, and obstacle (wall) nodes.
- **Multiple Algorithms**:
  - Press `1` to run **Dijkstra's Algorithm**.
  - Press `2` to run **A* Search Algorithm**.
- **Path Reset Option**: Press `0` to enter editing mode and alter the walls, start, and end nodes.

## How to Run
### Prerequisites
- Python 3.x
- ``pygame``

### Installation
1. Clone the repository or download the script.
2. Install Pygame if not already installed:
   ```sh
   pip install pygame
   ```
3. Run the script:
   ```sh
   python pathfinding_visualiser.py
   ```

## Controls
| Action              | Input           |
|---------------------|----------------|
| Set Start Node     | Left Click      |
| Set End Node       | Left Click (after setting start) |
| Place Obstacles    | Left Click on empty cells (after setting start and end) |
| Remove Obstacles   | Right Click on obstacles |
| Run Dijkstra       | Press `1`       |
| Run A* Search      | Press `2`       |
| Clear Algorithm's Path    | Press `0`       |
| Exit              | Close window or press `Esc` |
## Nodes
| Colour              | Node           |
|---------------------|----------------|
| White     | Empty Node      |
| Yellow       | Start Node |
| Red       | End Node |
| Black       | Obstacle Node |
| Blue       | Node Explored by Algorithm |
| Green       | Final Path |
## Algorithms Implemented
### Dijkstra's Algorithm
A **uniform-cost search** that finds the shortest path from start to end by exploring all possible routes in an optimal way.

### A* Search Algorithm
An **optimised pathfinding algorithm** that uses a heuristic function to prioritize paths that are likely to lead to the goal faster.

## Future Improvements
- Add more algorithms like BFS and DFS.
- Implement diagonal movement.
- Add more grid customisations.
- Enhance UI and animations.

## License
This project is licensed under the MIT License.

