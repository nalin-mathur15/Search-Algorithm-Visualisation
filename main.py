import tkinter as tk
import pygame
import sys
from queue import PriorityQueue

#Constants:
WIDTH, HEIGHT = 800, 800
ROWS = 50
CELL = WIDTH // ROWS

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
YELLOW = (255, 255, 0) # start
RED = (255, 0, 0) # end
BLUE = (0, 0, 255) #visited nodes
GREEN = (57, 255, 20) #final path

#set up the game
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pathfinding Algorithm') #replace with some options thingy

gridData = [[WHITE for _ in range(ROWS)] for _ in range(ROWS)]
start = None
end = None

def grid():
    for row in range(ROWS):
        for col in range(ROWS):
            colour = gridData[row][col]
            pygame.draw.rect(screen, colour, (col * CELL, row * CELL, CELL, CELL))
            pygame.draw.rect(screen, GRAY, (col * CELL, row * CELL, CELL, CELL), 1)

def getPosition(position):
    x, y = position
    row = int(y) // CELL
    col = int(x) // CELL
    return row, col

def neighbours(position):
    row, col = position
    neighbours = []

    if row > 0 and gridData[row - 1][col] != BLACK:
        neighbours.append((row - 1, col))
    if row < (ROWS - 1) and gridData[row + 1][col] != BLACK:
        neighbours.append((row + 1, col))
    if col > 0 and gridData[row][col - 1] != BLACK:
        neighbours.append((row, col - 1))
    if col < (ROWS - 1) and gridData[row][col + 1] != BLACK:
        neighbours.append((row, col + 1))
    
    return neighbours

def djikstra(start, end):
    queue = PriorityQueue()
    queue.put((0, start))
    cameFrom = {}
    visited = set()
    while not queue.empty():
        _, cur = queue.get()
        if cur == end:
            finalPath(cameFrom, end)
            return True
        if cur in visited:
            continue

        visited.add(cur)

        for neighbour in neighbours(cur):
            if neighbour not in visited:
                queue.put((1, neighbour))
                cameFrom[neighbour] = cur
                if gridData[neighbour[0]][neighbour[1]] != RED:
                    gridData[neighbour[0]][neighbour[1]] = BLUE
        grid()
        pygame.display.update()
    return False

def finalPath(cameFrom, cur):
    while cur in cameFrom:
        cur = cameFrom[cur]
        if gridData[cur[0]][cur[1]] != YELLOW:
            gridData[cur[0]][cur[1]] = GREEN
        grid()
        pygame.display.update()
    

def main():
    global start, end
    game = True
    while game:
        screen.fill(WHITE)
        grid()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game = False
        
            if pygame.mouse.get_pressed()[0]: #left click
                position = pygame.mouse.get_pos()
                row, col = getPosition(position)

                if not start:
                    start = (row, col)
                    gridData[row][col] = YELLOW
                elif not end and (row, col) != start:
                    end = (row, col)
                    gridData[row][col] = RED
                elif (row, col) != end and (row, col) != start:
                    gridData[row][col] = BLACK
            
            if pygame.mouse.get_pressed()[2]: #right click
                position = pygame.mouse.get_pos()
                row, col = getPosition(position)
                gridData[row][col] = WHITE

                if (row, col) == start:
                    start = None
                elif (row, col) == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    djikstra(start, end)
        
        pygame.display.update()
    
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
