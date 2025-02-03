import tkinter as tk
import pygame
import sys

#Constants:
WIDTH, HEIGHT = 800, 800
ROWS = 50
CELL = WIDTH // ROWS

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (57, 255, 20)

#set up the game
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pathfinding Algorithm') #replace with some options thingy

def grid():
    for row in range(ROWS):
        for col in range(ROWS):
            pygame.draw.rect(screen, WHITE, (col * CELL, row * CELL, CELL, CELL))
            pygame.draw.rect(screen, GRAY, (col * CELL, row * CELL, CELL, CELL), 1)

def main():
    game = True
    while game:
        screen.fill(WHITE)
        grid()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game = False
        
        pygame.display.update()
    
    pygame.quit()
    sys.quit()

if __name__ == '__main__':
    main()




