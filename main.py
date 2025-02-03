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





