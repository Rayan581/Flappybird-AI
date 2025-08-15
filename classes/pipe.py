import pygame
import random

GAP = 200


class Pipe:
    def __init__(self):
        self.x = 1000
        self.top_y = random.randint(100, 500 - GAP)
        self.bottom_y = self.top_y + GAP
        self.width = 100
        self.color = (0, 100, 0)
        self.passed = False

    def draw(self, screen):
        top_rect = pygame.Rect(self.x, 0, self.width, self.top_y)
        bottom_rect = pygame.Rect(self.x, self.bottom_y, self.width, 600)

        pygame.draw.rect(screen, self.color, top_rect)
        pygame.draw.rect(screen, self.color, bottom_rect)

    def update(self, dt):
        self.x -= int(300 * dt)

    def is_off_screen(self):
        return self.x + self.width <= 0
