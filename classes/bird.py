import pygame
import random
import math
from .neural_network import NeuralNetwork

GRAVITY = 1200
JUMP_STRENGTH = 800


def circle_rect_collision(circle_x, circle_y, circle_radius, rect):
    # Find closest point on rect to the circle center
    closest_x = max(rect.left, min(circle_x, rect.right))
    closest_y = max(rect.top, min(circle_y, rect.bottom))

    # Distance from circle center to closest point
    dx = circle_x - closest_x
    dy = circle_y - closest_y

    # If distance <= radius, collision
    return (dx * dx + dy * dy) <= (circle_radius * circle_radius)


class Bird:

    def __init__(self, brain=None):
        self.x = 50
        self.y = 300
        self.color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        self.radius = 15
        self.velocity = 0
        self.flap = False
        self.alive = True
        self.survival_time = 0
        self.score = 0

        self.brain = NeuralNetwork(
            [8, 16, 12, 8, 2], ['relu', 'relu', 'relu', 'sigmoid'])
        if brain:
            self.brain = brain

    def update(self, dt, pipes):
        self.survival_time += dt

        upcoming_pipes = [p for p in pipes if not p.passed]
        upcoming_pipes.sort(key=lambda p: p.x)

        nearest_pipe = upcoming_pipes[0] if upcoming_pipes else None
        second_nearest_pipe = upcoming_pipes[1] if len(
            upcoming_pipes) > 1 else None

        inputs = [0] * 8  # Initialize with zeros

        if nearest_pipe:
            pipe_x = nearest_pipe.x + nearest_pipe.width // 2
            top_y = nearest_pipe.top_y
            bottom_y = nearest_pipe.bottom_y

            inputs[0] = (pipe_x - self.x) / 400
            inputs[1] = self.y / 600
            inputs[2] = self.velocity / 1000
            inputs[3] = top_y / 600
            inputs[4] = bottom_y / 600

        if second_nearest_pipe:
            second_pipe_x = second_nearest_pipe.x + second_nearest_pipe.width // 2
            second_top_y = second_nearest_pipe.top_y
            second_bottom_y = second_nearest_pipe.bottom_y

            inputs[5] = (second_pipe_x - self.x) / 400
            inputs[6] = second_top_y / 600
            inputs[7] = second_bottom_y / 600

        output = self.brain.forward(inputs)
        flap_strength = output[0]
        should_flap = output[1] > 0.5

        self.flap = should_flap

        if self.flap:
            jump_power = JUMP_STRENGTH * (0.5 + flap_strength * 0.5)
            self.velocity = -jump_power

        self.velocity += GRAVITY * dt

        self.y += self.velocity * dt

        if self.y < 0:
            self.y = 0
            self.velocity = 0
        elif self.y + self.radius > 600:
            self.y = 600
            self.velocity = 0

        # Detect passing pipe center and increment score
        for pipe in pipes:
            pipe_center = pipe.x + pipe.width // 2
            if not pipe.passed and self.x > pipe_center:
                self.score += 1
                pipe.passed = True

        if self.__detect_collision(pipes):
            self.alive = False
        if self.y == 600 or self.y == 0:
            self.alive = False

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

    def __detect_collision(self, pipes):
        nearby_pipes = [p for p in pipes if p.x -
                        50 < self.x < p.x + p.width + 50]
        for pipe in nearby_pipes:
            top_rect = pygame.Rect(pipe.x, 0, pipe.width, pipe.top_y)
            bottom_rect = pygame.Rect(pipe.x, pipe.bottom_y, pipe.width, 600)

            if circle_rect_collision(self.x, self.y, self.radius, top_rect) or circle_rect_collision(self.x, self.y, self.radius, bottom_rect):
                return True

        return False

    def fitness(self):
        base_fitness = self.score * 1000 + self.survival_time * 50

        # Encourage staying in middle area
        middle_bonus = 0
        if 150 < self.y < 450:  # Middle 50% of screen
            middle_bonus = self.survival_time * 10

        return max(1, base_fitness + middle_bonus)  # Ensure positive
