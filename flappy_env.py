import pygame
import random
import numpy as np

class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 400, 600
        self.pipe_gap = 150
        self.pipe_width = 60
        self.gravity = 0.5
        self.jump = -8
        self.pipe_speed = 3
        self.reset()

    def reset(self):
        self.bird_x = 50
        self.bird_y = 250
        self.bird_speed = 0
        self.score = 0
        self.pipes = [self.create_pipe()]
        self.done = False
        return self.get_state()

    def create_pipe(self):
        top_height = random.randint(100, 400)
        return {'x': self.WIDTH, 'top': top_height, 'bottom': top_height + self.pipe_gap, 'scored': False}

    def get_state(self):
        pipe = self.pipes[0]
        return np.array([
            self.bird_y / self.HEIGHT,
            self.bird_speed / 10.0,
            (pipe['x'] - self.bird_x) / self.WIDTH,
            pipe['top'] / self.HEIGHT,
            pipe['bottom'] / self.HEIGHT
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        if action == 1:
            self.bird_speed = self.jump

        self.bird_speed += self.gravity
        self.bird_y += self.bird_speed

        for pipe in self.pipes:
            pipe['x'] -= self.pipe_speed

        if self.pipes[-1]['x'] < 200:
            self.pipes.append(self.create_pipe())

        if self.pipes[0]['x'] < -self.pipe_width:
            self.pipes.pop(0)

        pipe = self.pipes[0]
        if (self.bird_y < pipe['top'] or self.bird_y > pipe['bottom']) and \
           (pipe['x'] < self.bird_x + 34 < pipe['x'] + self.pipe_width):
            self.done = True
            return self.get_state(), -100, True

        if self.bird_y <= 0 or self.bird_y >= self.HEIGHT:
            self.done = True
            return self.get_state(), -100, True

        reward = 1
        if pipe['x'] + self.pipe_width < self.bird_x and not pipe['scored']:
            self.score += 1
            pipe['scored'] = True
            reward = 10

        return self.get_state(), reward, False
