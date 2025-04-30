import pygame
import torch
import numpy as np
from model import DQN
from flappy_env import FlappyBirdEnv

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(5, 2).to(device)
model.load_state_dict(torch.load("dqn_flappy.pt", map_location=device))
model.eval()

# Pygame hiển thị
pygame.init()
screen = pygame.display.set_mode((400, 600))
pygame.display.set_caption("Flappy Bird AI")
font = pygame.font.SysFont(None, 36)

env = FlappyBirdEnv()
clock = pygame.time.Clock()

running = True
state = env.reset()

while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # AI chọn hành động
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action = model(state_tensor).argmax().item()

    next_state, reward, done = env.step(action)
    state = next_state

    # Vẽ game
    screen.fill((135, 206, 235))  # nền
    pygame.draw.rect(screen, (255, 50, 50), (env.bird_x, env.bird_y, 34, 24))
    for pipe in env.pipes:
        pygame.draw.rect(screen, (0, 200, 0), (pipe['x'], 0, env.pipe_width, pipe['top']))
        pygame.draw.rect(screen, (0, 200, 0), (pipe['x'], pipe['bottom'], env.pipe_width, 600 - pipe['bottom']))
    score_text = font.render(f"Score: {env.score}", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))
    pygame.display.update()

    if done:
        pygame.time.wait(1000)
        state = env.reset()

pygame.quit()
