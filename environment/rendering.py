# rendering.py

import imageio
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from pygame.locals import *

# Grid and agent settings
grid_size = 5
cell_size = 100  # Size of each cell in pixels
agent_position = [0, 0]
frames = []  # Store frames for GIF

# Reward grid with environmental actions
reward_grid = np.zeros((grid_size, grid_size))
reward_grid[1, 1] = 1
reward_grid[3, 3] = 1
reward_grid[2, 2] = 1
reward_grid[0, 3] = -1
reward_grid[3, 1] = -1
reward_grid[4, 3] = -1


def draw_grid():
    """Draws the grid lines."""
    glColor3f(0, 0, 0)
    for i in range(grid_size + 1):
        glBegin(GL_LINES)
        glVertex2f(i * cell_size, 0)
        glVertex2f(i * cell_size, grid_size * cell_size)
        glVertex2f(0, i * cell_size)
        glVertex2f(grid_size * cell_size, i * cell_size)
        glEnd()


def draw_rewards():
    """Draws reward and penalty cells."""
    for i in range(grid_size):
        for j in range(grid_size):
            if reward_grid[i, j] == 1:
                glColor3f(0, 1, 0)  # Green for positive practices
            elif reward_grid[i, j] == -1:
                glColor3f(1, 0, 0)  # Red for negative practices
            else:
                continue
            glBegin(GL_QUADS)
            glVertex2f(j * cell_size, (grid_size - i - 1) * cell_size)
            glVertex2f((j + 1) * cell_size, (grid_size - i - 1) * cell_size)
            glVertex2f((j + 1) * cell_size, (grid_size - i) * cell_size)
            glVertex2f(j * cell_size, (grid_size - i) * cell_size)
            glEnd()


def draw_agent():
    """Draws the agent as a yellow circle."""
    glColor3f(1, 1, 0)
    x, y = agent_position
    cx, cy = (y + 0.5) * cell_size, (grid_size - x - 0.5) * cell_size
    radius = cell_size * 0.3
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(cx, cy)
    for angle in range(0, 361, 10):
        theta = np.radians(angle)
        glVertex2f(cx + radius * np.cos(theta), cy + radius * np.sin(theta))
    glEnd()


def capture_frame():
    """Captures the current frame for GIF generation."""
    width, height = grid_size * cell_size, grid_size * cell_size
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
    image = np.flipud(image)
    frames.append(image)


def main():
    global agent_position
    pygame.init()
    display = (grid_size * cell_size, grid_size * cell_size)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glOrtho(0, display[0], 0, display[1], -1, 1)
    glClearColor(1, 1, 1, 1)  # White background

    running = True
    move_sequence = [(0, 1), (1, 1), (1, 2), (2, 2), (3, 2), (3, 3)]  # Path
    move_index = 0
    clock = pygame.time.Clock()

    # Main loop for rendering
    while running and move_index < len(move_sequence):
        glClear(GL_COLOR_BUFFER_BIT)
        draw_rewards()
        draw_grid()
        agent_position = list(move_sequence[move_index])
        draw_agent()
        capture_frame()
        pygame.display.flip()
        pygame.time.wait(500)  # Delay between moves
        move_index += 1

    pygame.quit()

    # Save GIF
    gif_path = "first_env_visualization.gif"
    imageio.mimsave(gif_path, frames, fps=2)
    print(f"✅ GIF saved at {gif_path}")


if __name__ == "__main__":
    main()
