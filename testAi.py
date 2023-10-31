import pygame
import random

pygame.init()

# Set up the game window
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Tetris")

# Define game variables
block_size = 30
board_width = 10
board_height = 20
board_x = (screen_width - block_size * board_width) // 2
board_y = (screen_height - block_size * board_height) // 2
board = [[0] * board_width for _ in range(board_height)]
fall_speed = 0.5
fall_speed_increase = 0.025
score = 0
font = pygame.font.SysFont(None, 30)

# Define colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)
purple = (128, 0, 128)
orange = (255, 165, 0)
colors = [black, red, green, blue, yellow, purple, orange]

# Define the shapes of the blocks
shapes = [    [[1, 1, 1],
     [0, 1, 0]],
    [[0, 2, 2],
     [2, 2, 0]],
    [[3, 3, 0],
     [0, 3, 3]],
    [[4, 0, 0],
     [4, 4, 4]],
    [[0, 0, 5],
     [5, 5, 5]],
    [[6, 6],
     [6, 6]],
    [[7, 7, 7, 7]]
]

# Define a function to create a new block
def new_block():
    shape = random.choice(shapes)
    block = {
        "shape": shape,
        "color": random.choice(colors),
        "x": board_width // 2 - len(shape[0]) // 2,
        "y": -len(shape),
        "rotation": 0
    }
    return block

# Define a function to draw a block on the board
def draw_block(x, y, color):
    pygame.draw.rect(screen, color, (board_x + x * block_size, board_y + y * block_size, block_size, block_size))

# Define a function to draw the board
def draw_board():
    for y in range(board_height):
        for x in range(board_width):
            if board[y][x] != 0:
                draw_block(x, y, colors[board[y][x]])

# Define a function to check if a block is within the boundaries of the board
def in_bounds(block):
    for y, row in enumerate(block["shape"]):
        for x, value in enumerate(row):
            if value != 0 and (block["y"] + y < 0 or block["y"] + y >= board_height or block["x"] + x < 0 or block["x"] + x >= board_width):
                return False
    return True

# Define a function to check if a block is colliding with other blocks on the board
def collision(block):
    for y, row in enumerate(block["shape"]):
        for x, value in enumerate(row):
         if value != 0 and board[block["y"] + y][block["x"] + x] != 0:
            return True
    return False

def place_block(block):
    for y, row in enumerate(block["shape"]):
        for x, value in enumerate(row):
            if value != 0:
                board[block["y"] + y][block["x"] + x] = colors.index(block["color"])
                
                
def remove_rows():
    global score
    rows_removed = 0
    y = board_height - 1
    while y >= 0:
        if all(block != 0 for block in board[y]):
            for i in range(y, 0, -1):
                board[i] = board[i-1][:]
                board[0] = [0] * board_width
                rows_removed += 1
        else:
            y -= 1
            if rows_removed > 0:
                score += 100 * 2 ** (rows_removed - 1)
                
                
def main():
    global fall_speed, score
    clock = pygame.time.Clock()
    current_block = new_block()
next_block = new_block()
game_over = False
while not game_over:
# Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                current_block["x"] -= 1
                if not in_bounds(current_block) or collision(current_block):
                    current_block["x"] += 1
                elif event.key == pygame.K_RIGHT:
                    current_block["x"] += 1
                    if not in_bounds(current_block) or collision(current_block):
                        current_block["x"] -= 1
                elif event.key == pygame.K_DOWN:
                    current_block["y"] += 1
                    if not in_bounds(current_block) or collision(current_block):
                        current_block["y"] -= 1
                        place_block(current_block)
                        remove_rows()
                        current_block = next_block
                        next_block = new_block()
                        if collision(current_block):
                            game_over = True
                        elif event.key == pygame.K_UP:
                            rotated_block = {"shape": current_block["shape"], "color": current_block["color"], "x": current_block["x"], "y": current_block["y"], "rotation": (current_block["rotation"] + 1) % len(current_block["shape"])}
                            rotated_block["shape"] = rotated_block["shape"][rotated_block["rotation"]]
                            if in_bounds(rotated_block) and not collision(rotated_block):
                                current_block = rotated_block
# Move the current block down
                                current_block["y"] += fall_speed
# if not in_bounds(current_block) or collision(current_block):
# current_block["y"] -= fall_speed
# place_block(current_block)
# remove_rows()
# current_block = next_block
# next_block = new_block()
# if collision(current_block):
# game_over = True
# # Increase the fall speed
# fall_speed += fall_speed_increase
# # Draw the screen
# screen.fill(white)
# draw_board()
# for y, row in enumerate(next_block["shape"]):
# for x, value in enumerate(row