import pygame
import cv2
import numpy as np

import our_model

# --- Init ---
pygame.init()
WIDTH, HEIGHT = 1200, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Application")

# Font for text
font = pygame.font.SysFont("monospace", 24)

# OpenCV camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
if not camera.isOpened():
    raise RuntimeError("Could not open camera")

clock = pygame.time.Clock()
running = True
pressing_a = False
processed_image = []
top_three = []
graph = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                if pressing_a is False:
                    print('pressed!!')
                    our_model.plot_image(processed_image[0], our_model.to_category(top_three[0][0]), top_three[0][1], color=True)
                    pressing_a = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a:
                pressing_a = False


    # --- Capture frame ---
    ret, frame = camera.read()
    if not ret:
        continue

    # Convert BGR (cv2) -> RGB (pygame)
    cv2.rectangle(frame, (our_model.sq_start, our_model.sq_start), (our_model.sq_end, our_model.sq_end), (255, 0, 0), 2)
    frame_surface = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = np.rot90(frame_surface)

    # Convert to pygame surface
    frame_surface = pygame.surfarray.make_surface(frame_surface)

    # --- Draw ---
    screen.fill((30, 30, 30))  # background for text side

    processed_image = our_model.prepare_image(frame)
    model_out = our_model.predict_image(processed_image)
    top_three = sorted(enumerate(model_out), key=lambda x: x[1], reverse=True)[:3]
    # Draw text on left side
    text_lines = [
        f'1. {our_model.to_category(top_three[0][0]):<12} {100*top_three[0][1]:3.2f}%',
        f'2. {our_model.to_category(top_three[1][0]):<12} {100*top_three[1][1]:3.2f}%',
        f'3. {our_model.to_category(top_three[2][0]):<12} {100*top_three[2][1]:3.2f}%'
    ]
    y = 50
    for line in text_lines:
        txt_surface = font.render(line, True, (255, 255, 255))
        screen.blit(txt_surface, (20, y))
        y += 40

    # Draw camera feed on the right half
    screen.blit(frame_surface, (400, 0))
    updated_graph = our_model.plot_bar(model_out, pygame.time.get_ticks())
    if updated_graph:
        screen.blit(updated_graph, (0, y + 20))
        graph = updated_graph
    elif graph:
        screen.blit(graph, (0, y + 20))

    # Update display
    pygame.display.flip()
    clock.tick(30)  # limit to 30 FPS

# --- Cleanup ---
camera.release()
pygame.quit()
