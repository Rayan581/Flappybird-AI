import pygame
from classes import Game

# Window settings
WIDTH, HEIGHT = 1000, 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND = (155, 218, 255)  # Soft sky blue


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flappybird")

    game = Game('saves\\best_bird_brain.json')

    clock = pygame.time.Clock()

    running = True
    while running:
        dt = clock.get_time() / 1000

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    saved_file = game.save_best_bird()
                    if saved_file:
                        print(f'✅ Best bird saved to: {saved_file}')
                    else:
                        print("⚠️ No bird to save yet")

                    running = False
                elif event.key == pygame.K_UP:
                    game.increase_speed()
                elif event.key == pygame.K_DOWN:
                    game.decrease_speed()

        game.update(dt)

        screen.fill(BACKGROUND)

        game.draw(screen)

        font = pygame.font.Font(None, 36)
        gen_text = font.render(f"Generation: {game.generation}", True, BLACK)
        alive_text = font.render(f"Alive: {len(game.birds)}", True, BLACK)
        score_text = font.render(
            f"Score: {game.get_current_score()}", True, BLACK)
        screen.blit(gen_text, (10, 10))
        screen.blit(alive_text, (10, 50))
        screen.blit(score_text, (10, 90))

        pygame.display.flip()

        clock.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    main()
