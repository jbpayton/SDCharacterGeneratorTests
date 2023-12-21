import random
import pygame
from PIL import Image


class Transition:
    def apply(self, screen, old_surface, new_surface, duration):
        raise NotImplementedError("Must be implemented by subclasses")


class Dissolve(Transition):
    def apply(self, screen, old_surface, new_surface, duration=100):
        clock = pygame.time.Clock()
        num_frames = duration // (1000 // 60)  # Number of frames in the duration at 60 FPS
        alpha_step = max(1, 255 // num_frames)  # Calculate step size, ensure it's at least 1

        for alpha in range(0, 256, alpha_step):
            temp_surface = old_surface.copy()
            new_surface.set_alpha(alpha)
            temp_surface.blit(new_surface, (0, 0))
            screen.blit(temp_surface, (0, 0))
            pygame.display.flip()
            clock.tick(60)  # Maintain 60 FPS

        new_surface.set_alpha(255)  # Ensure the new surface is fully opaque at the end
        screen.blit(new_surface, (0, 0))
        pygame.display.flip()



class Fade(Transition):
    def apply(self, screen, old_surface, new_surface, duration=100):
        clock = pygame.time.Clock()
        half_duration = duration // 2
        for alpha in range(256):
            fade_surface = pygame.Surface(old_surface.get_size())
            fade_surface.fill((0, 0, 0))
            fade_surface.set_alpha(alpha)
            screen.blit(old_surface, (0, 0))
            screen.blit(fade_surface, (0, 0))
            pygame.display.flip()
            pygame.time.wait(half_duration // 255)
            clock.tick(60)
        for alpha in range(256):
            fade_surface.set_alpha(255 - alpha)
            screen.blit(new_surface, (0, 0))
            screen.blit(fade_surface, (0, 0))
            pygame.display.flip()
            pygame.time.wait(half_duration // 255)
            clock.tick(60)


class FadeInCharacter(Transition):
    def apply(self, game, character_surface, position, duration=250):
        clock = pygame.time.Clock()
        num_frames = duration // (1000 // 60)  # Number of frames in the duration at 60 FPS
        alpha_step = max(1, 255 // num_frames)  # Calculate step size, ensure it's at least 1

        for alpha in range(0, 256, alpha_step):
            character_surface.set_alpha(alpha)
            game.draw()  # Redraw the entire scene
            pygame.display.flip()
            clock.tick(60)  # Maintain 60 FPS

        character_surface.set_alpha(255)  # Ensure the surface is fully opaque at the end
        game.draw()  # Redraw the final state
        pygame.display.flip()


class FadeOutCharacter(Transition):
    def apply(self, game, character_surface, position, duration=250):
        clock = pygame.time.Clock()
        num_frames = duration // (1000 // 60)  # Number of frames in the duration at 60 FPS
        alpha_step = max(1, 255 // num_frames)  # Calculate step size, ensure it's at least 1

        for alpha in range(255, -1, -alpha_step):
            character_surface.set_alpha(alpha)
            game.draw()  # Redraw the entire scene
            pygame.display.flip()
            clock.tick(60)  # Maintain 60 FPS

        character_surface.set_alpha(0)  # Ensure the surface is fully transparent at the end
        game.draw()  # Redraw the final state
        pygame.display.flip()


