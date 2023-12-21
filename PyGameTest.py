import random
import pygame
from PIL import Image
from PyGameHelpers import scale_image, maintain_aspect_ratio, render_multiline_text

class VisualNovelGame:
    def __init__(self, initial_background_path, initial_music_path):
        pygame.init()

        self.screen_width, self.screen_height = 1152, 768
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)

        self.background_image = None
        self.original_bg_width, self.original_bg_height = 0, 0
        self.original_background_surface = None

        self.load_initial_background(initial_background_path)
        self.character_scale_factor = 1  # Scale factor for characters
        self.scale_images()

        self.font = pygame.font.SysFont(None, int(self.screen_height * 0.04))
        self.full_dialogue = "Initial dialogue..."
        self.dialogue_position = 0

        self.characters = {'left': None, 'center': None, 'right': None}

        self.play_initial_music(initial_music_path)

    def load_initial_background(self, background_path):
        self.background_image = Image.open(background_path)
        self.original_bg_width, self.original_bg_height = self.background_image.size
        self.original_background_surface = pygame.image.fromstring(self.background_image.tobytes(),
                                                                   self.background_image.size,
                                                                   self.background_image.mode)

    def play_initial_music(self, music_path):
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.play(-1)

    def load_images(self):
        background_image_path = 'output/VNImageGenerator-background-Final.png'
        self.background_image = Image.open(background_image_path)
        self.original_bg_width, self.original_bg_height = self.background_image.size
        self.original_background_surface = pygame.image.fromstring(self.background_image.tobytes(), self.background_image.size, self.background_image.mode)

    def scale_images(self):
        self.background_surface = scale_image(self.original_background_surface, self.screen_width, self.screen_height)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_click()
                elif event.type == pygame.VIDEORESIZE:
                    self.handle_resize(event)

            self.draw()
            pygame.display.flip()
            pygame.time.wait(50)

        pygame.quit()

    def handle_mouse_click(self):
        if self.dialogue_position < len(self.full_dialogue):
            self.dialogue_position = len(self.full_dialogue)
        else:
            self.load_next_dialogue()

    def load_next_dialogue(self):
        self.full_dialogue = "New piece of dialogue..."
        self.dialogue_position = 0
        # Randomly load or clear characters for demonstration
        if random.choice([True, False]):
            self.load_random_character('left')
        else:
            self.clear_character('left')
        if random.choice([True, False]):
            self.load_random_character('center')
        else:
            self.clear_character('center')
        if random.choice([True, False]):
            self.load_random_character('right')
        else:
            self.clear_character('right')

    def load_character(self, position, character_path, expression):
        image_path = f"{character_path}/dialogue-{expression}.png"
        character_image = Image.open(image_path)
        original_character_surface = pygame.image.fromstring(character_image.tobytes(), character_image.size, character_image.mode)
        character_surface = scale_image(original_character_surface, original_character_surface.get_width(), self.screen_height * self.character_scale_factor)
        self.characters[position] = character_surface

    def clear_character(self, position):
        self.characters[position] = None

    def change_background(self, new_background_path):
        self.background_image = Image.open(new_background_path)
        self.original_bg_width, self.original_bg_height = self.background_image.size
        self.original_background_surface = pygame.image.fromstring(self.background_image.tobytes(),
                                                                   self.background_image.size,
                                                                   self.background_image.mode)
        self.scale_images()  # Rescale the new background

    def change_music(self, music_path):
        pygame.mixer.music.stop()  # Stop the current music
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.play(-1)  # Play the new track

    def load_random_character(self, position):
        character_image_path = "output/Hikari_Yumeno"
        expressions = ["cry", "angry", "smile", "neutral", "disappointed",
                              "blush", "scared", "laugh", "yell"]
        expression = random.choice(expressions)
        self.load_character(position, character_image_path, expression)

    def handle_resize(self, event):
        new_width, new_height = event.size
        self.screen_width, self.screen_height = maintain_aspect_ratio(self.original_bg_width, self.original_bg_height, new_width, new_height)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        self.scale_images()

    def draw(self):
        self.screen.blit(self.background_surface, (0, 0))
        self.draw_characters()
        self.draw_text_box()

    def draw_characters(self):
        positions = {'left': 0.0, 'center': 0.3, 'right': 0.6}
        for position, character_surface in self.characters.items():
            if character_surface:
                char_x, char_y = self.screen_width * positions[position], self.screen_height - character_surface.get_height()
                self.screen.blit(character_surface, (char_x, char_y))

    def draw_text_box(self):
        text_box_x, text_box_y = self.screen_width * 0.05, self.screen_height * 0.75
        text_box_width, text_box_height = self.screen_width * 0.9, self.screen_height * 0.2
        text_box_rect = pygame.Rect(text_box_x, text_box_y, text_box_width, text_box_height)
        render_multiline_text(self.screen, self.full_dialogue[:self.dialogue_position], text_box_rect, self.font, padding=20)
        if self.dialogue_position < len(self.full_dialogue):
            self.dialogue_position += 1

if __name__ == "__main__":
    initial_background = 'output/VNImageGenerator-background-Final.png'
    initial_music = 'Shenanigans!.ogg'
    game = VisualNovelGame(initial_background, initial_music)
    game.run()
