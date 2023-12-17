import random

import pygame
from PIL import Image

def scale_image(image, target_width, target_height):
    """ Scale the image while maintaining aspect ratio. """
    original_width, original_height = image.get_size()
    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))
    return pygame.transform.scale(image, new_size)

def maintain_aspect_ratio(original_width, original_height, new_width, new_height):
    """ Adjust the new dimensions to maintain the aspect ratio. """
    original_aspect = original_width / original_height
    new_aspect = new_width / new_height

    if new_aspect > original_aspect:
        # Window too wide, adjust width
        return int(new_height * original_aspect), new_height
    else:
        # Window too tall, adjust height
        return new_width, int(new_width / original_aspect)


def render_multiline_text(surface, text, rect, font, color=(0, 0, 0), alpha=128, padding=10):
    """ Render multiline text within a given rectangle with word wrap and padding. """
    text_box_surface = pygame.Surface((rect.width, rect.height))
    text_box_surface.set_alpha(alpha)
    text_box_surface.fill((255, 255, 255))
    surface.blit(text_box_surface, rect.topleft)

    x, y = rect.topleft[0] + padding, rect.topleft[1] + padding
    max_width, max_height = rect.width - 2 * padding, rect.height - 2 * padding

    words = text.split(' ')
    space = font.size(' ')[0]
    for word in words:
        word_surface = font.render(word, True, color)
        word_width, word_height = word_surface.get_size()
        if x + word_width >= rect.right - padding:
            x = rect.left + padding  # Reset x to the left padding
            y += word_height  # Start on new row
        if y + word_height >= rect.bottom - padding:
            # No more vertical space for the text; stop rendering
            break
        surface.blit(word_surface, (x, y))
        x += word_width + space  # Move x to the right by the word width and space width


# Initialize Pygame
pygame.init()

# Load images (assuming you have PIL images)
background_image_path = 'VNImageGenerator-background-Final.png'
background_image = Image.open(background_image_path)
character_image = Image.open('VNImageGenerator-Character-Final.png')

# Original image size
original_bg_width, original_bg_height = background_image.size

# Create a resizable window with the initial size of 1152x768
screen_width, screen_height = 1152, 768
screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)

# Convert PIL images to Pygame surfaces and keep a reference to the original
original_background_surface = pygame.image.fromstring(background_image.tobytes(), background_image.size, background_image.mode)
original_character_surface = pygame.image.fromstring(character_image.tobytes(), character_image.size, character_image.mode)

character_scale_factor = 1  # Scale the character to 30% of the screen height

# Scale images initially
background_surface = scale_image(original_background_surface, screen_width, screen_height)
character_surface = scale_image(original_character_surface, original_character_surface.get_width(), screen_height * character_scale_factor)

# Set up the font once, so you don't have to create it every frame
font = pygame.font.SysFont(None, int(screen_height * 0.04))

# Define the full dialogue and a variable to keep track of the current position
full_dialogue = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
dialogue_position = 0  # Keeps track of which character we're up to in the typewriter effect

# play the music
pygame.mixer.music.load('Guitar on the Water.ogg')
pygame.mixer.music.play(-1)


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # If the mouse is clicked and we're not at the end of the dialogue
            if dialogue_position < len(full_dialogue):
                # Jump to the end of the current dialogue
                dialogue_position = len(full_dialogue)
            else:
                # Load the next piece of dialogue here
                full_dialogue = "New piece of dialogue..."
                dialogue_position = 0
                # load a different character image here (randomly, or based on the dialogue)
                # select a random character image between 1 and 3
                random_character_image = random.randint(1, 7)
                if random_character_image == 1:
                    character_image = Image.open('VNImageGenerator-Character-Happy.png')
                elif random_character_image == 2:
                    character_image = Image.open('VNImageGenerator-Character-sad.png')
                elif random_character_image == 3:
                    character_image = Image.open('VNImageGenerator-Character-embarrassed.png')
                elif random_character_image == 4:
                    character_image = Image.open('VNImageGenerator-Character-smug.png')
                elif random_character_image == 5:
                    character_image = Image.open('VNImageGenerator-Character-neutral.png')
                elif random_character_image == 6:
                    character_image = Image.open('VNImageGenerator-Character-angry.png')
                elif random_character_image == 7:
                    character_image = Image.open('VNImageGenerator-Character-surprised.png')

                # Convert PIL images to Pygame surfaces and keep a reference to the original
                original_character_surface = pygame.image.fromstring(character_image.tobytes(), character_image.size, character_image.mode)
                character_surface = scale_image(original_character_surface, original_character_surface.get_width(),
                                                screen_height * character_scale_factor)


        elif event.type == pygame.VIDEORESIZE:
            # Window has been resized, adjust to maintain aspect ratio
            new_width, new_height = event.size
            screen_width, screen_height = maintain_aspect_ratio(original_bg_width, original_bg_height, new_width, new_height)
            screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
            background_surface = scale_image(background_surface, screen_width, screen_height)
            # Re-scale the background and character from their original surfaces
            background_surface = scale_image(original_background_surface, screen_width, screen_height)
            character_surface = scale_image(original_character_surface, original_character_surface.get_width(),
                                            screen_height * character_scale_factor)

    # Drawing the background
    screen.blit(background_surface, (0, 0))

    # Drawing the character
    char_x, char_y = screen_width * 0.3, screen_height - character_surface.get_height()
    screen.blit(character_surface, (char_x, char_y))

    # Drawing the translucent text box with padding and multiline text
    text_box_x, text_box_y = screen_width * 0.05, screen_height * 0.75
    text_box_width, text_box_height = screen_width * 0.9, screen_height * 0.2
    text_box_rect = pygame.Rect(text_box_x, text_box_y, text_box_width, text_box_height)
    render_multiline_text(screen, full_dialogue[:dialogue_position], text_box_rect, font,
                          padding=20)  # Padding set to 20 pixels

    # Update the dialogue position, but not past the end of the dialogue
    if dialogue_position < len(full_dialogue):
        dialogue_position += 1

    # Update the display
    pygame.display.flip()
    pygame.time.wait(50)  # Wait a bit before drawing the next character

pygame.quit()
