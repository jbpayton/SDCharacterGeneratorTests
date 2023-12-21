import pygame


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