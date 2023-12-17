import cv2
import numpy as np
from PIL import Image

def remove_green_screen_pil(pil_img, threshold=2, blur=0, dilate=0):
    # Convert PIL image to OpenCV format (RGB to BGR)
    open_cv_image = np.array(pil_img.convert('RGB'))[:, :, ::-1]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)

    # Define base HSV range for green
    lower_green_base = np.array([40, 40, 40])
    upper_green_base = np.array([80, 255, 255])

    # Adjust the HSV range based on the threshold
    lower_green = lower_green_base + np.array([-threshold, -threshold, -threshold])
    upper_green = upper_green_base + np.array([threshold, threshold, threshold])

    # Limit the values to valid HSV range
    lower_green = np.clip(lower_green, 0, 255)
    upper_green = np.clip(upper_green, 0, 255)

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    if dilate > 0:
        mask = cv2.dilate(mask, None, iterations=dilate)
    # Optional: Apply blurring to the mask to soften edges
    if blur > 0:
        #mask = cv2.erode(mask, None, iterations=blur)

        # make sure blur kernel is odd
        if blur % 2 == 0:
            blur += 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)


    # Invert mask to get parts that are not green
    mask_inv = cv2.bitwise_not(mask)

    # Prepare an alpha channel with the inverted mask
    alpha_channel = mask_inv

    # Add the alpha channel to the BGR image
    bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2BGRA)
    bgr[:, :, 3] = alpha_channel

    # Convert back to RGB and then to PIL image
    rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
    result_img = Image.fromarray(rgb_image)

    return result_img

if __name__ == "__main__":
    pil_img = Image.open('SDControlNetTest0-Upscaled_2.png')
    pil_img.show()

    result_img = remove_green_screen_pil(pil_img)

    result_img.show()
    result_img.save('SDControlNetTest0-Upscaled_2_no_green.png')