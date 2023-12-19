import datetime

import cv2
import numpy as np
from PIL import Image

def remove_green_screen_pil(pil_img, threshold=2, blur=0, dilate=0, debug=False):
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

    #heal stray holes in the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None)

    # Optional: Apply blurring to the mask to soften edges
    if blur > 0:
        #mask = cv2.erode(mask, None, iterations=blur)

        # make sure blur kernel is odd
        if blur % 2 == 0:
            blur += 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    if debug:
        # get the percentage of the mask that is green
        total_pixels = mask.shape[0] * mask.shape[1]
        green_pixels = np.sum(mask == 255)
        green_percentage = green_pixels / total_pixels * 100
        #round to 2 decimal places
        green_percentage = round(green_percentage, 2)
        print("Green percentage: " + str(green_percentage))
        # save the mask to a png file
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        cv2.imwrite("./output/mask-" + timestamp + "-coveragepct-" + str(green_percentage) + ".png", mask)

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