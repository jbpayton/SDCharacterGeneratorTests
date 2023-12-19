import datetime

import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageFilter

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)


# Function to create a face mask from a PIL image.
def create_face_mask_pil(pil_image, debug=False):
    # Convert PIL Image to OpenCV format.
    image = np.array(pil_image)
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect facial landmarks.
    result = face_mesh.process(image_rgb)

    # Create an empty mask.
    mask = np.zeros((height, width), dtype=np.uint8)

    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:
            # Extract landmark points.
            points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in facial_landmarks.landmark]

            # Create a convex hull around the facial landmarks.
            hull = cv2.convexHull(np.array(points, dtype=np.int32))
            cv2.fillConvexPoly(mask, hull, (255, 255, 255))

    # Convert the mask back to PIL format.
    mask_pil = Image.fromarray(mask)

    # dilate the mask using opencv, then blur it to soften edges
    mask = cv2.dilate(mask, None, iterations=9)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # When debugging save the mask to png.
    if debug:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        cv2.imwrite("./output/face-mask-" + timestamp + ".png", mask)

    return mask_pil


if __name__ == '__main__':
    # Example usage.
    pil_image = Image.open("VNImageGenerator-Character-Final.png")
    mask = create_face_mask_pil(pil_image)
    mask.show()
