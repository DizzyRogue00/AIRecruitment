import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# STEP 2: Create an HandLandmarker object.
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
# configuration options, please refer to https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./Tasks/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2)

# STEP 3: Load the input image.
with HandLandmarker.create_from_options(options) as detector:
    image = mp.Image.create_from_file("./Pictures/woman_hands.jpg")
# STEP 4: Detect hand landmarks from the input image.
    detection_result=detector.detect(image)
    # HanLandmarkerResult
    #HandLandmarkerResult:
  # Handedness:
  #   Categories #0:
  #     index        : 0
  #     score        : 0.98396
  #     categoryName : Left
  # Landmarks:
  #   Landmark #0:
  #     x            : 0.638852
  #     y            : 0.671197
  #     z            : -3.41E-7
  #   Landmark #1:
  #     x            : 0.634599
  #     y            : 0.536441
  #     z            : -0.06984
  #   ... (21 landmarks for a hand)
  # WorldLandmarks:
  #   Landmark #0:
  #     x            : 0.067485
  #     y            : 0.031084
  #     z            : 0.055223
  #   Landmark #1:
  #     x            : 0.063209
  #     y            : -0.00382
  #     z            : 0.020920
  #   ... (21 world landmarks for a hand)
# STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    img_rgb = cv2.cvtColor(annotated_image,cv2.COLOR_RGB2BGR)
    # plt.imshow(img_rgb)
    # plt.axis('off')
    # plt.show()
    cv2.imshow('Image', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
import cv2

img=cv2.imread('./Pictures/woman_hands.jpg')
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# another method

import matplotlib.pyplot as plt

img = cv2.imread('./Pictures/woman_hands.jpg')

# OpenCV 使用 BGR，Matplotlib 使用 RGB，需要转换
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 显示图像
plt.imshow(img_rgb)
plt.axis('off')  # 不显示坐标轴
plt.show()
'''

