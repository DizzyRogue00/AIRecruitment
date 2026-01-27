import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import matplotlib.pyplot as plt


mp_poses = mp.tasks.vision.PoseLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

#Visualization utilities
def draw_landmarks_on_image(rgb_image,detection_result):
    pose_landmarks_list=detection_result.pose_landmarks
    annotated_image=np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks=pose_landmarks_list[idx]

        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks,
            mp_poses.POSE_LANDMARKS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    return annotated_image

# STEP 2: Create an PoseLandmarker object.
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configuration options, please refer to https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./Tasks/pose_landmarker_heavy.task'),
    running_mode=VisionRunningMode.IMAGE,
    output_segmentation_masks=True)

with PoseLandmarker.create_from_options(options) as landmarker:
# STEP 3: Load the input image.
    # Load the input image from an image file.
    mp_image = mp.Image.create_from_file('./Pictures/girl-4051811_960_720.jpg')

# STEP 4: Detect pose landmarks from the input image.
    # Perform pose landmarking on the provided single image.
    # The pose landmarker must be created with the image mode.
    pose_landmarker_result = landmarker.detect(mp_image)
# example of display result
# PoseLandmarkerResult:
#   Landmarks:
#     Landmark #0:
#       x            : 0.638852
#       y            : 0.671197
#       z            : 0.129959
#       visibility   : 0.9999997615814209
#       presence     : 0.9999984502792358
#     Landmark #1:
#       x            : 0.634599
#       y            : 0.536441
#       z            : -0.06984
#       visibility   : 0.999909
#       presence     : 0.999958
#     ... (33 landmarks per pose)
#   WorldLandmarks:
#     Landmark #0:
#       x            : 0.067485
#       y            : 0.031084
#       z            : 0.055223
#       visibility   : 0.9999997615814209
#       presence     : 0.9999984502792358
#     Landmark #1:
#       x            : 0.063209
#       y            : -0.00382
#       z            : 0.020920
#       visibility   : 0.999976
#       presence     : 0.999998
#     ... (33 world landmarks per pose)
#   SegmentationMasks:
#     ... (pictured below)
# STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)

    segmentation_mask=pose_landmarker_result.segmentation_masks[0].numpy_view()
    segmentation_mask_2d = np.squeeze(segmentation_mask)

    visualized_mask = np.repeat(segmentation_mask_2d[:, :, np.newaxis], 3, axis=2)* 255
    visualized_mask = visualized_mask.astype(np.uint8)
    visualized_mask = np.ascontiguousarray(visualized_mask)

    #img_visual = cv2.cvtColor(visualized_mask,cv2.COLOR_RGB2GRAY)
    cv2.imshow('Image1', visualized_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_rgb = cv2.cvtColor(annotated_image,cv2.COLOR_RGB2BGR)
    # plt.imshow(img_rgb)
    # plt.axis('off')
    # plt.show()
    cv2.imshow('Image', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()