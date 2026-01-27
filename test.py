import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# img=cv2.imread('./Pictures/girl-4051811_960_720.jpg')
# print(len(img.shape))
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./Tasks/pose_landmarker_heavy.task'),
    running_mode=VisionRunningMode.IMAGE,
    output_segmentation_masks=True)

mp_poses = mp.tasks.vision.PoseLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

with PoseLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image.create_from_file('./Pictures/girl-4051811_960_720.jpg')

    pose_landmarker_result = landmarker.detect(mp_image)

    # print(pose_landmarker_result.pose_landmarks)
    # print(len(pose_landmarker_result.pose_landmarks))
    pose_landmarks_list = pose_landmarker_result.pose_landmarks
    annotated_image = np.copy(mp_image.numpy_view())

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks,
            mp_poses.POSE_LANDMARKS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    img_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    cv2.imshow('Image', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()