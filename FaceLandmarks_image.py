import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt
import cv2

from HandLandmarks_image import img_rgb


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list=detection_result.face_landmarks
    annotated_image=np.copy(rgb_image)
    drawing_spec = drawing_utils.DrawingSpec(color=(255,0,0),thickness=1, circle_radius=1)
    # Loop through the detected faces to visualize.

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.

        # drawing_utils.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=None,
        #     landmark_drawing_spec=drawing_spec,
        #     connection_drawing_spec=None)
        # drawing_utils.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=[],
        #     landmark_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        #     connection_drawing_spec=None)
        # drawing_utils.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
        #     landmark_drawing_spec=drawing_spec,
        #     connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

        # drawing_utils.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,  # 使用嘴唇连接
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=drawing_utils.DrawingSpec(
        #         color=(255, 0, 0),  # 红色
        #         thickness=2,
        #         circle_radius=1
        #     )
        # )

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())


    return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title('Face Blendshapes')
    plt.tight_layout()
    plt.show()

# STEP 2: Create an FaceLandmarker object.
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./Tasks/face_landmarker.task'),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=VisionRunningMode.IMAGE)
# configuration options, please refer to https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python

with FaceLandmarker.create_from_options(options) as landmarker:
    # Load the input image from an image file.

# STEP 3: Load the input image.
    mp_image = mp.Image.create_from_file('./Pictures/business-person.png')

# STEP 4: Detect face landmarks from the input image.
    detection_result=landmarker.detect(mp_image)
#FaceLandmarkerResult
# FaceLandmarkerResult:
#     face_landmarks:
#     NormalizedLandmark  # 0:
#         x: 0.5971359014511108
#         y: 0.485361784696579
#         z: -0.038440968841314316
#     NormalizedLandmark  # 1:
#         x: 0.3302789330482483
#         y: 0.29289937019348145
#         z: -0.09489090740680695
#         ...(478 landmarks for each face)
#     face_blendshapes:
#         browDownLeft: 0.8296722769737244
#         browDownRight: 0.8096957206726074
#         browInnerUp: 0.00035583582939580083
#         browOuterUpLeft: 0.00035752105759456754
#         ...(52 blendshapes for each face)
#     facial_transformation_matrixes:
#         [9.99158978e-01, -1.23036895e-02, 3.91213447e-02, -3.70770246e-01]
#         [1.66496094e-02, 9.93480563e-01, -1.12779640e-01, 2.27719707e+01]
#         ...
# STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    imgFace_rgb=cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
    print(detection_result.facial_transformation_matrixes)

    cv2.imshow('Image', imgFace_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
