import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt
import cv2

class FaceLandmarksDetector:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.latest_result=None

        # Create a face landmarker instance with the live stream mode:
        def callback_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.latest_result=result
            print('face landmarker result: {}'.format(result))

        options=FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='./Tasks/face_landmarker.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            result_callback=callback_result)

        self.landmarker = FaceLandmarker.create_from_options(options)

        self.mp_drawing=mp.tasks.vision.drawing_utils
        self.mp_drawing_styles=mp.tasks.vision.drawing_styles
        self.mp_face_mesh=mp.tasks.vision.FaceLandmarksConnections

        self.drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0),thickness=1,circle_radius=1)

    def process_frame(self, frame, frame_timestamp_ms):
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        frame.flags.writeable=False
        # convert to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame received from OpenCV (numpy array) to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, frame_timestamp_ms)

        if self.latest_result and self.latest_result.face_landmarks:
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return self._draw_landmarks(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def _draw_landmarks(self,image):
        if not self.latest_result or not self.latest_result.face_landmarks:
            return image

        face_landmarks_list = self.latest_result.face_landmarks
        annotated_image = np.copy(image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())

            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            # self.mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=None,
            #     landmark_drawing_spec=self.drawing_spec,
            #     connection_drawing_spec=None)

        #image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        return annotated_image

def main():
    detector = FaceLandmarksDetector()
    cap = cv2.VideoCapture(1)

    frame_counter = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # calculate timestamp
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_timestamp_ms = int(frame_counter * (1000 / fps))
            process_frame = detector.process_frame(frame, frame_timestamp_ms)

            cv2.imshow('MediaPipe Faces', cv2.flip(process_frame, 1))

            frame_counter += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        detector.landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
