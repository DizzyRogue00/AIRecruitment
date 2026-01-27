import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

class HandLandmarksDetector:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.latest_result=None

        # Create a hand landmarker instance with the live stream mode:
        def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.latest_result=result
            print('hand landmarker result: {}'.format(self.latest_result))

        options=HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='./Tasks/hand_landmarker.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=print_result)

        # create landmarker
        self.landmarker=HandLandmarker.create_from_options(options)

        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

    def process_frame(self,frame,frame_timestamp_ms):
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        frame.flags.writeable=False
        # convert to rgb
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
        self.landmarker.detect_async(mp_image,frame_timestamp_ms)
        if self.latest_result and self.latest_result.hand_landmarks:
            frame.flags.writeable=True
            image=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            return self._draw_landmarks(image)

        return frame

    def _draw_landmarks(self,image):
        if not self.latest_result or not self.latest_result.hand_landmarks:
            return image
        hand_landmarks_list = self.latest_result.hand_landmarks
        handedness_list = self.latest_result.handedness
        annotated_image = np.copy(image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks=hand_landmarks_list[idx]
            handedness=handedness_list[idx]

            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
        image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return image


def main():
    detector=HandLandmarksDetector()
    cap=cv2.VideoCapture(1)

    frame_counter=0

    try:
        while cap.isOpened():
            success,frame=cap.read()
            if not success:
                print("ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # calculate timestamp
            fps=cap.get(cv2.CAP_PROP_FPS)
            frame_timestamp_ms=int(frame_counter*(1000/fps))
            process_frame=detector.process_frame(frame, frame_timestamp_ms)

            cv2.imshow('MediaPipe Hands',cv2.flip(process_frame,1))

            frame_counter+=1
            if cv2.waitKey(5) & 0xFF==27:
                break
    finally:
        detector.landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()



