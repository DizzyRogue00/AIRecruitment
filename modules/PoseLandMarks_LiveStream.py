import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import logging

#导入生成器模块
from VisualGenerators import VisualGeneratorFactory


logger=logging.getLogger(__name__)

class PoseDetector:
    def __init__(self,initial_visual_type='annotate'):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.latest_result = None
        self.visual_type = initial_visual_type

        self.logger=logging.getLogger('detector.pose')
        self.logger.info("姿态特征点初始器已初始化...")

        # Create a pose landmarker instance with the live stream mode:
        def callback_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.latest_result = result
            # print('pose landmarker result: {}'.format(result))

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='./Tasks/pose_landmarker_full.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True,
            result_callback=callback_result
        )

        # create landmarker
        self.landmarker=PoseLandmarker.create_from_options(options)

        self.mp_poses = mp.tasks.vision.PoseLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

    def process_frame(self, frame, frame_timestamp_ms):
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        frame.flags.writeable=False

        # convert to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # The pose landmarker must be created with the live stream mode.
        self.landmarker.detect_async(mp_image, frame_timestamp_ms)

        if self.latest_result and self.latest_result.pose_landmarks:
            frame.flags.writeable=True
            frame=self.visualize(frame,self.visual_type)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def visualize(self,image,output_type):
        '''
        pose visualization
        :param image: original image
        :param output_type: type of generated image, eg., 'annotate', 'segment'
        :return: the visualized image
        '''

        generator=VisualGeneratorFactory.get_generator_instance(output_type)

        visualized_image=generator.generate(self.latest_result,image)

        return visualized_image

    def close(self):
        self.landmarker.close()