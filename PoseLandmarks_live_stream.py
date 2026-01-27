import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from abc import ABC, abstractmethod



# define base class
class VisualGenerator(ABC):
    @abstractmethod
    def generate(self,pose_data:mp.tasks.vision.PoseLandmarkerResult,image:mp.Image):
        '''
        :param pose_data: result from PoseLandmarker.create_from_options(options)
        :param image:
        :return: generate a visual image
        '''
        pass

# specific generator
class AnnotatedImageGenerator(VisualGenerator):
    def generate(self,pose_data:mp.tasks.vision.PoseLandmarkerResult,image:mp.Image):
        if not pose_data or not pose_data.pose_landmarks:
            return image

        pose_landmarks_list = pose_data.pose_landmarks
        annotated_image = np.copy(image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            mp.tasks.vision.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks,
                mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=mp.tasks.vision.drawing_styles.get_default_pose_landmarks_style())

        return annotated_image

class SegmentationMaskGenerator(VisualGenerator):
    def generate(self,pose_data:mp.tasks.vision.PoseLandmarkerResult,image:mp.Image):
        if not pose_data or not pose_data.pose_landmarks:
            return image

        segmentation_mask = pose_data.segmentation_masks[0].numpy_view()
        segmentation_mask_2d = np.squeeze(segmentation_mask)

        visualized_mask = np.repeat(segmentation_mask_2d[:, :, np.newaxis], 3, axis=2) * 255
        visualized_mask = visualized_mask.astype(np.uint8)
        visualized_mask = np.ascontiguousarray(visualized_mask)

        return visualized_mask

# factory class, register mode
class VisualGeneratorFactory:

    _generators={}

    @classmethod
    def register(cls,visual_type:str,generator_class):
        '''
        :param visual_type: string name maps to generator class
        :param generator_class: specific generator class
        :return:
        '''

        cls._generators[visual_type]=generator_class

    @classmethod
    def get_generator_instance(cls,visual_type:str) -> VisualGenerator:
        '''
        :param visual_type: type of generator
        :return: instance of Visual
        :raises KeyError: if the visual type is not registered
        '''
        generator_class=cls._generators.get(visual_type)
        if not generator_class:
            raise KeyError(f"Unknown or unregistered visual type: '{visual_type}'. "
                           f"Available types:{list(cls._generators.keys())}")
        return generator_class()

# complete the registration when loading the module or initialization
VisualGeneratorFactory.register('annotate',AnnotatedImageGenerator)
VisualGeneratorFactory.register('segment',SegmentationMaskGenerator)

class PoseLandmarksDetector:
    def __init__(self,initial_visual_type='annotate'):
        # Create an PoseLandmarker object.

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.latest_result=None
        self.visual_type=initial_visual_type

        # Create a pose landmarker instance with the live stream mode:
        def callback_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.latest_result=result
            print('pose landmarker result: {}'.format(result))

        self.mp_poses=mp.tasks.vision.PoseLandmarksConnections
        self.mp_drawing=mp.tasks.vision.drawing_utils
        self.mp_drawing_styles=mp.tasks.vision.drawing_styles

        options=PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='./Tasks/pose_landmarker_full.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True,
            result_callback=callback_result
        )

        self.landmarker=PoseLandmarker.create_from_options(options)

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
        :param output_type: type of generated image, eg., 'skeleton', 'mask'
        :return: the visualized image
        '''

        generator=VisualGeneratorFactory.get_generator_instance(output_type)

        visualized_image=generator.generate(self.latest_result,image)

        return visualized_image

def main():
    detector = PoseLandmarksDetector(initial_visual_type='annotate')
    cap = cv2.VideoCapture(1)

    frame_counter = 0

    try:
        while cap.isOpened():
            success, frame=cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # calculate timestamp
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_timestamp_ms = int(frame_counter * (1000 / fps))
            process_frame = detector.process_frame(frame, frame_timestamp_ms)

            cv2.imshow('MediaPipe Faces', cv2.flip(process_frame, 1))

            frame_counter+=1

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        detector.landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__== "__main__":
    main()