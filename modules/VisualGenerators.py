import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from abc import ABC, abstractmethod

# define base class
class VisualGenerator(ABC):
    @abstractmethod
    def generate(self,pose_data:mp.tasks.vision.PoseLandmarkerResult,image):
        '''
        :param pose_data: result from PoseLandmarker.create_from_options(options)
        :param image:
        :return: generate a visual image
        '''
        pass

# specific generator
class AnnotatedImageGenerator(VisualGenerator):
    def generate(self,pose_data:mp.tasks.vision.PoseLandmarkerResult,image):
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

    @classmethod
    def get_registered_types(cls) -> list:
        return list(cls._generators.keys())

# complete the registration when loading the module or initialization
VisualGeneratorFactory.register('annotate',AnnotatedImageGenerator)
VisualGeneratorFactory.register('segment',SegmentationMaskGenerator)