import logging

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2
import os

from detector_base import BaseDetector
from architecture import DetectionType,DetectionResult
#导入生成器模块
from ..modules.VisualGenerators import VisualGeneratorFactory

class PoseDetector(BaseDetector):
    '''
    姿态检测器
    检测和可视化分离
    '''
    def __init__(self,result_buffer,model_path="../Tasks/pose_landmarker_full.task",visual_type='annotate'):
        '''

        :param result_buffer: 缓冲区域
        :param model_path: Optional["../Tasks/pose_landmarker_full.task","../Tasks/pose_landmarker_heavy.task","../Tasks/pose_landmarker_lite.task"]
        :param visual_type: Optional['annotate', 'segment']
        '''
        super().__init__(
            detection_type=DetectionType.POSE,
            model_path=model_path,
            result_buffer=result_buffer,
            logger_name='detector.pose',
            min_interval_ms=33.0 # 30fps
        )

        self._default_visual_type=visual_type
        self.logger.info(f"Pose visualizer default: '{visual_type}'")

    def _init_model(self):
        '''
        创建Mediapipe姿态检测器
        :return:
        '''
        try:
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
            VisionRunningMode = mp.tasks.vision.RunningMode

            if not hasattr(self,'_model_path_checked'):
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Pose model not found: {os.path.abspath(self.model_path)}")
                self._model_path_checked=True

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=True,
                result_callback=self._safe_callback_wrapper
            )

            # create detector
            self._detector = PoseLandmarker.create_from_options(options)
            self._is_initialized=True
            self.logger.info(f"Pose detector initialized | model: {self.model_path}")

        except Exception as e:
            self.logger.critical(f"Pose detector initialization failed: {e}", exc_info=True)
            raise

    def _is_result_valid(self, result) -> bool:
        '''
        验证姿态检测结果
        :param result:
        :return:
        '''
        return (result is not None and
                hasattr(result,'pose_landmarks') and
                len(result.pose_landmarks)>0)

    def _detect_async(self, rgb_frame: np.ndarray, timestamp_ms: int) -> None:
        '''
        提交帧到Mediapipe进行异步检测
        :param rgb_frame:
        :param timestamp_ms:
        :return:
        '''
        if not self._is_initialized or self._detector is None:
            raise RuntimeError("Detector not initialized")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._detector.detect_async(mp_image, timestamp_ms)

    def _release_detector(self) -> None:
        '''
        释放Mediapipe资源
        :return:
        '''
        if self._detector:
            self._detector.close()
            self.logger.debug("Pose landmarker closed")

    '''
    零拷贝可视化，整合（integrator.py)用
    '''
    @staticmethod
    def draw_on_frame(
            rgb_frame:np.ndarray,
            detection_result:DetectionResult,
            inplace:bool=False,
            visual_type:str="annotate"
    ) -> np.ndarray:
        '''
        静态方法，在BGR帧上可视化姿态,但Mediapipe通道需要rgb图像
        :param rgb_frame:
        :param detection_result:
        :param inplace: True=直接修改原帧，False=返回新帧
        :param visual_type: 'annotate','segment'（由聚合器）指定
        :return:
        '''
        if not detection_result.valid or detection_result.landmarks is None:
            return rgb_frame if inplace else rgb_frame.copy()

        output_frame = rgb_frame if inplace else rgb_frame.copy()

        try:
            '''
            整合器决定可视化策略，检测器提供数据
            '''
            generator=VisualGeneratorFactory.get_generator_instance(visual_type)

            return generator.generate(DetectionResult.landmarks,output_frame)

        except Exception as e:
            logger=logging.getLogger('detector.pose')
            logger.warning(f"Visualization failed with type '{visual_type}': {e}. Falling back to RGB frame.")
            return rgb_frame if inplace else rgb_frame.copy()

    @classmethod
    def get_registered_visual_type(cls) -> list:
        '''
        获取已经注册的类型
        :return:
        '''
        return VisualGeneratorFactory.get_registered_types()