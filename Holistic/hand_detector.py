import logging

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

import numpy as np
import cv2
import os

from detector_base import BaseDetector
from architecture import DetectionType,DetectionResult

class HandDetector(BaseDetector):
    '''
    手部检测
    纯检测职责和零拷贝可视化
    '''
    def __init__(self,result_buffer,model_path:str="../Tasks/hand_landmarker.task"):
        super().__init__(
            detection_type=DetectionType.HAND,
            model_path=model_path,
            result_buffer=result_buffer,
            logger_name='detector.hand',
            min_interval_ms=16.0
        )

    def _init_model(self):
        '''
        创建Mediapipe手部检测器
        :return:
        '''
        try:
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
            VisionRunningMode = mp.tasks.vision.RunningMode

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Hand model not found: {os.path.abspath(self.model_path)}")

            options=HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._safe_callback_wrapper
            )

            self._detector=HandLandmarker.create_from_options(options)
            self._is_initialized=True
            self.logger.info(f"Hand detector initialized | model: {self.model_path}")
        except Exception as e:
            self.logger.critical(f"Hand detector initialization failed: {e}",exc_info=True)
            raise

    def _detect_async(self,rgb_frame:np.ndarray,timestamp_ms:int) -> None:
        if not self._is_initialized or self._detector is None:
            raise RuntimeError("Detector not initialized")
        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
        self._detector.detect_async(mp_image,timestamp_ms)

    def _is_result_valid(self,result) -> bool:
        return (result is not None and
                hasattr(result,'hand_landmarks') and
                len(result.hand_landmarks)>0)

    def _release_detector(self) -> None:
        '''
        释放Mediapipe资源
        :return:
        '''
        if self._detector:
            self._detector.close()
            self.logger.debug("Hand lanmarker closed")

    @staticmethod
    def draw_on_frame(
            bgr_frame:np.ndarray,
            detection_result:DetectionResult,
            inplace:bool=False
    ) -> np.ndarray:
        '''
        静态方法，在BGR帧上绘制手部关键点
        :param bgr_frame: 原始BGR帧
        :param detection_result: 包含landmarks的DetectionResult
        :param inplace: True=直接修改原帧，False=返回新帧
        :return: 标注后的BGR帧
        '''
        if not detection_result.valid or detection_result.landmarks is None:
            return bgr_frame if inplace else bgr_frame.copy()

        output_frame=bgr_frame if inplace else bgr_frame.copy()

        try:
            mp_hands = mp.tasks.vision.HandLandmarksConnections
            mp_drawing = mp.tasks.vision.drawing_utils
            mp_drawing_styles = mp.tasks.vision.drawing_styles

            for idx,landmarks in enumerate(detection_result.landmarks.hand_landmarks):
                mp_drawing.draw_landmarks(
                    output_frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    #mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=4),  # 关键点
                    mp_drawing_styles.get_default_hand_connections_style()
                    #mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # 连接线
                )
            return output_frame # BGR
        except Exception as e:
            #可视化失败不应该阻塞主流程
            logging.getLogger('detector.hand').warning(f"Visualization failed: {e}")
            return bgr_frame if inplace else bgr_frame.copy()