import logging

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


import numpy as np
import cv2
import os

from detector_base import BaseDetector
from architecture import DetectionType,DetectionResult

class FaceDetector(BaseDetector):
    '''
    脸部检测
    纯检测职责和零拷贝可视化
    '''

    def __init__(self,result_buffer,model_path='../Tasks/face_landmarker.task'):
        super().__init__(
            detection_type=DetectionType.FACE,
            model_path=model_path,
            result_buffer=result_buffer,
            logger_name='detector.face',
            min_interval_ms=16.0
        )

    def _init_model(self):
        '''
        创建Mediapipe脸部检测器
        :return:
        '''
        try:
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
            VisionRunningMode = mp.tasks.vision.RunningMode

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Face model not found: {os.path.abspath(self.model_path)}")

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_faces=2,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                result_callback=self._safe_callback_wrapper
            )

            self._detector=FaceLandmarker.create_from_options(options)
            self._is_initialized=True
            self.logger.info(f"Face detector initialized | model: {self.model_path}")
        except Exception as e:
            self.logger.critical(f"Face detector initialization failed: {e}",exc_info=True)
            raise

    def _detect_async(self,rgb_frame:np.ndarray,timestamp_ms:int) -> None:
        if not self._is_initialized or self._detector is None:
            raise RuntimeError("Detector not initialized")
        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
        self._detector.detect_async(mp_image,timestamp_ms)

    def _is_result_valid(self,result) -> bool:
        return (result is not None and
                hasattr(result,'face_landmarks') and
                len(result.face_landmarks)>0)

    def _release_detector(self) -> None:
        '''
        释放Mediapipe资源
        :return:
        '''
        if self._detector:
            self._detector.close()
            self.logger.debug("Face lanmarker closed")

    @staticmethod
    def draw_on_frame(
            rgb_frame:np.ndarray,
            detection_result:DetectionResult,
            inplace:bool=False
    ) -> np.ndarray:
        '''
        静态方法，在BGR帧上绘制手部关键点,但Mediapipe通道需要rgb图像
        :param rgb_frame: 原始BGR帧
        :param detection_result: 包含landmarks的DetectionResult
        :param inplace: True=直接修改原帧，False=返回新帧
        :return: 标注后的BGR帧
        '''
        if not detection_result.valid or detection_result.landmarks is None:
            return rgb_frame if inplace else rgb_frame.copy()

        output_frame=rgb_frame if inplace else rgb_frame.copy()

        try:
            mp_drawing = mp.tasks.vision.drawing_utils
            mp_drawing_styles = mp.tasks.vision.drawing_styles
            mp_face_mesh = mp.tasks.vision.FaceLandmarksConnections
            drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

            for idx,landmarks in enumerate(detection_result.landmarks.face_landmarks):

                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACE_LANDMARKS_LEFT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACE_LANDMARKS_RIGHT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            return output_frame # BGR

        except Exception as e:
            #可视化失败不应该阻塞主流程
            logging.getLogger('detector.face').warning(f"Visualization failed: {e}")
            return rgb_frame if inplace else rgb_frame.copy()