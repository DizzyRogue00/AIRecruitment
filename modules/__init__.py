__version__='1.0.0'

from .HandLandMarks_LiveStream import HandDetector
from .FaceLandMarks_LiveStream import FaceDetector
from .PoseLandMarks_LiveStream import PoseDetector

__all__=["HandDetector","FaceDetector","PoseDetector"]
