import cv2
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np
import atexit

from utils import LogManager, registry_log_module
from modules import HandDetector,FaceDetector,PoseDetector

log_manager=LogManager()
