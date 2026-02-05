from dataclasses import dataclass
from enum import Enum
from typing import Dict,Optional,Tuple
import threading
import time
import numpy as np

class DetectionType(Enum):
    FACE="face"
    HAND="hand"
    POSE="pose"

@dataclass
class FrameData:
    frame_id: int
    timestamp: float
    bgr_frame: np.ndarray # 原始BGR帧
    rgb_frame: np.ndarray # RGB副本
    processed: bool=False

@dataclass
class DetectionResult:
    frame_id: int
    timestamp: float
    detection_type: DetectionType
    landmarks: Optional[any] # 结果对象
    processing_time_ms: float
    valid: bool=True

# 缓冲区，自动丢弃旧帧
class LatestFrameBuffer:
    def __init__(self,max_size=1):
        self.buffer={}
        self.lock=threading.RLock()
        self.max_size=max_size
        self.last_update=time.time()

    def put(self,key:str,value):
        '''
        放入缓冲区
        :param key: 缓冲区的键值
        :param value:
        :return:
        '''
        with self.lock:
            if len(self.buffer)>=self.max_size:
                oldest_key=min(self.buffer.keys(),key=lambda k:self.buffer[k][1])
                del self.buffer[oldest_key]
            self.buffer[key]=(value,time.time())
            self.last_update=time.time()

    def get(self,key:str):
        '''
        get 方法
        :param key:
        :return:
        '''
        with self.lock:
            item=self.buffer.get(key)
            return item[0] if item else None

    def clear_stale(self,max_age_sec=0.5):
        '''
        清理过期数据(防止内存泄漏)
        :param max_age_sec:
        :return:
        '''
        with self.lock:
            now=time.time()
            stale_keys=[k for k,(_,ts) in self.buffer.items() if now-ts>max_age_sec]
            for k in stale_keys:
                del self.buffer[k]