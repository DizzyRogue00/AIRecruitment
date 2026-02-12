import abc
import os.path
import threading
import time
import logging
import cv2
import numpy as np
from typing import Optional,Tuple,Dict
from architecture import DetectionType,FrameData,DetectionResult,LatestFrameBuffer

class BaseDetector(abc.ABC):
    '''
    检测基类
    '''

    def __init__(
            self,
            detection_type:DetectionType,
            model_path:str,
            result_buffer,
            logger_name:str,
            min_interval_ms:float=16.0,
            cleanup_threshold:Optional[float]=None
    ):
        '''
        初始化
        :param detection_type: 检测类型，hand、pose、face
        :param model_path: 模型文件，在Tasks文件夹
        :param result_buffer: LatestFrameBuffer实例
        :param logger_name: 日志器名字
        :param min_interval_ms: 16.0 是60fps上限
        '''

        self.logger=logging.getLogger(logger_name)
        self.detection_type=detection_type
        self.model_path=model_path
        self.result_buffer=result_buffer

        # 线程安全
        self._frame_map_lock=threading.RLock()
        self._frame_map:Dict[int,Tuple[int,float]]={} # {timestamp_ms:(frame_id,timestamp)}
        self._cleanup_threshold=cleanup_threshold if cleanup_threshold else 5.0 # 秒，清理过期的帧映射
        self._orphan_count=0
        self._orphan_warn_time=0.0
        self._submit_count=0

        # 流程控制与性能
        self._last_submit_time=0.0
        self._min_interval=min_interval_ms/1000.0
        self._frame_count=0
        self._total_latency=0.0
        self._last_stats_time=time.time()
        self._start_time=time.time()

        # 资源状态
        self._detector=None
        self._is_initialized=False
        self._shutdown=False

        self.logger.info(f"Initializing {detection_type.value.upper()} detector | model: {os.path.abspath(model_path)}")

        try:
            self._init_model()
            self._is_initialized=True
        except Exception as e:
            self.logger.exception(f"Detector initialization failed: {e}")
            self.cleanup()
            raise

    ORPHAN_WARN_THRESHOLD=10
    ORPHAN_WARN_INTERVAL=10.0 # 秒
    CLEANUP_EVERY_N_SUBMITS=20

    @abc.abstractmethod
    def _init_model(self) -> None:
        '''
        创建检测器实例并绑定安全回调（子类实现）
        :return:
        '''
        pass

    def _safe_callback_wrapper(self,result,output_image,timestamp_ms):
        '''
        mediapipe回调函数
        :param result:
        :param output_image: mp.Image
        :param timestamp_ms:  int 应与提交时一致
        :return:
        '''
        try:
            #从映射表获取元数据
            with self._frame_map_lock:
                if timestamp_ms not in self._frame_map:
                    self.logger.debug(f"Orphaned callback ts={timestamp_ms} (no frame mapping)")
                    self._orphan_count += 1
                    now=time.time()
                    if(self._orphan_count % self.ORPHAN_WARN_THRESHOLD==0 or
                        now-self._orphan_warn_time>self.ORPHAN_WARN_INTERVAL):
                        self.logger.warning(
                            f"High orphan callbacks: {self._orphan_count} | "
                            f"pending_mappings={len(self._frame_map)} | ts={timestamp_ms}"
                        )
                        self._orphan_warn_time=now
                    return
                frame_id,timestamp_sec=self._frame_map[timestamp_ms]

                # 清理已处理的帧
                del self._frame_map[timestamp_ms]

            # 构建DetectionResult
            is_valid=self._is_result_valid(result)
            det_result=DetectionResult(
                frame_id=frame_id,
                timestamp=timestamp_sec,
                detection_type=self.detection_type,
                landmarks=result if is_valid else None,
                processing_time_ms=0.0,
                valid=is_valid
            )

            # 写入结果缓冲区
            self.result_buffer.put(self.detection_type,det_result)

            # 性能统计(每100帧)
            self._frame_count+=1
            if self._frame_count % 100==0:
                elapsed=time.time()-self._last_stats_time
                avg_fps=100.0/elapsed if elapsed >0 else 0
                self.logger.info(
                    f"Detection stable |"
                    f"avg_fps={avg_fps:.1f},"
                    f"frames_processed={self._frame_count},"
                    f"pending_mappings={len(self._frame_map)}"
                )
                self._last_stats_time=time.time()
        except Exception as e:
            self.logger.error(f"Callback processing failed: {e}", exc_info=True)

    @abc.abstractmethod
    def _is_result_valid(self,result) -> bool:
        '''
        验证结果的有效性（子类实现）
        :param result:
        :return:
        '''
        pass

    def submit_frame(self,frame_data: FrameData) -> bool:
        '''
        提交帧进行异步检测，检测线程调用
        :param frame_data:
        :return: 是否成功提交
        '''
        if self._shutdown or not self._is_initialized:
            return False

        # 流控，防止生产过快
        now=time.time()
        if (now-self._last_submit_time)<self._min_interval:
            self.logger.debug(
                f"Frame dropped (rare limit) | interval={(now-self._last_submit_time)*1000:.1f}ms < {self._min_interval*1000:.1f}ms"
            )
            return False # 跳过本帧
        self._last_submit_time=now

        self._submit_count+=1
        if self._submit_count % self.CLEANUP_EVERY_N_SUBMITS ==0:
            self._cleanup_stale_record_mappings()
        # # 记录帧映射
        # with self._frame_map_lock:
        #     # 清理过期映射
        #     cutoff=now-self._cleanup_threshold
        #     stale_ts=[ts for ts,(_,ts_sec) in self._frame_map.items() if ts_sec<cutoff]
        #     for ts in stale_ts:
        #         del self._frame_map[ts]

            # 注册新映射
            self._frame_map[frame_data.timestamp_ms]=(frame_data.frame_id,frame_data.timestamp)

        # 执行异步检测,子类实现
        try:
            self._detect_async(frame_data.rgb_frame,frame_data.timestamp_ms)
            return True
        except Exception as e:
            self.logger.error(f"Frame submission failed: {e}",exc_info=True)
            with self._frame_map_lock:
                self._frame_map.pop(frame_data.timestamp_ms,None)
            return False

    def _cleanup_stale_record_mappings(self):
        with self._frame_map_lock:
            now=time.time()
            cutoff = now - self._cleanup_threshold
            stale_ts = [ts for ts, (_, ts_sec) in self._frame_map.items() if ts_sec < cutoff]
            for ts in stale_ts:
                del self._frame_map[ts]
            if stale_ts:
                self.logger.debug(f"Cleaned {len(stale_ts)} stale mappings (threshold={self._cleanup_threshold}s)")

    @abc.abstractmethod
    def _detect_async(self,rgb_frame:np.ndarray,timestamp_ms:int) -> None:
        '''
        执行mediapipe异步检测
        :param rgb_frame:
        :param timestamp_ms:
        :return:
        '''
        pass

    def cleanup(self):
        '''
        释放资源
        :return:
        '''

        if self._shutdown:
            return

        self._shutdown=True

        self.logger.info(f"Close operation: Shutting down {self.detection_type.value} detector...")

        try:
            #清理映射表
            with self._frame_map_lock:
                stale_count=len(self._frame_map)
                self._frame_map.clear()
                if stale_count>0:
                    self.logger.warning(f"Cleared {stale_count} stale frame mappings during shutdown")

            # 释放检测器（子类实现）
            if self._detector:
                self._release_detector()
                self._detector=None

            # 性能总结
            if self._frame_count>0:
                total_time=time.time()-self._start_time
                self.logger.info(
                    f"Shutdown summary | "
                    f"total_frame={self._frame_count},"
                    f"effective_runtime={total_time:.2f}s,"
                    f"avg_throughput={self._frame_count/total_time:.1f} fps"
                )
            self.logger.info(f"{self.detection_type.value} detector resources released")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}",exc_info=True)
        finally:
            self._is_initialized=False

    @abc.abstractmethod
    def _release_detector(self) -> None:
        '''
        释放Mediapipe资源（子类实现）
        :return:
        '''
        pass

    def __enter__(self):
        if not self._is_initialized:
            raise RuntimeError("Detector not initialized")
        return self

    def __exit__(self,exc_type,exc_val,exc_tb):
        self.cleanup()
        return False