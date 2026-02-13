from typing import Optional

import numpy as np
from abc import ABC, abstractmethod
import logging

logger=logging.getLogger('integrator')
class Strategy(ABC):
    @abstractmethod
    def integrate(self,frame_buffer,result_buffers,current_frame_id) -> Optional[np.ndarray]:
        pass

'''
两种策略：
策略1：同步策略
策略2：利用各个检测器的最新帧
'''
class FrameSyncStrategy(Strategy):

class LatestResultStrategy(Strategy):

class StrategyFactory():
    _generator={}
    


#  @classmethod
#     def register(cls,visual_type:str,generator_class):
#         '''
#         :param visual_type: string name maps to generator class
#         :param generator_class: specific generator class
#         :return:
#         '''
#
#         cls._generators[visual_type]=generator_class
#
#     @classmethod
#     def get_generator_instance(cls,visual_type:str) -> VisualGenerator:
#         '''
#         :param visual_type: type of generator
#         :return: instance of Visual
#         :raises KeyError: if the visual type is not registered
#         '''
#         generator_class=cls._generators.get(visual_type)
#         if not generator_class:
#             raise KeyError(f"Unknown or unregistered visual type: '{visual_type}'. "
#                            f"Available types:{list(cls._generators.keys())}")
#         return generator_class()
#
#     @classmethod
#     def get_registered_types(cls) -> list:
#         return list(cls._generators.keys())
#
# # complete the registration when loading the module or initialization
# VisualGeneratorFactory.register('annotate',AnnotatedImageGenerator)
# VisualGeneratorFactory.register('segment',SegmentationMaskGenerator)
