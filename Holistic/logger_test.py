import atexit
import logging

from utils import LogManager, registry_log_module,get_logger

log_mgr=LogManager('../config/logging_config.yaml')

if not log_mgr.setup_logging(log_dir='../logs/prod'):
    raise RuntimeError("日志系统初始化失败！")

# 动态注册新模块
registry_log_module('../config/logging_config.yaml',prefix="detector.gesture",filename="gesture.log",level=logging.DEBUG)

# 获取logger
logger=logging.getLogger("detector.face")
logger.info("面部检测模块启动") # 同时写入main.log 和 face.log

atexit.register(log_mgr.stop_logging)