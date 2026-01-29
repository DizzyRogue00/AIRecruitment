# logger_setup.py
import logging
import logging.config
import logging.handlers
import yaml
import os
import threading
import queue
from queue import Queue
import os
from typing import Optional, Tuple

class LogManager:
    def __init__(self,config_path:str='../Config/logging_config.yaml'):
        '''
        初始化日志管理器
        :param config_path: 日志配置文件(yaml)的路径
        '''
        self.config_path=config_path
        self._logging_queue:Optional[Queue]=None
        self._listener:Optional[logging.handlers.QueueListener]=None
        self._listener_thread:Optional[threading.Thread]=None
        self._stop_event:Optional[threading.Event]=None
        self.logger=logging.getLogger(self.__class__.__name__)

    def setup_logging(self):
        '''
        加载日志配置并启动队列监听器
        Raises:
            FileNotFoundError: 如果指定的配置文件不存在
            yaml.YAMLError: 如果配置文件格式错误
        :return:
        '''
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"日志配置文件未找到: {self.config_path}")

        with open(self.config_path,'rt',encoding='utf-8') as f:
            config=yaml.safe_load(f.read())

        # 创建全局队列
        # 队列必须在应用配置加载前创建，并注入到配置中
        self._logging_queue=queue.Queue(-1)

        if 'handlers' in config and 'queue_handler' in config['handlers']:
            config['handlers']['queue_handler']['queue']=self._logging_queue
        else:
            self.logger.warning("配置文件中可能缺少'queue_handler' handler.这可能导致队列监听器无法工作。")

        #加载日志配置
        logging.config.dictConfig(config)

        # 设置队列监听器
        # 创建一个独立的处理器
        file_handler=logging.handlers.RotatingFileHandler(
            '../Logs/main.log',maxBytes=10*1024*1024,backupCount=5,encoding='utf-8'
        )

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [%(processName)s/%(threadName)s] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        # 创建并启动监听线程
        self._listener=logging.handlers.QueueListener(self._logging_queue,file_handler)
        self._stop_event=threading.Event()

        def _run_listener():
            self.logger.info("日志队列监听器线程已启动。")
            self._listener.start()

            try:
                #监听停止事件，直到被设置
                while not self._stop_event.is_set():
                    self._stop_event.wait(timeout=0.2)
            finally:
                self._listener.stop()
                self.logger.info("日志队列监听器线程已停止。")

        self._listener_thread=threading.Thread(target=_run_listener,daemon=True,name="LogManagerListenerThread")
        self._listener_thread.start()

        self.logger.info(f"日志系统已通过 {self.__class__.__name__} 初始化并启动。")

    def stop_logging(self,timeout:Optional[float]=None):

