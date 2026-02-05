# logger_setup.py
import logging
import logging.config
import logging.handlers
from logging import Filter, LogRecord, Handler
import yaml
import os
import sys
import threading
import queue
from queue import Queue
from typing import Optional, Dict, Set
import atexit
from pathlib import Path


class ModuleFilter(Filter):
    '''
    精准模块过滤
    '''

    def __init__(self, module_prefix: str):
        super().__init__()
        self.module_prefix = module_prefix.rstrip('.') + '.'
        self.exact_match = module_prefix.endswith('*')  # 是否精确匹配
        if self.exact_match:
            self.module_prefix = module_prefix[:-1].rstrip('.')  # 移除'*'

    def filter(self, record: LogRecord) -> bool:
        if self.exact_match:
            return record.name == self.module_prefix.rstrip('.')
        return record.name.startswith(self.module_prefix)

class DynamicModuleRegistry:
    '''
    动态注册模块，运行时新增模块
    '''

    _instance=None
    _lock=threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance=super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self.registered_modules: Set[str]=set()
        self.module_configs:Dict[str,tuple]={} # {prefix: (filename,level)}
        self._logger=logging.getLogger('log.registry')

    def register_module(self,prefix:str,filename:str,level: int=logging.DEBUG):
        '''
        注册新模块
        :param prefix: 日志器
        :param filename: 文件存储位置
        :param level: 日志等
        :return:
        '''
        with self._lock:
            if prefix in self.registered_modules:
                self._logger.debug(f"Module '{prefix}' already registered")
                return

            self.registered_modules.add(prefix)
            self.module_configs[prefix]=(filename,level)
            self._logger.info(f"Registered new log module: {prefix} -> {filename} (level={logging.getLevelName(level)})")

    def get_all_configs(self) -> Dict[str,tuple]:
        with self._lock:
            return dict(self.module_configs)

class SmartQueueListener(logging.handlers.QueueListener):
    '''
    - 自动创建模块化日志文件
    - 线程安全处理器管理
    - 单个处理器失败不影响全局
    - 动态模块注册支持
    - atexit清理，防止资源泄漏
    '''
    def __init__(
            self,
            log_queue:queue.Queue,
            log_dir:str='../logs',
            main_log_config:tuple=("main.log",10*1024*1024,5,'utf-8'),
            module_log_config:tuple=(5*1024*1024,3,'utf-8'),
            console_handler:Optional[Handler]=None
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True,exist_ok=True)
        self.main_log_config=main_log_config
        self.module_log_config=module_log_config
        self.console_handler=console_handler

        self._handlers:Dict[str,Handler]={}
        self._lock=threading.RLock()
        self._shutdown=False
        self._registry=DynamicModuleRegistry()

        # 创建处理器
        self._create_all_handlers()

        # 注册退出清理
        atexit.register(self.stop)

        # 启动消费者线程
        self._thread=threading.Thread(
            target=self._monitor_queue,
            name="LogConsumerThread",
            daemon=True
        )
        self._thread.start()
        logging.getLogger('log.listener').info(f"SmartQueueListener started. Logs dir:{self.log_dir.absolute()}")

    def _create_all_handlers(self):
        '''
        创建所有日志处理器，主日志+模块日志+动态注册模块
        :return:
        '''


class LogManager:
    def __init__(self, config_path: str = '../config/logging_config.yaml'):
        '''
        初始化日志管理器
        :param config_path: 日志配置文件(yaml)的路径
        '''
        self.config_path = config_path
        self._logging_queue: Optional[Queue] = None
        self._listener: Optional[logging.handlers.QueueListener] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self.logger = logging.getLogger(self.__class__.__name__)

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

        with open(self.config_path, 'rt', encoding='utf-8') as f:
            config = yaml.safe_load(f.read())

        # 创建全局队列
        # 队列必须在应用配置加载前创建，并注入到配置中
        self._logging_queue = queue.Queue(-1)

        if 'handlers' in config and 'queue_handler' in config['handlers']:
            config['handlers']['queue_handler']['queue'] = self._logging_queue
        else:
            self.logger.warning("配置文件中可能缺少'queue_handler' handler.这可能导致队列监听器无法工作。")

        # 加载日志配置
        logging.config.dictConfig(config)
        # 设置队列监听器
        # 创建一个独立的处理器
        file_handler = logging.handlers.RotatingFileHandler(
            '../logs/main.log', maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [%(processName)s/%(threadName)s] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        # 创建并启动监听线程
        self._listener = logging.handlers.QueueListener(self._logging_queue, file_handler)

        self._stop_event = threading.Event()

        def _run_listener():
            self.logger.info("日志队列监听器线程已启动。")
            self._listener.start()

            try:
                # 监听停止事件，直到被设置
                while not self._stop_event.is_set():
                    self._stop_event.wait(timeout=0.2)
            finally:
                self._listener.stop()
                self.logger.info("日志队列监听器线程已停止。")

        self._listener_thread = threading.Thread(target=_run_listener, daemon=True, name="LogManagerListenerThread")
        self._listener_thread.start()

        self.logger.info(f"日志系统已通过 {self.__class__.__name__} 初始化并启动。")

    def stop_logging(self, timeout: Optional[float] = None):
        '''
        停止日志记录系统
        :param timeout: 等待监听器线程结束的最大秒数，如果为None，则无限等待。
        :return:
        '''

        if self._stop_event:
            self.logger.info("正在请求停止日志队列监听器...")
            self._stop_event.set()  # 发送停止信号

        if self._listener_thread and self._listener_thread.is_alive():
            self.logger.info(f"等待监听器线程 '{self._listener_thread.name}' 结束...")
            self._listener_thread.join(timeout=timeout)  # 等待线程结束

            if self._listener_thread.is_alive():
                self.logger.warning(f"监听器线程 '{self._listener_thread.name}' 在超时后仍未结束")
            else:
                self.logger.info("监听器线程已成功结束。")
        else:
            self.logger.info("监听器线程不存在或已经结束")

        # 清理资源
        self._logging_queue = None
        self._listener = None
        self._listener_thread = None
        self._stop_event = None
