# logger_setup.py
import inspect
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
from contextlib import suppress
import time

# 装饰器
def log_with_line_info(line_offset=0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_frame=None
            caller_frame=None
            try:
                # 获取当前帧和调用者信息
                current_frame = inspect.currentframe()
                caller_frame = current_frame.f_back

                # 获取装饰器调用位置的行号
                filename=caller_frame.f_code.co_filename
                lineno=caller_frame.f_lineno+line_offset
                funcname=caller_frame.f_code.co_name

            finally:
                del current_frame
                del caller_frame

            kwargs['caller_filename']=filename
            kwargs['caller_lineno']=lineno
            kwargs['caller_funcname']=funcname

            return func(*args, **kwargs)
        return wrapper
    return decorator

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
        #prefix_without_dot=self.module_prefix.rstrip('.')
        return (record.name == self.module_prefix.rstrip('.') or record.name.startswith(self.module_prefix))

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
        :param prefix: 处理器
        :param filename: 文件存储位置
        :param level: 日志等级
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
            main_log_config:tuple=("main.log",10*1024*1024,5),
            module_log_config:tuple=(5*1024*1024,3),
            console_handler:Optional[Handler]=None
    ):
        self._queue=log_queue or queue.Queue(-1)
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
        创建所有日志处理器，主日志handler+模块日志handler+动态注册模块
        :return:
        '''
        # 创建主日志处理器
        main_file,main_max_bytes,main_backup_count=self.main_log_config
        main_handler=logging.handlers.RotatingFileHandler(
            filename=self.log_dir / main_file,
            maxBytes=main_max_bytes,
            backupCount=main_backup_count,
            encoding='utf-8'
        )
        main_handler.setFormatter(self._get_formatter("detailed"))
        main_handler.setLevel(logging.DEBUG)
        self._handlers["main"]=main_handler

       # 预定义其他日志处理器
        base_configs=[
            ("detector.face","face.log",logging.DEBUG),
            ("detector.hand", "hand.log", logging.DEBUG),
            ("detector.pose", "pose.log", logging.DEBUG),
            ("integrator","integrator.log",logging.INFO),
            ("frame_publisher","frame_publisher.log",logging.INFO),
            ("system","system.log",logging.INFO),
            ("__main__","main_app.log",logging.INFO)
        ]

        # 添加动态注册的handler信息
        for prefix,(filename,level) in self._registry.get_all_configs():
            base_configs.append((prefix,filename,level))

        # 创建模块处理器
        for prefix, filename, level in base_configs:
            try:
                handler=logging.handlers.RotatingFileHandler(
                    filename=self.log_dir / filename,
                    maxBytes=self.module_log_config[0],
                    backupCount=self.module_log_config[1],
                    encoding='utf-8'
                )
                handler.setFormatter(self._get_formatter("detailed"))
                handler.setLevel(level)
                handler.addFilter(ModuleFilter(prefix))
                self._handlers[prefix]=handler
                logging.getLogger('log.listener').debug(f"Created handler for module: {prefix} -> {filename}")
            except Exception as e:
                logging.getLogger('log.listener').error(f"Failed to create handler for {prefix}: {e}",exc_info=True)

    def _get_formatter(self,style:str):
        fmts={
            "detailed": "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - [%(processName)s/%(threadName)s] - %(funcName)s:%(lineno)d - %(message)s",
            "simple": "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
        }
        return logging.Formatter(fmts[style],datefmt="%Y-%m-%d %H:%M:%S")

    def _monitor_queue(self):
        """
        带异常隔离的消费队列
        :return:
        """
        while not self._shutdown:
            try:
                record=self.queue.get(timeout=0.1)
                #record=self._queue.get()
                if record is None:
                    self._process_remaining_records()
                    break
                self._dispatch_record(record)
            except queue.Empty:
                continue
            except Exception as e:
                # 监听器自身发生故障
                try:
                    sys.stderr.write(f"[LOG-LISTENER-ERROR] {type(e).__name__}: {e}\n")
                except:
                    pass

        # 线程结束前的最后检查
        self._process_remaining_records()

    def _process_remaining_records(self):
        '''
        处理队列中剩余的记录
        :return:
        '''
        while True:
            try:
                record=self._queue.get_nowait()
                if record is not None:
                    self._dispatch_record(record)
            except queue.Empty:
                break
            except Exception as e:
                # 监听器自身发生故障
                try:
                    sys.stderr.write(f"[LOG-LISTENER-ERROR] {type(e).__name__}: {e}\n")
                except:
                    pass

    def _dispatch_record(self,record:LogRecord):
        """
        日志分发策略
        :param record:
        :return:
        """
        # 加锁获取当前处理器快照，避免迭代时字典被修改
        with self._lock:
             handlers_snapshot=list(self._handlers.items()) # 创建安全快照
        #for handler_name,handler in self._handlers.items():
        for handler_name, handler in handlers_snapshot:
            try:
                if handler.level<=record.levelno:
                    handler.handle(record)
            except Exception as e:
                # 单个处理器失效不影响其他处理器
                with suppress(BaseException):
                    sys.stderr.write(
                        f"[HANDLER-FAILURE] handler '{handler_name}' failed: {type(e).__name__}\n"
                        f"Record: {record.name}:{record.levelname} | {record.getMessage()}\n"
                    )

    def add_module_handler(self,prefix:str,filename:str,level:int=logging.DEBUG):
        '''
        运行时动态加载处理器模块
        :param prefix: 处理器模块名
        :param filename: 日志存储的文件
        :param level: 等级
        :return:
        '''
        with self._lock:
            if prefix in self._handlers:
                logging.getLogger('log.listener').warning(f"Handler for '{prefix}' already exists")
                return

            try:
                handler=logging.handlers.RotatingFileHandler(
                    filename=self.log_dir / filename,
                    maxBytes=self.module_log_config[0],
                    backupCount=self.module_log_config[1],
                    encoding='utf-8'
                )
                handler.setFormatter(self._get_formatter("detailed"))
                handler.setLevel(level)
                handler.addFilter(ModuleFilter(prefix))
                self._handlers[prefix]=handler
                self._registry.register_module(prefix,filename,level)
                logging.getLogger('log.listener').info(f"Dynamically added handler: {prefix} -> {filename}")
            except Exception as e:
                logging.getLogger('log.listener').error(f"Failed to add dynamic handler for {prefix}: {e}",exc_info=True)


    @log_with_line_info(line_offset=0)
    def _create_log_record(self,name,msg,caller_filename,caller_lineno,caller_funcname):
        return logging.LogRecord(
            name=name,
            level=logging.INFO,
            pathname=caller_filename,
            lineno=caller_lineno,
            msg=msg,
            args=(),
            exc_info=None,
            func=caller_funcname
        )

    def stop(self,timeout:float=3.0):
        '''
        停止，清理资源
        :return:
        '''
        if self._shutdown:
            return

        self._shutdown=True

        # 线程退出
        with suppress(BaseException):
            self._queue.put_nowait(None) #

        if self._thread.is_alive():
            self._thread.join(timeout=min(timeout,2.0) if timeout else 2.0)

        # 等待队列中剩余的日志被处理
        remaining=self._queue.qsize()
        if not self._thread.is_alive() and remaining>0:
            for _ in range(remaining):
                try:
                    #record = self.queue.get(timeout=0.1)
                    record = self.queue.get_nowait()
                    # record=self._queue.get()
                    if record is not None:
                        self._dispatch_record(record)
                except queue.Empty:
                    break
                except Exception as e:
                    # 监听器自身发生故障
                    try:
                        sys.stderr.write(f"[LOG-LISTENER-ERROR] {type(e).__name__}: {e}\n")
                    except:
                        pass

        # 在清理日志前，构造标准的LogRecord（绕过已停止的队列线程）
        try:
            stop_listener_record=self._create_log_record(
                name='log.listener',
                msg="SmartQueueListener stopped and resources cleaned (main.log)",
            )
            self._dispatch_record(stop_listener_record)
            stop_record = self._create_log_record(
                name='LogManager',
                msg="日志系统已安全停止 (main.log)",
            )
            self._dispatch_record(stop_record)
        except Exception as e:
            with suppress(BaseException):
                sys.stderr.write(f"[STOP-LOG-FAILURE] Failed to close handler '{type(e).__name__}': {e}\n")

        # 清理日志器
        with self._lock:
            for name,handler in self._handlers.items():
                try:
                    handler.close()
                except Exception as e:
                    if not isinstance(e,OSError):
                        sys.stderr.write(f"[CLEANUP-ERROR] Failed to close handler '{name}': {e}\n")
            self._handlers.clear()

        # 清理控制台
        if self.console_handler:
            with suppress(BaseException):
                self.console_handler.close()
        logging.getLogger('log.listener').info("SmartQueueListener stopped and resources cleaned (console)")

    @property
    def queue(self) -> queue.Queue:
        with self._lock:
            return self._queue

    @queue.setter
    def queue(self,new_queue:queue):
        '''
        替换正在使用的日志队列。
        注意：这是一个高级操作，应谨慎使用。通常在停止监听器后更换队列再重启，
        或确保新旧队列之间的平滑过渡，以避免日志丢失。
        :param new_queue:
        :return:
        '''
        if not isinstance(new_queue,queue.Queue):
            raise TypeError(f"新的队列必输是queue.Queue的实例")

        with self._lock:
            old_queue=self._queue
            self._queue=new_queue
            logging.getLogger('log.listener').info(f"Log queue replaced. Old: {id(old_queue)}, New: {id(new_queue)}")

class LogManager:
    _instances: Dict[str,'LogManager']={}
    _lock=threading.Lock()

    def __new__(cls,config_path:str=None):
        if config_path is None:
            config_path='../config/logging_config.yaml'

        config_path=os.path.abspath(config_path)

        if config_path not in cls._instances:
            with cls._lock:
                if config_path not in cls._instances:
                    instance=super().__new__(cls)
                    instance._config_path=config_path
                    instance._initialized=False
                    cls._instances[config_path]=instance
        return cls._instances[config_path]

    def __init__(self, config_path: str = None):
        '''
        初始化日志管理器
        :param config_path: 日志配置文件(yaml)的路径
        '''
        if self._initialized:
            # 单例模式
            return
        if config_path is None:
            config_path='../config/logging_config.yaml'

        self.config_path = config_path
        self._logging_queue: Optional[queue.Queue] = None
        self._listener: Optional[SmartQueueListener] = None
        #self._stop_event: Optional[threading.Event] = None
        self._stop_event = threading.Event()
        self._initialized=True
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def get_instance(cls,config_path:str=None) -> 'LogManager':
        '''
        获取实例
        :param config_path:
        :return:
        '''
        return cls(config_path)

    @classmethod
    def get_default_instance(cls) -> 'LogManager':
        '''
        获取默认配置的实例
        :return:
        '''
        return cls.get_instance('../config/logging_config.yaml')

    @classmethod
    def cleanup(cls,config_path:str=None):
        '''
        清理指定配置实例
        :param config_path:
        :return:
        '''
        with cls._lock:
            if config_path:
                config_path=os.path.abspath(config_path)
                if config_path in cls._instances:
                    del cls._instances[config_path]
            else:
                cls._instances.clear()

    @classmethod
    def count_instances(cls) -> int:
        '''
        统计有多少个不同配置的实例
        :return:
        '''
        return len(cls._instances)

    def setup_logging(self,log_dir:str="../logs") -> bool:
        '''
        加载日志配置并启动队列监听器
        Raises:
            FileNotFoundError: 如果指定的配置文件不存在
            yaml.YAMLError: 如果配置文件格式错误
        :return: 初始化是否成功
        '''
        try:
            # 验证配置文件
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"日志配置文件未找到: {os.path.abspath(self.config_path)}")
            # 加载yaml文件
            with open(self.config_path, 'rt', encoding='utf-8') as f:
                config = yaml.safe_load(f.read())

            # 创建全局队列
            # 队列必须在应用配置加载前创建，并注入到配置中
            self._logging_queue = queue.Queue(-1)
            #self._logging_queue=queue.Queue(maxsize=1000)

            if 'handlers' in config and 'queue_handler' in config['handlers']:
                config['handlers']['queue_handler']['queue'] = self._logging_queue
            else:
                self.logger.warning("配置文件中可能缺少'queue_handler' handler.这可能导致队列监听器无法工作。")
                if 'handlers' not in config:
                    config['handlers']={}
                config['handlers']['queue_handler']={
                    'class':'logging.handlers.QueueHandler',
                    'queue':self._logging_queue,
                    'level':'DEBUG'
                }
                if 'root' in config and 'handlers' in config['root']:
                    if 'queue_handler' not in config['root']['handlers']:
                        config['root']['handlers'].append('queue_handler')

            # 禁止第三方库冗余日志
            config.setdefault('loggers',{})
            noisy_loggers={
                'matplotlib':{'level':'WARNING','propagate':False},
                'PIL': {'level': 'WARNING', 'propagate': False},
                'urllib3': {'level': 'WARNING', 'propagate': False},
                'asyncio': {'level': 'WARNING', 'propagate': False},
            }

            for logger_name,cfg in noisy_loggers.items():
                if logger_name not in config['loggers']:
                    config['loggers'][logger_name]=cfg
            # 加载日志配置 root logger的处理器为console和queue_handlers
            logging.config.dictConfig(config)

            # 创建并启动监听线程
            console_handler=None
            root_logger=logging.getLogger()
            for h in root_logger.handlers:
                if isinstance(h,logging.StreamHandler) and h.stream==sys.stdout:
                    console_handler=h
                    break

            self._listener=SmartQueueListener(
                log_queue=self._logging_queue,
                log_dir=log_dir,
                console_handler=console_handler
            )

            # 验证初始化
            test_logger=logging.getLogger('system.init')
            test_logger.info("=" * 60)
            test_logger.info("企业级日志系统初始化成功")
            test_logger.info(f"日志目录: {Path(log_dir).resolve()}")
            test_logger.info(f"队列监听器: {self._listener.__class__.__name__}")
            test_logger.info(f"已注册模块: {len(self._listener._handlers)} 个处理器")
            test_logger.info(f"动态注册API: LogManager.instance().listener.add_module_handler()")
            test_logger.info("=" * 60)
            return True
        except Exception as e:
            # 初始化失败
            print(f"[CRITICAL] 日志系统初始化失败: {e}",file=sys.stderr)
            print("回退到基础控制台日志...",file=sys.stderr)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - [%(levelname)s] - %(message)s',
                handlers=[logging.StreamHandler(sys.stderr)]
            )
            return False

    def stop_logging(self, timeout: Optional[float] = None):
        '''
        停止日志记录系统
        :param timeout: 等待监听器线程结束的最大秒数，如果为None，则无限等待。
        :return:
        '''

        if not self._listener:
            return

        self.logger.info("正在停止日志系统...")
        self._listener.stop(timeout)
        self.logger.info("日志系统已安全停止")

    @property
    def listener(self) -> Optional[SmartQueueListener]:
        '''
        提供监听器的引用，便于全局动态注册模块
        :return:
        '''
        return self._listener

# 全局函数
def get_logger(name: str) -> logging.Logger:
    """
    获取已经配置的日志器
    :param name:
    :return:
    """
    return logging.getLogger(name)

def registry_log_module(config_path:str,prefix:str,filename:str,level:int=logging.DEBUG):
    """
    全局函数：动态注册新日志模块
    :param prefix:
    :param filename:
    :param level:
    :return:
    """
    manager=LogManager.get_instance(config_path)
    if manager.listener:
        manager.listener.add_module_handler(prefix,filename,level)
    else:
        logging.warning(f"LogManager 未初始化未加载日志器，无法注册模块： {prefix}")

