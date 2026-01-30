import logging

from logger_setup import LogManager
import time

if __name__=='__main__':
    log_manager=LogManager()

    try:
        log_manager.setup_logging()

        app_logger=logging.getLogger('MyApp')
        app_logger.info("应用程序启动了!")
        app_logger.error("发生了一个错误！")

        time.sleep(2)
    finally:
        log_manager.stop_logging(timeout=5)
        print("程序结束！")

