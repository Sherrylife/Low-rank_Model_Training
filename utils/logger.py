import logging
import os

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerCreator:
    @staticmethod
    def create_logger(log_path='./log', logging_name=None, level=logging.INFO, args=None):
        """
        :param log_path:
        :param logging_name:
        :param level:
        :param args:
        :return:
        """

        if logging_name is None:
            raise ValueError("name for logger cannot be None")

        # 获取logger对象,取名
        logger = logging.getLogger(logging_name)
        # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
        logger.setLevel(level=logging.DEBUG)
        # 获取文件日志句柄并设置日志级别，第二层过滤
        handler = logging.FileHandler(log_path, encoding='UTF-8', mode='w')
        handler.setLevel(logging.INFO)
        # 生成并设置文件日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # 为logger对象添加句柄
        logger.addHandler(handler)
        logger.addHandler(console)

        # logger_ = logging.getLogger(name)
        # logger_.setLevel(level)
        return logger

