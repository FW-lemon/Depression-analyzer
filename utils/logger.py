import logging
import sys
import os

os.makedirs("logs", exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO

# 自定义 ALERT 级别
ALERT_LEVEL = 35  # 比 WARNING 高，比 ERROR 低
logging.addLevelName(ALERT_LEVEL, "ALERT")

def alert(self, message, *args, **kwargs):
    if self.isEnabledFor(ALERT_LEVEL):
        self._log(ALERT_LEVEL, message, args, **kwargs)

logging.Logger.alert = alert

logger = logging.getLogger("FutureClient")
logger.setLevel(LOG_LEVEL)

# 文件日志（无颜色）
file_handler = logging.FileHandler("logs/app.log", encoding='utf-8')
file_handler.setLevel(LOG_LEVEL)
file_formatter = logging.Formatter(LOG_FORMAT)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 控制台日志（有颜色）
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(LOG_LEVEL)

try:
    from colorlog import ColoredFormatter

    class MessageColorFormatter(ColoredFormatter):
        def format(self, record):
            original_message = record.getMessage()
            if record.levelno == ALERT_LEVEL:
                record.msg = f"\033[31m{original_message}\033[0m"  # 红色
            elif record.levelno == logging.INFO:
                record.msg = f"\033[32m{original_message}\033[0m"  # 绿色
            elif record.levelno == logging.WARNING:
                record.msg = f"\033[33m{original_message}\033[0m"  # 黄色
            elif record.levelno == logging.ERROR:
                record.msg = f"\033[31m{original_message}\033[0m"  # 红色
            else:
                record.msg = original_message
            return super().format(record)

    color_formatter = MessageColorFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(color_formatter)

except ImportError:
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)

__all__ = ["logger"]
