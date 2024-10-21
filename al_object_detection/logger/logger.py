import logging
import os
import copy
from termcolor import colored

DEFAULT_LOGGER_NAME = "al_logger"
LOGGER_NAME = os.getenv("LOGGER_NAME", DEFAULT_LOGGER_NAME)


class MonitorFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno in (logging.WARNING, logging.ERROR):
            self._style._fmt = "%(levelname)s - %(message)s"
        else:
            self._style._fmt = "%(message)s"

        color = record.color if hasattr(record, "color") else None
        attrs = record.attrs if hasattr(record, "attrs") else None
        if color or attrs:
            colored_record = copy.copy(record)
            colored_record.msg = colored(record.msg, color, attrs=attrs)
            return super().format(colored_record)
        else:
            return super().format(record)


class ALLogger(logging.Logger):
    def __init__(self, log_file="al.log"):
        super().__init__("root")

        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file)

        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.INFO)

        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        console_formatter = MonitorFormatter()
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        self.addHandler(console_handler)
        self.addHandler(file_handler)


def get_logger(log_file=None):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        console_formatter = MonitorFormatter()
        console_handler.setFormatter(console_formatter)

        logger.addHandler(console_handler)

    if log_file is not None:       
        file_handler = logging.FileHandler(log_file)

        file_handler.setLevel(logging.INFO)

        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    return logger
