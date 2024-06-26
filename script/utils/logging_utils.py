import logging
import os

LOGGER_NAME = "my_logger"


def init_logger(log_level_for_console: str = "info", log_level_for_file: str = "debug", save_dir: str = None):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level=logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s %(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(log_level_for_console.upper())
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir is not None:
        fh = logging.FileHandler(os.path.join(save_dir, f"{LOGGER_NAME}.txt"))
        fh.setLevel(log_level_for_file.upper())
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def get_logger():
    return logging.getLogger(LOGGER_NAME)


if __name__ == "__main__":

    init_logger(save_dir="/path/to/somewhere")
    logger = get_logger()
    logger.info("Let's start!")
    # >>> 2020-04-12 14:49:47 [INFO] my_module.py 17 - Let's start!
