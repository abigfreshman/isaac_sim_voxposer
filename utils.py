import logging
import os
from datetime import datetime

def setup_logger(name="voxposer"):
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("output", "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{current_datetime}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")

    if not logger.handlers:  # 避免重复添加
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger