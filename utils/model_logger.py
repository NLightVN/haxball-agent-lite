"""
utils/model_logger.py — Per-model file logger.

Usage:
    from utils.model_logger import get_model_logger
    log = get_model_logger("a1.1_0")   # writes to logs/a1.1_0/training.log
    log.info("Training started")
"""

import logging
import os
import sys


def get_model_logger(model_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Returns a Logger that writes to:
        <log_dir>/<model_name>/training.log

    The logger also forwards every message to stdout so the console output
    is preserved exactly as before.

    Args:
        model_name: e.g. "a0", "a1.1_0", "a1.1_1", "a1.2"
        log_dir:    root logs directory (default "logs")

    Returns:
        A configured logging.Logger instance.
    """
    # Resolve absolute path so it works regardless of cwd
    log_folder = os.path.join(log_dir, model_name)
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, "training.log")

    logger_name = f"haxball.{model_name}"
    logger = logging.getLogger(logger_name)

    # Avoid adding duplicate handlers when the module is re-imported
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── File handler ───────────────────────────────────────────────────────────
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # ── Console (stdout) handler ───────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Prevent propagation to root logger (avoids double printing)
    logger.propagate = False

    logger.info(f"Logger initialised — writing to {os.path.abspath(log_path)}")
    return logger
