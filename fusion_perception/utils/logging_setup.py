"""Structured logging configuration for the pipeline."""
import logging
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> None:
    """Configure root logger for fusion_perception namespace."""
    root = logging.getLogger("fusion_perception")
    root.setLevel(getattr(logging, level.upper()))
    root.handlers.clear()

    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    datefmt = "%H:%M:%S"

    if use_rich and _RICH_AVAILABLE:
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True, show_path=False
        )
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root.addHandler(handler)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the fusion_perception namespace."""
    return logging.getLogger(f"fusion_perception.{name}")
