import logging
from fusion_perception.utils.logging_setup import get_logger, setup_logging

def test_get_logger_returns_named_logger():
    logger = get_logger("test_module")
    assert logger.name == "fusion_perception.test_module"

def test_setup_logging_sets_level():
    setup_logging(level="DEBUG", log_file=None)
    root = logging.getLogger("fusion_perception")
    assert root.level == logging.DEBUG
