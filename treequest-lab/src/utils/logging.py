"""
Logging utilities for TreeQuest Lab.

Centralized logging configuration with file and console output.
Supports structured JSON logging for analysis.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
    rich_tracebacks: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for TreeQuest.
    
    Args:
        level: Logging level (default: INFO)
        log_dir: Directory for log files (default: outputs/logs)
        console_output: Enable console output
        file_output: Enable file output
        json_format: Use JSON format for logs
        rich_tracebacks: Use rich library for better tracebacks
        
    Returns:
        Root logger
    """
    # Create logger
    logger = logging.getLogger("treequest")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        if log_dir is None:
            log_dir = "outputs/logs"
        
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"treequest_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Also create a JSON log file
        if json_format:
            json_log_file = log_path / f"treequest_{timestamp}.jsonl"
            json_handler = logging.FileHandler(json_log_file, encoding='utf-8')
            json_handler.setLevel(level)
            json_handler.setFormatter(JsonFormatter())
            logger.addHandler(json_handler)
    
    # Try to set up rich tracebacks
    if rich_tracebacks:
        try:
            from rich.logging import RichHandler
            
            # Replace console handler with rich handler
            if console_output:
                logger.removeHandler(console_handler)
                
                rich_handler = RichHandler(
                    rich_tracebacks=True,
                    tracebacks_show_locals=True,
                    markup=True
                )
                rich_handler.setLevel(level)
                logger.addHandler(rich_handler)
                
        except ImportError:
            pass  # Rich not available, use standard console handler
    
    logger.info(f"Logging initialized at level {logging.getLevelName(level)}")
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(
        self,
        level: int = logging.DEBUG,
        log_to_file: bool = False,
        log_dir: Optional[str] = None
    ):
        self.level = level
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self.old_handlers = None
        self.old_level = None
        
    def __enter__(self):
        logger = logging.getLogger("treequest")
        self.old_handlers = logger.handlers.copy()
        self.old_level = logger.level
        
        # Set up temporary logging
        setup_logging(
            level=self.level,
            log_dir=self.log_dir if self.log_to_file else None,
            console_output=True,
            file_output=self.log_to_file
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger("treequest")
        logger.handlers.clear()
        logger.handlers.extend(self.old_handlers)
        logger.setLevel(self.old_level)
        
        return False  # Don't suppress exceptions


if __name__ == "__main__":
    # Test logging setup
    print("=" * 60)
    print("Testing Logging Setup")
    print("=" * 60)
    
    # Test basic logging
    logger = setup_logging(level=logging.DEBUG, json_format=False)
    
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.exception("An exception occurred")
    
    # Test JSON formatting
    print("\n" + "=" * 60)
    print("Testing JSON Logging")
    print("=" * 60)
    
    json_logger = setup_logging(level=logging.INFO, json_format=True)
    json_logger.info("JSON formatted message", extra={"custom_field": "test_value"})
    
    print("\n✓ Logging tests complete!")
    print(f"Check outputs/logs/ for log files")
