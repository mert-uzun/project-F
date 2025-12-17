"""
Structured Logging with Rich.

Provides consistent, colorful logging across the application.
"""

import logging
from functools import lru_cache

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger with Rich handler.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    console = Console(stderr=True)

    logging.basicConfig(
        level=level,
        format="%(name)s | %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_time=True,
                show_path=False,
            )
        ],
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


@lru_cache(maxsize=128)
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    Uses lru_cache to avoid creating duplicate loggers.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for structured logging with extra context.

    Usage:
        with LogContext(logger, document_id="doc123"):
            logger.info("Processing document")
    """

    def __init__(self, logger: logging.Logger, **context: str | int | float) -> None:
        self.logger = logger
        self.context = context
        self._old_factory: logging.LogRecordFactory | None = None

    def __enter__(self) -> "LogContext":
        self._old_factory = logging.getLogRecordFactory()
        context = self.context

        def factory(*args, **kwargs) -> logging.LogRecord:  # type: ignore[no-untyped-def]
            record = self._old_factory(*args, **kwargs)  # type: ignore[misc]
            for key, value in context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(factory)
        return self

    def __exit__(self, *args: object) -> None:
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)
