"""Structured logging configuration for SDA.

Development: Pretty console output with rich
Production: JSON output for log aggregation
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence


def setup_logging(
    level: int | str = logging.INFO,
    json_logs: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (e.g., logging.INFO, logging.DEBUG, "INFO", "DEBUG")
        json_logs: If True, output JSON format (for production/log aggregation)
        log_file: Optional file path to write logs to
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Shared processors for all outputs
    shared_processors: Sequence[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        # Production: JSON output
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Pretty console output with rich
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard logging for libraries that use it
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # Optionally add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        A bound structlog logger

    Example:
        >>> from sda.logging import get_logger
        >>> log = get_logger(__name__)
        >>> log.info("training_started", epoch=1, lr=0.001)
    """
    return structlog.get_logger(name)


# Convenience function for quick setup
def quick_setup(debug: bool = False, json: bool = False) -> None:
    """Quick logging setup for scripts.

    Args:
        debug: Enable debug level logging
        json: Use JSON format (useful for log aggregation)

    Example:
        >>> from sda.logging import quick_setup, get_logger
        >>> quick_setup(debug=True)
        >>> log = get_logger(__name__)
        >>> log.debug("this will be shown")
    """
    level = logging.DEBUG if debug else logging.INFO
    setup_logging(level=level, json_logs=json)
