import logging
import structlog


def setup_logging(log_level: str = "INFO", environment: str = "development") -> None:
    """
    Initialize structlog. Call once at process startup.

    Args:
        log_level:   INFO / DEBUG / WARNING / ERROR
        environment: 'development' → pretty console output
                     anything else → JSON lines (production)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        level=level,
    )

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if environment == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def setup_logging_from_settings() -> None:
    """Initialise logging using values from pydantic Settings. Use inside the FastAPI app."""
    from app.core.config import get_settings
    s = get_settings()
    setup_logging(log_level=s.log_level, environment=s.environment)


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
