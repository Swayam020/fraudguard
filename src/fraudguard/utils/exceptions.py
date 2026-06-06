"""Custom exception hierarchy for FraudGuard.

All FraudGuard-specific failures inherit from FraudGuardError so callers
(especially the API layer) can catch the whole family with one `except`
clause while still distinguishing data vs model vs API failures.
"""


class FraudGuardError(Exception):
    """Base class for all FraudGuard exceptions.

    Inherit from this (not built-in Exception) for any project-specific
    error. The API layer catches this to convert internal errors into
    proper HTTP responses instead of crashing.
    """


class DataError(FraudGuardError):
    """Raised when something is wrong with the input data.

    Examples: missing CSV file, malformed columns, unexpected dtype,
    empty dataset after filtering.
    """


class ModelError(FraudGuardError):
    """Raised when model loading, training, or inference fails.

    Examples: checkpoint file missing, tensor shape mismatch,
    CUDA out of memory.
    """


class APIError(FraudGuardError):
    """Raised for API-layer problems before reaching business logic.

    Examples: malformed request body, missing required field,
    invalid auth token. Maps to HTTP 4xx in the FastAPI handler.
    """


class ConfigError(FraudGuardError):
    """Raised when configuration is invalid or missing.

    Examples: required env var not set, invalid path, port out of range.
    """
