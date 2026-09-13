class DiagAgentError(RuntimeError):
    """Base class for explicit trajectory failures."""


class ContractError(DiagAgentError):
    """Frozen contract or schema violation."""


class DataCoverageError(DiagAgentError):
    """Required source data or aligned dates are unavailable."""


class IsolationError(DiagAgentError):
    """A test-only path or value crossed into adaptive execution."""


class ResumeError(DiagAgentError):
    """Persisted trajectory state is inconsistent or stale."""


class ExternalServiceError(DiagAgentError):
    """The selected Agent provider could not satisfy a required call."""

