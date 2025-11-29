# -------------------------
# Base Exception
# -------------------------
class VeroException(Exception):
    """Base exception class for the Vero framework."""
    pass


# -------------------------
# LLM Exceptions
# -------------------------
class LLMConfigError(VeroException):
    """Raised when there is a configuration error for the LLM client (missing API key, model, or base URL)."""
    pass


class LLMCallError(VeroException):
    """Raised when the LLM API call fails."""
    pass


# -------------------------
# Agent Exceptions
# -------------------------
class AgentError(VeroException):
    """Base exception for Agent-related errors."""
    pass


class AgentExecutionError(AgentError):
    """Raised when the agent fails to execute a planned action."""
    pass


class AgentPlanningError(AgentError):
    """Raised when the agent fails to generate or follow a plan."""
    pass


# -------------------------
# Tool / Integration Exceptions
# -------------------------
class ToolError(VeroException):
    """Base exception for external tool or API call errors."""
    pass


class ToolCallError(ToolError):
    """Raised when a tool fails during execution or returns an error."""
    pass


class ToolNotFoundError(ToolError):
    """Raised when the requested tool is not available in the agent's toolkit."""
    pass
