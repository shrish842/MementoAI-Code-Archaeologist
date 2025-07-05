class MementoAIException(Exception):
    """Base exception for MementoAI application errors."""
    pass

class ModelLoadingError(MementoAIException):
    """Raised when an AI model fails to load."""
    pass

class PineconeConnectionError(MementoAIException):
    """Raised when there's an issue connecting to Pinecone."""
    pass

class GitOperationError(MementoAIException):
    """Raised when a Git command fails."""
    pass

class InvalidInputError(MementoAIException):
    """Raised for invalid user input."""
    pass

