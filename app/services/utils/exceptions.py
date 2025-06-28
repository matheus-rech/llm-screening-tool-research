"""Custom exceptions for the LLM screening tool."""

class ScreeningError(Exception):
    """Base exception for screening-related errors."""
    pass

class APIError(ScreeningError):
    """Exception raised for API-related errors."""
    def __init__(self, message, status_code=None, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after

class FileParsingError(ScreeningError):
    """Exception raised for file parsing errors."""
    def __init__(self, message, file_path=None, line_number=None):
        super().__init__(message)
        self.file_path = file_path
        self.line_number = line_number

class ConfigurationError(ScreeningError):
    """Exception raised for configuration-related errors."""
    pass

class DatabaseError(ScreeningError):
    """Exception raised for database-related errors."""
    pass

class ValidationError(ScreeningError):
    """Exception raised for data validation errors."""
    def __init__(self, message, field=None, value=None):
        super().__init__(message)
        self.field = field
        self.value = value