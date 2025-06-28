"""Error handling utilities with retry logic and logging."""

import time
import logging
import os
from functools import wraps
from typing import Callable, Any
from app.services.utils.exceptions import APIError, FileParsingError

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/screening.log', mode='a') if os.path.exists('logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except APIError as e:
                    last_exception = e
                    
                    # Don't retry on certain errors
                    if e.status_code and e.status_code in [400, 401, 403]:
                        logger.error(f"Non-retryable API error: {e}")
                        raise
                    
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        if e.retry_after:
                            delay = max(delay, e.retry_after)
                        
                        logger.warning(f"API error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay}s")
                        logger.debug(f"Full error details: {type(e).__name__}: {str(e)}")
                        time.sleep(delay)
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded for API call. Final error: {e}")
                        logger.debug(f"Function: {func.__name__}, Args: {args}, Kwargs: {kwargs}")
                        raise
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.warning(f"Error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded")
                        raise
            
            raise last_exception
        return wrapper
    return decorator

def safe_json_parse(json_string: str, fallback_value: dict = None) -> dict:
    """Safely parse JSON with fallback value."""
    import json
    
    if fallback_value is None:
        fallback_value = {"decision": "error", "reasoning": "Invalid JSON response"}
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return fallback_value

def handle_file_parsing_error(file_path: str, error: Exception) -> dict:
    """Handle file parsing errors with detailed logging."""
    error_msg = f"Failed to parse {file_path}: {error}"
    logger.error(error_msg)
    
    return {
        "error": True,
        "message": error_msg,
        "file_path": file_path,
        "exception_type": type(error).__name__
    }

def validate_api_response(response: dict, required_fields: list = None) -> bool:
    """Validate API response structure."""
    if required_fields is None:
        required_fields = ["decision", "reasoning"]
    
    if not isinstance(response, dict):
        return False
    
    return all(field in response for field in required_fields)

def setup_screening_logger(name: str) -> logging.Logger:
    """Set up a logger specifically for screening operations."""
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if logs directory exists)
    if os.path.exists('logs'):
        file_handler = logging.FileHandler('logs/screening_detailed.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger