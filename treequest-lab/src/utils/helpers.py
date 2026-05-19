"""
Helper utilities for TreeQuest Lab.

Common utility functions used throughout the project.
"""

import re
import time
from typing import Any


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def sanitize_text(text: str, max_length: int = 200) -> str:
    """
    Sanitize text for use in filenames or safe display.
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Sanitized text
    """
    # Remove special characters
    sanitized = re.sub(r'[^\w\s\-\.]', '', text)
    
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Strip and truncate
    sanitized = sanitized.strip()[:max_length]
    
    return sanitized


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Input text
        max_length: Maximum length (including suffix)
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    Retry function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential: Use exponential backoff
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorator function
        
    Example:
        @retry_with_backoff(max_retries=3)
        def api_call():
            ...
    """
    import functools
    import random
    
    def decorator(func_to_decorate):
        @functools.wraps(func_to_decorate)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func_to_decorate(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Calculate delay
                    if exponential:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay
                    
                    # Add jitter
                    delay = delay * (0.5 + random.random())
                    
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    
    # If called without parentheses, func is the function to decorate
    if callable(func):
        return decorator(func)
    
    return decorator


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary to merge on top
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator between keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def safe_get(obj: Any, *keys, default: Any = None) -> Any:
    """
    Safely get nested dictionary/list values.
    
    Args:
        obj: Object to traverse
        *keys: Keys to traverse
        default: Default value if not found
        
    Returns:
        Value or default
        
    Example:
        safe_get(data, 'user', 'name', default='Unknown')
    """
    current = obj
    
    for key in keys:
        try:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, (list, tuple)):
                current = current[int(key)]
            else:
                return default
            
            if current is None:
                return default
                
        except (KeyError, IndexError, ValueError, TypeError):
            return default
    
    return current


def batch_iterable(iterable, batch_size: int):
    """
    Iterate over iterable in batches.
    
    Args:
        iterable: Iterable to batch
        batch_size: Size of each batch
        
    Yields:
        Batches as lists
        
    Example:
        for batch in batch_iterable(range(10), 3):
            print(batch)  # [0,1,2], [3,4,5], [6,7,8], [9]
    """
    batch = []
    
    for item in iterable:
        batch.append(item)
        
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:
        yield batch


if __name__ == "__main__":
    # Test helper functions
    print("=" * 60)
    print("Testing Helper Functions")
    print("=" * 60)
    
    # Test format_duration
    print("\nDuration formatting:")
    for secs in [30, 125, 3725]:
        print(f"  {secs}s -> {format_duration(secs)}")
    
    # Test sanitize_text
    print("\nText sanitization:")
    test_text = "Hello! This is a test@#$% string with   spaces"
    print(f"  Original: {test_text}")
    print(f"  Sanitized: {sanitize_text(test_text)}")
    
    # Test truncate_text
    print("\nText truncation:")
    long_text = "This is a very long hypothesis that needs to be truncated for display purposes"
    print(f"  Original: {long_text}")
    print(f"  Truncated: {truncate_text(long_text, max_length=40)}")
    
    # Test deep_merge
    print("\nDeep merge:")
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 20, "e": 5}, "f": 6}
    merged = deep_merge(base, override)
    print(f"  Base: {base}")
    print(f"  Override: {override}")
    print(f"  Merged: {merged}")
    
    # Test safe_get
    print("\nSafe get:")
    data = {"user": {"name": "Alice", "age": 30}}
    print(f"  Data: {data}")
    print(f"  user.name: {safe_get(data, 'user', 'name')}")
    print(f"  user.email: {safe_get(data, 'user', 'email', default='N/A')}")
    
    # Test batch_iterable
    print("\nBatch iteration:")
    for i, batch in enumerate(batch_iterable(range(10), 3)):
        print(f"  Batch {i}: {batch}")
    
    print("\n✓ All helper tests complete!")
