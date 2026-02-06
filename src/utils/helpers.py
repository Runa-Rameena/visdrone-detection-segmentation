import os
import json
import time
from pathlib import Path
from typing import Any, Dict
from functools import wraps

def ensure_dir(directory: str):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def save_json(data: Dict, filepath: str):
    """Save dictionary to JSON file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def timer(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

class ProgressTracker:
    """Simple progress tracker"""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        
        print(f"\r{self.desc}: {self.current}/{self.total} ({percent:.1f}%) "
              f"[{elapsed:.1f}s < {eta:.1f}s]", end='', flush=True)
    
    def close(self):
        """Finish progress tracking"""
        print()  # New line

def format_bytes(bytes_size: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def get_file_size(filepath: str) -> str:
    """Get file size in human readable format"""
    size = os.path.getsize(filepath)
    return format_bytes(size)