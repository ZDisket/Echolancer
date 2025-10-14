import torch
import time
import functools
from typing import Callable, Any
from contextlib import contextmanager

class Profiler:
    """
    A simple profiler for timing model components and operations.
    """
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
    
    def time_function(self, func_name: str):
        """
        Decorator to time a function.
        
        Args:
            func_name: Name to identify the function in timing reports
            
        Returns:
            Decorated function that records execution time
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                # Update timing statistics
                elapsed = end_time - start_time
                if func_name in self.timings:
                    self.timings[func_name] += elapsed
                    self.call_counts[func_name] += 1
                else:
                    self.timings[func_name] = elapsed
                    self.call_counts[func_name] = 1
                    
                return result
            return wrapper
        return decorator
    
    @contextmanager
    def time_block(self, block_name: str):
        """
        Context manager to time a block of code.
        
        Args:
            block_name: Name to identify the code block in timing reports
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            
            # Update timing statistics
            if block_name in self.timings:
                self.timings[block_name] += elapsed
                self.call_counts[block_name] += 1
            else:
                self.timings[block_name] = elapsed
                self.call_counts[block_name] = 1
    
    def get_timing_stats(self) -> dict:
        """
        Get timing statistics.
        
        Returns:
            Dictionary with timing statistics for all tracked functions/blocks
        """
        stats = {}
        for name in self.timings:
            total_time = self.timings[name]
            call_count = self.call_counts[name]
            avg_time = total_time / call_count if call_count > 0 else 0
            stats[name] = {
                'total_time': total_time,
                'call_count': call_count,
                'avg_time': avg_time
            }
        return stats
    
    def reset(self):
        """Reset all timing statistics."""
        self.timings.clear()
        self.call_counts.clear()
    
    def print_stats(self, sort_by='total_time'):
        """
        Print formatted timing statistics.
        
        Args:
            sort_by: Column to sort by ('total_time', 'call_count', 'avg_time')
        """
        stats = self.get_timing_stats()
        if not stats:
            print("No timing data collected.")
            return
        
        # Sort by selected column
        sorted_items = sorted(stats.items(), key=lambda x: x[1][sort_by], reverse=True)
        
        # Print header
        print(f"{'Function/Block':<30} {'Total Time (s)':<15} {'Calls':<10} {'Avg Time (s)':<15}")
        print("-" * 70)
        
        # Print each entry
        for name, data in sorted_items:
            print(f"{name:<30} {data['total_time']:<15.6f} {data['call_count']:<10} {data['avg_time']:<15.6f}")

# Global profiler instance
_GLOBAL_PROFILER = None

def get_global_profiler() -> Profiler:
    """
    Get the global profiler instance, creating it if it doesn't exist.
    
    Returns:
        Global Profiler instance
    """
    global _GLOBAL_PROFILER
    if _GLOBAL_PROFILER is None:
        _GLOBAL_PROFILER = Profiler()
    return _GLOBAL_PROFILER

def profile_function(func_name: str):
    """
    Decorator to profile a function using the global profiler.
    
    Args:
        func_name: Name to identify the function in timing reports
        
    Returns:
        Decorated function that records execution time
    """
    def decorator(func: Callable) -> Callable:
        profiler = get_global_profiler()
        return profiler.time_function(func_name)(func)
    return decorator

@contextmanager
def profile_block(block_name: str):
    """
    Context manager to profile a block of code using the global profiler.
    
    Args:
        block_name: Name to identify the code block in timing reports
    """
    profiler = get_global_profiler()
    with profiler.time_block(block_name):
        yield

def print_profiling_stats(sort_by='total_time'):
    """
    Print profiling statistics from the global profiler.
    
    Args:
        sort_by: Column to sort by ('total_time', 'call_count', 'avg_time')
    """
    profiler = get_global_profiler()
    profiler.print_stats(sort_by=sort_by)

def reset_profiling():
    """Reset profiling statistics in the global profiler."""
    profiler = get_global_profiler()
    profiler.reset()