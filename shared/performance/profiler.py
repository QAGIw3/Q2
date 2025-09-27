"""
Performance profiling utilities for identifying bottlenecks.
"""

import cProfile
import pstats
import io
import time
import logging
import functools
from contextlib import contextmanager
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Profile execution results"""
    function_name: str
    total_time: float
    call_count: int
    per_call_time: float
    cumulative_time: float
    stats_output: str


class ProfilerContext:
    """Context manager for profiling code blocks"""
    
    def __init__(self, name: str = "profile", enable: bool = True):
        self.name = name
        self.enable = enable
        self.profiler: Optional[cProfile.Profile] = None
        self.start_time: float = 0
        
    def __enter__(self):
        if not self.enable:
            return self
            
        self.profiler = cProfile.Profile()
        self.start_time = time.time()
        self.profiler.enable()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable or not self.profiler:
            return
            
        self.profiler.disable()
        end_time = time.time()
        
        # Generate profile report
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        total_time = end_time - self.start_time
        logger.info(f"Profile '{self.name}' completed in {total_time:.4f}s")
        logger.debug(f"Profile details:\n{s.getvalue()}")
        
    def get_stats(self) -> Optional[str]:
        """Get profile statistics as string"""
        if not self.profiler:
            return None
            
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()
        return s.getvalue()


def profile_function(
    name: Optional[str] = None,
    enable: bool = True,
    log_results: bool = True
) -> Callable:
    """
    Decorator to profile function execution
    
    Args:
        name: Custom name for the profile (defaults to function name)
        enable: Whether profiling is enabled
        log_results: Whether to log profile results
    """
    def decorator(func: Callable) -> Callable:
        profile_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enable:
                return func(*args, **kwargs)
                
            with ProfilerContext(profile_name, enable=enable) as profiler:
                result = func(*args, **kwargs)
                
            if log_results and profiler.profiler:
                stats_output = profiler.get_stats()
                logger.debug(f"Profile results for {profile_name}:\n{stats_output}")
                
            return result
            
        return wrapper
    return decorator


@contextmanager
def memory_profiler(name: str = "memory_profile"):
    """Context manager for memory profiling using tracemalloc"""
    try:
        import tracemalloc
        tracemalloc.start()
        
        # Take snapshot before
        snapshot1 = tracemalloc.take_snapshot()
        start_time = time.time()
        
        yield
        
        # Take snapshot after
        snapshot2 = tracemalloc.take_snapshot()
        end_time = time.time()
        
        # Calculate memory diff
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        total_memory = sum(stat.size_diff for stat in top_stats)
        duration = end_time - start_time
        
        logger.info(f"Memory profile '{name}' - Duration: {duration:.4f}s, Memory diff: {total_memory / 1024 / 1024:.2f}MB")
        
        # Log top memory allocations
        logger.debug(f"Top memory allocations for {name}:")
        for stat in top_stats[:10]:
            logger.debug(f"  {stat}")
            
    except ImportError:
        logger.warning("tracemalloc not available for memory profiling")
        yield
    except Exception as e:
        logger.error(f"Error in memory profiling: {e}")
        yield
    finally:
        try:
            tracemalloc.stop()
        except:
            pass


class PerformanceProfiler:
    """
    Comprehensive performance profiler combining CPU and memory profiling
    """
    
    def __init__(self, name: str):
        self.name = name
        self.cpu_profiler: Optional[cProfile.Profile] = None
        self.start_time: float = 0
        self.memory_snapshots: list = []
        self.enabled = True
        
    def start(self):
        """Start profiling"""
        if not self.enabled:
            return
            
        self.start_time = time.time()
        
        # Start CPU profiling
        self.cpu_profiler = cProfile.Profile()
        self.cpu_profiler.enable()
        
        # Start memory tracking
        try:
            import tracemalloc
            tracemalloc.start()
            self.memory_snapshots.append(tracemalloc.take_snapshot())
        except ImportError:
            logger.warning("tracemalloc not available for memory profiling")
            
    def stop(self) -> Dict[str, Any]:
        """Stop profiling and return results"""
        if not self.enabled:
            return {}
            
        end_time = time.time()
        total_time = end_time - self.start_time
        
        results = {
            "name": self.name,
            "total_time": total_time
        }
        
        # Stop CPU profiling
        if self.cpu_profiler:
            self.cpu_profiler.disable()
            
            s = io.StringIO()
            ps = pstats.Stats(self.cpu_profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)
            
            results["cpu_profile"] = s.getvalue()
            
        # Stop memory tracking
        try:
            import tracemalloc
            if self.memory_snapshots:
                final_snapshot = tracemalloc.take_snapshot()
                top_stats = final_snapshot.compare_to(self.memory_snapshots[0], 'lineno')
                
                total_memory_diff = sum(stat.size_diff for stat in top_stats)
                results["memory_diff_mb"] = total_memory_diff / 1024 / 1024
                results["memory_stats"] = [str(stat) for stat in top_stats[:5]]
                
            tracemalloc.stop()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error collecting memory stats: {e}")
            
        logger.info(f"Performance profile '{self.name}' completed:")
        logger.info(f"  Total time: {total_time:.4f}s")
        if "memory_diff_mb" in results:
            logger.info(f"  Memory diff: {results['memory_diff_mb']:.2f}MB")
            
        return results


# Convenience decorators and functions
def quick_profile(func: Callable) -> Callable:
    """Quick profiling decorator for development"""
    return profile_function(enable=True, log_results=True)(func)


def production_profile(func: Callable) -> Callable:
    """Production-safe profiling decorator (disabled by default)"""
    import os
    enable_profiling = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
    return profile_function(enable=enable_profiling, log_results=False)(func)
