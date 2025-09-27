"""
Performance metrics collection and monitoring utilities.
"""

import logging
import time
import threading
import psutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
import functools

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class TimingMetric:
    """Timing metric with statistical data"""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_time(self) -> float:
        """Average execution time"""
        return self.total_time / self.count if self.count > 0 else 0.0
        
    @property
    def p95_time(self) -> float:
        """95th percentile execution time"""
        if not self.recent_times:
            return 0.0
        sorted_times = sorted(self.recent_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]
        
    def record(self, duration: float):
        """Record a new timing measurement"""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.recent_times.append(duration)


class MetricsCollector:
    """
    Thread-safe performance metrics collector with automatic system monitoring.
    """
    
    def __init__(self, collect_system_metrics: bool = True):
        self._metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self._timing_metrics: Dict[str, TimingMetric] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._collect_system_metrics = collect_system_metrics
        
        if collect_system_metrics:
            self._start_system_monitoring()
            
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Record a performance metric"""
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                unit=unit
            )
            self._metrics[name].append(metric)
            
            # Keep only recent metrics to prevent memory growth
            if len(self._metrics[name]) > 1000:
                self._metrics[name] = self._metrics[name][-500:]
                
    def record_timing(self, name: str, duration: float):
        """Record a timing metric"""
        with self._lock:
            if name not in self._timing_metrics:
                self._timing_metrics[name] = TimingMetric(name=name)
            self._timing_metrics[name].record(duration)
            
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric"""
        with self._lock:
            self._counters[name] += value
            
    def set_gauge(self, name: str, value: float):
        """Set a gauge metric"""
        with self._lock:
            self._gauges[name] = value
            
    def get_timing_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get timing statistics for a metric"""
        with self._lock:
            if name not in self._timing_metrics:
                return None
                
            metric = self._timing_metrics[name]
            return {
                "count": metric.count,
                "avg_time": metric.avg_time,
                "min_time": metric.min_time,
                "max_time": metric.max_time,
                "p95_time": metric.p95_time,
                "total_time": metric.total_time
            }
            
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        with self._lock:
            return {
                "metrics": dict(self._metrics),
                "timing_metrics": {
                    name: {
                        "count": metric.count,
                        "avg_time": metric.avg_time,
                        "min_time": metric.min_time,
                        "max_time": metric.max_time,
                        "p95_time": metric.p95_time,
                        "total_time": metric.total_time
                    }
                    for name, metric in self._timing_metrics.items()
                },
                "counters": dict(self._counters),
                "gauges": dict(self._gauges)
            }
            
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key performance metrics"""
        with self._lock:
            summary = {
                "total_metrics": sum(len(metrics) for metrics in self._metrics.values()),
                "timing_metrics_count": len(self._timing_metrics),
                "counters_count": len(self._counters),
                "gauges_count": len(self._gauges)
            }
            
            # Add top slow operations
            if self._timing_metrics:
                sorted_timings = sorted(
                    self._timing_metrics.items(),
                    key=lambda x: x[1].avg_time,
                    reverse=True
                )
                summary["slowest_operations"] = [
                    {
                        "name": name,
                        "avg_time": metric.avg_time,
                        "count": metric.count
                    }
                    for name, metric in sorted_timings[:5]
                ]
                
            return summary
            
    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._metrics.clear()
            self._timing_metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            
    def _start_system_monitoring(self):
        """Start background system metrics collection"""
        def collect_system_metrics():
            while True:
                try:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.set_gauge("system.cpu_percent", cpu_percent)
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.set_gauge("system.memory_percent", memory.percent)
                    self.set_gauge("system.memory_available_mb", memory.available / 1024 / 1024)
                    
                    # Disk I/O metrics
                    disk_io = psutil.disk_io_counters()
                    if disk_io:
                        self.set_gauge("system.disk_read_mb_per_sec", disk_io.read_bytes / 1024 / 1024)
                        self.set_gauge("system.disk_write_mb_per_sec", disk_io.write_bytes / 1024 / 1024)
                    
                    # Network I/O metrics  
                    net_io = psutil.net_io_counters()
                    if net_io:
                        self.set_gauge("system.network_sent_mb_per_sec", net_io.bytes_sent / 1024 / 1024)
                        self.set_gauge("system.network_recv_mb_per_sec", net_io.bytes_recv / 1024 / 1024)
                        
                except Exception as e:
                    logger.warning(f"Error collecting system metrics: {e}")
                    
                time.sleep(30)  # Collect every 30 seconds
                
        # Run in daemon thread
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()


@contextmanager
def measure_time(metrics: MetricsCollector, operation_name: str):
    """Context manager to measure execution time"""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        metrics.record_timing(operation_name, duration)


def timed(metrics: MetricsCollector, operation_name: Optional[str] = None):
    """Decorator to measure function execution time"""
    def decorator(func: Callable) -> Callable:
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics.record_timing(name, duration)
                
        return wrapper
    return decorator


# Global metrics collector instance
_global_metrics: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def reset_metrics():
    """Reset the global metrics collector"""
    global _global_metrics
    if _global_metrics:
        _global_metrics.reset()
