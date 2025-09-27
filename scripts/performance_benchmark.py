#!/usr/bin/env python3
"""
Performance benchmark script for Q2 Platform optimizations.

This script benchmarks the performance improvements from:
- Connection pooling
- Caching implementations
- Memory optimization
- Concurrent processing
"""

import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our performance modules
from shared.performance.cache import LRUCache, TTLCache, MultiLevelCache, memoize
from shared.performance.metrics import get_metrics_collector, measure_time
from shared.performance.profiler import ProfilerContext, profile_function

# Global metrics collector
metrics = get_metrics_collector()


def benchmark_cache_performance():
    """Benchmark different cache implementations"""
    print("=" * 60)
    print("CACHE PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    cache_configs = [
        ("LRU Cache (100)", LRUCache(max_size=100)),
        ("LRU Cache (1000)", LRUCache(max_size=1000)),
        ("TTL Cache (100, 60s)", TTLCache(max_size=100, default_ttl=60)),
        ("Multi-Level Cache", MultiLevelCache(l1_size=50, l2_size=500)),
    ]
    
    operations = 10000
    
    for name, cache in cache_configs:
        print(f"\nTesting {name}:")
        
        # Warm up cache
        for i in range(cache.max_size // 2):
            cache.put(f"key-{i}", f"value-{i}" * 10)
            
        start_time = time.time()
        
        # Mixed operations (70% reads, 30% writes)
        for i in range(operations):
            if i % 10 < 3:  # 30% writes
                cache.put(f"key-{i % 1000}", f"value-{i}")
            else:  # 70% reads
                cache.get(f"key-{i % 1000}")
                
        end_time = time.time()
        duration = end_time - start_time
        ops_per_second = operations / duration
        
        print(f"  Operations: {operations}")
        print(f"  Duration: {duration:.4f}s")
        print(f"  Throughput: {ops_per_second:.0f} ops/sec")
        print(f"  Hit rate: {cache.stats.hit_rate:.1f}%")
        print(f"  Cache size: {cache.size()}")


def benchmark_memoization():
    """Benchmark memoization performance"""
    print("\n" + "=" * 60)
    print("MEMOIZATION BENCHMARKS")
    print("=" * 60)
    
    # Test expensive function caching
    compute_count = 0
    
    @memoize()
    def expensive_computation(n):
        nonlocal compute_count
        compute_count += 1
        # Simulate expensive computation
        result = sum(i * i for i in range(min(n, 1000)))
        return result
    
    print("\nTesting function memoization:")
    
    # First run (cache misses)
    compute_count = 0
    start_time = time.time()
    
    results = []
    for i in range(100):
        results.append(expensive_computation(i % 20))  # Reuse some inputs
        
    end_time = time.time()
    first_duration = end_time - start_time
    first_compute_count = compute_count
    
    # Second run (cache hits)
    compute_count = 0
    start_time = time.time()
    
    results2 = []
    for i in range(100):
        results2.append(expensive_computation(i % 20))
        
    end_time = time.time()
    second_duration = end_time - start_time
    second_compute_count = compute_count
    
    print(f"  First run (cold cache): {first_duration:.4f}s, {first_compute_count} computations")
    print(f"  Second run (warm cache): {second_duration:.4f}s, {second_compute_count} computations")
    print(f"  Speedup: {first_duration / second_duration:.1f}x")
    print(f"  Results match: {results == results2}")


def benchmark_concurrent_performance():
    """Benchmark concurrent cache access"""
    print("\n" + "=" * 60)
    print("CONCURRENT ACCESS BENCHMARKS")
    print("=" * 60)
    
    cache = LRUCache(max_size=1000)
    
    # Populate cache
    for i in range(500):
        cache.put(f"key-{i}", f"value-{i}")
        
    def worker(worker_id):
        """Worker function for concurrent testing"""
        for i in range(1000):
            key = f"key-{i % 100}"
            if i % 4 == 0:
                cache.put(f"{key}-{worker_id}", f"value-{i}-{worker_id}")
            else:
                cache.get(key)
                
    print("\nTesting concurrent cache access:")
    
    thread_counts = [1, 2, 4, 8, 16]
    
    for num_threads in thread_counts:
        cache.stats.reset()
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker, i) 
                for i in range(num_threads)
            ]
            for future in futures:
                future.result()
                
        end_time = time.time()
        duration = end_time - start_time
        
        total_ops = num_threads * 1000
        ops_per_second = total_ops / duration
        
        print(f"  {num_threads:2d} threads: {duration:.4f}s, {ops_per_second:.0f} ops/sec, "
              f"hit rate: {cache.stats.hit_rate:.1f}%")


@profile_function()
def simulate_react_loop():
    """Simulate ReAct loop operations for profiling"""
    cache = LRUCache(max_size=100)
    
    # Simulate memory lookups
    for i in range(50):
        cache.put(f"memory-{i}", f"memory-data-{i}")
        
    # Simulate tool executions with caching
    tool_cache = LRUCache(max_size=200)
    
    for turn in range(5):
        # LLM call simulation
        time.sleep(0.001)  # Simulate network delay
        
        # Tool execution with caching
        tool_key = f"tool-search-{turn % 3}"
        result = tool_cache.get(tool_key)
        
        if result is None:
            # Simulate tool execution
            time.sleep(0.002)
            result = f"tool-result-{turn}"
            tool_cache.put(tool_key, result)
            
        # Memory retrieval
        memory_result = cache.get(f"memory-{turn % 10}")
        
    return "Final answer"


def benchmark_react_loop_simulation():
    """Benchmark simulated ReAct loop performance"""
    print("\n" + "=" * 60)
    print("REACT LOOP SIMULATION BENCHMARK")
    print("=" * 60)
    
    iterations = 100
    
    print(f"\nRunning {iterations} simulated ReAct loops:")
    
    start_time = time.time()
    
    results = []
    for i in range(iterations):
        result = simulate_react_loop()
        results.append(result)
        
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"  Total iterations: {iterations}")
    print(f"  Total duration: {duration:.4f}s")
    print(f"  Average per loop: {duration / iterations * 1000:.2f}ms")
    print(f"  Loops per second: {iterations / duration:.1f}")
    
    # Show profiling results
    print(f"\nAll {iterations} loops completed successfully")


def benchmark_memory_usage():
    """Benchmark memory usage of cache implementations"""
    print("\n" + "=" * 60)
    print("MEMORY USAGE BENCHMARKS")
    print("=" * 60)
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"\nBaseline memory usage: {baseline_memory:.1f} MB")
        
        # Test large cache memory usage
        cache = LRUCache(max_size=10000)
        
        # Fill cache with data
        for i in range(10000):
            cache.put(f"key-{i}", "x" * 1000)  # 1KB values
            
        filled_memory = process.memory_info().rss / 1024 / 1024  # MB
        cache_memory = filled_memory - baseline_memory
        
        print(f"Memory with 10k cached items: {filled_memory:.1f} MB")
        print(f"Cache memory overhead: {cache_memory:.1f} MB")
        print(f"Memory per cached item: {cache_memory / 10000 * 1024:.1f} KB")
        
        # Test cache cleanup
        cache.clear()
        cleared_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Memory after cache clear: {cleared_memory:.1f} MB")
        print(f"Memory recovered: {filled_memory - cleared_memory:.1f} MB")
        
    except ImportError:
        print("\npsutil not available - skipping memory benchmarks")


def main():
    """Run all benchmarks"""
    print("Q2 Platform Performance Benchmarks")
    print("Starting benchmark suite...")
    
    start_time = time.time()
    
    try:
        benchmark_cache_performance()
        benchmark_memoization()
        benchmark_concurrent_performance()
        benchmark_react_loop_simulation()
        benchmark_memory_usage()
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
        
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total benchmark duration: {total_duration:.2f}s")
    print("All benchmarks completed successfully!")
    
    # Show global metrics if available
    try:
        summary = metrics.get_summary()
        if summary:
            print(f"\nMetrics collected:")
            print(f"  Total metrics: {summary.get('total_metrics', 0)}")
            print(f"  Timing metrics: {summary.get('timing_metrics_count', 0)}")
    except Exception:
        pass
        
    return 0


if __name__ == "__main__":
    exit(main())