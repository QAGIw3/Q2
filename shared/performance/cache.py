"""
High-performance caching implementations with TTL, LRU, and multi-level support.
"""

import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Callable, Generic, TypeVar
import pickle
import hashlib

logger = logging.getLogger(__name__)

K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type


class CacheStats:
    """Cache statistics tracking"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size = 0
        self.max_size = 0
        
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
        
    def reset(self):
        """Reset all statistics"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
        
    def touch(self):
        """Update access time and count"""
        self.last_accessed = time.time()
        self.access_count += 1


class BaseCache(ABC, Generic[K, V]):
    """Abstract base class for cache implementations"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.stats = CacheStats()
        self.stats.max_size = max_size
        self._lock = threading.RLock()
        
    @abstractmethod
    def get(self, key: K) -> Optional[V]:
        """Get value by key"""
        pass
        
    @abstractmethod
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Put key-value pair with optional TTL"""
        pass
        
    @abstractmethod
    def delete(self, key: K) -> bool:
        """Delete key-value pair"""
        pass
        
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries"""
        pass
        
    @abstractmethod
    def size(self) -> int:
        """Get current cache size"""
        pass
        
    def contains(self, key: K) -> bool:
        """Check if key exists in cache"""
        return self.get(key) is not None


class LRUCache(BaseCache[K, V]):
    """
    Least Recently Used (LRU) cache implementation with thread safety.
    """
    
    def __init__(self, max_size: int = 1000):
        super().__init__(max_size)
        self._cache: OrderedDict[K, CacheEntry] = OrderedDict()
        
    def get(self, key: K) -> Optional[V]:
        """Get value by key, moving it to end (most recently used)"""
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None
                
            entry = self._cache[key]
            
            # Check TTL expiration
            if entry.is_expired:
                del self._cache[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
                
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self.stats.hits += 1
            return entry.value
            
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Put key-value pair, evicting least recently used if necessary"""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Update existing entry
                entry = self._cache[key]
                entry.value = value
                entry.created_at = current_time
                entry.last_accessed = current_time
                entry.ttl = ttl
                self._cache.move_to_end(key)
            else:
                # Add new entry
                entry = CacheEntry(
                    value=value,
                    created_at=current_time,
                    last_accessed=current_time,
                    ttl=ttl
                )
                
                # Evict if at capacity
                if len(self._cache) >= self.max_size:
                    self._evict_lru()
                    
                self._cache[key] = entry
                
            self.stats.size = len(self._cache)
            
    def delete(self, key: K) -> bool:
        """Delete key-value pair"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.stats.size = len(self._cache)
                return True
            return False
            
    def clear(self) -> None:
        """Clear all entries"""
        with self._lock:
            self._cache.clear()
            self.stats.size = 0
            
    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)
        
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self._cache:
            self._cache.popitem(last=False)  # Remove first (oldest) item
            self.stats.evictions += 1


class TTLCache(BaseCache[K, V]):
    """
    Time-To-Live (TTL) cache implementation with automatic expiration.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        super().__init__(max_size)
        self._cache: Dict[K, CacheEntry] = {}
        self.default_ttl = default_ttl
        self._cleanup_interval = 60  # seconds
        self._last_cleanup = time.time()
        
    def get(self, key: K) -> Optional[V]:
        """Get value by key, checking TTL expiration"""
        with self._lock:
            self._cleanup_expired()
            
            if key not in self._cache:
                self.stats.misses += 1
                return None
                
            entry = self._cache[key]
            
            if entry.is_expired:
                del self._cache[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
                
            entry.touch()
            self.stats.hits += 1
            return entry.value
            
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Put key-value pair with TTL"""
        with self._lock:
            self._cleanup_expired()
            
            current_time = time.time()
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            entry = CacheEntry(
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                ttl=effective_ttl
            )
            
            # Evict random entry if at capacity (TTL cache doesn't use LRU)
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_random()
                
            self._cache[key] = entry
            self.stats.size = len(self._cache)
            
    def delete(self, key: K) -> bool:
        """Delete key-value pair"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.stats.size = len(self._cache)
                return True
            return False
            
    def clear(self) -> None:
        """Clear all entries"""
        with self._lock:
            self._cache.clear()
            self.stats.size = 0
            
    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)
        
    def _cleanup_expired(self):
        """Clean up expired entries periodically"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
            
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            del self._cache[key]
            self.stats.evictions += 1
            
        self._last_cleanup = current_time
        self.stats.size = len(self._cache)
        
    def _evict_random(self):
        """Evict a random entry when at capacity"""
        if self._cache:
            key = next(iter(self._cache))
            del self._cache[key]
            self.stats.evictions += 1


class MultiLevelCache(BaseCache[K, V]):
    """
    Multi-level cache with L1 (LRU) and L2 (TTL) levels for optimal performance.
    """
    
    def __init__(self, l1_size: int = 100, l2_size: int = 1000, l2_ttl: float = 1800):
        super().__init__(l1_size + l2_size)
        self.l1_cache = LRUCache[K, V](l1_size)  # Fast access
        self.l2_cache = TTLCache[K, V](l2_size, l2_ttl)  # Larger capacity
        
    def get(self, key: K) -> Optional[V]:
        """Get value, checking L1 first, then L2"""
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats.hits += 1
            return value
            
        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.put(key, value)
            self.stats.hits += 1
            return value
            
        self.stats.misses += 1
        return None
        
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Put value in both L1 and L2"""
        self.l1_cache.put(key, value)
        self.l2_cache.put(key, value, ttl)
        self.stats.size = self.l1_cache.size() + self.l2_cache.size()
        
    def delete(self, key: K) -> bool:
        """Delete from both levels"""
        deleted_l1 = self.l1_cache.delete(key)
        deleted_l2 = self.l2_cache.delete(key)
        self.stats.size = self.l1_cache.size() + self.l2_cache.size()
        return deleted_l1 or deleted_l2
        
    def clear(self) -> None:
        """Clear both levels"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.stats.size = 0
        
    def size(self) -> int:
        """Get total size across both levels"""
        return self.l1_cache.size() + self.l2_cache.size()


def memoize(cache: Optional[BaseCache] = None, ttl: Optional[float] = None):
    """
    Decorator to cache function results using specified cache implementation.
    
    Args:
        cache: Cache instance to use (defaults to LRUCache)
        ttl: Time-to-live for cached results
    """
    if cache is None:
        cache = LRUCache(max_size=1000)
        
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
                
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            return result
            
        wrapper._cache = cache  # Allow access to cache for testing
        return wrapper
    return decorator


# Convenience function for creating optimized caches
def create_cache(cache_type: str = "lru", **kwargs) -> BaseCache:
    """
    Factory function to create optimized cache instances.
    
    Args:
        cache_type: Type of cache ('lru', 'ttl', 'multi')
        **kwargs: Configuration options for the cache
    """
    if cache_type == "lru":
        return LRUCache(**kwargs)
    elif cache_type == "ttl":
        return TTLCache(**kwargs)
    elif cache_type == "multi":
        return MultiLevelCache(**kwargs)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
