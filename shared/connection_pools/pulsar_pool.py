"""
High-performance connection pool for Apache Pulsar.
Provides connection reuse, batching, and circuit breaker patterns.
"""

import logging
import threading
from typing import Dict, Optional, List, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import time
import os
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Soft dependency on pulsar - we'll handle import errors gracefully
try:
    import pulsar
    PULSAR_AVAILABLE = True
except ImportError:
    PULSAR_AVAILABLE = False
    logger.warning("Pulsar client not available - connection pooling disabled")


@dataclass
class PulsarPoolConfig:
    """Configuration for Pulsar connection pool"""
    service_url: str = "pulsar://pulsar:6650"
    max_connections: int = 10
    min_connections: int = 2
    connection_timeout_ms: int = 30000
    operation_timeout_ms: int = 30000
    max_retries: int = 3
    batch_size: int = 100
    batch_timeout_ms: int = 1000
    enable_batching: bool = True
    enable_compression: bool = True
    producer_cache_size: int = 50
    consumer_cache_size: int = 20


class PulsarConnectionPool:
    """
    High-performance connection pool for Apache Pulsar with:
    - Connection reuse and pooling
    - Producer/consumer caching
    - Message batching for throughput
    - Circuit breaker pattern
    - Automatic reconnection
    """
    
    def __init__(self, config: PulsarPoolConfig):
        if not PULSAR_AVAILABLE:
            raise ImportError("Pulsar client not available. Install with: pip install pulsar-client")
            
        self.config = config
        self._client_pool: List[pulsar.Client] = []
        self._producer_cache: Dict[str, pulsar.Producer] = {}
        self._consumer_cache: Dict[str, pulsar.Consumer] = {}
        self._lock = threading.RLock()
        self._thread_pool = ThreadPoolExecutor(max_workers=config.max_connections)
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 60  # seconds
        self._message_batch: Dict[str, List[bytes]] = {}
        self._batch_lock = threading.Lock()
        self._batch_timers: Dict[str, threading.Timer] = {}
        
        # Initialize connection pool
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections"""
        logger.info(f"Initializing Pulsar connection pool with {self.config.min_connections} connections")
        
        for _ in range(self.config.min_connections):
            try:
                client = self._create_client()
                self._client_pool.append(client)
            except Exception as e:
                logger.error(f"Failed to create initial Pulsar connection: {e}")
                
    def _create_client(self) -> pulsar.Client:
        """Create a new Pulsar client with optimized settings"""
        return pulsar.Client(
            self.config.service_url,
            connection_timeout_millis=self.config.connection_timeout_ms,
            operation_timeout_seconds=self.config.operation_timeout_ms // 1000,
            io_threads=2,
            message_listener_threads=4,
            concurrent_lookup_requests=50000,
            use_tls=False
        )
        
    def _get_client(self) -> pulsar.Client:
        """Get an available client from the pool"""
        with self._lock:
            if self._is_circuit_breaker_open():
                raise Exception("Circuit breaker is open")
                
            if self._client_pool:
                return self._client_pool.pop(0)
            elif len(self._client_pool) < self.config.max_connections:
                return self._create_client()
            else:
                # Wait for a connection to become available
                time.sleep(0.01)
                return self._get_client()
                
    def _return_client(self, client: pulsar.Client):
        """Return a client to the pool"""
        with self._lock:
            if len(self._client_pool) < self.config.max_connections:
                self._client_pool.append(client)
            else:
                client.close()
                
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            if time.time() - self._circuit_breaker_last_failure < self._circuit_breaker_timeout:
                return True
            else:
                # Reset circuit breaker
                self._circuit_breaker_failures = 0
        return False
        
    def _record_failure(self):
        """Record a failure for circuit breaker"""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = time.time()
        
    def _record_success(self):
        """Record a success - reset circuit breaker if it was failing"""
        if self._circuit_breaker_failures > 0:
            self._circuit_breaker_failures = max(0, self._circuit_breaker_failures - 1)
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_producer(self, topic: str, **kwargs) -> pulsar.Producer:
        """
        Get a cached producer for a topic with retry logic
        
        Args:
            topic: Pulsar topic name
            **kwargs: Additional producer configuration
        """
        cache_key = f"{topic}:{hash(str(sorted(kwargs.items())))}"
        
        with self._lock:
            if cache_key in self._producer_cache:
                return self._producer_cache[cache_key]
                
            if len(self._producer_cache) >= self.config.producer_cache_size:
                # Remove oldest producer
                oldest_key = next(iter(self._producer_cache))
                old_producer = self._producer_cache.pop(oldest_key)
                old_producer.close()
                
        try:
            client = self._get_client()
            
            producer_config = {
                'topic': topic,
                'batching_enabled': self.config.enable_batching,
                'batching_max_messages': self.config.batch_size,
                'batching_max_allowed_size_in_bytes': 128 * 1024,  # 128KB
                'batching_max_publish_delay_millis': self.config.batch_timeout_ms,
                'compression_type': pulsar.CompressionType.LZ4 if self.config.enable_compression else pulsar.CompressionType.NONE,
                'send_timeout_millis': self.config.operation_timeout_ms,
                'max_pending_messages': 10000,
                **kwargs
            }
            
            producer = client.create_producer(**producer_config)
            
            with self._lock:
                self._producer_cache[cache_key] = producer
                
            self._return_client(client)
            self._record_success()
            return producer
            
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to create producer for topic {topic}: {e}")
            raise
            
    def send_async(self, topic: str, message: bytes, callback: Optional[Callable] = None):
        """
        Send message asynchronously with batching support
        
        Args:
            topic: Pulsar topic
            message: Message bytes
            callback: Optional callback function
        """
        try:
            producer = self.get_producer(topic)
            producer.send_async(message, callback=callback)
                
        except Exception as e:
            logger.error(f"Failed to send message to topic {topic}: {e}")
            if callback:
                callback(None, e)
            
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the connection pool"""
        with self._lock:
            return {
                "total_connections": len(self._client_pool),
                "cached_producers": len(self._producer_cache),
                "cached_consumers": len(self._consumer_cache),
                "circuit_breaker_failures": self._circuit_breaker_failures,
                "circuit_breaker_open": self._is_circuit_breaker_open(),
                "pending_batches": sum(len(batch) for batch in self._message_batch.values())
            }
            
    def close(self):
        """Close all connections and clean up resources"""
        logger.info("Closing Pulsar connection pool")
        
        # Close all producers
        for producer in self._producer_cache.values():
            try:
                producer.close()
            except Exception as e:
                logger.warning(f"Error closing producer: {e}")
                
        # Close all consumers
        for consumer in self._consumer_cache.values():
            try:
                consumer.close()
            except Exception as e:
                logger.warning(f"Error closing consumer: {e}")
                
        # Close all clients
        for client in self._client_pool:
            try:
                client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
                
        # Cancel batch timers
        for timer in self._batch_timers.values():
            timer.cancel()
            
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        logger.info("Pulsar connection pool closed")


# Global instance
_pulsar_pool: Optional[PulsarConnectionPool] = None


def get_pulsar_pool(config: Optional[PulsarPoolConfig] = None) -> PulsarConnectionPool:
    """Get or create the global Pulsar connection pool"""
    global _pulsar_pool
    if _pulsar_pool is None:
        if config is None:
            config = PulsarPoolConfig()
            # Override with environment variables
            config.service_url = os.getenv("PULSAR_SERVICE_URL", config.service_url)
        _pulsar_pool = PulsarConnectionPool(config)
    return _pulsar_pool


def close_pulsar_pool():
    """Close the global Pulsar connection pool"""
    global _pulsar_pool
    if _pulsar_pool:
        _pulsar_pool.close()
        _pulsar_pool = None
