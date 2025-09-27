"""
Advanced Model Manager for Q2 Platform.

Extends the basic ModelManager with enterprise features including:
- Model versioning and lifecycle management
- Multi-tenant model isolation
- Advanced caching and resource optimization
- Integration with monitoring and A/B testing
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import logging

try:
    from shared.error_handling import Q2Exception, with_retry
    from shared.observability import get_logger, get_tracer
    logger = get_logger(__name__)
    tracer = get_tracer(__name__)
except ImportError:
    # Fallback for testing
    logger = logging.getLogger(__name__)
    tracer = None
    
    class Q2Exception(Exception):
        pass
    
    def with_retry(max_attempts=3, delay=1.0):
        def decorator(func):
            return func
        return decorator


class ModelStatus(Enum):
    """Model lifecycle status."""
    LOADING = "loading"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    requests_count: int = 0
    average_latency: float = 0.0
    error_rate: float = 0.0
    memory_usage: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    version: str
    tenant_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class ModelInfo:
    """Complete model information."""
    config: ModelConfig
    status: ModelStatus
    metrics: ModelMetrics
    model_instance: Any = None
    tokenizer_instance: Any = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelNotFoundError(Q2Exception):
    """Raised when a requested model is not found."""
    pass


class ModelLoadError(Q2Exception):
    """Raised when model loading fails."""
    pass


class AdvancedModelManager:
    """
    Advanced model management with enterprise features.
    
    Features:
    - Multi-tenant model isolation
    - Model versioning and lifecycle management
    - Performance monitoring and metrics
    - Resource optimization and caching
    - Integration with A/B testing framework
    """
    
    def __init__(self, max_cache_size: int = 10, enable_monitoring: bool = True):
        self._models: Dict[str, ModelInfo] = {}
        self._tenant_models: Dict[str, Set[str]] = {}
        self._max_cache_size = max_cache_size
        self._enable_monitoring = enable_monitoring
        self._lock = asyncio.Lock()
        
    def _get_model_key(self, name: str, version: str, tenant_id: Optional[str] = None) -> str:
        """Generate unique model key."""
        if tenant_id:
            return f"{tenant_id}:{name}:{version}"
        return f"{name}:{version}"
    
    async def register_model(self, config: ModelConfig) -> str:
        """Register a new model configuration."""
        async with self._lock:
            model_key = self._get_model_key(config.name, config.version, config.tenant_id)
            
            if model_key in self._models:
                logger.warning(f"Model {model_key} already registered, updating configuration")
            
            model_info = ModelInfo(
                config=config,
                status=ModelStatus.INACTIVE,
                metrics=ModelMetrics(),
            )
            
            self._models[model_key] = model_info
            
            # Track tenant models
            if config.tenant_id:
                if config.tenant_id not in self._tenant_models:
                    self._tenant_models[config.tenant_id] = set()
                self._tenant_models[config.tenant_id].add(model_key)
            
            logger.info(f"Registered model: {model_key}")
            return model_key
    
    @with_retry(max_attempts=3, delay=1.0)
    async def load_model(self, name: str, version: str, tenant_id: Optional[str] = None) -> None:
        """Load a model into memory with retry logic."""
        model_key = self._get_model_key(name, version, tenant_id)
        
        async with self._lock:
            if model_key not in self._models:
                raise ModelNotFoundError(f"Model {model_key} not registered")
            
            model_info = self._models[model_key]
            
            if model_info.status == ModelStatus.ACTIVE:
                logger.info(f"Model {model_key} already loaded")
                return
            
            if model_info.status == ModelStatus.LOADING:
                logger.info(f"Model {model_key} is already being loaded")
                return
            
            # Check cache size and evict if necessary
            await self._manage_cache()
            
            model_info.status = ModelStatus.LOADING
            
        try:
            with tracer.start_as_current_span("load_model") as span:
                span.set_attribute("model.name", name)
                span.set_attribute("model.version", version)
                span.set_attribute("model.tenant_id", tenant_id or "global")
                
                # Simulate model loading (replace with actual implementation)
                await asyncio.sleep(0.1)  # Simulate async loading
                
                # In real implementation, this would load from HuggingFace, local storage, etc.
                model_info.model_instance = f"mock_model_{model_key}"
                model_info.tokenizer_instance = f"mock_tokenizer_{model_key}"
                
                async with self._lock:
                    model_info.status = ModelStatus.ACTIVE
                    model_info.last_accessed = datetime.now(timezone.utc)
                
                logger.info(f"Successfully loaded model: {model_key}")
                
        except Exception as e:
            async with self._lock:
                model_info.status = ModelStatus.FAILED
            logger.error(f"Failed to load model {model_key}: {e}")
            raise ModelLoadError(f"Failed to load model {model_key}: {e}")
    
    async def get_model(self, name: str, version: str, tenant_id: Optional[str] = None) -> Tuple[Any, Any]:
        """Get model and tokenizer instances."""
        model_key = self._get_model_key(name, version, tenant_id)
        
        async with self._lock:
            if model_key not in self._models:
                raise ModelNotFoundError(f"Model {model_key} not found")
            
            model_info = self._models[model_key]
            
            if model_info.status != ModelStatus.ACTIVE:
                # Auto-load if not active
                await self.load_model(name, version, tenant_id)
            
            # Update access time and metrics
            model_info.last_accessed = datetime.now(timezone.utc)
            model_info.metrics.requests_count += 1
            
            return model_info.model_instance, model_info.tokenizer_instance
    
    async def unload_model(self, name: str, version: str, tenant_id: Optional[str] = None) -> None:
        """Unload a model from memory."""
        model_key = self._get_model_key(name, version, tenant_id)
        
        async with self._lock:
            if model_key not in self._models:
                logger.warning(f"Model {model_key} not found for unloading")
                return
            
            model_info = self._models[model_key]
            model_info.status = ModelStatus.INACTIVE
            model_info.model_instance = None
            model_info.tokenizer_instance = None
            
            logger.info(f"Unloaded model: {model_key}")
    
    async def get_model_info(self, name: str, version: str, tenant_id: Optional[str] = None) -> ModelInfo:
        """Get complete model information."""
        model_key = self._get_model_key(name, version, tenant_id)
        
        async with self._lock:
            if model_key not in self._models:
                raise ModelNotFoundError(f"Model {model_key} not found")
            
            return self._models[model_key]
    
    async def list_models(self, tenant_id: Optional[str] = None) -> List[ModelInfo]:
        """List all models, optionally filtered by tenant."""
        async with self._lock:
            if tenant_id:
                if tenant_id not in self._tenant_models:
                    return []
                return [self._models[key] for key in self._tenant_models[tenant_id]]
            
            return list(self._models.values())
    
    async def update_metrics(self, name: str, version: str, tenant_id: Optional[str] = None, 
                           latency: Optional[float] = None, error: bool = False) -> None:
        """Update model performance metrics."""
        if not self._enable_monitoring:
            return
            
        model_key = self._get_model_key(name, version, tenant_id)
        
        async with self._lock:
            if model_key not in self._models:
                return
            
            model_info = self._models[model_key]
            metrics = model_info.metrics
            
            if latency is not None:
                # Update average latency using exponential moving average
                alpha = 0.1
                metrics.average_latency = (1 - alpha) * metrics.average_latency + alpha * latency
            
            if error:
                # Update error rate
                total_requests = metrics.requests_count
                error_count = metrics.error_rate * total_requests + 1
                metrics.error_rate = error_count / total_requests if total_requests > 0 else 1.0
            
            metrics.last_updated = datetime.now(timezone.utc)
    
    async def _manage_cache(self) -> None:
        """Manage model cache size by evicting least recently used models."""
        active_models = [
            (key, info) for key, info in self._models.items()
            if info.status == ModelStatus.ACTIVE
        ]
        
        if len(active_models) >= self._max_cache_size:
            # Sort by last accessed time and evict oldest
            active_models.sort(key=lambda x: x[1].last_accessed)
            
            models_to_evict = active_models[:len(active_models) - self._max_cache_size + 1]
            
            for model_key, model_info in models_to_evict:
                logger.info(f"Evicting model from cache: {model_key}")
                model_info.status = ModelStatus.INACTIVE
                model_info.model_instance = None
                model_info.tokenizer_instance = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all active models."""
        async with self._lock:
            active_count = sum(1 for info in self._models.values() if info.status == ModelStatus.ACTIVE)
            failed_count = sum(1 for info in self._models.values() if info.status == ModelStatus.FAILED)
            
            return {
                "total_models": len(self._models),
                "active_models": active_count,
                "failed_models": failed_count,
                "cache_utilization": active_count / self._max_cache_size if self._max_cache_size > 0 else 0,
                "tenants": len(self._tenant_models),
            }


# Global instance
model_manager = AdvancedModelManager()