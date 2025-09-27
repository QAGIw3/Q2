"""
A/B Testing Framework for AI Models in Q2 Platform.

Provides comprehensive A/B testing capabilities including:
- Traffic splitting and routing
- Statistical significance testing
- Performance comparison
- Automatic winner selection
"""

import asyncio
import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import logging

try:
    from shared.error_handling import Q2Exception
    from shared.observability import get_logger, get_tracer
    logger = get_logger(__name__)
    tracer = get_tracer(__name__)
except ImportError:
    # Fallback for testing
    logger = logging.getLogger(__name__)
    tracer = None
    
    class Q2Exception(Exception):
        pass


class TestStatus(Enum):
    """A/B test status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantStatus(Enum):
    """Test variant status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    WINNER = "winner"
    LOSER = "loser"


@dataclass
class TestVariant:
    """A/B test variant configuration."""
    id: str
    name: str
    model_name: str
    model_version: str
    traffic_percentage: float
    status: VariantStatus = VariantStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate variant after initialization."""
        if not 0 <= self.traffic_percentage <= 100:
            raise ValueError("Traffic percentage must be between 0 and 100")


@dataclass
class ABTestMetrics:
    """A/B test performance metrics."""
    variant_id: str
    requests_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    average_latency: float = 0.0
    conversion_rate: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update_metrics(self, latency: Optional[float] = None, success: bool = True, 
                      converted: bool = False, custom: Optional[Dict[str, float]] = None):
        """Update metrics with new data point."""
        self.requests_count += 1
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        if latency is not None:
            self.total_latency += latency
            self.average_latency = self.total_latency / self.requests_count
        
        if converted:
            # Update conversion rate
            conversion_count = self.conversion_rate * (self.requests_count - 1) + 1
            self.conversion_rate = conversion_count / self.requests_count
        
        if custom:
            for metric_name, value in custom.items():
                if metric_name not in self.custom_metrics:
                    self.custom_metrics[metric_name] = value
                else:
                    # Use exponential moving average
                    alpha = 0.1
                    self.custom_metrics[metric_name] = (
                        (1 - alpha) * self.custom_metrics[metric_name] + alpha * value
                    )
        
        self.last_updated = datetime.now(timezone.utc)


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    id: str
    name: str
    description: str
    variants: List[TestVariant]
    status: TestStatus = TestStatus.DRAFT
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    success_metric: str = "conversion_rate"  # Which metric to optimize for
    tenant_filter: Optional[Set[str]] = None  # Limit test to specific tenants
    user_filter: Optional[Dict[str, Any]] = None  # Additional user filtering
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    
    def __post_init__(self):
        """Validate test configuration."""
        if len(self.variants) < 2:
            raise ValueError("A/B test must have at least 2 variants")
        
        total_traffic = sum(v.traffic_percentage for v in self.variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Total traffic percentage must equal 100%, got {total_traffic}%")
        
        if not 0.5 <= self.confidence_level <= 0.99:
            raise ValueError("Confidence level must be between 0.5 and 0.99")


class ABTestError(Q2Exception):
    """Base exception for A/B testing errors."""
    pass


class TestNotFoundError(ABTestError):
    """Raised when a test is not found."""
    pass


class TestConfigError(ABTestError):
    """Raised when test configuration is invalid."""
    pass


class ABTestManager:
    """
    A/B Testing Manager for AI Models.
    
    Provides comprehensive A/B testing capabilities including:
    - Test creation and management
    - Traffic routing and splitting
    - Statistical analysis
    - Automatic winner selection
    """
    
    def __init__(self):
        self._tests: Dict[str, ABTestConfig] = {}
        self._metrics: Dict[str, Dict[str, ABTestMetrics]] = {}  # test_id -> {variant_id -> metrics}
        self._user_assignments: Dict[str, Dict[str, str]] = {}  # test_id -> {user_id -> variant_id}
        self._lock = asyncio.Lock()
    
    async def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test."""
        async with self._lock:
            if config.id in self._tests:
                raise TestConfigError(f"Test with id {config.id} already exists")
            
            # Validate variant models exist (this would integrate with ModelManager)
            for variant in config.variants:
                # TODO: Validate model exists in ModelManager
                pass
            
            self._tests[config.id] = config
            self._metrics[config.id] = {
                variant.id: ABTestMetrics(variant_id=variant.id)
                for variant in config.variants
            }
            self._user_assignments[config.id] = {}
            
            logger.info(f"Created A/B test: {config.name} ({config.id})")
            return config.id
    
    async def start_test(self, test_id: str) -> None:
        """Start an A/B test."""
        async with self._lock:
            if test_id not in self._tests:
                raise TestNotFoundError(f"Test {test_id} not found")
            
            test_config = self._tests[test_id]
            
            if test_config.status != TestStatus.DRAFT:
                raise TestConfigError(f"Can only start tests in DRAFT status, current: {test_config.status}")
            
            test_config.status = TestStatus.RUNNING
            test_config.start_time = datetime.now(timezone.utc)
            
            logger.info(f"Started A/B test: {test_config.name}")
    
    async def stop_test(self, test_id: str, reason: str = "manual") -> None:
        """Stop an A/B test."""
        async with self._lock:
            if test_id not in self._tests:
                raise TestNotFoundError(f"Test {test_id} not found")
            
            test_config = self._tests[test_id]
            
            if test_config.status != TestStatus.RUNNING:
                raise TestConfigError(f"Can only stop running tests, current: {test_config.status}")
            
            test_config.status = TestStatus.COMPLETED
            test_config.end_time = datetime.now(timezone.utc)
            
            logger.info(f"Stopped A/B test: {test_config.name} (reason: {reason})")
    
    async def assign_variant(self, test_id: str, user_id: str, 
                           context: Optional[Dict[str, Any]] = None) -> TestVariant:
        """Assign a user to a test variant."""
        async with self._lock:
            if test_id not in self._tests:
                raise TestNotFoundError(f"Test {test_id} not found")
            
            test_config = self._tests[test_id]
            
            if test_config.status != TestStatus.RUNNING:
                raise TestConfigError(f"Test {test_id} is not running")
            
            # Check if user is already assigned
            if user_id in self._user_assignments[test_id]:
                variant_id = self._user_assignments[test_id][user_id]
                return next(v for v in test_config.variants if v.id == variant_id)
            
            # Apply filters
            if not self._passes_filters(test_config, user_id, context):
                # Return control variant (first variant)
                return test_config.variants[0]
            
            # Assign variant based on traffic percentages
            variant = self._select_variant(test_config, user_id)
            self._user_assignments[test_id][user_id] = variant.id
            
            logger.debug(f"Assigned user {user_id} to variant {variant.name} in test {test_config.name}")
            return variant
    
    async def record_event(self, test_id: str, user_id: str, 
                          latency: Optional[float] = None, success: bool = True,
                          converted: bool = False, custom_metrics: Optional[Dict[str, float]] = None) -> None:
        """Record an event for A/B test metrics."""
        async with self._lock:
            if test_id not in self._tests or user_id not in self._user_assignments[test_id]:
                return  # Silently ignore if test doesn't exist or user not assigned
            
            variant_id = self._user_assignments[test_id][user_id]
            
            if test_id in self._metrics and variant_id in self._metrics[test_id]:
                metrics = self._metrics[test_id][variant_id]
                metrics.update_metrics(latency, success, converted, custom_metrics)
    
    async def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get comprehensive test results including statistical analysis."""
        async with self._lock:
            if test_id not in self._tests:
                raise TestNotFoundError(f"Test {test_id} not found")
            
            test_config = self._tests[test_id]
            test_metrics = self._metrics[test_id]
            
            results = {
                "test_id": test_id,
                "test_name": test_config.name,
                "status": test_config.status.value,
                "start_time": test_config.start_time.isoformat() if test_config.start_time else None,
                "end_time": test_config.end_time.isoformat() if test_config.end_time else None,
                "variants": {},
                "statistical_analysis": {},
                "recommendations": []
            }
            
            # Collect variant results
            for variant in test_config.variants:
                metrics = test_metrics[variant.id]
                results["variants"][variant.id] = {
                    "name": variant.name,
                    "model_name": variant.model_name,
                    "model_version": variant.model_version,
                    "traffic_percentage": variant.traffic_percentage,
                    "status": variant.status.value,
                    "metrics": {
                        "requests_count": metrics.requests_count,
                        "success_rate": metrics.success_count / metrics.requests_count if metrics.requests_count > 0 else 0,
                        "error_rate": metrics.error_count / metrics.requests_count if metrics.requests_count > 0 else 0,
                        "average_latency": metrics.average_latency,
                        "conversion_rate": metrics.conversion_rate,
                        "custom_metrics": metrics.custom_metrics
                    }
                }
            
            # Perform statistical analysis
            results["statistical_analysis"] = await self._perform_statistical_analysis(test_config, test_metrics)
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(test_config, test_metrics, results["statistical_analysis"])
            
            return results
    
    async def get_winning_variant(self, test_id: str) -> Optional[TestVariant]:
        """Get the winning variant based on statistical analysis."""
        results = await self.get_test_results(test_id)
        
        analysis = results["statistical_analysis"]
        if analysis.get("has_significant_winner"):
            winner_id = analysis["winner_variant_id"]
            test_config = self._tests[test_id]
            return next(v for v in test_config.variants if v.id == winner_id)
        
        return None
    
    async def auto_select_winner(self, test_id: str, min_runtime_hours: int = 24) -> Optional[TestVariant]:
        """Automatically select winner if test meets criteria."""
        async with self._lock:
            test_config = self._tests[test_id]
            
            # Check minimum runtime
            if test_config.start_time:
                runtime = datetime.now(timezone.utc) - test_config.start_time
                if runtime < timedelta(hours=min_runtime_hours):
                    return None
            
            # Check minimum sample size
            total_samples = sum(
                metrics.requests_count 
                for metrics in self._metrics[test_id].values()
            )
            
            if total_samples < test_config.minimum_sample_size:
                return None
            
            # Get statistical analysis
            winner = await self.get_winning_variant(test_id)
            
            if winner:
                # Mark winner and stop test
                for variant in test_config.variants:
                    if variant.id == winner.id:
                        variant.status = VariantStatus.WINNER
                    else:
                        variant.status = VariantStatus.LOSER
                
                await self.stop_test(test_id, "auto_winner_selected")
                logger.info(f"Auto-selected winner for test {test_config.name}: {winner.name}")
                
                return winner
        
        return None
    
    def _passes_filters(self, config: ABTestConfig, user_id: str, 
                       context: Optional[Dict[str, Any]]) -> bool:
        """Check if user passes test filters."""
        # Tenant filter
        if config.tenant_filter and context:
            user_tenant = context.get("tenant_id")
            if user_tenant not in config.tenant_filter:
                return False
        
        # Additional user filters
        if config.user_filter and context:
            for key, expected_value in config.user_filter.items():
                if context.get(key) != expected_value:
                    return False
        
        return True
    
    def _select_variant(self, config: ABTestConfig, user_id: str) -> TestVariant:
        """Select variant based on consistent hashing and traffic percentages."""
        # Create deterministic hash for consistent assignment
        hash_input = f"{config.id}:{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        # Map hash to 0-100 range
        percentage = (hash_value % 10000) / 100.0
        
        # Select variant based on cumulative traffic percentages
        cumulative = 0.0
        for variant in config.variants:
            if variant.status != VariantStatus.ACTIVE:
                continue
            
            cumulative += variant.traffic_percentage
            if percentage <= cumulative:
                return variant
        
        # Fallback to first active variant
        return next(v for v in config.variants if v.status == VariantStatus.ACTIVE)
    
    async def _perform_statistical_analysis(self, config: ABTestConfig, 
                                          metrics: Dict[str, ABTestMetrics]) -> Dict[str, Any]:
        """Perform statistical analysis on test results."""
        # This is a simplified statistical analysis
        # In production, you'd use proper statistical libraries like scipy
        
        success_metric = config.success_metric
        variants = list(metrics.values())
        
        if len(variants) < 2:
            return {"has_significant_winner": False}
        
        # Get metric values for each variant
        metric_values = []
        for variant_metrics in variants:
            if success_metric == "conversion_rate":
                value = variant_metrics.conversion_rate
            elif success_metric == "average_latency":
                value = -variant_metrics.average_latency  # Lower is better, so negate
            else:
                value = variant_metrics.custom_metrics.get(success_metric, 0)
            
            metric_values.append((variant_metrics.variant_id, value, variant_metrics.requests_count))
        
        # Find best performing variant
        best_variant = max(metric_values, key=lambda x: x[1])
        
        # Simple significance test (in production, use proper statistical tests)
        min_improvement = 0.05  # 5% minimum improvement
        min_confidence = config.confidence_level
        
        is_significant = (
            best_variant[1] > metric_values[0][1] * (1 + min_improvement) and
            best_variant[2] >= config.minimum_sample_size / len(variants)
        )
        
        return {
            "has_significant_winner": is_significant,
            "winner_variant_id": best_variant[0] if is_significant else None,
            "confidence_level": min_confidence,
            "minimum_improvement_threshold": min_improvement,
            "metric_used": success_metric
        }
    
    def _generate_recommendations(self, config: ABTestConfig, 
                                metrics: Dict[str, ABTestMetrics],
                                analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check sample sizes
        total_samples = sum(m.requests_count for m in metrics.values())
        if total_samples < config.minimum_sample_size:
            recommendations.append(f"Continue test to reach minimum sample size of {config.minimum_sample_size}")
        
        # Check for significant winner
        if analysis.get("has_significant_winner"):
            winner_id = analysis["winner_variant_id"]
            winner_variant = next(v for v in config.variants if v.id == winner_id)
            recommendations.append(f"Deploy variant '{winner_variant.name}' as the winner")
        else:
            recommendations.append("No statistically significant winner found")
        
        # Check error rates
        for variant_id, variant_metrics in metrics.items():
            error_rate = variant_metrics.error_count / max(variant_metrics.requests_count, 1)
            if error_rate > 0.05:  # 5% error rate threshold
                variant = next(v for v in config.variants if v.id == variant_id)
                recommendations.append(f"High error rate ({error_rate:.2%}) in variant '{variant.name}' - investigate issues")
        
        return recommendations


# Global instance
ab_test_manager = ABTestManager()