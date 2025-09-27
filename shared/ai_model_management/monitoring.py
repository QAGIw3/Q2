"""
AI Model Monitoring and Performance Tracking for Q2 Platform.

Provides comprehensive monitoring capabilities including:
- Real-time performance metrics
- Model health checking
- Anomaly detection
- Predictive maintenance alerts
"""

import asyncio
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

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


class HealthStatus(Enum):
    """Model health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricThreshold:
    """Metric threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str = ">"  # >, <, >=, <=, ==, !=
    time_window_minutes: int = 5
    
    def evaluate(self, value: float) -> AlertSeverity:
        """Evaluate metric value against thresholds."""
        if self.comparison_operator == ">":
            if value > self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value > self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.comparison_operator == "<":
            if value < self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value < self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.comparison_operator == ">=":
            if value >= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.comparison_operator == "<=":
            if value <= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value <= self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.comparison_operator == "==":
            if value == self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value == self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.comparison_operator == "!=":
            if value != self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value != self.warning_threshold:
                return AlertSeverity.WARNING
        
        return AlertSeverity.INFO


@dataclass
class MonitoringConfig:
    """Monitoring configuration for a model."""
    model_name: str
    model_version: str
    tenant_id: Optional[str] = None
    metric_collection_interval: int = 60  # seconds
    metric_retention_hours: int = 168  # 7 days
    thresholds: List[MetricThreshold] = field(default_factory=list)
    custom_health_checks: List[str] = field(default_factory=list)
    alert_webhooks: List[str] = field(default_factory=list)
    enable_anomaly_detection: bool = True
    anomaly_sensitivity: float = 0.95  # Higher = more sensitive


@dataclass
class ModelAlert:
    """Model monitoring alert."""
    id: str
    model_name: str
    model_version: str
    tenant_id: Optional[str]
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold: Optional[MetricThreshold]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ModelHealthReport:
    """Comprehensive model health report."""
    model_name: str
    model_version: str
    tenant_id: Optional[str]
    overall_status: HealthStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics_summary: Dict[str, float] = field(default_factory=dict)
    active_alerts: List[ModelAlert] = field(default_factory=list)
    health_checks: Dict[str, bool] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class MonitoringError(Q2Exception):
    """Base exception for monitoring errors."""
    pass


class ModelMonitor:
    """
    Real-time model monitoring and alerting system.
    
    Provides:
    - Continuous metric collection
    - Threshold-based alerting
    - Anomaly detection
    - Health status tracking
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._metrics_history: Dict[str, Deque[Tuple[datetime, float]]] = {}
        self._alerts: List[ModelAlert] = []
        self._last_health_check: Optional[datetime] = None
        self._health_status = HealthStatus.UNKNOWN
        self._alert_callbacks: List[Callable[[ModelAlert], None]] = []
        self._anomaly_baselines: Dict[str, Tuple[float, float]] = {}  # metric -> (mean, std)
        self._is_running = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    def add_alert_callback(self, callback: Callable[[ModelAlert], None]) -> None:
        """Add callback for alert notifications."""
        self._alert_callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        if self._is_running:
            logger.warning(f"Monitoring already running for {self.config.model_name}")
            return
        
        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started monitoring for model {self.config.model_name}:{self.config.model_version}")
    
    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped monitoring for model {self.config.model_name}:{self.config.model_version}")
    
    async def record_metric(self, metric_name: str, value: float, 
                           timestamp: Optional[datetime] = None) -> None:
        """Record a metric value."""
        if not timestamp:
            timestamp = datetime.now(timezone.utc)
        
        # Initialize metric history if needed
        if metric_name not in self._metrics_history:
            self._metrics_history[metric_name] = deque(maxlen=self._calculate_max_samples())
        
        # Add to history
        self._metrics_history[metric_name].append((timestamp, value))
        
        # Check thresholds
        await self._check_thresholds(metric_name, value, timestamp)
        
        # Update anomaly baselines if enabled
        if self.config.enable_anomaly_detection:
            await self._update_anomaly_baseline(metric_name, value)
    
    async def get_metric_statistics(self, metric_name: str, 
                                  time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get statistical summary for a metric."""
        if metric_name not in self._metrics_history:
            return {}
        
        cutoff_time = None
        if time_window:
            cutoff_time = datetime.now(timezone.utc) - time_window
        
        # Filter data points within time window
        values = []
        for timestamp, value in self._metrics_history[metric_name]:
            if cutoff_time is None or timestamp >= cutoff_time:
                values.append(value)
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "percentile_95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            "percentile_99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
        }
    
    async def perform_health_check(self) -> ModelHealthReport:
        """Perform comprehensive health check."""
        with tracer.start_as_current_span("health_check") as span:
            span.set_attribute("model.name", self.config.model_name)
            span.set_attribute("model.version", self.config.model_version)
            
            # Collect current metrics
            metrics_summary = {}
            for metric_name in self._metrics_history:
                stats = await self.get_metric_statistics(
                    metric_name, 
                    timedelta(minutes=5)  # Last 5 minutes
                )
                if stats:
                    metrics_summary[metric_name] = stats["mean"]
            
            # Get active alerts
            active_alerts = [alert for alert in self._alerts if not alert.resolved]
            
            # Determine overall health status
            overall_status = self._calculate_overall_health_status(active_alerts)
            
            # Generate recommendations
            recommendations = await self._generate_health_recommendations(metrics_summary, active_alerts)
            
            # Perform custom health checks
            health_checks = await self._perform_custom_health_checks()
            
            report = ModelHealthReport(
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                tenant_id=self.config.tenant_id,
                overall_status=overall_status,
                metrics_summary=metrics_summary,
                active_alerts=active_alerts,
                health_checks=health_checks,
                recommendations=recommendations
            )
            
            self._last_health_check = datetime.now(timezone.utc)
            self._health_status = overall_status
            
            return report
    
    async def detect_anomalies(self, metric_name: str, 
                             sensitivity: Optional[float] = None) -> List[Tuple[datetime, float, float]]:
        """Detect anomalies in metric data."""
        if metric_name not in self._metrics_history:
            return []
        
        if metric_name not in self._anomaly_baselines:
            return []
        
        sensitivity = sensitivity or self.config.anomaly_sensitivity
        mean, std = self._anomaly_baselines[metric_name]
        
        # Calculate anomaly threshold (Z-score based)
        threshold_multiplier = self._sensitivity_to_zscore(sensitivity)
        upper_bound = mean + threshold_multiplier * std
        lower_bound = mean - threshold_multiplier * std
        
        # Find anomalies
        anomalies = []
        for timestamp, value in self._metrics_history[metric_name]:
            if value > upper_bound or value < lower_bound:
                anomaly_score = abs(value - mean) / std if std > 0 else 0
                anomalies.append((timestamp, value, anomaly_score))
        
        return anomalies
    
    async def get_alerts(self, resolved: Optional[bool] = None) -> List[ModelAlert]:
        """Get alerts, optionally filtered by resolved status."""
        if resolved is None:
            return self._alerts.copy()
        
        return [alert for alert in self._alerts if alert.resolved == resolved]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self._alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                logger.info(f"Resolved alert {alert_id} for model {self.config.model_name}")
                return True
        
        return False
    
    def _calculate_max_samples(self) -> int:
        """Calculate maximum samples to retain based on retention hours and collection interval."""
        return int((self.config.metric_retention_hours * 3600) / self.config.metric_collection_interval)
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self._is_running:
                await asyncio.sleep(self.config.metric_collection_interval)
                
                if not self._is_running:
                    break
                
                # Perform periodic health check
                try:
                    await self.perform_health_check()
                except Exception as e:
                    logger.error(f"Health check failed for model {self.config.model_name}: {e}")
                
                # Clean up old data
                await self._cleanup_old_data()
                
        except asyncio.CancelledError:
            logger.info(f"Monitoring loop cancelled for model {self.config.model_name}")
        except Exception as e:
            logger.error(f"Monitoring loop error for model {self.config.model_name}: {e}")
    
    async def _check_thresholds(self, metric_name: str, value: float, timestamp: datetime) -> None:
        """Check metric value against configured thresholds."""
        for threshold in self.config.thresholds:
            if threshold.metric_name == metric_name:
                severity = threshold.evaluate(value)
                
                if severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]:
                    alert = ModelAlert(
                        id=f"{self.config.model_name}_{metric_name}_{timestamp.isoformat()}",
                        model_name=self.config.model_name,
                        model_version=self.config.model_version,
                        tenant_id=self.config.tenant_id,
                        severity=severity,
                        message=f"Metric {metric_name} value {value} exceeded {severity.value} threshold",
                        metric_name=metric_name,
                        metric_value=value,
                        threshold=threshold,
                        timestamp=timestamp
                    )
                    
                    self._alerts.append(alert)
                    
                    # Notify callbacks
                    for callback in self._alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")
    
    async def _update_anomaly_baseline(self, metric_name: str, value: float) -> None:
        """Update anomaly detection baseline for a metric."""
        if metric_name not in self._metrics_history:
            return
        
        # Use last N samples to calculate baseline
        recent_values = [val for _, val in list(self._metrics_history[metric_name])[-100:]]
        
        if len(recent_values) >= 10:  # Need minimum samples
            mean = statistics.mean(recent_values)
            std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
            self._anomaly_baselines[metric_name] = (mean, std)
    
    def _sensitivity_to_zscore(self, sensitivity: float) -> float:
        """Convert sensitivity to Z-score threshold."""
        # Higher sensitivity = lower Z-score threshold
        # Map 0.8-0.99 sensitivity to 3.0-1.5 Z-score
        return 4.5 - 3.0 * sensitivity
    
    def _calculate_overall_health_status(self, active_alerts: List[ModelAlert]) -> HealthStatus:
        """Calculate overall health status from active alerts."""
        if not active_alerts:
            return HealthStatus.HEALTHY
        
        has_critical = any(alert.severity == AlertSeverity.CRITICAL for alert in active_alerts)
        if has_critical:
            return HealthStatus.CRITICAL
        
        has_warning = any(alert.severity == AlertSeverity.WARNING for alert in active_alerts)
        if has_warning:
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    async def _generate_health_recommendations(self, metrics_summary: Dict[str, float], 
                                             active_alerts: List[ModelAlert]) -> List[str]:
        """Generate health recommendations based on current state."""
        recommendations = []
        
        # Check for high error rates
        if "error_rate" in metrics_summary and metrics_summary["error_rate"] > 0.05:
            recommendations.append("High error rate detected - investigate model performance")
        
        # Check for high latency
        if "average_latency" in metrics_summary and metrics_summary["average_latency"] > 1000:
            recommendations.append("High latency detected - consider model optimization or scaling")
        
        # Check for low request volume
        if "requests_per_minute" in metrics_summary and metrics_summary["requests_per_minute"] < 1:
            recommendations.append("Low request volume - verify model is receiving traffic")
        
        # Alert-based recommendations
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        if len(critical_alerts) > 3:
            recommendations.append("Multiple critical alerts active - consider rolling back to previous version")
        
        if not recommendations:
            recommendations.append("Model is performing within expected parameters")
        
        return recommendations
    
    async def _perform_custom_health_checks(self) -> Dict[str, bool]:
        """Perform custom health checks defined in configuration."""
        # This would integrate with external health check systems
        # For now, return mock results
        results = {}
        for check_name in self.config.custom_health_checks:
            # Mock health check - in production, this would call actual health check functions
            results[check_name] = True
        
        return results
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metric data beyond retention period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.config.metric_retention_hours)
        
        for metric_name in self._metrics_history:
            history = self._metrics_history[metric_name]
            # Remove old entries
            while history and history[0][0] < cutoff_time:
                history.popleft()
        
        # Clean up old resolved alerts
        cutoff_time_alerts = datetime.now(timezone.utc) - timedelta(days=7)
        self._alerts = [
            alert for alert in self._alerts
            if not alert.resolved or (alert.resolved_at and alert.resolved_at >= cutoff_time_alerts)
        ]


class PerformanceTracker:
    """
    Performance tracking utility for model inference operations.
    
    Provides context managers and decorators for automatic performance tracking.
    """
    
    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.start_time = datetime.now(timezone.utc)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        end_time = datetime.now(timezone.utc)
        latency = (end_time - self.start_time).total_seconds() * 1000  # milliseconds
        
        success = exc_type is None
        await self.monitor.record_metric("latency", latency)
        await self.monitor.record_metric("request_count", 1)
        
        if not success:
            await self.monitor.record_metric("error_count", 1)
    
    def track_performance(self, func):
        """Decorator for tracking function performance."""
        async def wrapper(*args, **kwargs):
            async with self:
                return await func(*args, **kwargs)
        return wrapper


class ModelHealthChecker:
    """
    Centralized health checking service for all models.
    
    Manages health checks across multiple models and provides
    aggregated health status reporting.
    """
    
    def __init__(self):
        self._monitors: Dict[str, ModelMonitor] = {}
        self._global_alert_callbacks: List[Callable[[ModelAlert], None]] = []
    
    def register_model(self, config: MonitoringConfig) -> ModelMonitor:
        """Register a model for health monitoring."""
        model_key = f"{config.model_name}:{config.model_version}"
        if config.tenant_id:
            model_key = f"{config.tenant_id}:{model_key}"
        
        monitor = ModelMonitor(config)
        
        # Add global alert callbacks
        for callback in self._global_alert_callbacks:
            monitor.add_alert_callback(callback)
        
        self._monitors[model_key] = monitor
        logger.info(f"Registered model for health monitoring: {model_key}")
        
        return monitor
    
    def add_global_alert_callback(self, callback: Callable[[ModelAlert], None]) -> None:
        """Add global alert callback for all models."""
        self._global_alert_callbacks.append(callback)
        
        # Add to existing monitors
        for monitor in self._monitors.values():
            monitor.add_alert_callback(callback)
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        reports = {}
        total_models = len(self._monitors)
        healthy_count = 0
        warning_count = 0
        critical_count = 0
        
        for model_key, monitor in self._monitors.items():
            try:
                report = await monitor.perform_health_check()
                reports[model_key] = report
                
                if report.overall_status == HealthStatus.HEALTHY:
                    healthy_count += 1
                elif report.overall_status == HealthStatus.WARNING:
                    warning_count += 1
                elif report.overall_status == HealthStatus.CRITICAL:
                    critical_count += 1
                    
            except Exception as e:
                logger.error(f"Health check failed for {model_key}: {e}")
                critical_count += 1
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_models": total_models,
            "healthy_models": healthy_count,
            "warning_models": warning_count,
            "critical_models": critical_count,
            "system_health_percentage": (healthy_count / total_models * 100) if total_models > 0 else 0,
            "model_reports": reports
        }
    
    async def start_all_monitoring(self) -> None:
        """Start monitoring for all registered models."""
        tasks = []
        for monitor in self._monitors.values():
            tasks.append(monitor.start_monitoring())
        
        await asyncio.gather(*tasks)
        logger.info(f"Started monitoring for {len(self._monitors)} models")
    
    async def stop_all_monitoring(self) -> None:
        """Stop monitoring for all registered models."""
        tasks = []
        for monitor in self._monitors.values():
            tasks.append(monitor.stop_monitoring())
        
        await asyncio.gather(*tasks)
        logger.info(f"Stopped monitoring for {len(self._monitors)} models")


# Global instances
health_checker = ModelHealthChecker()