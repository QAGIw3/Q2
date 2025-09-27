"""
Tests for AI Model Management system.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from shared.ai_model_management import (
    AdvancedModelManager,
    ModelConfig,
    ModelStatus,
    ModelVersion,
    VersionStatus,
    ModelRepository,
    ABTestConfig,
    TestVariant,
    ABTestManager,
    MonitoringConfig,
    ModelMonitor,
    HealthStatus,
    AlertSeverity,
    MetricThreshold,
)


class TestAdvancedModelManager:
    """Test cases for AdvancedModelManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a test model manager."""
        return AdvancedModelManager(max_cache_size=3, enable_monitoring=True)
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample model configuration."""
        return ModelConfig(
            name="test-model",
            version="1.0.0",
            tenant_id="test-tenant",
            parameters={"temperature": 0.7},
            tags={"environment", "test"}
        )
    
    @pytest.mark.asyncio
    async def test_register_model(self, manager, sample_config):
        """Test model registration."""
        model_key = await manager.register_model(sample_config)
        
        assert model_key == "test-tenant:test-model:1.0.0"
        
        # Verify model is registered
        models = await manager.list_models(tenant_id="test-tenant")
        assert len(models) == 1
        assert models[0].config.name == "test-model"
        assert models[0].status == ModelStatus.INACTIVE
    
    @pytest.mark.asyncio
    async def test_load_model(self, manager, sample_config):
        """Test model loading."""
        await manager.register_model(sample_config)
        
        # Load the model
        await manager.load_model("test-model", "1.0.0", "test-tenant")
        
        # Verify model is loaded
        model_info = await manager.get_model_info("test-model", "1.0.0", "test-tenant")
        assert model_info.status == ModelStatus.ACTIVE
        assert model_info.model_instance is not None
        assert model_info.tokenizer_instance is not None
    
    @pytest.mark.asyncio
    async def test_get_model(self, manager, sample_config):
        """Test getting model instances."""
        await manager.register_model(sample_config)
        
        # Get model (should auto-load)
        model, tokenizer = await manager.get_model("test-model", "1.0.0", "test-tenant")
        
        assert model is not None
        assert tokenizer is not None
        
        # Verify metrics updated
        model_info = await manager.get_model_info("test-model", "1.0.0", "test-tenant")
        assert model_info.metrics.requests_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_management(self, manager):
        """Test model cache eviction."""
        # Register models exceeding cache size
        configs = []
        for i in range(5):
            config = ModelConfig(name=f"model-{i}", version="1.0.0")
            configs.append(config)
            await manager.register_model(config)
            await manager.load_model(f"model-{i}", "1.0.0")
        
        # Check that only max_cache_size models are active
        active_count = 0
        for i in range(5):
            model_info = await manager.get_model_info(f"model-{i}", "1.0.0")
            if model_info.status == ModelStatus.ACTIVE:
                active_count += 1
        
        assert active_count <= manager._max_cache_size
    
    @pytest.mark.asyncio
    async def test_health_check(self, manager, sample_config):
        """Test health check functionality."""
        await manager.register_model(sample_config)
        await manager.load_model("test-model", "1.0.0", "test-tenant")
        
        health = await manager.health_check()
        
        assert health["total_models"] == 1
        assert health["active_models"] == 1
        assert health["failed_models"] == 0
        assert health["tenants"] == 1


class TestModelRepository:
    """Test cases for ModelRepository."""
    
    @pytest.fixture
    def repository(self, tmp_path):
        """Create a test repository."""
        return ModelRepository(base_path=tmp_path)
    
    def test_create_version(self, repository):
        """Test version creation."""
        artifacts = {"model": {"type": "huggingface", "path": "model.bin"}}
        metadata = {"accuracy": 0.95, "dataset": "test"}
        
        version = repository.create_version(
            name="test-model",
            version="1.0.0",
            artifacts=artifacts,
            metadata=metadata,
            created_by="test-user",
            tags={"production"}
        )
        
        assert version.name == "test-model"
        assert version.version == "1.0.0"
        assert version.status == VersionStatus.DRAFT
        assert version.created_by == "test-user"
        assert "production" in version.tags
        assert version.checksum is not None
    
    def test_promote_version(self, repository):
        """Test version promotion."""
        artifacts = {"model": {"type": "test"}}
        
        # Create and promote version
        version = repository.create_version("test-model", "1.0.0", artifacts)
        promoted = repository.promote_version("test-model", "1.0.0", VersionStatus.STAGING)
        
        assert promoted.status == VersionStatus.STAGING
    
    def test_production_rollback(self, repository):
        """Test production rollback."""
        artifacts = {"model": {"type": "test"}}
        
        # Create two versions
        v1 = repository.create_version("test-model", "1.0.0", artifacts)
        v2 = repository.create_version("test-model", "2.0.0", artifacts)
        
        # Promote both to production (simulating version history)
        repository.promote_version("test-model", "1.0.0", VersionStatus.STAGING)
        repository.promote_version("test-model", "1.0.0", VersionStatus.PRODUCTION)
        repository.promote_version("test-model", "2.0.0", VersionStatus.STAGING)
        repository.promote_version("test-model", "2.0.0", VersionStatus.PRODUCTION)
        
        # Rollback
        rolled_back = repository.rollback_production("test-model", "1.0.0")
        
        assert rolled_back.version == "1.0.0"
        assert rolled_back.status == VersionStatus.PRODUCTION
        
        # Verify current production version is deprecated
        v2_current = repository.get_version("test-model", "2.0.0")
        assert v2_current.status == VersionStatus.DEPRECATED


class TestABTestManager:
    """Test cases for ABTestManager."""
    
    @pytest.fixture
    def ab_manager(self):
        """Create a test A/B test manager."""
        return ABTestManager()
    
    @pytest.fixture
    def sample_test_config(self):
        """Create a sample A/B test configuration."""
        variants = [
            TestVariant(
                id="control",
                name="Control",
                model_name="test-model",
                model_version="1.0.0",
                traffic_percentage=50.0
            ),
            TestVariant(
                id="treatment",
                name="Treatment",
                model_name="test-model",
                model_version="2.0.0", 
                traffic_percentage=50.0
            )
        ]
        
        return ABTestConfig(
            id="test-experiment",
            name="Model Version Test",
            description="Testing new model version",
            variants=variants,
            minimum_sample_size=100,
            confidence_level=0.95
        )
    
    @pytest.mark.asyncio
    async def test_create_test(self, ab_manager, sample_test_config):
        """Test A/B test creation."""
        test_id = await ab_manager.create_test(sample_test_config)
        
        assert test_id == "test-experiment"
        assert "test-experiment" in ab_manager._tests
        assert "test-experiment" in ab_manager._metrics
    
    @pytest.mark.asyncio
    async def test_start_stop_test(self, ab_manager, sample_test_config):
        """Test starting and stopping A/B tests."""
        test_id = await ab_manager.create_test(sample_test_config)
        
        # Start test
        await ab_manager.start_test(test_id)
        test_config = ab_manager._tests[test_id]
        assert test_config.status == TestStatus.RUNNING
        assert test_config.start_time is not None
        
        # Stop test
        await ab_manager.stop_test(test_id)
        assert test_config.status == TestStatus.COMPLETED
        assert test_config.end_time is not None
    
    @pytest.mark.asyncio
    async def test_variant_assignment(self, ab_manager, sample_test_config):
        """Test user variant assignment."""
        test_id = await ab_manager.create_test(sample_test_config)
        await ab_manager.start_test(test_id)
        
        # Assign users to variants
        variant1 = await ab_manager.assign_variant(test_id, "user1")
        variant2 = await ab_manager.assign_variant(test_id, "user2")
        
        # Verify assignments are consistent
        variant1_again = await ab_manager.assign_variant(test_id, "user1")
        assert variant1.id == variant1_again.id
        
        # Verify users are tracked
        assert "user1" in ab_manager._user_assignments[test_id]
        assert "user2" in ab_manager._user_assignments[test_id]
    
    @pytest.mark.asyncio
    async def test_metrics_recording(self, ab_manager, sample_test_config):
        """Test metrics recording for A/B tests."""
        test_id = await ab_manager.create_test(sample_test_config)
        await ab_manager.start_test(test_id)
        
        # Assign user and record events
        variant = await ab_manager.assign_variant(test_id, "user1")
        
        await ab_manager.record_event(test_id, "user1", latency=100.0, success=True, converted=True)
        await ab_manager.record_event(test_id, "user1", latency=150.0, success=True, converted=False)
        
        # Check metrics
        metrics = ab_manager._metrics[test_id][variant.id]
        assert metrics.requests_count == 2
        assert metrics.success_count == 2
        assert metrics.error_count == 0
        assert metrics.conversion_rate > 0


class TestModelMonitor:
    """Test cases for ModelMonitor."""
    
    @pytest.fixture
    def monitoring_config(self):
        """Create a test monitoring configuration."""
        thresholds = [
            MetricThreshold(
                metric_name="latency",
                warning_threshold=500.0,
                critical_threshold=1000.0,
                comparison_operator=">"
            ),
            MetricThreshold(
                metric_name="error_rate",
                warning_threshold=0.05,
                critical_threshold=0.10,
                comparison_operator=">"
            )
        ]
        
        return MonitoringConfig(
            model_name="test-model",
            model_version="1.0.0",
            tenant_id="test-tenant",
            thresholds=thresholds,
            metric_collection_interval=1,  # 1 second for testing
            enable_anomaly_detection=True
        )
    
    @pytest.fixture
    def monitor(self, monitoring_config):
        """Create a test model monitor."""
        return ModelMonitor(monitoring_config)
    
    @pytest.mark.asyncio
    async def test_record_metrics(self, monitor):
        """Test metric recording."""
        await monitor.record_metric("latency", 250.0)
        await monitor.record_metric("latency", 300.0)
        await monitor.record_metric("error_rate", 0.02)
        
        # Check statistics
        latency_stats = await monitor.get_metric_statistics("latency")
        assert latency_stats["count"] == 2
        assert latency_stats["mean"] == 275.0
        assert latency_stats["min"] == 250.0
        assert latency_stats["max"] == 300.0
    
    @pytest.mark.asyncio
    async def test_threshold_alerts(self, monitor):
        """Test threshold-based alerting."""
        # Record metric that exceeds warning threshold
        await monitor.record_metric("latency", 750.0)
        
        alerts = await monitor.get_alerts(resolved=False)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert alerts[0].metric_name == "latency"
        
        # Record metric that exceeds critical threshold
        await monitor.record_metric("latency", 1200.0)
        
        alerts = await monitor.get_alerts(resolved=False)
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) == 1
    
    @pytest.mark.asyncio
    async def test_health_check(self, monitor):
        """Test health check functionality."""
        # Record some metrics
        await monitor.record_metric("latency", 200.0)
        await monitor.record_metric("error_rate", 0.01)
        
        # Perform health check
        report = await monitor.perform_health_check()
        
        assert report.model_name == "test-model"
        assert report.model_version == "1.0.0"
        assert report.overall_status == HealthStatus.HEALTHY
        assert "latency" in report.metrics_summary
        assert "error_rate" in report.metrics_summary
        assert len(report.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, monitor):
        """Test anomaly detection."""
        # Record baseline data
        for i in range(50):
            await monitor.record_metric("latency", 200.0 + (i % 10))  # Values around 200-210
        
        # Record anomalous values
        await monitor.record_metric("latency", 500.0)  # Anomaly
        await monitor.record_metric("latency", 205.0)  # Normal
        await monitor.record_metric("latency", 50.0)   # Anomaly
        
        # Detect anomalies
        anomalies = await monitor.detect_anomalies("latency")
        
        # Should detect the anomalous values
        anomaly_values = [anomaly[1] for anomaly in anomalies]
        assert 500.0 in anomaly_values
        assert 50.0 in anomaly_values
        assert 205.0 not in anomaly_values
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, monitor):
        """Test monitoring start/stop lifecycle."""
        # Start monitoring
        await monitor.start_monitoring()
        assert monitor._is_running
        assert monitor._monitoring_task is not None
        
        # Give it a moment to run
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        assert not monitor._is_running


if __name__ == "__main__":
    pytest.main([__file__])