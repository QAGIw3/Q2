"""
Edge AI Deployment Manager

Manages deployment of AI models to edge devices with:
- Intelligent model optimization for resource constraints
- Over-the-air deployment and updates
- Performance monitoring and auto-scaling
- Device fleet management
- Real-time inference coordination
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import json
import numpy as np
from collections import defaultdict, deque
import hashlib
# Optional HTTP client - would be installed when needed for actual deployments
try:
    import aiohttp
except ImportError:
    aiohttp = None
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    """Edge deployment strategies"""
    SINGLE_DEVICE = "single_device"          # Deploy to one device
    REPLICATED = "replicated"                # Deploy to multiple devices
    SHARDED = "sharded"                      # Distribute model across devices
    HIERARCHICAL = "hierarchical"            # Hierarchical edge-cloud deployment
    CANARY = "canary"                        # Gradual rollout deployment
    A_B_TESTING = "a_b_testing"             # A/B testing deployment

class DeviceType(Enum):
    """Types of edge devices"""
    MOBILE_PHONE = "mobile_phone"
    TABLET = "tablet"
    IOT_SENSOR = "iot_sensor"
    EDGE_SERVER = "edge_server"
    RASPBERRY_PI = "raspberry_pi"
    NVIDIA_JETSON = "nvidia_jetson"
    INDUSTRIAL_PC = "industrial_pc"
    AUTONOMOUS_VEHICLE = "autonomous_vehicle"

class DeploymentStatus(Enum):
    """Deployment status states"""
    PENDING = "pending"
    OPTIMIZING = "optimizing"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    UPDATING = "updating"
    FAILED = "failed"
    RETIRED = "retired"

class OptimizationTechnique(Enum):
    """Model optimization techniques"""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    COMPRESSION = "compression"
    BATCHING = "batching"
    CACHING = "caching"
    PIPELINE_OPTIMIZATION = "pipeline_optimization"

@dataclass
class DeviceCapabilities:
    """Edge device capabilities"""
    cpu_cores: int
    cpu_frequency: float  # GHz
    memory_mb: int
    storage_gb: int
    gpu_available: bool
    gpu_memory_mb: int = 0
    network_bandwidth: float = 1.0  # Mbps
    battery_life: float = 0.0  # Hours (0 if plugged in)
    operating_system: str = "linux"
    architecture: str = "arm64"
    accelerators: List[str] = None
    
    def __post_init__(self):
        if self.accelerators is None:
            self.accelerators = []

@dataclass
class ResourceConstraints:
    """Resource constraints for deployment"""
    max_memory_mb: int
    max_storage_mb: int
    max_inference_latency_ms: float
    max_power_consumption_watts: float = 10.0
    min_accuracy: float = 0.9
    max_model_size_mb: float = 100.0
    bandwidth_limit_mbps: float = 1.0

@dataclass
class EdgeDevice:
    """Represents an edge device"""
    device_id: str
    device_type: DeviceType
    name: str
    location: str
    capabilities: DeviceCapabilities
    constraints: ResourceConstraints
    status: str = "online"
    last_heartbeat: datetime = None
    deployed_models: List[str] = None
    performance_metrics: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.utcnow()
        if self.deployed_models is None:
            self.deployed_models = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ModelDeployment:
    """Represents a model deployment to edge devices"""
    deployment_id: str
    model_id: str
    model_version: str
    target_devices: List[str]
    deployment_strategy: DeploymentStrategy
    optimization_config: Dict[str, Any]
    status: DeploymentStatus
    created_at: datetime
    deployed_at: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = None
    error_message: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass
class InferenceRequest:
    """Edge inference request"""
    request_id: str
    model_id: str
    device_id: str
    input_data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 5=high
    timeout_ms: int = 5000

@dataclass
class InferenceResult:
    """Edge inference result"""
    request_id: str
    model_id: str
    device_id: str
    prediction: Dict[str, Any]
    confidence: float
    inference_time_ms: float
    timestamp: datetime
    success: bool = True
    error_message: str = ""

class EdgeDeploymentManager:
    """
    Advanced Edge AI Deployment Manager with cutting-edge capabilities
    """
    
    def __init__(self):
        self.devices: Dict[str, EdgeDevice] = {}
        self.deployments: Dict[str, ModelDeployment] = {}
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.optimization_pipelines: Dict[str, Callable] = {}
        
        # Performance tracking
        self.inference_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.deployment_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.device_health: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Background tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Edge AI Deployment Manager initialized")
    
    async def initialize(self):
        """Initialize the edge deployment manager"""
        await self._setup_optimization_pipelines()
        await self._start_background_tasks()
        logger.info("Edge deployment manager initialization complete")
    
    # ===== DEVICE MANAGEMENT =====
    
    async def register_device(
        self,
        device_type: DeviceType,
        name: str,
        location: str,
        capabilities: DeviceCapabilities,
        constraints: ResourceConstraints,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Register a new edge device"""
        
        device_id = str(uuid.uuid4())
        
        device = EdgeDevice(
            device_id=device_id,
            device_type=device_type,
            name=name,
            location=location,
            capabilities=capabilities,
            constraints=constraints,
            metadata=metadata or {}
        )
        
        self.devices[device_id] = device
        
        # Start device monitoring
        monitoring_task = asyncio.create_task(
            self._monitor_device(device_id)
        )
        self.monitoring_tasks[device_id] = monitoring_task
        
        # Start heartbeat monitoring
        heartbeat_task = asyncio.create_task(
            self._device_heartbeat_monitor(device_id)
        )
        self.heartbeat_tasks[device_id] = heartbeat_task
        
        logger.info(f"Registered edge device: {name} ({device_type.value})")
        return device_id
    
    async def update_device_status(
        self,
        device_id: str,
        status: str,
        performance_metrics: Dict[str, float] = None
    ) -> bool:
        """Update device status and metrics"""
        
        if device_id not in self.devices:
            logger.error(f"Device {device_id} not found")
            return False
        
        device = self.devices[device_id]
        device.status = status
        device.last_heartbeat = datetime.utcnow()
        
        if performance_metrics:
            device.performance_metrics.update(performance_metrics)
            
            # Store metrics for analysis
            for metric, value in performance_metrics.items():
                self.inference_metrics[f"{device_id}_{metric}"].append({
                    "timestamp": datetime.utcnow(),
                    "value": value
                })
        
        return True
    
    # ===== MODEL DEPLOYMENT =====
    
    async def deploy_model(
        self,
        model_id: str,
        model_version: str,
        target_devices: List[str],
        deployment_strategy: DeploymentStrategy = DeploymentStrategy.REPLICATED,
        optimization_config: Dict[str, Any] = None
    ) -> str:
        """Deploy a model to edge devices"""
        
        deployment_id = str(uuid.uuid4())
        
        # Validate target devices
        valid_devices = []
        for device_id in target_devices:
            if device_id in self.devices and self.devices[device_id].status == "online":
                valid_devices.append(device_id)
            else:
                logger.warning(f"Device {device_id} not available for deployment")
        
        if not valid_devices:
            logger.error("No valid devices available for deployment")
            return None
        
        # Create deployment record
        deployment = ModelDeployment(
            deployment_id=deployment_id,
            model_id=model_id,
            model_version=model_version,
            target_devices=valid_devices,
            deployment_strategy=deployment_strategy,
            optimization_config=optimization_config or {},
            status=DeploymentStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        self.deployments[deployment_id] = deployment
        
        # Start deployment process
        deployment_task = asyncio.create_task(
            self._execute_deployment(deployment_id)
        )
        
        logger.info(f"Started deployment {deployment_id} for model {model_id}")
        return deployment_id
    
    async def _execute_deployment(self, deployment_id: str):
        """Execute the deployment process"""
        
        deployment = self.deployments[deployment_id]
        
        try:
            # Phase 1: Model optimization
            deployment.status = DeploymentStatus.OPTIMIZING
            optimized_models = await self._optimize_model_for_devices(deployment)
            
            # Phase 2: Deploy to devices
            deployment.status = DeploymentStatus.DEPLOYING
            deployment_results = await self._deploy_to_devices(deployment, optimized_models)
            
            # Phase 3: Verify deployment
            if all(deployment_results.values()):
                deployment.status = DeploymentStatus.ACTIVE
                deployment.deployed_at = datetime.utcnow()
                
                # Update device records
                for device_id in deployment.target_devices:
                    if device_id in self.devices:
                        self.devices[device_id].deployed_models.append(
                            f"{deployment.model_id}:{deployment.model_version}"
                        )
                
                logger.info(f"Deployment {deployment_id} completed successfully")
            else:
                deployment.status = DeploymentStatus.FAILED
                deployment.error_message = "Failed to deploy to some devices"
                logger.error(f"Deployment {deployment_id} partially failed")
        
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            logger.error(f"Deployment {deployment_id} failed: {e}")
    
    async def _optimize_model_for_devices(
        self, 
        deployment: ModelDeployment
    ) -> Dict[str, Dict[str, Any]]:
        """Optimize model for target devices"""
        
        optimized_models = {}
        
        for device_id in deployment.target_devices:
            device = self.devices[device_id]
            
            # Determine optimization strategy based on device constraints
            optimization_techniques = await self._select_optimization_techniques(
                device, deployment.optimization_config
            )
            
            # Apply optimizations
            optimized_model = await self._apply_optimizations(
                deployment.model_id,
                deployment.model_version,
                optimization_techniques,
                device.constraints
            )
            
            optimized_models[device_id] = optimized_model
        
        return optimized_models
    
    async def _deploy_to_devices(
        self, 
        deployment: ModelDeployment, 
        optimized_models: Dict[str, Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Deploy optimized models to devices"""
        
        deployment_results = {}
        
        for device_id in deployment.target_devices:
            try:
                # Deploy to device
                success = await self._deploy_to_single_device(
                    device_id, 
                    optimized_models[device_id], 
                    deployment
                )
                deployment_results[device_id] = success
                
            except Exception as e:
                logger.error(f"Failed to deploy to device {device_id}: {e}")
                deployment_results[device_id] = False
        
        return deployment_results
    
    # ===== INFERENCE MANAGEMENT =====
    
    async def submit_inference_request(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        device_id: str = None,
        priority: int = 1,
        timeout_ms: int = 5000
    ) -> str:
        """Submit an inference request to edge devices"""
        
        request_id = str(uuid.uuid4())
        
        # Select device if not specified
        if device_id is None:
            device_id = await self._select_optimal_device(model_id, input_data, priority)
        
        if device_id is None:
            logger.error(f"No suitable device found for model {model_id}")
            return None
        
        # Create inference request
        request = InferenceRequest(
            request_id=request_id,
            model_id=model_id,
            device_id=device_id,
            input_data=input_data,
            timestamp=datetime.utcnow(),
            priority=priority,
            timeout_ms=timeout_ms
        )
        
        # Submit to device
        result = await self._execute_inference(request)
        
        return request_id
    
    async def _execute_inference(self, request: InferenceRequest) -> InferenceResult:
        """Execute inference on edge device"""
        
        start_time = time.time()
        
        try:
            # Mock inference execution
            # In practice, this would communicate with actual edge devices
            prediction = await self._mock_edge_inference(
                request.device_id, 
                request.model_id, 
                request.input_data
            )
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            result = InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                device_id=request.device_id,
                prediction=prediction,
                confidence=prediction.get("confidence", 0.9),
                inference_time_ms=inference_time,
                timestamp=datetime.utcnow(),
                success=True
            )
            
            # Record performance metrics
            await self._record_inference_metrics(result)
            
            return result
            
        except Exception as e:
            result = InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                device_id=request.device_id,
                prediction={},
                confidence=0.0,
                inference_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Inference failed: {e}")
            return result
    
    # ===== OPTIMIZATION METHODS =====
    
    async def _select_optimization_techniques(
        self, 
        device: EdgeDevice, 
        config: Dict[str, Any]
    ) -> List[OptimizationTechnique]:
        """Select optimization techniques for a device"""
        
        techniques = []
        
        # Memory constraints
        if device.capabilities.memory_mb < 1000:
            techniques.extend([
                OptimizationTechnique.QUANTIZATION,
                OptimizationTechnique.PRUNING,
                OptimizationTechnique.COMPRESSION
            ])
        
        # CPU constraints
        if device.capabilities.cpu_cores < 4:
            techniques.append(OptimizationTechnique.DISTILLATION)
        
        # Latency constraints
        if device.constraints.max_inference_latency_ms < 100:
            techniques.extend([
                OptimizationTechnique.BATCHING,
                OptimizationTechnique.CACHING,
                OptimizationTechnique.PIPELINE_OPTIMIZATION
            ])
        
        # Remove duplicates
        return list(set(techniques))
    
    async def _apply_optimizations(
        self,
        model_id: str,
        model_version: str,
        techniques: List[OptimizationTechnique],
        constraints: ResourceConstraints
    ) -> Dict[str, Any]:
        """Apply optimization techniques to a model"""
        
        optimized_model = {
            "model_id": model_id,
            "version": model_version,
            "optimizations": [],
            "size_mb": 50.0,  # Mock size
            "inference_time_ms": 150.0  # Mock inference time
        }
        
        for technique in techniques:
            if technique == OptimizationTechnique.QUANTIZATION:
                optimized_model = await self._apply_quantization(optimized_model, constraints)
            elif technique == OptimizationTechnique.PRUNING:
                optimized_model = await self._apply_pruning(optimized_model, constraints)
            elif technique == OptimizationTechnique.DISTILLATION:
                optimized_model = await self._apply_distillation(optimized_model, constraints)
            elif technique == OptimizationTechnique.COMPRESSION:
                optimized_model = await self._apply_compression(optimized_model, constraints)
            
            optimized_model["optimizations"].append(technique.value)
        
        return optimized_model
    
    async def _apply_quantization(
        self, 
        model: Dict[str, Any], 
        constraints: ResourceConstraints
    ) -> Dict[str, Any]:
        """Apply quantization to reduce model size"""
        # Mock quantization - reduces size by ~75%, increases inference speed
        model["size_mb"] *= 0.25
        model["inference_time_ms"] *= 0.7
        return model
    
    async def _apply_pruning(
        self, 
        model: Dict[str, Any], 
        constraints: ResourceConstraints
    ) -> Dict[str, Any]:
        """Apply pruning to remove unnecessary parameters"""
        # Mock pruning - reduces size by ~50%, slight speed increase
        model["size_mb"] *= 0.5
        model["inference_time_ms"] *= 0.9
        return model
    
    async def _apply_distillation(
        self, 
        model: Dict[str, Any], 
        constraints: ResourceConstraints
    ) -> Dict[str, Any]:
        """Apply knowledge distillation"""
        # Mock distillation - creates smaller, faster model
        model["size_mb"] *= 0.3
        model["inference_time_ms"] *= 0.6
        return model
    
    async def _apply_compression(
        self, 
        model: Dict[str, Any], 
        constraints: ResourceConstraints
    ) -> Dict[str, Any]:
        """Apply model compression"""
        # Mock compression - reduces size with minimal performance impact
        model["size_mb"] *= 0.6
        model["inference_time_ms"] *= 1.05
        return model
    
    # ===== DEVICE SELECTION AND MONITORING =====
    
    async def _select_optimal_device(
        self, 
        model_id: str, 
        input_data: Dict[str, Any],
        priority: int
    ) -> Optional[str]:
        """Select optimal device for inference"""
        
        # Find devices with the model deployed
        candidate_devices = []
        model_key = f"{model_id}:"
        
        for device_id, device in self.devices.items():
            if (device.status == "online" and 
                any(model_key in deployed for deployed in device.deployed_models)):
                candidate_devices.append(device_id)
        
        if not candidate_devices:
            return None
        
        # Score devices based on performance and load
        device_scores = {}
        for device_id in candidate_devices:
            score = await self._calculate_device_score(device_id, priority)
            device_scores[device_id] = score
        
        # Select device with highest score
        best_device = max(device_scores.keys(), key=lambda d: device_scores[d])
        return best_device
    
    async def _calculate_device_score(self, device_id: str, priority: int) -> float:
        """Calculate device score for selection"""
        
        device = self.devices[device_id]
        score = 0.0
        
        # Base score from capabilities
        score += device.capabilities.cpu_cores * 10
        score += device.capabilities.memory_mb / 100
        
        if device.capabilities.gpu_available:
            score += 50
        
        # Adjust for current performance
        recent_metrics = list(self.inference_metrics[f"{device_id}_inference_time"])[-10:]
        if recent_metrics:
            avg_inference_time = np.mean([m["value"] for m in recent_metrics])
            score += max(0, 100 - avg_inference_time)  # Prefer faster devices
        
        # Battery consideration for mobile devices
        if device.device_type in [DeviceType.MOBILE_PHONE, DeviceType.TABLET]:
            if device.capabilities.battery_life > 0:
                score *= min(1.0, device.capabilities.battery_life / 2.0)
        
        # Network bandwidth
        score += device.capabilities.network_bandwidth * 5
        
        return score
    
    async def _monitor_device(self, device_id: str):
        """Monitor device health and performance"""
        
        while device_id in self.devices:
            try:
                device = self.devices[device_id]
                
                # Check device health
                health_metrics = await self._collect_device_health(device_id)
                self.device_health[device_id].update(health_metrics)
                
                # Detect issues
                issues = await self._detect_device_issues(device_id, health_metrics)
                if issues:
                    logger.warning(f"Device {device_id} issues: {issues}")
                    await self._handle_device_issues(device_id, issues)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error monitoring device {device_id}: {e}")
                await asyncio.sleep(60)
    
    async def _device_heartbeat_monitor(self, device_id: str):
        """Monitor device heartbeat"""
        
        while device_id in self.devices:
            try:
                device = self.devices[device_id]
                
                # Check if device has been inactive too long
                time_since_heartbeat = datetime.utcnow() - device.last_heartbeat
                
                if time_since_heartbeat > timedelta(minutes=5):
                    logger.warning(f"Device {device_id} heartbeat timeout")
                    device.status = "offline"
                    await self._handle_device_offline(device_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor for {device_id}: {e}")
                await asyncio.sleep(30)
    
    # ===== HELPER METHODS =====
    
    async def _setup_optimization_pipelines(self):
        """Setup model optimization pipelines"""
        self.optimization_pipelines = {
            OptimizationTechnique.QUANTIZATION.value: self._apply_quantization,
            OptimizationTechnique.PRUNING.value: self._apply_pruning,
            OptimizationTechnique.DISTILLATION.value: self._apply_distillation,
            OptimizationTechnique.COMPRESSION.value: self._apply_compression
        }
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        # Performance monitoring
        asyncio.create_task(self._performance_monitoring_loop())
        
        # Deployment health check
        asyncio.create_task(self._deployment_health_loop())
        
        # Metric collection
        asyncio.create_task(self._metric_collection_loop())
    
    async def _deploy_to_single_device(
        self, 
        device_id: str, 
        optimized_model: Dict[str, Any], 
        deployment: ModelDeployment
    ) -> bool:
        """Deploy model to a single device"""
        
        try:
            # Mock deployment to device
            # In practice, this would use device-specific deployment protocols
            device = self.devices[device_id]
            
            logger.info(f"Deploying model to {device.name} ({device_id})")
            
            # Simulate deployment time
            await asyncio.sleep(np.random.uniform(2, 5))
            
            # Check if deployment would succeed based on constraints
            model_size = optimized_model.get("size_mb", 0)
            if model_size > device.constraints.max_model_size_mb:
                logger.error(f"Model too large for device {device_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy to device {device_id}: {e}")
            return False
    
    async def _mock_edge_inference(
        self, 
        device_id: str, 
        model_id: str, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock edge inference execution"""
        
        # Simulate inference processing
        await asyncio.sleep(np.random.uniform(0.05, 0.2))
        
        # Mock prediction result
        prediction = {
            "classification": np.random.choice(["class_a", "class_b", "class_c"]),
            "confidence": np.random.uniform(0.7, 0.95),
            "scores": {
                "class_a": np.random.uniform(0, 1),
                "class_b": np.random.uniform(0, 1),
                "class_c": np.random.uniform(0, 1)
            }
        }
        
        return prediction
    
    async def _record_inference_metrics(self, result: InferenceResult):
        """Record inference performance metrics"""
        
        metrics = {
            "inference_time": result.inference_time_ms,
            "success": 1 if result.success else 0,
            "confidence": result.confidence
        }
        
        for metric, value in metrics.items():
            self.inference_metrics[f"{result.device_id}_{metric}"].append({
                "timestamp": result.timestamp,
                "value": value
            })
    
    async def _collect_device_health(self, device_id: str) -> Dict[str, Any]:
        """Collect device health metrics"""
        
        # Mock health metrics
        return {
            "cpu_usage": np.random.uniform(20, 80),
            "memory_usage": np.random.uniform(30, 90),
            "temperature": np.random.uniform(40, 75),
            "battery_level": np.random.uniform(20, 100) if self.devices[device_id].capabilities.battery_life > 0 else 100,
            "network_latency": np.random.uniform(10, 100),
            "disk_usage": np.random.uniform(10, 80)
        }
    
    async def _detect_device_issues(
        self, 
        device_id: str, 
        health_metrics: Dict[str, Any]
    ) -> List[str]:
        """Detect device issues from health metrics"""
        
        issues = []
        
        if health_metrics.get("cpu_usage", 0) > 90:
            issues.append("high_cpu_usage")
        
        if health_metrics.get("memory_usage", 0) > 95:
            issues.append("high_memory_usage")
        
        if health_metrics.get("temperature", 0) > 80:
            issues.append("overheating")
        
        if health_metrics.get("battery_level", 100) < 15:
            issues.append("low_battery")
        
        if health_metrics.get("network_latency", 0) > 200:
            issues.append("high_network_latency")
        
        return issues
    
    async def _handle_device_issues(self, device_id: str, issues: List[str]):
        """Handle detected device issues"""
        
        for issue in issues:
            if issue == "high_cpu_usage":
                # Could implement load balancing
                logger.info(f"High CPU usage on {device_id}, considering load redistribution")
            elif issue == "low_battery":
                # Could reduce inference load or switch to power-saving mode
                logger.info(f"Low battery on {device_id}, considering power optimization")
            elif issue == "overheating":
                # Could throttle inference requests
                logger.info(f"Overheating detected on {device_id}, considering throttling")
    
    async def _handle_device_offline(self, device_id: str):
        """Handle device going offline"""
        
        logger.warning(f"Device {device_id} is offline")
        
        # Could implement failover to other devices
        # or pause deployments targeting this device
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring"""
        
        while True:
            try:
                # Analyze inference performance across all devices
                for device_id in self.devices.keys():
                    recent_metrics = list(self.inference_metrics[f"{device_id}_inference_time"])[-100:]
                    if recent_metrics:
                        avg_time = np.mean([m["value"] for m in recent_metrics])
                        logger.info(f"Device {device_id} avg inference time: {avg_time:.2f}ms")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _deployment_health_loop(self):
        """Background deployment health monitoring"""
        
        while True:
            try:
                # Check health of active deployments
                for deployment_id, deployment in self.deployments.items():
                    if deployment.status == DeploymentStatus.ACTIVE:
                        await self._check_deployment_health(deployment_id)
                
                await asyncio.sleep(180)  # Every 3 minutes
                
            except Exception as e:
                logger.error(f"Error in deployment health monitoring: {e}")
                await asyncio.sleep(180)
    
    async def _metric_collection_loop(self):
        """Background metric collection"""
        
        while True:
            try:
                # Collect and aggregate metrics
                for device_id in self.devices.keys():
                    await self._collect_device_health(device_id)
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Error in metric collection: {e}")
                await asyncio.sleep(60)
    
    async def _check_deployment_health(self, deployment_id: str):
        """Check health of a specific deployment"""
        
        deployment = self.deployments[deployment_id]
        
        # Check if all target devices are still available
        available_devices = 0
        for device_id in deployment.target_devices:
            if device_id in self.devices and self.devices[device_id].status == "online":
                available_devices += 1
        
        availability_ratio = available_devices / len(deployment.target_devices)
        
        if availability_ratio < 0.5:
            logger.warning(f"Deployment {deployment_id} has low device availability: {availability_ratio:.2f}")
            # Could trigger redeployment or scaling actions
    
    # ===== PUBLIC API METHODS =====
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a deployment"""
        
        if deployment_id not in self.deployments:
            return None
        
        deployment = self.deployments[deployment_id]
        
        # Collect current status from devices
        device_statuses = {}
        for device_id in deployment.target_devices:
            if device_id in self.devices:
                device_statuses[device_id] = self.devices[device_id].status
        
        return {
            "deployment_id": deployment_id,
            "status": deployment.status.value,
            "model_id": deployment.model_id,
            "model_version": deployment.model_version,
            "created_at": deployment.created_at.isoformat(),
            "deployed_at": deployment.deployed_at.isoformat() if deployment.deployed_at else None,
            "target_devices": deployment.target_devices,
            "device_statuses": device_statuses,
            "performance_metrics": deployment.performance_metrics,
            "error_message": deployment.error_message
        }
    
    async def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a device"""
        
        if device_id not in self.devices:
            return None
        
        device = self.devices[device_id]
        health_metrics = self.device_health.get(device_id, {})
        
        return {
            "device_id": device_id,
            "name": device.name,
            "type": device.device_type.value,
            "status": device.status,
            "location": device.location,
            "capabilities": asdict(device.capabilities),
            "constraints": asdict(device.constraints),
            "deployed_models": device.deployed_models,
            "performance_metrics": device.performance_metrics,
            "health_metrics": health_metrics,
            "last_heartbeat": device.last_heartbeat.isoformat()
        }
    
    async def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments"""
        
        deployments = []
        for deployment_id in self.deployments.keys():
            deployment_status = await self.get_deployment_status(deployment_id)
            if deployment_status:
                deployments.append(deployment_status)
        
        return deployments
    
    async def list_devices(self) -> List[Dict[str, Any]]:
        """List all devices"""
        
        devices = []
        for device_id in self.devices.keys():
            device_status = await self.get_device_status(device_id)
            if device_status:
                devices.append(device_status)
        
        return devices