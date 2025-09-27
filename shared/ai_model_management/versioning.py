"""
Model Versioning and Repository Management for Q2 Platform.

Provides advanced model versioning capabilities including:
- Semantic versioning
- Model rollback and promotion
- Version-based deployment strategies
- Model artifact management
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import logging

try:
    from shared.error_handling import Q2Exception
    from shared.observability import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback for testing
    logger = logging.getLogger(__name__)
    
    class Q2Exception(Exception):
        pass


class VersionStatus(Enum):
    """Version lifecycle status."""
    DRAFT = "draft"
    STAGING = "staging" 
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Model version information."""
    name: str
    version: str
    status: VersionStatus
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_name -> path
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    parent_version: Optional[str] = None
    
    def __post_init__(self):
        """Validate version after initialization."""
        if not self.version:
            raise ValueError("Version cannot be empty")
        if not self.name:
            raise ValueError("Model name cannot be empty")


class VersionNotFoundError(Q2Exception):
    """Raised when a version is not found."""
    pass


class VersionConflictError(Q2Exception):
    """Raised when there's a version conflict."""
    pass


class ModelRepository:
    """
    Model repository for managing versions and artifacts.
    
    Provides:
    - Version management and history
    - Artifact storage and retrieval
    - Rollback and promotion capabilities
    - Version-based deployment strategies
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        self._versions: Dict[str, Dict[str, ModelVersion]] = {}  # model_name -> {version -> ModelVersion}
        self._latest_versions: Dict[str, str] = {}  # model_name -> latest_version
        self._production_versions: Dict[str, str] = {}  # model_name -> production_version
        self._base_path = base_path or Path("/tmp/model_repository")
        self._base_path.mkdir(parents=True, exist_ok=True)
    
    def create_version(self, name: str, version: str, artifacts: Dict[str, Any], 
                      metadata: Optional[Dict[str, Any]] = None, 
                      created_by: Optional[str] = None,
                      tags: Optional[Set[str]] = None,
                      parent_version: Optional[str] = None) -> ModelVersion:
        """Create a new model version."""
        
        # Validate input
        if not name or not version:
            raise ValueError("Model name and version are required")
        
        if name in self._versions and version in self._versions[name]:
            raise VersionConflictError(f"Version {version} already exists for model {name}")
        
        # Validate parent version exists if specified  
        if parent_version and name in self._versions:
            if parent_version not in self._versions[name]:
                raise VersionNotFoundError(f"Parent version {parent_version} not found for model {name}")
        
        # Calculate checksum from artifacts
        checksum = self._calculate_checksum(artifacts, metadata or {})
        
        # Create version directory
        version_path = self._base_path / name / version
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Store artifacts
        artifact_paths = {}
        for artifact_name, artifact_data in artifacts.items():
            artifact_path = version_path / f"{artifact_name}.json"
            with open(artifact_path, 'w') as f:
                json.dump(artifact_data, f, indent=2)
            artifact_paths[artifact_name] = str(artifact_path)
        
        # Create model version
        model_version = ModelVersion(
            name=name,
            version=version,
            status=VersionStatus.DRAFT,
            checksum=checksum,
            metadata=metadata or {},
            artifacts=artifact_paths,
            created_by=created_by,
            tags=tags or set(),
            parent_version=parent_version
        )
        
        # Store version
        if name not in self._versions:
            self._versions[name] = {}
        
        self._versions[name][version] = model_version
        self._latest_versions[name] = version
        
        logger.info(f"Created version {version} for model {name}")
        return model_version
    
    def get_version(self, name: str, version: str) -> ModelVersion:
        """Get a specific model version."""
        if name not in self._versions or version not in self._versions[name]:
            raise VersionNotFoundError(f"Version {version} not found for model {name}")
        
        return self._versions[name][version]
    
    def get_latest_version(self, name: str) -> ModelVersion:
        """Get the latest version of a model."""
        if name not in self._latest_versions:
            raise VersionNotFoundError(f"No versions found for model {name}")
        
        latest_version = self._latest_versions[name]
        return self._versions[name][latest_version]
    
    def get_production_version(self, name: str) -> Optional[ModelVersion]:
        """Get the current production version of a model."""
        if name not in self._production_versions:
            return None
        
        production_version = self._production_versions[name]
        return self._versions[name][production_version]
    
    def list_versions(self, name: str, status: Optional[VersionStatus] = None) -> List[ModelVersion]:
        """List all versions of a model, optionally filtered by status."""
        if name not in self._versions:
            return []
        
        versions = list(self._versions[name].values())
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions
    
    def list_models(self) -> List[str]:
        """List all models in the repository."""
        return list(self._versions.keys())
    
    def promote_version(self, name: str, version: str, target_status: VersionStatus) -> ModelVersion:
        """Promote a version to a different status."""
        model_version = self.get_version(name, version)
        
        # Validate promotion path
        valid_promotions = {
            VersionStatus.DRAFT: [VersionStatus.STAGING, VersionStatus.ARCHIVED],
            VersionStatus.STAGING: [VersionStatus.PRODUCTION, VersionStatus.DEPRECATED],
            VersionStatus.PRODUCTION: [VersionStatus.DEPRECATED],
            VersionStatus.DEPRECATED: [VersionStatus.ARCHIVED],
        }
        
        if target_status not in valid_promotions.get(model_version.status, []):
            raise ValueError(f"Cannot promote from {model_version.status.value} to {target_status.value}")
        
        # If promoting to production, demote current production version
        if target_status == VersionStatus.PRODUCTION and name in self._production_versions:
            current_prod_version = self._production_versions[name]
            if current_prod_version != version:
                current_prod = self._versions[name][current_prod_version]
                current_prod.status = VersionStatus.DEPRECATED
                logger.info(f"Demoted version {current_prod_version} from production to deprecated")
        
        # Update status
        model_version.status = target_status
        
        if target_status == VersionStatus.PRODUCTION:
            self._production_versions[name] = version
        
        logger.info(f"Promoted version {version} of model {name} to {target_status.value}")
        return model_version
    
    def rollback_production(self, name: str, target_version: Optional[str] = None) -> ModelVersion:
        """
        Rollback production to a previous version.
        If target_version is not specified, rollback to the previous production version.
        """
        if name not in self._production_versions:
            raise VersionNotFoundError(f"No production version found for model {name}")
        
        current_prod_version = self._production_versions[name]
        
        if target_version:
            # Rollback to specific version
            target = self.get_version(name, target_version)
            if target.status == VersionStatus.ARCHIVED:
                raise ValueError("Cannot rollback to archived version")
        else:
            # Find previous production version from history
            versions = self.list_versions(name)
            production_versions = [
                v for v in versions 
                if v.status in [VersionStatus.PRODUCTION, VersionStatus.DEPRECATED]
                and v.version != current_prod_version
            ]
            
            if not production_versions:
                raise VersionNotFoundError(f"No previous production version found for model {name}")
            
            target = production_versions[0]  # Most recent non-current production version
        
        # Perform rollback
        current_prod = self._versions[name][current_prod_version]
        current_prod.status = VersionStatus.DEPRECATED
        
        target.status = VersionStatus.PRODUCTION
        self._production_versions[name] = target.version
        
        logger.info(f"Rolled back model {name} from version {current_prod_version} to {target.version}")
        return target
    
    def delete_version(self, name: str, version: str, force: bool = False) -> None:
        """Delete a model version."""
        model_version = self.get_version(name, version)
        
        # Prevent deletion of production versions unless forced
        if model_version.status == VersionStatus.PRODUCTION and not force:
            raise ValueError("Cannot delete production version without force=True")
        
        # Remove from production tracking if necessary
        if name in self._production_versions and self._production_versions[name] == version:
            del self._production_versions[name]
        
        # Update latest version pointer if necessary
        if name in self._latest_versions and self._latest_versions[name] == version:
            remaining_versions = [v for k, v in self._versions[name].items() if k != version]
            if remaining_versions:
                # Set latest to most recently created version
                latest = max(remaining_versions, key=lambda v: v.created_at)
                self._latest_versions[name] = latest.version
            else:
                del self._latest_versions[name]
        
        # Remove artifacts from filesystem
        version_path = self._base_path / name / version
        if version_path.exists():
            import shutil
            shutil.rmtree(version_path)
        
        # Remove from memory
        del self._versions[name][version]
        
        # Clean up model entry if no versions remain
        if not self._versions[name]:
            del self._versions[name]
        
        logger.info(f"Deleted version {version} of model {name}")
    
    def get_version_history(self, name: str) -> List[Dict[str, Any]]:
        """Get version history for a model."""
        if name not in self._versions:
            return []
        
        versions = list(self._versions[name].values())
        versions.sort(key=lambda v: v.created_at)
        
        history = []
        for version in versions:
            history.append({
                "version": version.version,
                "status": version.status.value,
                "created_at": version.created_at.isoformat(),
                "created_by": version.created_by,
                "checksum": version.checksum,
                "tags": list(version.tags),
                "parent_version": version.parent_version,
            })
        
        return history
    
    def compare_versions(self, name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a model."""
        v1 = self.get_version(name, version1)
        v2 = self.get_version(name, version2)
        
        return {
            "version1": {
                "version": v1.version,
                "status": v1.status.value,
                "checksum": v1.checksum,
                "created_at": v1.created_at.isoformat(),
                "metadata": v1.metadata,
                "tags": list(v1.tags),
            },
            "version2": {
                "version": v2.version,
                "status": v2.status.value,
                "checksum": v2.checksum,
                "created_at": v2.created_at.isoformat(),
                "metadata": v2.metadata,
                "tags": list(v2.tags),
            },
            "differences": {
                "checksum_changed": v1.checksum != v2.checksum,
                "metadata_changed": v1.metadata != v2.metadata,
                "tags_changed": v1.tags != v2.tags,
            }
        }
    
    def _calculate_checksum(self, artifacts: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Calculate checksum for artifacts and metadata."""
        combined_data = {
            "artifacts": artifacts,
            "metadata": metadata
        }
        
        # Create deterministic JSON string
        json_str = json.dumps(combined_data, sort_keys=True, separators=(',', ':'))
        
        # Calculate SHA-256 hash
        return hashlib.sha256(json_str.encode()).hexdigest()


class VersionManager:
    """
    High-level version management interface.
    
    Provides simplified access to versioning operations and integrates
    with the Advanced Model Manager.
    """
    
    def __init__(self, repository: Optional[ModelRepository] = None):
        self._repository = repository or ModelRepository()
    
    @property
    def repository(self) -> ModelRepository:
        """Get the underlying repository."""
        return self._repository
    
    async def create_model_version(self, name: str, version: str, 
                                 model_artifacts: Dict[str, Any],
                                 metadata: Optional[Dict[str, Any]] = None,
                                 created_by: Optional[str] = None,
                                 tags: Optional[Set[str]] = None) -> ModelVersion:
        """Create a new model version with async support."""
        return self._repository.create_version(
            name=name,
            version=version,
            artifacts=model_artifacts,
            metadata=metadata,
            created_by=created_by,
            tags=tags
        )
    
    async def deploy_version(self, name: str, version: str, 
                           environment: str = "production") -> ModelVersion:
        """Deploy a version to an environment."""
        if environment == "production":
            return self._repository.promote_version(name, version, VersionStatus.PRODUCTION)
        elif environment == "staging":
            return self._repository.promote_version(name, version, VersionStatus.STAGING)
        else:
            raise ValueError(f"Unknown environment: {environment}")
    
    async def rollback_model(self, name: str, target_version: Optional[str] = None) -> ModelVersion:
        """Rollback a model to a previous version."""
        return self._repository.rollback_production(name, target_version)
    
    async def get_deployable_versions(self, name: str) -> List[ModelVersion]:
        """Get versions that can be deployed to production."""
        return self._repository.list_versions(name, VersionStatus.STAGING)


# Global instances
default_repository = ModelRepository()
version_manager = VersionManager(default_repository)