"""
Disaster Recovery and Backup Strategies for Q2 Platform

Implements automated backup, recovery procedures, and disaster recovery planning.
"""

import os
import json
import asyncio
import shutil
import tarfile
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from shared.security_config import get_security_audit_logger
from shared.vault_client import VaultClient


class BackupType(Enum):
    """Types of backups supported"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):
    """Backup operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackupJob:
    """Represents a backup job"""
    job_id: str
    job_name: str
    backup_type: BackupType
    source_paths: List[str]
    destination: str
    schedule: str  # Cron-like schedule
    retention_days: int
    compression: bool = True
    encryption: bool = True
    status: BackupStatus = BackupStatus.PENDING
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    size_bytes: Optional[int] = None
    error_message: Optional[str] = None


class DisasterRecoveryManager:
    """Manages disaster recovery and backup operations"""
    
    def __init__(
        self,
        backup_root: str = "/var/q2-backups",
        vault_client: Optional[VaultClient] = None
    ):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        self.vault_client = vault_client
        self.audit_logger = get_security_audit_logger()
        self.logger = logging.getLogger(__name__)
        
        # Active backup jobs
        self.backup_jobs: Dict[str, BackupJob] = {}
        
        # Recovery points
        self.recovery_catalog = self._load_recovery_catalog()
        
    def _load_recovery_catalog(self) -> Dict[str, Any]:
        """Load the recovery catalog from disk"""
        catalog_path = self.backup_root / "recovery_catalog.json"
        
        if catalog_path.exists():
            try:
                with open(catalog_path) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load recovery catalog: {e}")
        
        return {"backups": [], "last_updated": None}
    
    def _save_recovery_catalog(self):
        """Save the recovery catalog to disk"""
        catalog_path = self.backup_root / "recovery_catalog.json"
        
        try:
            with open(catalog_path, 'w') as f:
                json.dump(self.recovery_catalog, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save recovery catalog: {e}")
    
    async def create_backup_job(
        self,
        job_name: str,
        backup_type: BackupType,
        source_paths: List[str],
        destination: str,
        retention_days: int = 30,
        compression: bool = True,
        encryption: bool = True
    ) -> BackupJob:
        """Create a new backup job"""
        
        job_id = f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{job_name}"
        
        job = BackupJob(
            job_id=job_id,
            job_name=job_name,
            backup_type=backup_type,
            source_paths=source_paths,
            destination=destination,
            schedule="",  # Set externally if needed
            retention_days=retention_days,
            compression=compression,
            encryption=encryption,
            created_at=datetime.now(timezone.utc)
        )
        
        self.backup_jobs[job_id] = job
        
        self.audit_logger.log_security_event(
            "backup_job_created",
            "INFO",
            {
                "job_id": job_id,
                "job_name": job_name,
                "backup_type": backup_type.value,
                "source_paths": source_paths
            }
        )
        
        return job
    
    async def execute_backup(self, job_id: str) -> bool:
        """Execute a backup job"""
        if job_id not in self.backup_jobs:
            self.logger.error(f"Backup job {job_id} not found")
            return False
        
        job = self.backup_jobs[job_id]
        job.status = BackupStatus.IN_PROGRESS
        
        try:
            self.audit_logger.log_security_event(
                "backup_started",
                "INFO",
                {"job_id": job_id, "job_name": job.job_name}
            )
            
            # Create backup directory
            backup_dir = self.backup_root / job.job_name / job_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup archive
            archive_path = backup_dir / f"{job_id}.tar"
            if job.compression:
                archive_path = backup_dir / f"{job_id}.tar.gz"
                
            total_size = 0
            
            with tarfile.open(archive_path, 'w:gz' if job.compression else 'w') as tar:
                for source_path in job.source_paths:
                    source = Path(source_path)
                    
                    if source.exists():
                        if source.is_file():
                            tar.add(source, arcname=source.name)
                            total_size += source.stat().st_size
                        elif source.is_dir():
                            tar.add(source, arcname=source.name)
                            total_size += sum(
                                f.stat().st_size 
                                for f in source.rglob('*') 
                                if f.is_file()
                            )
                        
                        self.logger.info(f"Added {source} to backup")
                    else:
                        self.logger.warning(f"Source path {source} does not exist")
            
            # Encrypt backup if requested
            if job.encryption:
                await self._encrypt_backup(archive_path)
            
            # Create metadata file
            metadata = {
                "job_id": job_id,
                "job_name": job.job_name,
                "backup_type": job.backup_type.value,
                "created_at": job.created_at.isoformat(),
                "source_paths": job.source_paths,
                "size_bytes": total_size,
                "compressed": job.compression,
                "encrypted": job.encryption
            }
            
            metadata_path = backup_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update job status
            job.status = BackupStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.size_bytes = total_size
            
            # Update recovery catalog
            self.recovery_catalog["backups"].append(metadata)
            self.recovery_catalog["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._save_recovery_catalog()
            
            self.audit_logger.log_security_event(
                "backup_completed",
                "INFO",
                {
                    "job_id": job_id,
                    "size_bytes": total_size,
                    "duration_seconds": (
                        job.completed_at - job.created_at
                    ).total_seconds()
                }
            )
            
            # Schedule cleanup of old backups
            await self._cleanup_old_backups(job.job_name, job.retention_days)
            
            return True
            
        except Exception as e:
            job.status = BackupStatus.FAILED
            job.error_message = str(e)
            
            self.audit_logger.log_security_event(
                "backup_failed",
                "ERROR",
                {"job_id": job_id, "error": str(e)}
            )
            
            self.logger.error(f"Backup job {job_id} failed: {e}")
            return False
    
    async def restore_from_backup(
        self,
        backup_id: str,
        restore_path: str,
        selective_restore: Optional[List[str]] = None
    ) -> bool:
        """Restore from a backup"""
        
        # Find backup in catalog
        backup_metadata = None
        for backup in self.recovery_catalog["backups"]:
            if backup["job_id"] == backup_id:
                backup_metadata = backup
                break
        
        if not backup_metadata:
            self.logger.error(f"Backup {backup_id} not found in catalog")
            return False
        
        try:
            self.audit_logger.log_security_event(
                "restore_started",
                "INFO",
                {"backup_id": backup_id, "restore_path": restore_path}
            )
            
            # Locate backup archive
            backup_dir = self.backup_root / backup_metadata["job_name"] / backup_id
            archive_files = list(backup_dir.glob("*.tar*"))
            
            if not archive_files:
                self.logger.error(f"No archive found for backup {backup_id}")
                return False
            
            archive_path = archive_files[0]
            
            # Decrypt if necessary
            if backup_metadata.get("encrypted", False):
                archive_path = await self._decrypt_backup(archive_path)
            
            # Create restore directory
            restore_dir = Path(restore_path)
            restore_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract archive with security validation
            with tarfile.open(archive_path, 'r:gz' if 'gz' in archive_path.name else 'r') as tar:
                if selective_restore:
                    # Extract only specified files/directories
                    for member in tar.getmembers():
                        if any(pattern in member.name for pattern in selective_restore):
                            # Security validation
                            if self._is_safe_member(member, restore_dir):
                                tar.extract(member, restore_dir)
                else:
                    # Extract all with validation
                    safe_members = [m for m in tar.getmembers() if self._is_safe_member(m, restore_dir)]
                    tar.extractall(restore_dir, members=safe_members)
            
            self.audit_logger.log_security_event(
                "restore_completed",
                "INFO",
                {"backup_id": backup_id, "restore_path": restore_path}
            )
            
            self.logger.info(f"Restore from backup {backup_id} completed")
            return True
            
        except Exception as e:
            self.audit_logger.log_security_event(
                "restore_failed",
                "ERROR",
                {"backup_id": backup_id, "error": str(e)}
            )
            
            self.logger.error(f"Restore from backup {backup_id} failed: {e}")
            return False
    
    async def list_backups(self, job_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = self.recovery_catalog["backups"]
        
        if job_name:
            backups = [b for b in backups if b["job_name"] == job_name]
        
        return sorted(backups, key=lambda x: x["created_at"], reverse=True)
    
    async def verify_backup_integrity(self, backup_id: str) -> bool:
        """Verify the integrity of a backup"""
        try:
            # Find backup metadata
            backup_metadata = None
            for backup in self.recovery_catalog["backups"]:
                if backup["job_id"] == backup_id:
                    backup_metadata = backup
                    break
            
            if not backup_metadata:
                return False
            
            # Check if backup files exist
            backup_dir = self.backup_root / backup_metadata["job_name"] / backup_id
            
            if not backup_dir.exists():
                return False
            
            # Check archive integrity
            archive_files = list(backup_dir.glob("*.tar*"))
            if not archive_files:
                return False
            
            archive_path = archive_files[0]
            
            # Verify archive can be opened
            with tarfile.open(archive_path, 'r:gz' if 'gz' in archive_path.name else 'r') as tar:
                # Try to list all members
                members = tar.getmembers()
                
            self.logger.info(f"Backup {backup_id} integrity verified ({len(members)} files)")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup integrity verification failed for {backup_id}: {e}")
            return False
    
    async def _encrypt_backup(self, archive_path: Path) -> Path:
        """Encrypt a backup archive"""
        if not self.vault_client:
            self.logger.warning("No vault client available for encryption")
            return archive_path
        
        try:
            from shared.security_config import get_encryption_manager
            encryption_manager = get_encryption_manager()
            
            # Read archive
            with open(archive_path, 'rb') as f:
                archive_data = f.read()
            
            # Encrypt
            encrypted_data = encryption_manager.encrypt(archive_data.decode('latin-1'))
            
            # Write encrypted archive
            encrypted_path = archive_path.with_suffix(archive_path.suffix + '.encrypted')
            with open(encrypted_path, 'w') as f:
                f.write(encrypted_data)
            
            # Remove original
            archive_path.unlink()
            
            return encrypted_path
            
        except Exception as e:
            self.logger.error(f"Backup encryption failed: {e}")
            return archive_path
    
    async def _decrypt_backup(self, encrypted_path: Path) -> Path:
        """Decrypt a backup archive"""
        try:
            from shared.security_config import get_encryption_manager
            encryption_manager = get_encryption_manager()
            
            # Read encrypted data
            with open(encrypted_path, 'r') as f:
                encrypted_data = f.read()
            
            # Decrypt
            decrypted_data = encryption_manager.decrypt(encrypted_data)
            
            # Write decrypted archive
            decrypted_path = encrypted_path.with_suffix('')  # Remove .encrypted
            with open(decrypted_path, 'wb') as f:
                f.write(decrypted_data.encode('latin-1'))
            
            return decrypted_path
            
        except Exception as e:
            self.logger.error(f"Backup decryption failed: {e}")
            return encrypted_path
    
    async def _cleanup_old_backups(self, job_name: str, retention_days: int):
        """Clean up old backups based on retention policy"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        backups_to_remove = []
        
        for backup in self.recovery_catalog["backups"]:
            if backup["job_name"] == job_name:
                backup_date = datetime.fromisoformat(backup["created_at"])
                if backup_date < cutoff_date:
                    backups_to_remove.append(backup)
        
        for backup in backups_to_remove:
            try:
                # Remove backup directory
                backup_dir = self.backup_root / backup["job_name"] / backup["job_id"]
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                
                # Remove from catalog
                self.recovery_catalog["backups"].remove(backup)
                
                self.audit_logger.log_security_event(
                    "backup_cleaned_up",
                    "INFO",
                    {"backup_id": backup["job_id"], "age_days": (datetime.now(timezone.utc) - datetime.fromisoformat(backup["created_at"])).days}
                )
                
            except Exception as e:
                self.logger.error(f"Failed to clean up backup {backup['job_id']}: {e}")
        
        if backups_to_remove:
            self._save_recovery_catalog()
    
    def _is_safe_member(self, member: tarfile.TarInfo, extract_path: Path) -> bool:
        """Validate that a tar member is safe to extract"""
        # Check for directory traversal attacks
        if member.name.startswith('/') or '..' in member.name:
            self.logger.warning(f"Unsafe member path detected: {member.name}")
            return False
        
        # Check if the resolved path is within the extraction directory
        member_path = (extract_path / member.name).resolve()
        if not str(member_path).startswith(str(extract_path.resolve())):
            self.logger.warning(f"Member path outside extraction directory: {member.name}")
            return False
        
        # Check for suspicious file types
        if member.name.endswith(('.exe', '.bat', '.cmd', '.scr')):
            self.logger.warning(f"Potentially dangerous file type: {member.name}")
            return False
        
        return True
    
    async def create_disaster_recovery_plan(self) -> Dict[str, Any]:
        """Create a comprehensive disaster recovery plan"""
        
        plan = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "recovery_time_objective_hours": 4,  # RTO
            "recovery_point_objective_hours": 1,  # RPO
            "critical_services": [
                "agentQ",
                "managerQ", 
                "VectorStoreQ",
                "KnowledgeGraphQ",
                "AuthQ"
            ],
            "backup_schedule": {
                "full_backup": "weekly",
                "incremental_backup": "daily",
                "config_backup": "daily"
            },
            "recovery_procedures": {
                "service_failure": {
                    "detection": "Automated monitoring alerts",
                    "response_time": "< 15 minutes",
                    "steps": [
                        "Identify failed service",
                        "Check service health endpoints",
                        "Restart service containers",
                        "If restart fails, restore from backup",
                        "Notify operations team"
                    ]
                },
                "data_corruption": {
                    "detection": "Data integrity checks",
                    "response_time": "< 30 minutes",
                    "steps": [
                        "Isolate corrupted data",
                        "Identify last known good backup",
                        "Restore from backup to staging",
                        "Validate restored data",
                        "Promote to production",
                        "Analyze root cause"
                    ]
                },
                "infrastructure_failure": {
                    "detection": "Infrastructure monitoring",
                    "response_time": "< 1 hour",
                    "steps": [
                        "Assess scope of failure",
                        "Activate secondary infrastructure",
                        "Restore services from backups",
                        "Update DNS/load balancers",
                        "Validate all services operational"
                    ]
                }
            },
            "contact_information": {
                "primary_oncall": "ops-team@q-platform.com",
                "secondary_oncall": "engineering-lead@q-platform.com",
                "escalation": "cto@q-platform.com"
            },
            "testing_schedule": {
                "recovery_drill": "monthly",
                "backup_verification": "weekly",
                "failover_test": "quarterly"
            }
        }
        
        # Save plan to disk
        plan_path = self.backup_root / "disaster_recovery_plan.json"
        with open(plan_path, 'w') as f:
            json.dump(plan, f, indent=2)
        
        self.audit_logger.log_security_event(
            "disaster_recovery_plan_created",
            "INFO",
            {"plan_path": str(plan_path)}
        )
        
        return plan


# Global disaster recovery manager
_dr_manager: Optional[DisasterRecoveryManager] = None


def get_disaster_recovery_manager() -> DisasterRecoveryManager:
    """Get the global disaster recovery manager instance"""
    global _dr_manager
    if _dr_manager is None:
        vault_client = None
        try:
            vault_client = VaultClient()
        except Exception:
            pass  # Vault not available
        
        _dr_manager = DisasterRecoveryManager(vault_client=vault_client)
    
    return _dr_manager


# Utility functions for common backup scenarios
async def backup_service_data(service_name: str, data_paths: List[str]) -> bool:
    """Backup data for a specific service"""
    dr_manager = get_disaster_recovery_manager()
    
    job = await dr_manager.create_backup_job(
        job_name=f"{service_name}_data",
        backup_type=BackupType.FULL,
        source_paths=data_paths,
        destination=f"/var/q2-backups/{service_name}",
        retention_days=30
    )
    
    return await dr_manager.execute_backup(job.job_id)


async def backup_service_config(service_name: str, config_paths: List[str]) -> bool:
    """Backup configuration for a specific service"""
    dr_manager = get_disaster_recovery_manager()
    
    job = await dr_manager.create_backup_job(
        job_name=f"{service_name}_config",
        backup_type=BackupType.INCREMENTAL,
        source_paths=config_paths,
        destination=f"/var/q2-backups/{service_name}",
        retention_days=90
    )
    
    return await dr_manager.execute_backup(job.job_id)