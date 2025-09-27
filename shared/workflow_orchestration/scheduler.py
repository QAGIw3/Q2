"""
Advanced Workflow Scheduler for Q2 Platform.

Provides sophisticated scheduling capabilities including:
- Cron-based scheduling
- Event-driven triggers
- Dependency-based scheduling
- Resource-aware scheduling
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

try:
    from croniter import croniter
except ImportError:
    # Fallback implementation for basic cron parsing
    class croniter:
        def __init__(self, cron_expr, start_time=None):
            self.cron_expr = cron_expr
            self.start_time = start_time or datetime.now(timezone.utc)
        
        def get_next(self, ret_type=datetime):
            # Simple fallback - just add 1 hour
            return self.start_time + timedelta(hours=1)

from .engine import WorkflowEngine, WorkflowDefinition

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of workflow triggers."""
    CRON = "cron"
    EVENT = "event"
    DEPENDENCY = "dependency"
    MANUAL = "manual"
    RECURRING = "recurring"


class TriggerStatus(Enum):
    """Trigger activation status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ScheduleTrigger:
    """Base class for workflow triggers."""
    id: str
    name: str
    trigger_type: TriggerType
    workflow_id: str
    status: TriggerStatus = TriggerStatus.ACTIVE
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_triggered: Optional[datetime] = None
    next_trigger_time: Optional[datetime] = None
    trigger_count: int = 0
    max_triggers: Optional[int] = None
    
    def is_ready(self) -> bool:
        """Check if trigger is ready to fire."""
        if self.status != TriggerStatus.ACTIVE:
            return False
        
        if self.max_triggers and self.trigger_count >= self.max_triggers:
            return False
        
        return True
    
    def should_trigger(self, current_time: datetime, context: Dict[str, Any]) -> bool:
        """Check if trigger should fire at current time."""
        raise NotImplementedError


@dataclass 
class CronTrigger(ScheduleTrigger):
    """Cron-based workflow trigger."""
    cron_expression: str = ""
    timezone: str = "UTC"
    
    def __post_init__(self):
        """Initialize cron trigger."""
        if not self.cron_expression:
            raise ValueError("Cron expression is required")
        
        self.trigger_type = TriggerType.CRON
        self._update_next_trigger_time()
    
    def should_trigger(self, current_time: datetime, context: Dict[str, Any]) -> bool:
        """Check if cron trigger should fire."""
        if not self.is_ready():
            return False
        
        return self.next_trigger_time and current_time >= self.next_trigger_time
    
    def _update_next_trigger_time(self) -> None:
        """Update next trigger time based on cron expression."""
        try:
            base_time = self.last_triggered or datetime.now(timezone.utc)
            cron = croniter(self.cron_expression, base_time)
            self.next_trigger_time = cron.get_next(datetime)
        except Exception as e:
            logger.error(f"Failed to parse cron expression {self.cron_expression}: {e}")
            self.status = TriggerStatus.ERROR


@dataclass
class EventTrigger(ScheduleTrigger):
    """Event-driven workflow trigger."""
    event_type: str = ""
    event_filters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize event trigger."""
        if not self.event_type:
            raise ValueError("Event type is required")
        
        self.trigger_type = TriggerType.EVENT
    
    def should_trigger(self, current_time: datetime, context: Dict[str, Any]) -> bool:
        """Check if event trigger should fire."""
        if not self.is_ready():
            return False
        
        # Check if matching event is in context
        events = context.get("events", [])
        for event in events:
            if self._matches_event(event):
                return True
        
        return False
    
    def _matches_event(self, event: Dict[str, Any]) -> bool:
        """Check if event matches trigger criteria."""
        if event.get("type") != self.event_type:
            return False
        
        # Apply event filters
        for filter_key, filter_value in self.event_filters.items():
            if event.get(filter_key) != filter_value:
                return False
        
        return True


@dataclass
class DependencyTrigger(ScheduleTrigger):
    """Dependency-based workflow trigger."""
    dependent_workflows: List[str] = field(default_factory=list)
    dependency_mode: str = "all"  # "all" or "any"
    
    def __post_init__(self):
        """Initialize dependency trigger."""
        if not self.dependent_workflows:
            raise ValueError("Dependent workflows are required")
        
        self.trigger_type = TriggerType.DEPENDENCY
    
    def should_trigger(self, current_time: datetime, context: Dict[str, Any]) -> bool:
        """Check if dependency trigger should fire."""
        if not self.is_ready():
            return False
        
        completed_workflows = context.get("completed_workflows", set())
        
        if self.dependency_mode == "all":
            return all(workflow_id in completed_workflows for workflow_id in self.dependent_workflows)
        else:  # "any"
            return any(workflow_id in completed_workflows for workflow_id in self.dependent_workflows)


@dataclass
class RecurringTrigger(ScheduleTrigger):
    """Recurring workflow trigger with interval."""
    interval_seconds: int = 3600  # Default 1 hour
    max_occurrences: Optional[int] = None
    
    def __post_init__(self):
        """Initialize recurring trigger."""
        self.trigger_type = TriggerType.RECURRING
        if not self.next_trigger_time:
            self.next_trigger_time = datetime.now(timezone.utc) + timedelta(seconds=self.interval_seconds)
    
    def should_trigger(self, current_time: datetime, context: Dict[str, Any]) -> bool:
        """Check if recurring trigger should fire."""
        if not self.is_ready():
            return False
        
        if self.max_occurrences and self.trigger_count >= self.max_occurrences:
            return False
        
        return self.next_trigger_time and current_time >= self.next_trigger_time
    
    def _update_next_trigger_time(self) -> None:
        """Update next trigger time."""
        if self.next_trigger_time:
            self.next_trigger_time += timedelta(seconds=self.interval_seconds)


@dataclass
class ScheduleEntry:
    """Scheduled workflow execution entry."""
    id: str
    trigger_id: str
    workflow_id: str
    scheduled_time: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    execution_id: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowScheduler:
    """
    Advanced workflow scheduler with multiple trigger types.
    
    Features:
    - Cron-based scheduling
    - Event-driven triggers
    - Dependency-based triggers
    - Resource-aware scheduling
    - Trigger management and monitoring
    """
    
    def __init__(self, workflow_engine: Optional[WorkflowEngine] = None):
        self._workflow_engine = workflow_engine
        self._triggers: Dict[str, ScheduleTrigger] = {}
        self._schedule_entries: Dict[str, ScheduleEntry] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._is_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._check_interval = 60  # Check triggers every 60 seconds
        
    def set_workflow_engine(self, engine: WorkflowEngine) -> None:
        """Set the workflow engine."""
        self._workflow_engine = engine
    
    def register_trigger(self, trigger: ScheduleTrigger) -> None:
        """Register a workflow trigger."""
        self._triggers[trigger.id] = trigger
        logger.info(f"Registered trigger: {trigger.name} ({trigger.id})")
    
    def unregister_trigger(self, trigger_id: str) -> None:
        """Unregister a workflow trigger."""
        if trigger_id in self._triggers:
            del self._triggers[trigger_id]
            logger.info(f"Unregistered trigger: {trigger_id}")
    
    def get_trigger(self, trigger_id: str) -> Optional[ScheduleTrigger]:
        """Get a trigger by ID."""
        return self._triggers.get(trigger_id)
    
    def list_triggers(self, status: Optional[TriggerStatus] = None) -> List[ScheduleTrigger]:
        """List all triggers, optionally filtered by status."""
        triggers = list(self._triggers.values())
        
        if status:
            triggers = [t for t in triggers if t.status == status]
        
        return triggers
    
    def pause_trigger(self, trigger_id: str) -> None:
        """Pause a trigger."""
        if trigger_id in self._triggers:
            self._triggers[trigger_id].status = TriggerStatus.PAUSED
            logger.info(f"Paused trigger: {trigger_id}")
    
    def resume_trigger(self, trigger_id: str) -> None:
        """Resume a paused trigger.""" 
        if trigger_id in self._triggers:
            trigger = self._triggers[trigger_id]
            if trigger.status == TriggerStatus.PAUSED:
                trigger.status = TriggerStatus.ACTIVE
                logger.info(f"Resumed trigger: {trigger_id}")
    
    def schedule_workflow(self, workflow_id: str, scheduled_time: datetime,
                         parameters: Optional[Dict[str, Any]] = None) -> str:
        """Schedule a one-time workflow execution."""
        entry_id = str(uuid4())
        
        entry = ScheduleEntry(
            id=entry_id,
            trigger_id="manual",
            workflow_id=workflow_id,
            scheduled_time=scheduled_time,
            parameters=parameters or {}
        )
        
        self._schedule_entries[entry_id] = entry
        logger.info(f"Scheduled workflow {workflow_id} for {scheduled_time}")
        
        return entry_id
    
    def cancel_scheduled_workflow(self, entry_id: str) -> None:
        """Cancel a scheduled workflow execution."""
        if entry_id in self._schedule_entries:
            entry = self._schedule_entries[entry_id]
            if entry.status == "pending":
                entry.status = "cancelled"
                logger.info(f"Cancelled scheduled workflow: {entry_id}")
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Emit an event to trigger event-based workflows."""
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc),
            "data": event_data
        }
        
        # Call event handlers
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed for {event_type}: {e}")
        
        # Check event triggers
        await self._process_event_triggers([event])
    
    async def start(self) -> None:
        """Start the scheduler."""
        if self._is_running:
            logger.warning("Scheduler is already running")
            return
        
        self._is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Started workflow scheduler")
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        self._is_running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped workflow scheduler")
    
    async def get_schedule_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics."""
        active_triggers = len([t for t in self._triggers.values() if t.status == TriggerStatus.ACTIVE])
        pending_entries = len([e for e in self._schedule_entries.values() if e.status == "pending"])
        
        return {
            "is_running": self._is_running,
            "total_triggers": len(self._triggers),
            "active_triggers": active_triggers,
            "pending_schedules": pending_entries,
            "next_check": self._get_next_check_time(),
            "uptime_seconds": (datetime.now(timezone.utc) - self._start_time).total_seconds() if hasattr(self, '_start_time') else 0
        }
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        self._start_time = datetime.now(timezone.utc)
        
        try:
            while self._is_running:
                current_time = datetime.now(timezone.utc)
                
                # Process scheduled entries
                await self._process_scheduled_entries(current_time)
                
                # Process triggers
                await self._process_triggers(current_time)
                
                # Sleep until next check
                await asyncio.sleep(self._check_interval)
                
        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
    
    async def _process_scheduled_entries(self, current_time: datetime) -> None:
        """Process pending scheduled entries."""
        pending_entries = [
            entry for entry in self._schedule_entries.values()
            if entry.status == "pending" and entry.scheduled_time <= current_time
        ]
        
        for entry in pending_entries:
            try:
                if self._workflow_engine:
                    execution_id = await self._workflow_engine.start_execution(
                        entry.workflow_id,
                        entry.parameters
                    )
                    entry.execution_id = execution_id
                    entry.status = "running"
                    
                    logger.info(f"Started scheduled workflow: {entry.workflow_id} (execution: {execution_id})")
                else:
                    logger.warning(f"No workflow engine available for scheduled entry: {entry.id}")
                    entry.status = "failed"
                    
            except Exception as e:
                logger.error(f"Failed to start scheduled workflow {entry.id}: {e}")
                entry.status = "failed"
    
    async def _process_triggers(self, current_time: datetime) -> None:
        """Process all active triggers."""
        context = await self._build_trigger_context(current_time)
        
        for trigger in self._triggers.values():
            try:
                if trigger.should_trigger(current_time, context):
                    await self._fire_trigger(trigger, current_time)
            except Exception as e:
                logger.error(f"Error processing trigger {trigger.id}: {e}")
                trigger.status = TriggerStatus.ERROR
    
    async def _process_event_triggers(self, events: List[Dict[str, Any]]) -> None:
        """Process event-based triggers."""
        context = {"events": events}
        current_time = datetime.now(timezone.utc)
        
        event_triggers = [
            trigger for trigger in self._triggers.values()
            if trigger.trigger_type == TriggerType.EVENT
        ]
        
        for trigger in event_triggers:
            try:
                if trigger.should_trigger(current_time, context):
                    await self._fire_trigger(trigger, current_time)
            except Exception as e:
                logger.error(f"Error processing event trigger {trigger.id}: {e}")
                trigger.status = TriggerStatus.ERROR
    
    async def _fire_trigger(self, trigger: ScheduleTrigger, current_time: datetime) -> None:
        """Fire a trigger and start workflow execution."""
        try:
            if self._workflow_engine:
                execution_id = await self._workflow_engine.start_execution(
                    trigger.workflow_id,
                    trigger.parameters
                )
                
                # Update trigger state
                trigger.last_triggered = current_time
                trigger.trigger_count += 1
                
                # Update next trigger time for recurring triggers
                if hasattr(trigger, '_update_next_trigger_time'):
                    trigger._update_next_trigger_time()
                
                logger.info(f"Triggered workflow {trigger.workflow_id} via trigger {trigger.id} (execution: {execution_id})")
            else:
                logger.warning(f"No workflow engine available for trigger: {trigger.id}")
                
        except Exception as e:
            logger.error(f"Failed to fire trigger {trigger.id}: {e}")
            trigger.status = TriggerStatus.ERROR
    
    async def _build_trigger_context(self, current_time: datetime) -> Dict[str, Any]:
        """Build context for trigger evaluation."""
        # This would integrate with workflow engine to get completed workflows
        completed_workflows = set()
        
        if self._workflow_engine:
            try:
                executions = await self._workflow_engine.list_executions()
                completed_workflows = {
                    exec.workflow_id for exec in executions
                    if exec.status.value == "completed"
                }
            except Exception as e:
                logger.error(f"Failed to get completed workflows: {e}")
        
        return {
            "current_time": current_time,
            "completed_workflows": completed_workflows,
        }
    
    def _get_next_check_time(self) -> Optional[datetime]:
        """Get the next scheduled check time."""
        if self._is_running:
            return datetime.now(timezone.utc) + timedelta(seconds=self._check_interval)
        return None
    
    def create_cron_trigger(self, name: str, workflow_id: str, cron_expression: str,
                          parameters: Optional[Dict[str, Any]] = None) -> CronTrigger:
        """Create and register a cron trigger."""
        trigger = CronTrigger(
            id=str(uuid4()),
            name=name,
            workflow_id=workflow_id,
            cron_expression=cron_expression,
            parameters=parameters or {}
        )
        
        self.register_trigger(trigger)
        return trigger
    
    def create_event_trigger(self, name: str, workflow_id: str, event_type: str,
                           event_filters: Optional[Dict[str, Any]] = None,
                           parameters: Optional[Dict[str, Any]] = None) -> EventTrigger:
        """Create and register an event trigger."""
        trigger = EventTrigger(
            id=str(uuid4()),
            name=name,
            workflow_id=workflow_id,
            event_type=event_type,
            event_filters=event_filters or {},
            parameters=parameters or {}
        )
        
        self.register_trigger(trigger)
        return trigger
    
    def create_dependency_trigger(self, name: str, workflow_id: str, 
                                dependent_workflows: List[str],
                                dependency_mode: str = "all",
                                parameters: Optional[Dict[str, Any]] = None) -> DependencyTrigger:
        """Create and register a dependency trigger."""
        trigger = DependencyTrigger(
            id=str(uuid4()),
            name=name,
            workflow_id=workflow_id,
            dependent_workflows=dependent_workflows,
            dependency_mode=dependency_mode,
            parameters=parameters or {}
        )
        
        self.register_trigger(trigger)
        return trigger
    
    def create_recurring_trigger(self, name: str, workflow_id: str, interval_seconds: int,
                               max_occurrences: Optional[int] = None,
                               parameters: Optional[Dict[str, Any]] = None) -> RecurringTrigger:
        """Create and register a recurring trigger."""
        trigger = RecurringTrigger(
            id=str(uuid4()),
            name=name,
            workflow_id=workflow_id,
            interval_seconds=interval_seconds,
            max_occurrences=max_occurrences,
            parameters=parameters or {}
        )
        
        self.register_trigger(trigger)
        return trigger


# Global instance
workflow_scheduler = WorkflowScheduler()