"""
Tests for Advanced Workflow Orchestration system.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from shared.workflow_orchestration import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowStep,
    StepType,
    ExecutionStatus,
    WorkflowTemplate,
    ParameterDefinition,
    ParameterType,
    TemplateManager,
    WorkflowScheduler,
    CronTrigger,
    EventTrigger,
    RecurringTrigger,
    TriggerStatus,
    Condition,
    ConditionOperator,
    ConditionGroup,
    ConditionalStep,
    ParallelStep,
    LoopStep,
    ConditionEvaluator,
)


class TestWorkflowEngine:
    """Test cases for WorkflowEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create a test workflow engine."""
        return WorkflowEngine(max_concurrent_workflows=5)
    
    @pytest.fixture
    def simple_workflow(self):
        """Create a simple test workflow."""
        steps = [
            WorkflowStep(
                id="step1",
                name="Log Step",
                type=StepType.ACTION,
                parameters={"action_name": "log", "message": "Hello World", "level": "info"}
            ),
            WorkflowStep(
                id="step2", 
                name="Delay Step",
                type=StepType.ACTION,
                parameters={"action_name": "delay", "seconds": 0.1},
                dependencies=["step1"]
            )
        ]
        
        return WorkflowDefinition(
            id="test_workflow",
            name="Test Workflow",
            description="Simple test workflow",
            steps=steps
        )
    
    def test_register_workflow(self, engine, simple_workflow):
        """Test workflow registration."""
        engine.register_workflow(simple_workflow)
        
        assert simple_workflow.id in engine._workflows
        assert engine._workflows[simple_workflow.id].name == "Test Workflow"
    
    def test_register_action(self, engine):
        """Test action registration."""
        def test_action(message: str):
            return {"result": message}
        
        engine.register_action("test_action", test_action)
        
        assert "test_action" in engine._action_registry
        assert engine._action_registry["test_action"] == test_action
    
    @pytest.mark.asyncio
    async def test_start_execution(self, engine, simple_workflow):
        """Test starting workflow execution."""
        engine.register_workflow(simple_workflow)
        
        execution_id = await engine.start_execution(
            simple_workflow.id,
            {"test_param": "test_value"}
        )
        
        assert execution_id is not None
        assert execution_id in engine._executions
        
        execution = engine._executions[execution_id]
        assert execution.workflow_id == simple_workflow.id
        assert execution.input_parameters["test_param"] == "test_value"
        assert execution.status == ExecutionStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_get_execution_status(self, engine, simple_workflow):
        """Test getting execution status."""
        engine.register_workflow(simple_workflow)
        execution_id = await engine.start_execution(simple_workflow.id)
        
        # Give execution time to start
        await asyncio.sleep(0.2)
        
        execution = await engine.get_execution_status(execution_id)
        
        assert execution.id == execution_id
        assert execution.status in [ExecutionStatus.RUNNING, ExecutionStatus.COMPLETED]
    
    @pytest.mark.asyncio
    async def test_cancel_execution(self, engine, simple_workflow):
        """Test cancelling execution."""
        engine.register_workflow(simple_workflow)
        execution_id = await engine.start_execution(simple_workflow.id)
        
        await engine.cancel_execution(execution_id)
        
        execution = await engine.get_execution_status(execution_id)
        assert execution.status == ExecutionStatus.CANCELLED


class TestWorkflowTemplate:
    """Test cases for WorkflowTemplate."""
    
    @pytest.fixture
    def sample_template(self):
        """Create a sample workflow template."""
        parameters = [
            ParameterDefinition(
                name="input_message",
                type=ParameterType.STRING,
                description="Message to log",
                required=True
            ),
            ParameterDefinition(
                name="delay_seconds",
                type=ParameterType.FLOAT,
                description="Delay in seconds",
                default_value=1.0,
                required=False,
                validation_rules={"min_value": 0.1, "max_value": 10.0}
            )
        ]
        
        steps_template = [
            {
                "id": "log_step",
                "name": "Log Message",
                "type": "action",
                "parameters": {
                    "action_name": "log",
                    "message": "${input_message}",
                    "level": "info"
                }
            },
            {
                "id": "delay_step",
                "name": "Delay",
                "type": "action", 
                "parameters": {
                    "action_name": "delay",
                    "seconds": "${delay_seconds}"
                },
                "dependencies": ["log_step"]
            }
        ]
        
        return WorkflowTemplate(
            id="test_template",
            name="Test Template",
            description="Template for testing",
            parameters=parameters,
            steps_template=steps_template
        )
    
    def test_parameter_validation(self, sample_template):
        """Test parameter validation."""
        # Valid parameters
        valid_params = {
            "input_message": "Hello World",
            "delay_seconds": 2.5
        }
        errors = sample_template.validate_parameters(valid_params)
        assert not errors
        
        # Missing required parameter
        invalid_params = {
            "delay_seconds": 1.0
        }
        errors = sample_template.validate_parameters(invalid_params)
        assert "input_message" in errors
        
        # Invalid parameter value
        invalid_params = {
            "input_message": "Hello",
            "delay_seconds": 15.0  # Exceeds max_value
        }
        errors = sample_template.validate_parameters(invalid_params)
        assert "delay_seconds" in errors
    
    def test_instantiate_template(self, sample_template):
        """Test template instantiation."""
        parameters = {
            "input_message": "Test Message",
            "delay_seconds": 0.5
        }
        
        workflow = sample_template.instantiate(parameters)
        
        assert workflow.name == "Test Template_instance"
        assert len(workflow.steps) == 2
        
        # Check parameter substitution
        log_step = next(step for step in workflow.steps if step.id == "log_step")
        assert log_step.parameters["message"] == "Test Message"
        
        delay_step = next(step for step in workflow.steps if step.id == "delay_step")
        assert delay_step.parameters["seconds"] == 0.5


class TestTemplateManager:
    """Test cases for TemplateManager."""
    
    @pytest.fixture
    def template_manager(self):
        """Create a test template manager."""
        return TemplateManager()
    
    def test_create_common_templates(self, template_manager):
        """Test creating common templates."""
        template_manager.create_common_templates()
        
        templates = template_manager.registry.list_templates()
        assert len(templates) >= 2
        
        # Check data pipeline template exists
        data_pipeline = next(
            (t for t in templates if "Data Processing Pipeline" in t.name), 
            None
        )
        assert data_pipeline is not None
        assert data_pipeline.category == "data_processing"
        
        # Check ML training template exists
        ml_template = next(
            (t for t in templates if "ML Model Training" in t.name),
            None
        )
        assert ml_template is not None
        assert ml_template.category == "machine_learning"


class TestWorkflowScheduler:
    """Test cases for WorkflowScheduler."""
    
    @pytest.fixture
    def scheduler(self):
        """Create a test workflow scheduler."""
        return WorkflowScheduler()
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock workflow engine."""
        engine = Mock()
        engine.start_execution = Mock(return_value=asyncio.Future())
        engine.start_execution.return_value.set_result("mock_execution_id")
        return engine
    
    def test_register_trigger(self, scheduler):
        """Test trigger registration."""
        trigger = CronTrigger(
            id="test_trigger",
            name="Test Cron Trigger",
            workflow_id="test_workflow",
            cron_expression="0 * * * *"  # Every hour
        )
        
        scheduler.register_trigger(trigger)
        
        assert trigger.id in scheduler._triggers
        assert scheduler._triggers[trigger.id].name == "Test Cron Trigger"
    
    def test_create_cron_trigger(self, scheduler):
        """Test creating cron trigger."""
        trigger = scheduler.create_cron_trigger(
            name="Hourly Task",
            workflow_id="hourly_workflow", 
            cron_expression="0 * * * *"
        )
        
        assert trigger.trigger_type.value == "cron"
        assert trigger.cron_expression == "0 * * * *"
        assert trigger.id in scheduler._triggers
    
    def test_create_event_trigger(self, scheduler):
        """Test creating event trigger."""
        trigger = scheduler.create_event_trigger(
            name="File Upload Handler",
            workflow_id="file_workflow",
            event_type="file_uploaded",
            event_filters={"file_type": "csv"}
        )
        
        assert trigger.trigger_type.value == "event"
        assert trigger.event_type == "file_uploaded"
        assert trigger.event_filters["file_type"] == "csv"
    
    def test_create_recurring_trigger(self, scheduler):
        """Test creating recurring trigger."""
        trigger = scheduler.create_recurring_trigger(
            name="Every 5 Minutes",
            workflow_id="monitoring_workflow",
            interval_seconds=300,
            max_occurrences=10
        )
        
        assert trigger.trigger_type.value == "recurring"
        assert trigger.interval_seconds == 300
        assert trigger.max_occurrences == 10
    
    @pytest.mark.asyncio
    async def test_emit_event(self, scheduler, mock_engine):
        """Test event emission and trigger processing."""
        scheduler.set_workflow_engine(mock_engine)
        
        # Create event trigger
        trigger = scheduler.create_event_trigger(
            name="Test Event Trigger",
            workflow_id="event_workflow",
            event_type="test_event"
        )
        
        # Emit matching event
        await scheduler.emit_event("test_event", {"data": "test"})
        
        # Verify workflow was triggered
        assert mock_engine.start_execution.called
    
    @pytest.mark.asyncio
    async def test_schedule_workflow(self, scheduler):
        """Test one-time workflow scheduling."""
        scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        
        entry_id = scheduler.schedule_workflow(
            workflow_id="test_workflow",
            scheduled_time=scheduled_time,
            parameters={"param1": "value1"}
        )
        
        assert entry_id in scheduler._schedule_entries
        entry = scheduler._schedule_entries[entry_id]
        assert entry.workflow_id == "test_workflow"
        assert entry.scheduled_time == scheduled_time
        assert entry.parameters["param1"] == "value1"


class TestConditions:
    """Test cases for conditional logic."""
    
    def test_simple_condition(self):
        """Test simple condition evaluation."""
        condition = Condition(
            field="user.age",
            operator=ConditionOperator.GREATER_THAN,
            value=18
        )
        
        context = {"user": {"age": 25, "name": "John"}}
        assert condition.evaluate(context) is True
        
        context = {"user": {"age": 16, "name": "Jane"}}
        assert condition.evaluate(context) is False
    
    def test_condition_group(self):
        """Test condition group evaluation."""
        conditions = [
            Condition("user.age", ConditionOperator.GREATER_EQUAL, 18),
            Condition("user.status", ConditionOperator.EQUALS, "active")
        ]
        
        group = ConditionGroup(conditions, ConditionOperator.AND)
        
        # Both conditions true
        context = {"user": {"age": 25, "status": "active"}}
        assert group.evaluate(context) is True
        
        # One condition false
        context = {"user": {"age": 16, "status": "active"}}
        assert group.evaluate(context) is False
        
        # Test OR group
        or_group = ConditionGroup(conditions, ConditionOperator.OR)
        assert or_group.evaluate(context) is True  # status is active
    
    def test_condition_evaluator(self):
        """Test condition evaluator utility."""
        condition = ConditionEvaluator.create_simple_condition(
            "score", ">=", 80
        )
        
        context = {"score": 85}
        assert condition.evaluate(context) is True
        
        context = {"score": 75}
        assert condition.evaluate(context) is False
    
    def test_parse_condition_expression(self):
        """Test parsing condition expressions."""
        expression = "score >= 80 AND status == 'active'"
        group = ConditionEvaluator.parse_condition_expression(expression)
        
        context = {"score": 85, "status": "active"}
        assert group.evaluate(context) is True
        
        context = {"score": 75, "status": "active"}
        assert group.evaluate(context) is False


class TestConditionalStep:
    """Test cases for ConditionalStep."""
    
    @pytest.fixture
    def conditional_step(self):
        """Create a conditional step."""
        condition = ConditionGroup([
            Condition("user.role", ConditionOperator.EQUALS, "admin")
        ])
        
        return ConditionalStep(
            id="admin_check",
            name="Admin Check Step", 
            condition_group=condition,
            if_true_steps=["admin_task"],
            if_false_steps=["user_task"]
        )
    
    def test_should_execute(self, conditional_step):
        """Test conditional execution logic."""
        # Admin user
        admin_context = {"user": {"role": "admin"}}
        assert conditional_step.should_execute(admin_context) is True
        
        # Regular user
        user_context = {"user": {"role": "user"}}
        assert conditional_step.should_execute(user_context) is False
    
    def test_get_next_steps(self, conditional_step):
        """Test getting next steps based on condition."""
        admin_context = {"user": {"role": "admin"}}
        next_steps = conditional_step.get_next_steps(admin_context)
        assert next_steps == ["admin_task"]
        
        user_context = {"user": {"role": "user"}}
        next_steps = conditional_step.get_next_steps(user_context)
        assert next_steps == ["user_task"]


class TestParallelStep:
    """Test cases for ParallelStep."""
    
    @pytest.fixture
    def parallel_step(self):
        """Create a parallel step."""
        sub_steps = [
            WorkflowStep(
                id="task1",
                name="Task 1",
                type=StepType.ACTION,
                parameters={"action_name": "log", "message": "Task 1"}
            ),
            WorkflowStep(
                id="task2", 
                name="Task 2",
                type=StepType.ACTION,
                parameters={"action_name": "log", "message": "Task 2"}
            ),
            WorkflowStep(
                id="task3",
                name="Task 3", 
                type=StepType.ACTION,
                parameters={"action_name": "delay", "seconds": 0.1}
            )
        ]
        
        return ParallelStep(
            id="parallel_tasks",
            name="Parallel Tasks",
            parallel_steps=sub_steps,
            wait_for_all=True,
            failure_threshold=0.2
        )
    
    @pytest.mark.asyncio
    async def test_execute_parallel(self, parallel_step):
        """Test parallel execution."""
        results = await parallel_step.execute_parallel()
        
        assert len(results) == 3
        assert "task1" in results
        assert "task2" in results
        assert "task3" in results
        
        # All should be completed
        for result in results.values():
            assert result.status == ExecutionStatus.COMPLETED
    
    def test_check_success_criteria(self, parallel_step):
        """Test success criteria checking."""
        from shared.workflow_orchestration.engine import StepResult, ExecutionStatus
        
        # All successful
        results = {
            "task1": StepResult("task1", ExecutionStatus.COMPLETED),
            "task2": StepResult("task2", ExecutionStatus.COMPLETED),  
            "task3": StepResult("task3", ExecutionStatus.COMPLETED)
        }
        assert parallel_step.check_success_criteria(results) is True
        
        # One failure (within threshold)
        results["task1"] = StepResult("task1", ExecutionStatus.FAILED)
        failure_rate = 1/3  # 33% failure rate
        assert failure_rate > parallel_step.failure_threshold
        assert parallel_step.check_success_criteria(results) is False


class TestLoopStep:
    """Test cases for LoopStep."""
    
    @pytest.fixture
    def for_loop_step(self):
        """Create a for loop step."""
        loop_steps = [
            WorkflowStep(
                id="process_item",
                name="Process Item",
                type=StepType.ACTION,
                parameters={"action_name": "log", "message": "Processing item"}
            )
        ]
        
        return LoopStep(
            id="process_list",
            name="Process List",
            loop_type="for",
            loop_steps=loop_steps,
            iterate_over="items",
            max_iterations=5
        )
    
    @pytest.mark.asyncio
    async def test_execute_for_loop(self, for_loop_step):
        """Test for loop execution."""
        context = {"items": ["item1", "item2", "item3"]}
        
        results = await for_loop_step.execute_loop(context)
        
        assert len(results) == 3  # One iteration per item
        
        # Each iteration should have results for the loop steps
        for iteration_result in results:
            assert "process_item" in iteration_result
            assert iteration_result["process_item"].status == ExecutionStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__])