"""
Advanced Conditional Logic and Control Flow for Q2 Platform Workflows.

Provides sophisticated conditional steps, parallel execution,
loops, and control flow constructs.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from .engine import WorkflowStep, StepType, StepResult, ExecutionStatus

logger = logging.getLogger(__name__)


class ConditionOperator(Enum):
    """Logical operators for conditions."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    NOT_IN = "not_in"
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class Condition:
    """Individual condition definition."""
    field: str
    operator: ConditionOperator
    value: Any
    negate: bool = False
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        try:
            field_value = self._get_field_value(self.field, context)
            result = self._apply_operator(field_value, self.operator, self.value)
            
            return not result if self.negate else result
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    def _get_field_value(self, field_path: str, context: Dict[str, Any]) -> Any:
        """Get field value from context using dot notation."""
        keys = field_path.split('.')
        value = context
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _apply_operator(self, field_value: Any, operator: ConditionOperator, 
                       expected_value: Any) -> bool:
        """Apply operator to field value and expected value."""
        if operator == ConditionOperator.EQUALS:
            return field_value == expected_value
        elif operator == ConditionOperator.NOT_EQUALS:
            return field_value != expected_value
        elif operator == ConditionOperator.GREATER_THAN:
            return field_value > expected_value
        elif operator == ConditionOperator.LESS_THAN:
            return field_value < expected_value
        elif operator == ConditionOperator.GREATER_EQUAL:
            return field_value >= expected_value
        elif operator == ConditionOperator.LESS_EQUAL:
            return field_value <= expected_value
        elif operator == ConditionOperator.CONTAINS:
            return expected_value in str(field_value)
        elif operator == ConditionOperator.NOT_CONTAINS:
            return expected_value not in str(field_value)
        elif operator == ConditionOperator.STARTS_WITH:
            return str(field_value).startswith(str(expected_value))
        elif operator == ConditionOperator.ENDS_WITH:
            return str(field_value).endswith(str(expected_value))
        elif operator == ConditionOperator.IN:
            return field_value in expected_value
        elif operator == ConditionOperator.NOT_IN:
            return field_value not in expected_value
        else:
            return False


@dataclass
class ConditionGroup:
    """Group of conditions with logical operators."""
    conditions: List[Union[Condition, 'ConditionGroup']]
    operator: ConditionOperator = ConditionOperator.AND
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition group."""
        if not self.conditions:
            return True
        
        results = [cond.evaluate(context) for cond in self.conditions]
        
        if self.operator == ConditionOperator.AND:
            return all(results)
        elif self.operator == ConditionOperator.OR:
            return any(results)
        elif self.operator == ConditionOperator.NOT:
            return not all(results)
        else:
            return all(results)  # Default to AND


@dataclass
class ConditionalStep(WorkflowStep):
    """Workflow step with conditional execution."""
    condition_group: Optional[ConditionGroup] = None
    if_true_steps: List[str] = field(default_factory=list)
    if_false_steps: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize conditional step."""
        super().__post_init__()
        self.type = StepType.CONDITION
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        """Check if step should execute based on conditions."""
        if not self.condition_group:
            return True
        
        return self.condition_group.evaluate(context)
    
    def get_next_steps(self, context: Dict[str, Any]) -> List[str]:
        """Get next steps based on condition evaluation."""
        if self.should_execute(context):
            return self.if_true_steps
        else:
            return self.if_false_steps


@dataclass
class ParallelStep(WorkflowStep):
    """Step that executes multiple sub-steps in parallel."""
    parallel_steps: List[WorkflowStep] = field(default_factory=list)
    wait_for_all: bool = True  # Wait for all steps or just one
    failure_threshold: float = 0.0  # Percentage of failures allowed (0.0 = no failures)
    
    def __post_init__(self):
        """Initialize parallel step."""
        super().__post_init__()
        self.type = StepType.PARALLEL
    
    async def execute_parallel(self) -> Dict[str, StepResult]:
        """Execute all parallel steps concurrently."""
        if not self.parallel_steps:
            return {}
        
        # Create tasks for all parallel steps
        tasks = []
        step_ids = []
        
        for step in self.parallel_steps:
            task = asyncio.create_task(self._execute_single_step(step))
            tasks.append(task)
            step_ids.append(step.id)
        
        results = {}
        
        if self.wait_for_all:
            # Wait for all steps to complete
            step_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for step_id, result in zip(step_ids, step_results):
                if isinstance(result, Exception):
                    results[step_id] = StepResult(
                        step_id=step_id,
                        status=ExecutionStatus.FAILED,
                        error=str(result)
                    )
                else:
                    results[step_id] = result
        
        else:
            # Wait for first completion
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Get first completed result
            for task in done:
                try:
                    result = await task
                    step_id = step_ids[tasks.index(task)]
                    results[step_id] = result
                    break
                except Exception as e:
                    step_id = step_ids[tasks.index(task)]
                    results[step_id] = StepResult(
                        step_id=step_id,
                        status=ExecutionStatus.FAILED,
                        error=str(e)
                    )
        
        return results
    
    async def _execute_single_step(self, step: WorkflowStep) -> StepResult:
        """Execute a single step (mock implementation)."""
        # This would integrate with the actual workflow engine
        await asyncio.sleep(0.1)  # Simulate execution
        
        return StepResult(
            step_id=step.id,
            status=ExecutionStatus.COMPLETED,
            output={"step_name": step.name, "executed": True}
        )
    
    def check_success_criteria(self, results: Dict[str, StepResult]) -> bool:
        """Check if parallel execution meets success criteria."""
        if not results:
            return False
        
        total_steps = len(results)
        failed_steps = sum(1 for result in results.values() 
                          if result.status == ExecutionStatus.FAILED)
        
        failure_rate = failed_steps / total_steps
        
        return failure_rate <= self.failure_threshold


@dataclass
class LoopStep(WorkflowStep):
    """Step that executes a loop with various termination conditions."""
    loop_steps: List[WorkflowStep] = field(default_factory=list)
    loop_type: str = "for"  # "for", "while", "until"
    loop_condition: Optional[ConditionGroup] = None
    max_iterations: Optional[int] = None
    iteration_variable: str = "loop_index"
    
    # For 'for' loops
    iterate_over: Optional[str] = None  # Field path to iterate over
    
    def __post_init__(self):
        """Initialize loop step."""
        super().__post_init__()
        self.type = StepType.LOOP
    
    async def execute_loop(self, context: Dict[str, Any]) -> List[Dict[str, StepResult]]:
        """Execute loop with specified conditions."""
        results = []
        iteration_count = 0
        
        if self.loop_type == "for":
            results = await self._execute_for_loop(context)
        elif self.loop_type == "while":
            results = await self._execute_while_loop(context)
        elif self.loop_type == "until":
            results = await self._execute_until_loop(context)
        
        return results
    
    async def _execute_for_loop(self, context: Dict[str, Any]) -> List[Dict[str, StepResult]]:
        """Execute for loop."""
        results = []
        
        if self.iterate_over:
            # Iterate over collection
            collection = self._get_field_value(self.iterate_over, context)
            if not isinstance(collection, (list, tuple, dict)):
                logger.error(f"Cannot iterate over non-collection: {type(collection)}")
                return results
            
            items = collection.items() if isinstance(collection, dict) else enumerate(collection)
            
            for index, item in items:
                if self.max_iterations and len(results) >= self.max_iterations:
                    break
                
                # Set iteration variables in context
                loop_context = dict(context)
                loop_context[self.iteration_variable] = index
                loop_context['loop_item'] = item
                
                iteration_results = await self._execute_loop_iteration(loop_context)
                results.append(iteration_results)
        
        else:
            # Simple counter loop
            max_iter = self.max_iterations or 10
            for i in range(max_iter):
                loop_context = dict(context)
                loop_context[self.iteration_variable] = i
                
                iteration_results = await self._execute_loop_iteration(loop_context)
                results.append(iteration_results)
        
        return results
    
    async def _execute_while_loop(self, context: Dict[str, Any]) -> List[Dict[str, StepResult]]:
        """Execute while loop."""
        results = []
        iteration_count = 0
        
        while (not self.loop_condition or self.loop_condition.evaluate(context)):
            if self.max_iterations and iteration_count >= self.max_iterations:
                break
            
            loop_context = dict(context)
            loop_context[self.iteration_variable] = iteration_count
            
            iteration_results = await self._execute_loop_iteration(loop_context)
            results.append(iteration_results)
            
            # Update context with results for next iteration
            context.update({"last_iteration_results": iteration_results})
            iteration_count += 1
        
        return results
    
    async def _execute_until_loop(self, context: Dict[str, Any]) -> List[Dict[str, StepResult]]:
        """Execute until loop (opposite of while)."""
        results = []
        iteration_count = 0
        
        while True:
            if self.max_iterations and iteration_count >= self.max_iterations:
                break
            
            loop_context = dict(context)
            loop_context[self.iteration_variable] = iteration_count
            
            iteration_results = await self._execute_loop_iteration(loop_context)
            results.append(iteration_results)
            
            # Update context and check termination condition
            context.update({"last_iteration_results": iteration_results})
            
            if self.loop_condition and self.loop_condition.evaluate(context):
                break
            
            iteration_count += 1
        
        return results
    
    async def _execute_loop_iteration(self, context: Dict[str, Any]) -> Dict[str, StepResult]:
        """Execute one loop iteration."""
        results = {}
        
        for step in self.loop_steps:
            result = await self._execute_single_step(step, context)
            results[step.id] = result
            
            # Update context with step result
            context[f"step_{step.id}_result"] = result.output
        
        return results
    
    async def _execute_single_step(self, step: WorkflowStep, 
                                 context: Dict[str, Any]) -> StepResult:
        """Execute a single step within loop."""
        # Mock implementation - would integrate with workflow engine
        await asyncio.sleep(0.05)
        
        return StepResult(
            step_id=step.id,
            status=ExecutionStatus.COMPLETED,
            output={"step_name": step.name, "context_keys": list(context.keys())}
        )
    
    def _get_field_value(self, field_path: str, context: Dict[str, Any]) -> Any:
        """Get field value from context using dot notation."""
        keys = field_path.split('.')
        value = context
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value


class ConditionEvaluator:
    """Utility class for evaluating complex conditions."""
    
    @staticmethod
    def create_simple_condition(field: str, operator: str, value: Any) -> Condition:
        """Create a simple condition."""
        op_enum = ConditionOperator(operator)
        return Condition(field=field, operator=op_enum, value=value)
    
    @staticmethod
    def create_and_group(conditions: List[Condition]) -> ConditionGroup:
        """Create an AND condition group."""
        return ConditionGroup(conditions=conditions, operator=ConditionOperator.AND)
    
    @staticmethod
    def create_or_group(conditions: List[Condition]) -> ConditionGroup:
        """Create an OR condition group."""
        return ConditionGroup(conditions=conditions, operator=ConditionOperator.OR)
    
    @staticmethod
    def parse_condition_expression(expression: str) -> ConditionGroup:
        """Parse a condition expression string (simplified parser)."""
        # This is a simplified implementation
        # In production, you'd use a proper expression parser like pyparsing
        
        # Example: "field1 == 'value1' AND field2 > 10"
        # For now, create a simple condition
        
        parts = expression.split(' AND ')
        conditions = []
        
        for part in parts:
            part = part.strip()
            
            # Simple parsing for basic conditions
            for op in ['==', '!=', '>=', '<=', '>', '<']:
                if op in part:
                    field, value_str = part.split(op, 1)
                    field = field.strip()
                    value_str = value_str.strip().strip("'\"")
                    
                    # Try to convert to appropriate type
                    try:
                        value = int(value_str)
                    except ValueError:
                        try:
                            value = float(value_str)
                        except ValueError:
                            if value_str.lower() in ['true', 'false']:
                                value = value_str.lower() == 'true'
                            else:
                                value = value_str
                    
                    condition = Condition(
                        field=field,
                        operator=ConditionOperator(op),
                        value=value
                    )
                    conditions.append(condition)
                    break
        
        return ConditionGroup(conditions=conditions, operator=ConditionOperator.AND)
    
    @staticmethod
    def evaluate_expression(expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression against context."""
        condition_group = ConditionEvaluator.parse_condition_expression(expression)
        return condition_group.evaluate(context)


# Utility functions for creating common workflow patterns
def create_if_then_else_workflow(condition: ConditionGroup, 
                                if_steps: List[WorkflowStep],
                                else_steps: Optional[List[WorkflowStep]] = None) -> ConditionalStep:
    """Create an if-then-else workflow pattern."""
    return ConditionalStep(
        id=str(uuid4()),
        name="If-Then-Else Step",
        condition_group=condition,
        if_true_steps=[step.id for step in if_steps],
        if_false_steps=[step.id for step in else_steps] if else_steps else []
    )

def create_parallel_workflow(steps: List[WorkflowStep], 
                           wait_for_all: bool = True,
                           failure_threshold: float = 0.0) -> ParallelStep:
    """Create a parallel execution workflow pattern."""
    return ParallelStep(
        id=str(uuid4()),
        name="Parallel Execution Step",
        parallel_steps=steps,
        wait_for_all=wait_for_all,
        failure_threshold=failure_threshold
    )

def create_for_each_workflow(collection_field: str, 
                           loop_steps: List[WorkflowStep],
                           max_iterations: Optional[int] = None) -> LoopStep:
    """Create a for-each workflow pattern."""
    return LoopStep(
        id=str(uuid4()),
        name="For-Each Loop Step",
        loop_type="for",
        loop_steps=loop_steps,
        iterate_over=collection_field,
        max_iterations=max_iterations
    )

def create_retry_workflow(step: WorkflowStep, 
                         max_attempts: int = 3,
                         retry_condition: Optional[str] = None) -> LoopStep:
    """Create a retry workflow pattern."""
    retry_cond = None
    if retry_condition:
        retry_cond = ConditionEvaluator.parse_condition_expression(retry_condition)
    
    return LoopStep(
        id=str(uuid4()),
        name=f"Retry {step.name}",
        loop_type="while",
        loop_steps=[step],
        loop_condition=retry_cond,
        max_iterations=max_attempts
    )