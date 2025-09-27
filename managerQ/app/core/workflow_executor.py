import logging
import threading
from typing import Optional, List, Set
import jinja2
import json
from datetime import datetime
import time # Added for workflow duration metric

from managerQ.app.core.workflow_manager import workflow_manager
from managerQ.app.core.task_dispatcher import task_dispatcher
from managerQ.app.models import TaskStatus, WorkflowStatus, Workflow, TaskBlock, WorkflowTask, ConditionalBlock, ApprovalBlock, LoopBlock, WorkflowEvent
from managerQ.app.api.dashboard_ws import broadcast_workflow_event
import asyncio
import pulsar
from managerQ.app.config import settings
from shared.pulsar_tracing import inject_trace_context, extract_trace_context
from opentelemetry import trace
from shared.observability.metrics import WORKFLOW_COMPLETED_COUNTER, WORKFLOW_DURATION_HISTOGRAM, TASK_COMPLETED_COUNTER
from managerQ.app.dependencies import get_kg_client # Import the dependency provider
from managerQ.app.core.observability_manager import observability_manager

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class WorkflowExecutor:
    """
    An event-driven process that listens for task status changes and advances
    workflows accordingly.
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pulsar_client: pulsar.Client = None
        self._task_producer: pulsar.Producer = None
        self._conditional_producer: pulsar.Producer = None
        self._status_consumer: pulsar.Consumer = None
        self._jinja_env = jinja2.Environment(autoescape=True)
        self.task_dispatcher = task_dispatcher

    def start(self):
        """Starts the executor in a background thread to listen for events."""
        if self._running:
            logger.warning("WorkflowExecutor is already running.")
            return
            
        self._pulsar_client = pulsar.Client(settings.pulsar.service_url)
        self._task_producer = self._pulsar_client.create_producer(settings.pulsar.topics.tasks_dispatch)
        self._conditional_producer = self._pulsar_client.create_producer(settings.pulsar.topics.tasks_conditional)

        # Subscribe to the task status update topic
        self._status_consumer = self._pulsar_client.subscribe(
            settings.pulsar.topics.tasks_status_update,
            subscription_name="managerq-workflow-executor-status-sub",
            consumer_type=pulsar.ConsumerType.Shared
        )

        self._running = True
        # The loop will now be an asyncio task
        self._thread = threading.Thread(target=self._start_async_loop, daemon=True)
        self._thread.start()
        logger.info("WorkflowExecutor started.")

    def pause_workflow(self, workflow_id: str):
        """Pauses a running workflow."""
        workflow = workflow_manager.get_workflow(workflow_id)
        if workflow and workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.PAUSED
            workflow_manager.update_workflow(workflow)
            logger.info(f"Workflow '{workflow_id}' has been paused.")
            asyncio.run(observability_manager.broadcast({"type": "WORKFLOW_UPDATE", "payload": workflow.dict()}))
    
    def resume_workflow(self, workflow_id: str):
        """Resumes a paused workflow."""
        workflow = workflow_manager.get_workflow(workflow_id)
        if workflow and workflow.status == WorkflowStatus.PAUSED:
            workflow.status = WorkflowStatus.RUNNING
            workflow_manager.update_workflow(workflow)
            logger.info(f"Workflow '{workflow_id}' has been resumed.")
            # Kick off processing again
            self.process_workflow(workflow)
            asyncio.run(observability_manager.broadcast({"type": "WORKFLOW_UPDATE", "payload": workflow.dict()}))

    def cancel_workflow(self, workflow_id: str):
        """Cancels a workflow."""
        workflow = workflow_manager.get_workflow(workflow_id)
        if workflow and workflow.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED, WorkflowStatus.PENDING]:
            workflow.status = WorkflowStatus.CANCELLED
            workflow_manager.update_workflow(workflow)
            logger.info(f"Workflow '{workflow_id}' has been cancelled.")
            asyncio.run(observability_manager.broadcast({"type": "WORKFLOW_UPDATE", "payload": workflow.dict()}))

    async def execute_workflow(self, workflow_id: str, user_id: str):
        """
        Starts the execution of a new workflow.
        This is the primary entry point for kicking off a workflow.
        """
        with tracer.start_as_current_span("execute_workflow", attributes={"workflow_id": workflow_id, "user_id": user_id}) as span:
            logger.info(f"Starting execution for workflow '{workflow_id}' for user '{user_id}'.")
            
            workflow = workflow_manager.get_workflow(workflow_id, user_id)
            if not workflow:
                logger.error(f"Cannot execute workflow '{workflow_id}': Not found for user '{user_id}'.")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Workflow not found"))
                return

            if workflow.status != WorkflowStatus.PENDING:
                logger.warning(f"Workflow '{workflow_id}' is already in status '{workflow.status.value}' and cannot be started again.")
                return

            workflow.status = WorkflowStatus.RUNNING
            workflow_manager.update_workflow(workflow)

            # Broadcast the start event
            await broadcast_workflow_event(WorkflowEvent(
                event_type="WORKFLOW_STARTED",
                workflow_id=workflow_id,
                data=workflow.dict()
            ))
            await observability_manager.broadcast({
                "type": "WORKFLOW_UPDATE",
                "payload": workflow.dict()
            })
            
            # This triggers the first pass to dispatch initial tasks
            await self.process_workflow(workflow)
            span.add_event("Workflow processing initiated.")


    def _start_async_loop(self):
        """Creates and runs the asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._consumer_loop())

    def stop(self):
        """Stops the executor loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join()
        if self._task_producer:
            self._task_producer.close()
        if self._conditional_producer:
            self._conditional_producer.close()
        if self._status_consumer:
            self._status_consumer.close()
        if self._pulsar_client:
            self._pulsar_client.close()
        logger.info("WorkflowExecutor stopped.")

    async def _consumer_loop(self):
        """The main async loop for consuming task status updates."""
        while self._running:
            try:
                msg = self._status_consumer.receive(timeout_millis=1000) # This is blocking, will be addressed
                await self._handle_status_update(msg)
                self._status_consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error(f"Error in WorkflowExecutor consumer loop: {e}", exc_info=True)
                if 'msg' in locals():
                    self._status_consumer.negative_acknowledge(msg)
                asyncio.sleep(5)

    async def _handle_status_update(self, msg: pulsar.Message):
        """Processes a task status update message and triggers workflow progression."""
        context = extract_trace_context(msg.properties())
        with tracer.start_as_current_span("handle_status_update", context=context) as span:
            payload = json.loads(msg.data().decode('utf-8'))
            workflow_id = payload.get("workflow_id")
            task_id = payload.get("task_id")
            status_str = payload.get("status")
            result = payload.get("result")

            if not all([workflow_id, task_id, status_str]):
                logger.error(f"Invalid status update message received: {payload}")
                return

            span.set_attributes({
                "workflow_id": workflow_id,
                "task_id": task_id,
                "status": status_str
            })
            logger.info(f"Received status update for task {task_id}: {status_str}")

            try:
                status = TaskStatus(status_str) if status_str else None
            except ValueError:
                logger.error(f"Invalid status '{status_str}' for task {task_id}.")
                return

            # NEW: Intercept failed status to trigger self-correction
            if status == TaskStatus.FAILED:
                logger.warning(f"Task {task_id} in workflow {workflow_id} has failed. Initiating self-correction process.")
                await self._handle_task_failure(workflow_id, task_id, result)
                return # Stop normal processing for this failed task

            # Instrument task status metric
            if status and status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                TASK_COMPLETED_COUNTER.labels(status=status.value).inc()

            workflow_manager.update_task_status(workflow_id, task_id, status, result)

            workflow = workflow_manager.get_workflow(workflow_id)
            if workflow and workflow.status == WorkflowStatus.RUNNING:
                await self.process_workflow(workflow)

    async def _handle_task_failure(self, workflow_id: str, task_id: str, result: str):
        """
        Handles a failed task by dispatching to the reflector agent to generate a corrective plan.
        """
        workflow = workflow_manager.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Cannot handle failure for workflow '{workflow_id}': not found.")
            return

        failed_task = workflow.get_task(task_id)
        if not failed_task:
            logger.error(f"Cannot handle failure for task '{task_id}': not found in workflow.")
            return

        logger.info(f"Generating corrective plan for failed task '{task_id}'.")
        
        try:
            # Create the prompt for the reflector agent
            prompt = f"""
            Original Goal: {workflow.original_prompt}
            Failed Task: {failed_task.json()}
            """
            
            # Dispatch to the reflector agent and await the new plan
            correction_task_id = self.task_dispatcher.dispatch_task(
                prompt=prompt,
                agent_personality="reflector_agent"
            )
            new_plan_json_str = await self.task_dispatcher.await_task_result(correction_task_id, timeout=60)
            
            new_tasks_data = json.loads(new_plan_json_str)
            
            # Patch the workflow with the new plan
            if new_tasks_data:
                workflow_manager.patch_workflow(workflow_id, task_id, new_tasks_data)
                logger.info(f"Successfully patched workflow '{workflow_id}'. Resuming execution.")
                
                # Re-process the now-patched workflow to dispatch the new tasks
                patched_workflow = workflow_manager.get_workflow(workflow_id)
                await self.process_workflow(patched_workflow)
            else:
                logger.warning("Reflector agent returned an empty plan. Failing workflow.")
                workflow.status = WorkflowStatus.FAILED
                workflow_manager.update_workflow(workflow)

        except Exception as e:
            logger.error(f"Self-correction failed for task '{task_id}': {e}", exc_info=True)
            workflow.status = WorkflowStatus.FAILED
            workflow_manager.update_workflow(workflow)

    async def process_workflow(self, workflow: Workflow):
        """
        Processes a single workflow's execution state, dispatching any new tasks that are ready.
        """
        logger.info(f"Processing workflow '{workflow.workflow_id}'...")
        await self._process_blocks(workflow.tasks, workflow)

        all_blocks_after = workflow.get_all_tasks_recursive()
        is_complete = all(block.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED} for block in all_blocks_after)
        
        if is_complete:
            final_status = WorkflowStatus.FAILED if any(b.status == TaskStatus.FAILED for b in all_blocks_after) else WorkflowStatus.COMPLETED
            
            # --- New: Extract and save the final result ---
            if final_status == WorkflowStatus.COMPLETED:
                final_tasks = [task for task in all_blocks_after if not any(other_task.task_id in task.dependencies for other_task in all_blocks_after if other_task is not task)]
                # For simplicity, if there are multiple final tasks, we'll concatenate their results.
                final_results = [t.result for t in final_tasks if t.result]
                workflow.final_result = "\n".join(final_results)
            # -----------------------------------------

            # Instrument workflow metrics
            WORKFLOW_COMPLETED_COUNTER.labels(status=final_status.value).inc()
            duration_seconds = time.time() - workflow.created_at.timestamp()
            WORKFLOW_DURATION_HISTOGRAM.observe(duration_seconds)

            workflow.status = final_status
            workflow_manager.update_workflow(workflow)
            logger.info(f"Workflow '{workflow.workflow_id}' has finished with status '{final_status.value}'.")
            
            # Broadcast to the old dashboard and the new observability dashboard
            await broadcast_workflow_event(WorkflowEvent(
                event_type="WORKFLOW_COMPLETED",
                workflow_id=workflow.workflow_id,
                data={"status": final_status.value}
            ))
            await observability_manager.broadcast({
                "type": "WORKFLOW_UPDATE",
                "payload": workflow.dict()
            })
            
            # Check if this was an AIOps workflow and handle the final report
            if workflow.event_id and final_status == WorkflowStatus.COMPLETED:
                await self._handle_completed_aiops_workflow(workflow)
            else:
                await self._trigger_reflection_task(workflow)


    async def _handle_completed_aiops_workflow(self, workflow: Workflow):
        """
        Handles the completion of a event-driven workflow, like the AIOps one,
        by ingesting the final report into the Knowledge Graph.
        """
        logger.info(f"Handling completed AIOps workflow '{workflow.workflow_id}'.")
        
        # Find the final task (a task with no other tasks depending on it)
        all_task_ids = {task.task_id for task in workflow.get_all_tasks_recursive()}
        dependency_ids = set()
        for task in workflow.get_all_tasks_recursive():
            dependency_ids.update(task.dependencies)
            
        final_task_ids = all_task_ids - dependency_ids
        
        if not final_task_ids:
            logger.warning(f"Could not determine final task for AIOps workflow '{workflow.workflow_id}'.")
            return
            
        # Assuming one final report task for simplicity
        final_task_id = final_task_ids.pop()
        final_task = workflow.get_task(final_task_id)
        
        if not final_task or not final_task.result:
            logger.warning(f"Final task '{final_task_id}' for AIOps workflow has no result to ingest.")
            return
            
        service_name = workflow.shared_context.get("service_name")
        if not service_name:
            logger.warning("AIOps workflow is missing 'service_name' in shared_context.")
            return

        logger.info(f"Ingesting final report from task '{final_task_id}' into Knowledge Graph.")
        
        try:
            # This is not ideal, but a workaround for the client management issue
            kg_client = get_kg_client()
            
            report_content = final_task.result.replace("'", "\\'") # Simple escaping for Gremlin
            
            # Gremlin query to create the report and link it
            query = f"""
            def report = g.addV('Report')
                           .property('source', 'AIOpsWorkflow')
                           .property('content', '{report_content}')
                           .property('createdAt', '{datetime.utcnow().isoformat()}')
                           .next()
            
            def service = g.V().has('Service', 'name', '{service_name}').tryNext().orElse(null)
            if (service != null) {{
                g.V(report).addE('REPORT_FOR').to(service).iterate()
            }}
            
            def event = g.V().has('Event', 'eventId', '{workflow.event_id}').tryNext().orElse(null)
            if (event != null) {{
                g.V(report).addE('GENERATED_FROM').to(event).iterate()
            }}
            
            return report
            """
            
            # We need to run this async function in our sync thread
            await kg_client.execute_gremlin_query(query)
            logger.info("Successfully ingested AIOps report into Knowledge Graph.")
        
        except Exception as e:
            logger.error(f"Failed to ingest AIOps report into Knowledge Graph: {e}", exc_info=True)


    async def _trigger_reflection_task(self, workflow: Workflow):
        """Creates and dispatches a task for a 'reflector' agent to analyze a completed workflow."""
        logger.info(f"Triggering reflection for workflow '{workflow.workflow_id}'.")
        
        # Serialize the workflow to JSON to pass to the reflector
        workflow_dump = workflow.json(indent=2)

        prompt = (
            "You are a Reflector Agent. Your purpose is to analyze a completed workflow to find insights and lessons.\n\n"
            "Analyze the following workflow execution record. Identify key successes, failures, and reasons for the outcome. "
            "Formulate a concise 'lesson learned' that can be stored in our knowledge graph to improve future planning.\n\n"
            f"Workflow Analysis Request for: {workflow.workflow_id}\n"
            f"Original Goal: {workflow.original_prompt}\n"
            f"Final Status: {workflow.status.value}\n\n"
            f"Full Workflow Record:\n{workflow_dump}"
        )

        try:
            # Dispatch to a reflector agent. This is a "fire-and-forget" task.
            self.task_dispatcher.dispatch_task(
                prompt=prompt,
                agent_personality='reflector'
            )
            logger.info(f"Successfully dispatched reflection task for workflow '{workflow.workflow_id}'.")
        except RuntimeError as e:
            logger.error(f"Failed to dispatch reflection task for workflow '{workflow.workflow_id}': {e}", exc_info=True)


    async def _process_blocks(self, blocks: List[TaskBlock], workflow: Workflow):
        """Recursively processes a list of tasks, dispatching all that are ready."""
        all_blocks = workflow.get_all_tasks_recursive()
        completed_ids = {block.task_id for block in all_blocks if block.status == TaskStatus.COMPLETED}

        for block in blocks:
            if block.status == TaskStatus.PENDING and set(block.dependencies).issubset(completed_ids):
                if isinstance(block, WorkflowTask):
                    # NEW: Check for task-level condition before dispatching
                    if block.condition:
                        eval_context = self._get_evaluation_context(workflow)
                        try:
                            template = self._jinja_env.from_string(block.condition)
                            if template.render(eval_context):
                                await self._dispatch_task(block, workflow)
                            else:
                                # If condition is not met, mark the task as cancelled (or a new 'SKIPPED' status)
                                logger.info(f"Skipping task '{block.task_id}' due to unmet condition.")
                                workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.CANCELLED, result="Condition not met.")
                        except Exception as e:
                            logger.error(f"Failed to evaluate condition for task '{block.task_id}': {e}", exc_info=True)
                            workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.FAILED, result=f"Condition evaluation failed: {e}")
                    else:
                        await self._dispatch_task(block, workflow)
                elif isinstance(block, ConditionalBlock):
                    self._evaluate_conditional(block, workflow)
                elif isinstance(block, ApprovalBlock):
                    await self._handle_approval_block(block, workflow)
                elif isinstance(block, LoopBlock):
                    await self._handle_loop_block(block, workflow)
            
            # This recursive call is problematic and can lead to re-processing.
            # A better approach would be a single pass. Refactoring this is out of scope for now.
            if hasattr(block, 'tasks') and block.tasks:
                await self._process_blocks(block.tasks, workflow)


    async def _handle_loop_block(self, block: LoopBlock, workflow: Workflow):
        """Handles the logic for a loop block."""
        logger.info(f"Evaluating loop block '{block.task_id}' for workflow '{workflow.workflow_id}'.")
        
        eval_context = self._get_evaluation_context(workflow)
        
        # Track iterations in the shared context
        iteration_key = f"loop_{block.task_id}_iterations"
        current_iteration = workflow.shared_context.get(iteration_key, 0)
        
        if current_iteration >= block.max_iterations:
            logger.warning(f"Loop block '{block.task_id}' exceeded max iterations. Marking as complete.")
            workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.COMPLETED, result="Max iterations reached.")
            return

        try:
            template = self._jinja_env.from_string(block.condition)
            should_continue = template.render(eval_context)
            
            # Jinja render can return empty strings for falsey conditions
            should_continue = str(should_continue).lower() in ['true']

            if should_continue:
                logger.info(f"Loop condition for '{block.task_id}' is true. Starting iteration {current_iteration + 1}.")
                # Increment iteration count
                workflow.shared_context[iteration_key] = current_iteration + 1
                
                # Reset all tasks within the loop to PENDING to allow re-execution
                for task in block.tasks:
                    workflow_manager.update_task_status(workflow.workflow_id, task.task_id, TaskStatus.PENDING)
                
                # Re-process the blocks within the loop
                await self._process_blocks(block.tasks, workflow)
            else:
                logger.info(f"Loop condition for '{block.task_id}' is false. Marking as complete.")
                workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.COMPLETED, result="Condition became false.")

        except Exception as e:
            logger.error(f"Failed to evaluate condition for loop block '{block.task_id}': {e}", exc_info=True)
            workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.FAILED, result=f"Loop condition evaluation failed: {e}")


    async def _handle_approval_block(self, block: ApprovalBlock, workflow: Workflow):
        """Handles a workflow block that requires human approval."""
        logger.info(f"Pausing workflow '{workflow.workflow_id}' for human approval on task '{block.task_id}'.")
        
        # Update the task's status to indicate it's waiting for a decision.
        # The workflow will not proceed down this path until an external API call
        # changes this status to 'COMPLETED' (approved) or 'FAILED' (rejected).
        workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.PENDING_APPROVAL)
        
        # Broadcast the event so the UI can update
        await broadcast_workflow_event(WorkflowEvent(
            event_type="APPROVAL_REQUIRED",
            workflow_id=workflow.workflow_id,
            task_id=block.task_id,
            data={"message": block.message}
        ))

    async def _dispatch_task(self, task: WorkflowTask, workflow: Workflow):
        """Renders a task's prompt and dispatches it to an agent."""
        logger.info(f"Dispatching task '{task.task_id}' for workflow '{workflow.workflow_id}' to Pulsar.")
        
        try:
            # Render prompt using Jinja2 and the workflow's shared context
            template = self._jinja_env.from_string(task.prompt)
            rendered_prompt = template.render(self._get_evaluation_context(workflow))
            
            task_payload = {
                "task_id": task.task_id,
                "workflow_id": workflow.workflow_id,
                "agent_personality": task.agent_personality,
                "prompt": rendered_prompt,
            }
            
            properties = inject_trace_context({})
            self._task_producer.send(
                json.dumps(task_payload).encode('utf-8'),
                properties=properties
            )
            
            workflow_manager.update_task_status(workflow.workflow_id, task.task_id, TaskStatus.DISPATCHED)
            
            # Broadcast dispatch event
            await broadcast_workflow_event(WorkflowEvent(
                event_type="TASK_STATUS_UPDATE",
                workflow_id=workflow.workflow_id,
                task_id=task.task_id,
                data={"status": TaskStatus.DISPATCHED.value}
            ))
            await observability_manager.broadcast({
                "type": "WORKFLOW_UPDATE",
                "payload": workflow.dict()
            })

        except jinja2.TemplateError as e:
            logger.error(f"Failed to render prompt for task '{task.task_id}': {e}", exc_info=True)
            workflow_manager.update_task_status(workflow.workflow_id, task.task_id, TaskStatus.FAILED, result=f"Prompt rendering failed: {e}")
        except Exception as e:
            logger.error(f"Failed to publish task '{task.task_id}' to Pulsar: {e}", exc_info=True)
            # Optionally, set the task to FAILED here as well
            workflow_manager.update_task_status(workflow.workflow_id, task.task_id, TaskStatus.FAILED, result="Failed to publish to message queue.")


    def _evaluate_conditional(self, block: ConditionalBlock, workflow: Workflow):
        """Publishes a conditional evaluation task to Pulsar."""
        logger.info(f"Publishing conditional evaluation for block '{block.task_id}' to Pulsar.")
        
        eval_context = self._get_evaluation_context(workflow)

        conditional_payload = {
            "block_id": block.task_id,
            "workflow_id": workflow.workflow_id,
            "evaluation_context": eval_context,
            "branches": [branch.dict() for branch in block.branches]
        }
        
        properties = inject_trace_context({})
        self._conditional_producer.send(
            json.dumps(conditional_payload).encode('utf-8'),
            properties=properties
        )
        
        # We don't mark as complete here anymore. The worker will do that.
        # We can, however, mark it as "evaluating" if we add such a status.
        # For now, we'll leave it as PENDING until the worker picks it up.

    def _get_evaluation_context(self, workflow: Workflow) -> dict:
        """
        Creates a context for Jinja2 rendering, combining workflow's shared_context
        with the results of all completed tasks.
        """
        context = workflow.shared_context.copy()
        
        # Add task results to the context under a 'tasks' key for easy access.
        # e.g., {{ tasks.task_1.result }}
        task_results = {}
        for task in workflow.get_all_tasks_recursive():
            if task.status == TaskStatus.COMPLETED and isinstance(task, WorkflowTask):
                 # Try to parse JSON result, otherwise use the raw string
                try:
                    task_results[task.task_id] = json.loads(task.result) if task.result and task.result.strip().startswith(('{', '[')) else task.result
                except (json.JSONDecodeError, TypeError):
                    task_results[task.task_id] = task.result

        context['tasks'] = task_results
        return context

# Singleton instance
workflow_executor = WorkflowExecutor() 