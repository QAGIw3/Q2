# WorkflowWorker/main.py
import logging
import json
import pulsar
import threading
import time
from typing import Optional, Dict, Any
import jinja2
from opentelemetry import trace, context as trace_context
from pulsar.schema import AvroSchema

from shared.pulsar_tracing import extract_trace_context, inject_trace_context
from shared.opentelemetry.tracing import setup_tracing
from shared.q_messaging_schemas.schemas import PromptMessage
from shared.q_memory_schemas.models import TaskStatus  # Assuming TaskStatus is moved here
from .config import settings
import io
# import fastavro # No longer needed

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("WorkflowWorker")
setup_tracing(app=None, service_name="WorkflowWorker")
tracer = trace.get_tracer(__name__)


class WorkflowExecutor:
    """
    Manages the execution of a single workflow instance.
    This is not thread-safe and should be managed by the main Worker.
    """
    def __init__(self, workflow_data: dict, worker: 'Worker'):
        self.workflow_id = workflow_data['workflow_id']
        self.tasks = {task['task_id']: task for task in workflow_data['tasks']}
        self.task_states = {task_id: TaskStatus.PENDING for task_id in self.tasks}
        self.task_results = {}
        self.worker = worker
        self.jinja_env = jinja2.Environment(autoescape=True)
        logger.info(f"[{self.workflow_id}] Initialized workflow executor.")

    def start(self):
        """Starts the workflow by finding and dispatching initial tasks."""
        logger.info(f"[{self.workflow_id}] Starting workflow execution.")
        self.check_and_dispatch_tasks()

    def handle_task_status_update(self, update: dict):
        """
        Handles an incoming status update for a task in this workflow.
        """
        task_id = update['task_id']
        status = TaskStatus(update['status'])
        result = update.get('result')
        
        if task_id not in self.tasks:
            return # Not for this workflow

        logger.info(f"[{self.workflow_id}] Received update for task {task_id}: {status.value}")
        
        self.task_states[task_id] = status
        if result:
            self.task_results[task_id] = result

        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            self.check_and_dispatch_tasks()

    def check_and_dispatch_tasks(self):
        """
        Iterates through all tasks to find ones that are ready to run.
        """
        logger.info(f"[{self.workflow_id}] Checking for ready tasks...")
        for task_id, task in self.tasks.items():
            if self.task_states[task_id] == TaskStatus.PENDING:
                dependencies = task.get('dependencies', [])
                if all(self.task_states.get(dep_id) == TaskStatus.COMPLETED for dep_id in dependencies):
                    self.dispatch(task)
        
        # After attempting to dispatch, check if the workflow is complete
        self.check_for_completion()

    def check_for_completion(self):
        """Checks if all tasks are in a terminal state and triggers reflection if so."""
        all_tasks = self.tasks.values()
        if not all(task_data['task_id'] in self.task_states for task_data in all_tasks):
             # Not all tasks have a state yet, so it can't be complete.
            return

        is_complete = all(self.task_states[task_data['task_id']] in [TaskStatus.COMPLETED, TaskStatus.FAILED] for task_data in all_tasks)

        if is_complete:
            logger.info(f"[{self.workflow_id}] Workflow is complete.")
            final_status = TaskStatus.FAILED if any(s == TaskStatus.FAILED for s in self.task_states.values()) else TaskStatus.COMPLETED
            
            # Construct the final workflow data object
            final_workflow_data = {
                "workflow_id": self.workflow_id,
                "final_status": final_status.value,
                "tasks": [dict(task, status=self.task_states[task['task_id']].value, result=self.task_results.get(task['task_id'])) for task in self.tasks.values()]
            }
            
            self.worker.trigger_reflection(final_workflow_data)
            # Potentially remove from active executors dict to save memory
            # self.worker.remove_executor(self.workflow_id)


    def dispatch(self, task: dict):
        """Dispatches a single task or conditional block."""
        task_id = task['task_id']
        task_type = task.get('type', 'task')
        
        logger.info(f"[{self.workflow_id}] Dispatching task {task_id} of type {task_type}.")
        self.task_states[task_id] = TaskStatus.RUNNING
        
        if task_type == 'task':
            self.worker.dispatch_agent_task(self.workflow_id, task, self.task_results)
        elif task_type == 'conditional':
            self.evaluate_conditional(task)
        else:
            logger.error(f"[{self.workflow_id}] Unknown task type '{task_type}' for task {task_id}.")
            self.task_states[task_id] = TaskStatus.FAILED

    def evaluate_conditional(self, block: dict):
        """Evaluates a conditional block and triggers the next tasks."""
        block_id = block['block_id']
        logger.info(f"[{self.workflow_id}] Evaluating conditional block {block_id}.")
        
        context = {"tasks": self.task_results} # Jinja context
        
        for branch in block.get('branches', []):
            try:
                condition = branch['condition']
                template = self.jinja_env.from_string(condition)
                if template.render(context).lower() in ['true', '1', 'yes']:
                    logger.info(f"[{self.workflow_id}] Condition '{condition}' is TRUE. Activating branch.")
                    # The tasks within this branch are now part of the main workflow
                    for task in branch.get('tasks', []):
                        self.tasks[task['task_id']] = task
                        self.task_states[task['task_id']] = TaskStatus.PENDING
                    self.task_states[block_id] = TaskStatus.COMPLETED
                    self.check_and_dispatch_tasks()
                    return
            except Exception as e:
                logger.error(f"[{self.workflow_id}] Failed to evaluate condition '{condition}': {e}", exc_info=True)
                self.task_states[block_id] = TaskStatus.FAILED
                return
        
        logger.info(f"[{self.workflow_id}] No conditions met for block {block_id}.")
        self.task_states[block_id] = TaskStatus.COMPLETED
        self.check_and_dispatch_tasks()


class Worker:
    def __init__(self):
        self._client: Optional[pulsar.Client] = None
        self._workflow_consumer: Optional[pulsar.Consumer] = None
        self._status_update_consumer: Optional[pulsar.Consumer] = None
        self._agent_task_producer: Optional[pulsar.Producer] = None
        self._reflection_producer: Optional[pulsar.Producer] = None
        self._running = False
        self._executors: Dict[str, WorkflowExecutor] = {}
        self._lock = threading.Lock()

    def start(self):
        if self._running:
            return
        
        logger.info("Starting WorkflowWorker...")
        self._client = pulsar.Client(settings.PULSAR_SERVICE_URL)
        
        dead_letter_policy = pulsar.DeadLetterPolicy(max_redeliver_count=settings.MAX_REDELIVER_COUNT)

        self._workflow_consumer = self._client.subscribe(
            settings.WORKFLOW_EXECUTION_TOPIC,
            subscription_name=settings.WORKFLOW_EXECUTION_SUBSCRIPTION,
            consumer_type=pulsar.ConsumerType.Shared,
            dead_letter_policy=dead_letter_policy
        )
        
        self._status_update_consumer = self._client.subscribe(
            settings.TASK_STATUS_UPDATE_TOPIC,
            subscription_name=settings.TASK_STATUS_UPDATE_SUBSCRIPTION,
            consumer_type=pulsar.ConsumerType.Shared
        )

        self._agent_task_producer = self._client.create_producer(settings.AGENT_TASK_TOPIC)
        self._reflection_producer = self._client.create_producer(
            getattr(settings, 'REFLECTION_TOPIC', 'persistent://public/default/q.reflector.tasks')
        )
        
        self._running = True
        
        threading.Thread(target=self._run_consumer, args=(self._workflow_consumer, self.handle_workflow_message), daemon=True).start()
        threading.Thread(target=self._run_consumer, args=(self._status_update_consumer, self.handle_status_update_message), daemon=True).start()
        
        logger.info("WorkflowWorker started successfully.")

    def stop(self):
        self._running = False
        logger.info("Stopping WorkflowWorker...")
        if self._reflection_producer:
            self._reflection_producer.close()
        # ... close consumers and producers ...
        logger.info("WorkflowWorker stopped.")

    def _run_consumer(self, consumer: pulsar.Consumer, handler):
        while self._running:
            try:
                msg = consumer.receive(timeout_millis=1000)
                handler(msg)
                consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}", exc_info=True)
                if 'msg' in locals():
                    consumer.negative_acknowledge(msg)
                time.sleep(5)

    def handle_workflow_message(self, msg: pulsar.Message):
        """Handles a new workflow to be executed."""
        try:
            workflow_data = json.loads(msg.data().decode('utf-8'))
            workflow_id = workflow_data.get('workflow_id')
            if not workflow_id:
                logger.error("Received workflow message without a workflow_id.")
                return

            with self._lock:
                if workflow_id in self._executors:
                    logger.warning(f"Workflow {workflow_id} is already being executed.")
                    return
                executor = WorkflowExecutor(workflow_data, self)
                self._executors[workflow_id] = executor

            executor.start()
        except Exception as e:
            logger.error(f"Failed to process new workflow message: {e}", exc_info=True)
            raise

    def handle_status_update_message(self, msg: pulsar.Message):
        """Handles a task status update."""
        try:
            update_data = json.loads(msg.data().decode('utf-8'))
            workflow_id = update_data.get('workflow_id')
            if not workflow_id:
                return # Not a message we can route

            with self._lock:
                executor = self._executors.get(workflow_id)
            
            if executor:
                executor.handle_task_status_update(update_data)
        except Exception as e:
            logger.error(f"Failed to process status update message: {e}", exc_info=True)
            raise

    def trigger_reflection(self, workflow_data: dict):
        """Sends the completed workflow data to the reflector agent."""
        if not self._reflection_producer:
            logger.error("Reflection producer not initialized.")
            return
        try:
            self._reflection_producer.send(json.dumps(workflow_data).encode('utf-8'))
            logger.info(f"Sent workflow {workflow_data.get('workflow_id')} for reflection.")
        except Exception as e:
            logger.error(f"Failed to send workflow for reflection: {e}", exc_info=True)

    def dispatch_agent_task(self, workflow_id: str, task: dict, task_results: dict):
        """Sends a task to the agent task topic."""
        jinja_env = jinja2.Environment(autoescape=True)
        
        try:
            # Render the prompt using Jinja2 to insert results from previous tasks
            template = jinja_env.from_string(task['prompt'])
            rendered_prompt = template.render({"tasks": task_results})
            
            agent_personality = task['agent_personality']
            task_topic = f"persistent://public/default/q.agentq.tasks.{agent_personality}"
            
            payload = PromptMessage(
                id=f"{workflow_id}-{task['task_id']}",
                workflow_id=workflow_id,
                task_id=task['task_id'],
                agent_personality=agent_personality,
                prompt=rendered_prompt,
                model=task.get('model', 'default'), # Assuming a model can be specified in the task
                timestamp=int(time.time() * 1000)
            )

            producer = self._client.create_producer(
                topic=task_topic,
                schema=AvroSchema(PromptMessage)
            )
            producer.send(payload)
            producer.close()
            
            logger.info(f"[{workflow_id}] Dispatched task {task['task_id']} to agent {agent_personality} on topic {task_topic}.")

        except Exception as e:
            logger.error(f"[{workflow_id}] Failed to dispatch agent task {task['task_id']}: {e}", exc_info=True)
            # Mark the task as failed
            # This requires a mechanism to update the state from here.
            # For simplicity, we assume this dispatch will not fail often.

if __name__ == "__main__":
    worker = Worker()
    worker.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop() 