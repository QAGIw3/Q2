import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# --- Search Models ---


class SearchQuery(BaseModel):
    """Represents a user's search query."""

    query: str = Field(..., description="The search term or question from the user.")
    session_id: Optional[str] = Field(None, description="An optional session ID for conversational context.")


class VectorStoreResult(BaseModel):
    """Represents a single search result from the vector store."""

    source: str = Field(..., description="The origin of the document (e.g., file path, URL).")
    content: str = Field(..., description="The text content of the search result chunk.")
    score: float = Field(..., description="The similarity score of the result.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Any additional metadata.")


class KGNode(BaseModel):
    """Represents a node in the knowledge graph."""

    id: str
    label: str
    properties: Dict[str, Any]


class KGEdge(BaseModel):
    """Represents an edge between two nodes in the knowledge graph."""

    source: str
    target: str
    label: str


class KnowledgeGraphResult(BaseModel):
    """Represents a subgraph from the knowledge graph relevant to the query."""

    nodes: List[KGNode]
    edges: List[KGEdge]


class SearchResponse(BaseModel):
    """The final, aggregated response for a search query."""

    ai_summary: Optional[str] = Field(None, description="An AI-generated summary of the search results.")
    vector_results: List[VectorStoreResult] = Field(default_factory=list, description="Results from the vector store.")
    knowledge_graph_result: Optional[KnowledgeGraphResult] = Field(
        None, description="Results from the knowledge graph."
    )
    model_version: Optional[str] = Field(None, description="The version of the model that generated the summary.")


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    CANCELLED = "CANCELLED"


class WorkflowTask(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4()}")
    type: Literal["task"] = "task"
    agent_personality: str = Field(description="The type of agent needed, e.g., 'default', 'devops'.")
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(
        default_factory=list, description="List of task_ids that must be completed before this one can start."
    )
    result: Optional[str] = None
    condition: Optional[str] = Field(None, description="A Jinja2 condition to evaluate before dispatching the task.")


# --- NEW: Models for Conditional Logic ---


class ConditionalBranch(BaseModel):
    """A branch of tasks to be executed if a condition is met."""

    condition: str = Field(
        description="A Jinja2-like condition to evaluate against the workflow's shared_context, e.g., '{{ task_1.result.status }} == \"success\"'"
    )
    tasks: List["TaskBlock"] = Field(
        default_factory=list, description="A list of tasks to execute if the condition is true."
    )


class ConditionalBlock(BaseModel):
    """A block that allows for conditional execution paths in a workflow."""

    task_id: str = Field(default_factory=lambda: f"cond_{uuid.uuid4()}")
    type: Literal["conditional"] = "conditional"
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of task_ids that must be completed before this conditional block can be evaluated.",
    )
    branches: List[ConditionalBranch]


class ApprovalBlock(BaseModel):
    """A block that pauses the workflow to wait for human approval."""

    task_id: str = Field(default_factory=lambda: f"approve_{uuid.uuid4()}")
    type: Literal["approval"] = "approval"
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(
        default_factory=list, description="List of task_ids that must be completed before this one can start."
    )
    message: str = Field(
        description="The message to display to the user for approval, e.g., 'Do you approve this action?'"
    )
    required_roles: List[str] = Field(
        default_factory=list,
        description="A list of roles required to approve this step, e.g., ['sre', 'admin']. If empty, any user can approve.",
    )
    result: Optional[Literal["approved", "rejected"]] = Field(
        None, description="The final decision of the approval step."
    )


# A Union type representing any execution block in the workflow graph.
TaskBlock = Union[WorkflowTask, ConditionalBlock, ApprovalBlock]

# --- NEW: Model for Loop Logic ---


class LoopBlock(BaseModel):
    """A block that allows for iterative execution of tasks."""

    task_id: str = Field(default_factory=lambda: f"loop_{uuid.uuid4()}")
    type: Literal["loop"] = "loop"
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(default_factory=list)
    condition: str = Field(description="A Jinja2 condition. The loop continues as long as this evaluates to true.")
    tasks: List["TaskBlock"] = Field(
        default_factory=list, description="The list of tasks to execute in each iteration."
    )
    max_iterations: int = Field(10, description="A safeguard to prevent infinite loops.")


# Update the TaskBlock to include the new LoopBlock
TaskBlock = Union[WorkflowTask, ConditionalBlock, ApprovalBlock, LoopBlock]

# Update Pydantic v2 forward references to handle all recursive TaskBlock definitions
ConditionalBranch.model_rebuild()
LoopBlock.model_rebuild()


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PENDING_CLARIFICATION = "PENDING_CLARIFICATION"
    CANCELLED = "CANCELLED"


class Workflow(BaseModel):
    workflow_id: str = Field(default_factory=lambda: f"wf_{uuid.uuid4()}")
    original_prompt: str
    status: WorkflowStatus = WorkflowStatus.RUNNING
    tasks: List[TaskBlock]
    shared_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="A shared dictionary for agents in this workflow to read/write intermediate results.",
    )
    event_id: Optional[str] = Field(None, description="The ID of the event that triggered this workflow.")
    final_result: Optional[str] = Field(
        None, description="The result of the final task in the workflow, for parent workflows to consume."
    )

    def get_task(self, task_id: str) -> Optional[TaskBlock]:
        """Recursively finds a task or block by its ID."""

        def find_task_recursive(search_id: str, blocks: List[TaskBlock]) -> Optional[TaskBlock]:
            for block in blocks:
                if block.task_id == search_id:
                    return block
                if isinstance(block, ConditionalBlock):
                    for branch in block.branches:
                        found = find_task_recursive(search_id, branch.tasks)
                        if found:
                            return found
            return None

        return find_task_recursive(task_id, self.tasks)

    def get_all_tasks_recursive(self) -> List[TaskBlock]:
        """Returns a flattened list of all tasks and blocks in the workflow."""
        all_blocks = []

        def gather_blocks(blocks: List[TaskBlock]):
            for block in blocks:
                all_blocks.append(block)
                if isinstance(block, ConditionalBlock):
                    for branch in block.branches:
                        gather_blocks(branch.tasks)

        gather_blocks(self.tasks)
        return all_blocks

    def get_ready_tasks(self) -> List[TaskBlock]:
        """
        Returns a list of tasks or blocks whose dependencies are met.
        NOTE: This implementation is now more complex and tightly coupled with the executor's logic.
        The executor will need to handle the recursive nature of the workflow.
        """
        all_blocks = self.get_all_tasks_recursive()
        completed_task_ids = {block.task_id for block in all_blocks if block.status == TaskStatus.COMPLETED}

        ready_tasks = []
        for task in all_blocks:
            if task.status == TaskStatus.PENDING and set(task.dependencies).issubset(completed_task_ids):
                ready_tasks.append(task)
        return ready_tasks


# --- Goal Models ---


class Condition(BaseModel):
    metric: str = Field(description="The name of the metric to monitor, e.g., 'cpu_usage', 'error_rate'.")
    operator: Literal["<", ">", "==", "!=", "<=", ">="]
    value: float
    service: str = Field(description="The service the metric applies to.")


class ClarificationResponse(BaseModel):
    """Represents the user's answer to a clarifying question."""

    answer: str


class WorkflowEvent(BaseModel):
    """Represents a real-time event in a workflow's lifecycle."""

    event_type: str  # e.g., "TASK_STATUS_UPDATE", "WORKFLOW_COMPLETED"
    workflow_id: str
    task_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class Goal(BaseModel):
    goal_id: str = Field(default_factory=lambda: f"goal_{uuid.uuid4()}")
    objective: str = Field(..., description="A high-level description of the desired state.")
    is_active: bool = True
    conditions: List[Condition] = Field(
        default_factory=list, description="A list of conditions that define the goal. Can be empty for proactive goals."
    )
    remediation_workflow_id: Optional[str] = Field(
        None, description="The ID of a specific workflow to trigger if this goal is breached or proactively triggered."
    )
    trigger: Optional[Dict[str, Any]] = Field(None, description="A proactive trigger, e.g., {'type': 'on_startup'}.")
    context_overrides: Optional[Dict[str, Any]] = Field(
        None, description="A dictionary of values to override the workflow's shared_context."
    )


# --- Exceptions ---


class AmbiguousGoalError(Exception):
    """Raised when a goal is ambiguous and requires user clarification."""

    def __init__(self, message: str, clarifying_question: str):
        super().__init__(message)
        self.message = message
        self.clarifying_question = clarifying_question


class GoalStatus(BaseModel):
    goal_id: str
    workflow_id: str
    status: str
