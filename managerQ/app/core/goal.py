import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from managerQ.app.core.workflow_manager import workflow_manager
from managerQ.app.models import Workflow, WorkflowTask


class Goal(BaseModel):
    goal_id: str = Field(default_factory=lambda: f"goal_{uuid.uuid4()}")
    prompt: str
    created_by: str
    context: Dict[str, Any] = Field(default_factory=dict)

    async def create_and_run_workflow(self) -> Optional[Workflow]:
        """Creates a trivial workflow tied to this goal (offline/dev stub)."""
        # Minimal single task workflow blueprint
        task = WorkflowTask(agent_personality="default", prompt=self.prompt, dependencies=[])
        wf = Workflow(original_prompt=self.prompt, tasks=[task], shared_context=self.context)
        workflow_manager.create_workflow(wf)
        return wf


__all__ = ["Goal"]
