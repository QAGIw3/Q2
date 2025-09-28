import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from pyignite import Client
from pyignite.exceptions import AuthenticationError, CacheError, ClusterError, HandshakeError, SocketError, SQLError

from managerQ.app.config import settings
from managerQ.app.core.task_dispatcher import task_dispatcher
from managerQ.app.core.user_workflow_store import user_workflow_store
from managerQ.app.core.workflow_executor import workflow_executor
from managerQ.app.models import AmbiguousGoalError  # Import the exception
from managerQ.app.models import Goal, GoalStatus, Workflow

logger = logging.getLogger(__name__)


class GoalManager:
    """
    Manages the lifecycle of platform goals in an Ignite cache.
    """

    def __init__(self):
        self._offline = os.environ.get("MANAGERQ_OFFLINE", "0") == "1"
        self._client = Client() if not self._offline else None
        self._cache = None
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        if not self._offline:
            self.connect()
        else:
            logger.warning("GoalManager running in OFFLINE mode: Ignite skipped; using in-memory goal store.")

    def connect(self):
        if self._offline:
            return
        try:
            self._client.connect(settings.ignite.addresses)
            self._cache = self._client.get_or_create_cache("goals")
            logger.info("GoalManager connected to Ignite and got cache 'goals'.")
        except (CacheError, ClusterError, SocketError, SQLError, AuthenticationError, HandshakeError) as e:
            logger.error(f"Failed to connect GoalManager to Ignite: {e}", exc_info=True)
            raise

    def close(self):
        if not self._offline and self._client and self._client.is_connected():
            self._client.close()

    def create_goal(self, goal: Goal) -> None:
        """Saves a new goal to the cache."""
        logger.info(f"Creating goal: {goal.goal_id} - '{goal.objective}'")
        if self._offline:
            self._memory_store[goal.goal_id] = goal.dict()
        else:
            self._cache.put(goal.goal_id, goal.dict())

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Retrieves a goal from the cache."""
        goal_data = self._memory_store.get(goal_id) if self._offline else self._cache.get(goal_id)
        if goal_data:
            return Goal(**goal_data)
        return None

    def get_all_active_goals(self) -> List[Goal]:
        """Retrieves all active goals using a SQL query."""
        if self._offline:
            return [Goal(**g) for g in self._memory_store.values() if g.get("is_active")]
        query = "SELECT * FROM Goal WHERE is_active = true"
        try:
            cursor = self._cache.sql(query, include_field_names=False)
            goals = [Goal(**row) for row in cursor]
            return goals
        except (CacheError, ClusterError, SocketError, SQLError, AuthenticationError, HandshakeError) as e:
            logger.error(f"Failed to query for active goals: {e}", exc_info=True)
            return []

    async def process_new_goal(self, prompt: str, user_id: str) -> GoalStatus:
        """
        Processes a new user-submitted goal by dispatching it to the PlannerAgent.
        """
        goal_id = str(uuid.uuid4())
        logger.info(f"Dispatching new goal '{prompt}' to PlannerAgent.")

        try:
            # Dispatch the prompt to the dedicated planner agent
            task_id = await task_dispatcher.dispatch_task(
                prompt=prompt,
                agent_personality="planner_agent",
            )

            # Wait for the planner agent to return the workflow JSON
            plan_json_str = await task_dispatcher.await_task_result(task_id, timeout=120)

            if not plan_json_str:
                raise ValueError("PlannerAgent returned an empty result.")

            plan_json = json.loads(plan_json_str)

            # --- New: Handle structured errors from PlannerAgent ---
            if "error" in plan_json and plan_json["error"] == "AMBIGUOUS_GOAL":
                raise AmbiguousGoalError(
                    message="The user's goal is ambiguous and requires clarification.",
                    clarifying_question=plan_json.get("clarifying_question", "Please provide more details."),
                )
            # ----------------------------------------------------

            workflow = Workflow(**plan_json)

            # If planning succeeds, store and execute
            await user_workflow_store.save_workflow(user_id, workflow)
            asyncio.create_task(workflow_executor.execute_workflow(workflow.workflow_id, user_id))

            new_goal = Goal(
                goal_id=goal_id,
                user_id=user_id,
                objective=prompt,
                workflow_id=workflow.workflow_id,
                status="IN_PROGRESS",
            )
            self.create_goal(new_goal)

            return GoalStatus(goal_id=goal_id, workflow_id=workflow.workflow_id, status=new_goal.status)

        except Exception as e:
            logger.error(f"Failed to process new goal via PlannerAgent: {e}", exc_info=True)
            # This is a simplified error handling. A real system might create a "failed" goal.
            raise

    async def process_clarification(self, goal_id: str, user_clarification: str, user_id: str) -> GoalStatus:
        """
        Processes a user's clarification for a previously ambiguous goal.
        """
        original_goal = self.get_goal(goal_id)
        if not original_goal:
            raise ValueError(f"Goal with ID '{goal_id}' not found.")

        if original_goal.user_id != user_id:
            raise ValueError("User is not authorized to clarify this goal.")

        if original_goal.status != "PENDING_CLARIFICATION":
            raise ValueError(f"Goal '{goal_id}' is not awaiting clarification.")

        # Re-plan with the new information.
        new_prompt = f"Original Goal: {original_goal.objective}\nMy Clarifying Answer: {user_clarification}"

        # We can re-use the same process_new_goal logic with the clarified prompt.
        # Note: This creates a new goal/workflow rather than updating the old one.
        # A more complex implementation could patch the original.
        return await self.process_new_goal(prompt=new_prompt, user_id=user_id)


# Singleton instance
goal_manager = GoalManager()
