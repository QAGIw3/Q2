# managerQ/app/core/user_workflow_store.py
import logging
import os
from typing import Dict, List, Optional

from pyignite import AioClient
from pyignite.exceptions import AuthenticationError, CacheError, ClusterError, HandshakeError, SocketError, SQLError

from managerQ.app.config import settings
from managerQ.app.models import Workflow

logger = logging.getLogger(__name__)

USER_WORKFLOWS_CACHE_NAME = "user_workflows"


class UserWorkflowStore:
    def __init__(self):
        self._offline = os.environ.get("MANAGERQ_OFFLINE", "0") == "1"
        self._client = AioClient() if not self._offline else None
        self._memory_store: Dict[str, Dict] = {}

    async def connect(self):
        if self._offline:
            return
        if not self._client.is_connected():
            try:
                await self._client.connect(settings.ignite.addresses)
                await self._client.get_or_create_cache(USER_WORKFLOWS_CACHE_NAME)
                logger.info("Connected to Ignite and ensured user_workflows cache exists.")
            except (CacheError, ClusterError, SocketError, SQLError, AuthenticationError, HandshakeError) as e:
                logger.error(f"Failed to connect UserWorkflowStore to Ignite: {e}", exc_info=True)
                raise

    async def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        if self._offline:
            wf = self._memory_store.get(workflow_id)
            return Workflow(**wf) if wf else None
        cache = await self._client.get_cache(USER_WORKFLOWS_CACHE_NAME)
        workflow_dict = await cache.get(workflow_id)
        return Workflow(**workflow_dict) if workflow_dict else None

    async def save_workflow(self, workflow: Workflow):
        if self._offline:
            self._memory_store[workflow.workflow_id] = workflow.dict()
            return
        cache = await self._client.get_cache(USER_WORKFLOWS_CACHE_NAME)
        await cache.put(workflow.workflow_id, workflow.dict())

    async def get_workflows_by_owner(self, owner_id: str) -> List[Workflow]:
        if self._offline:
            return [
                Workflow(**wf)
                for wf in self._memory_store.values()
                if wf.get("shared_context", {}).get("owner_id") == owner_id
            ]
        cache = await self._client.get_cache(USER_WORKFLOWS_CACHE_NAME)
        workflows = []
        cursor = await cache.scan()
        for _, workflow_dict in cursor:
            if workflow_dict.get("shared_context", {}).get("owner_id") == owner_id:
                workflows.append(Workflow(**workflow_dict))
        return workflows

    async def delete_workflow(self, workflow_id: str, owner_id: str):
        workflow = await self.get_workflow(workflow_id)
        if workflow and workflow.shared_context.get("owner_id") == owner_id:
            if self._offline:
                self._memory_store.pop(workflow_id, None)
            else:
                cache = await self._client.get_cache(USER_WORKFLOWS_CACHE_NAME)
                await cache.remove_key(workflow_id)
            logger.info(f"Deleted workflow '{workflow_id}' for user '{owner_id}'.")


# Singleton instance
user_workflow_store = UserWorkflowStore()
