import logging
import asyncio
import json
import re
from typing import Dict, Any, Optional, List
import jinja2
from collections import deque

from app.models.flow import Flow, FlowStep
from app.models.connector import Connector, ConnectorAction, BaseConnector
from app.connectors.zulip.zulip_connector import zulip_connector
from app.connectors.smtp.email_connector import email_connector
from app.connectors.pulsar.pulsar_connector import pulsar_publisher_connector
from app.connectors.http.http_connector import http_connector
from app.connectors.github.github_connector import github_connector
from app.core.vault_client import vault_client

# This is a bit of a hack, ideally the engine would have access
# to a persistent flow store instead of importing from the API layer.
from app.api.flows import PREDEFINED_FLOWS


# A registry of all available connector instances
AVAILABLE_CONNECTORS: Dict[str, Connector] = {
    zulip_connector.connector_id: zulip_connector,
    email_connector.connector_id: email_connector,
    pulsar_publisher_connector.connector_id: pulsar_publisher_connector,
    http_connector.connector_id: http_connector,
    github_connector.connector_id: github_connector,
}

logger = logging.getLogger(__name__)

class FlowExecutionEngine:
    def __init__(self):
        self._connectors: Dict[str, 'BaseConnector'] = {}
        self.logger = logger
        self._jinja_env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=True)

    def register_connector(self, connector: 'BaseConnector'):
        if connector.connector_id in self._connectors:
            self.logger.warning(f"Connector '{connector.connector_id}' is already registered. Overwriting.")
        self.logger.info(f"Registering connector: {connector.connector_id}")
        self._connectors[connector.connector_id] = connector

    def _get_connector(self, connector_id: str) -> Optional['BaseConnector']:
        return self._connectors.get(connector_id)

    def _topologically_sort_steps(self, steps: List[FlowStep]) -> List[FlowStep]:
        """
        Performs a topological sort on the flow steps based on their dependencies.
        Returns a list of steps in a valid execution order.
        Raises ValueError if a cycle is detected or a dependency is not found.
        """
        step_map = {step.name: step for step in steps}
        in_degree = {step.name: 0 for step in steps}
        adj = {step.name: [] for step in steps}

        for step in steps:
            for dep in step.dependencies:
                if dep not in step_map:
                    raise ValueError(f"Step '{step.name}' has an unknown dependency: '{dep}'")
                adj[dep].append(step.name)
                in_degree[step.name] += 1

        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        sorted_order = []

        while queue:
            u = queue.popleft()
            sorted_order.append(step_map[u])
            for v_name in adj[u]:
                in_degree[v_name] -= 1
                if in_degree[v_name] == 0:
                    queue.append(v_name)

        if len(sorted_order) != len(steps):
            raise ValueError("A cycle was detected in the flow's dependencies.")

        return sorted_order

    def _render_template(self, template_str: Optional[str], context: Dict[str, Any]) -> Any:
        """Renders a Jinja2 template string with the given context."""
        if template_str is None:
            return context
        try:
            template = self._jinja_env.from_string(template_str)
            rendered_str = template.render(context)
            # Try to parse as JSON, fall back to raw string if it fails
            try:
                return json.loads(rendered_str)
            except json.JSONDecodeError:
                return rendered_str
        except Exception as e:
            self.logger.error(f"Failed to render input template: {e}", exc_info=True)
            raise

    async def _execute_step(self, step_config: FlowStep, flow_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        connector_id = step_config.connector_id
        
        # 1. Render the input for this step from the overall flow context
        step_input = self._render_template(step_config.input_template, flow_context)

        # 2. Prepare the action for the connector
        action = ConnectorAction(
            action_id=step_config.configuration.get("action_id", "default_action"),
            credential_id=step_config.credential_id,
            configuration=step_config.configuration
        )
        
        self.logger.info(f"Executing action '{action.action_id}' on connector '{connector_id}'")

        if connector_id not in AVAILABLE_CONNECTORS:
            raise ValueError(f"Connector '{connector_id}' not found.")
        
        connector = AVAILABLE_CONNECTORS[connector_id]
        
        # 3. Execute and return the result
        result = await connector.execute(action, configuration=action.configuration, data_context=step_input)
        return result

    async def run_flow(self, flow_model: Flow, initial_context: Dict[str, Any]):
        self.logger.info(f"--- Running Flow: {flow_model.name} ---")
        
        try:
            sorted_steps = self._topologically_sort_steps(flow_model.steps)
            self.logger.info(f"Execution order: {[s.name for s in sorted_steps]}")
        except ValueError as e:
            self.logger.error(f"Flow validation failed: {e}", exc_info=True)
            return

        # This context holds the results from all completed steps
        flow_context = {"trigger": initial_context}

        for step in sorted_steps:
            self.logger.info(f"  - Executing step: {step.name}")
            try:
                step_result = await self._execute_step(step, flow_context)
                
                if step_result:
                    # Add the result of this step to the global flow context
                    flow_context[step.name] = step_result
                    self.logger.info(f"    Step '{step.name}' completed.")

            except Exception as e:
                self.logger.error(f"    ERROR: Step '{step.name}' failed: {e}", exc_info=True)
                # In a real engine, we'd have error handling, retries, etc.
                break # Stop flow on first failure
        self.logger.info(f"--- Flow Finished: {flow_model.name} ---")

    async def run_flow_by_id(self, flow_id: str, data_context: Dict[str, Any]):
        """Finds a pre-defined flow by its ID and runs it."""
        if flow_id not in PREDEFINED_FLOWS:
            self.logger.error(f"Attempted to run non-existent flow with ID: {flow_id}")
            raise ValueError(f"Flow with ID '{flow_id}' not found.")
        
        flow_data = PREDEFINED_FLOWS[flow_id]
        flow_model = Flow(**flow_data) # Validate with Pydantic model
        await self.run_flow(flow_model, data_context)

    async def run_action(self, connector_id: str, action_id: str, credential_id: str, configuration: dict, data_context: dict) -> Optional[dict]:
        """
        Executes a single connector action directly, outside of a flow.
        """
        self.logger.info(f"Executing single action '{action_id}' for connector '{connector_id}'.")
        
        connector = self._get_connector(connector_id)
        if not connector:
            raise ValueError(f"Connector '{connector_id}' not found.")
        
        action = ConnectorAction(action_id=action_id, credential_id=credential_id)
        
        try:
            result = await connector.execute(
                action=action,
                configuration=configuration,
                data_context=data_context
            )
            return result
        except Exception as e:
            self.logger.error(f"Error executing single action '{action_id}': {e}", exc_info=True)
            # Re-raise the exception to be handled by the API layer
            raise


engine = FlowExecutionEngine()

# --- Register Connectors ---
from ..connectors.github.github_connector import github_connector
from ..connectors.http.http_connector import http_connector
from ..connectors.smtp.email_connector import email_connector
from ..connectors.zulip.zulip_connector import zulip_connector
from ..connectors.openproject.openproject_connector import openproject_connector
from ..connectors.kubernetes.kubernetes_connector import kubernetes_connector

engine.register_connector(github_connector)
engine.register_connector(http_connector)
engine.register_connector(email_connector)
engine.register_connector(zulip_connector)
engine.register_connector(openproject_connector)
engine.register_connector(kubernetes_connector) 