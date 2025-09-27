import yaml
import os
import httpx
from shared.vault_client import VaultClient
from jinja2 import Environment

class FlowEngine:
    def __init__(self):
        self.flows_dir = os.path.join(os.path.dirname(__file__), '..', 'flows')
        self.jinja_env = Environment(autoescape=True)
        self.vault_client = VaultClient() # This should be a singleton in a real app

    async def trigger_flow(self, flow_id: str, trigger_event: dict):
        flow_path = os.path.join(self.flows_dir, f"{flow_id}.yaml")
        if not os.path.exists(flow_path):
            raise ValueError(f"Flow '{flow_id}' not found.")

        with open(flow_path, 'r') as f:
            flow_definition = yaml.safe_load(f)

        context = {"trigger": trigger_event}

        for step in flow_definition.get("steps", []):
            if step["type"] == "task":
                await self._execute_task_step(step, context)

    async def _execute_task_step(self, step: dict, context: dict):
        task_def = step.get("task", {})
        
        prompt_template = self.jinja_env.from_string(task_def.get("prompt", ""))
        rendered_prompt = prompt_template.render(context)
        
        # Delegate to managerQ
        manager_url = os.environ.get("MANAGERQ_URL", "http://managerq.q-platform.svc.cluster.local:8003")
        service_token = self.vault_client.read_secret_data("secret/data/managerq")["service_token"]
        
        headers = {"Authorization": f"Bearer {service_token}"}
        
        payload = {
            "prompt": rendered_prompt,
            "agent_personality": task_def.get("agent_personality", "default"),
            "workflow_id": context["trigger"].get("issue", {}).get("key") # Use issue key as workflow id
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{manager_url}/v1/tasks", json=payload, headers=headers)
            response.raise_for_status()

flow_engine = FlowEngine() 