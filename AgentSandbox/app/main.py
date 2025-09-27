# AgentSandbox/app/main.py
import os
import logging
import time
import yaml
import httpx
import threading
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import structlog
from jinja2 import Environment, FileSystemLoader
from dpath.util import get as dpath_get, set as dpath_set
import json

from shared.observability.logging_config import setup_logging
from shared.observability.metrics import setup_metrics
from .api import router as sandbox_api_router

# --- Configuration & Setup ---
setup_logging()
logger = structlog.get_logger("agentsandbox")
app = FastAPI(title="Agent Sandbox", version="0.1.0")
setup_metrics(app, app_name="AgentSandbox")
app.include_router(sandbox_api_router, prefix="/api/v1", tags=["Sandbox"])
jinja_env = Environment(loader=FileSystemLoader('scenarios/'), autoescape=True)

# In-memory storage for simulation results
simulations: Dict[str, Dict[str, Any]] = {}

# --- Action Executor ---
def execute_http_request(params: Dict[str, Any], context: Dict[str, Any]) -> httpx.Response:
    """Executes an HTTP request based on rendered step parameters."""
    rendered_params = json.loads(jinja_env.from_string(json.dumps(params)).render(context))
    
    method = rendered_params.get("method", "GET").upper()
    url = rendered_params.get("url")
    
    if not url:
        raise ValueError("URL is missing from http_request parameters.")

    with httpx.Client(timeout=30.0) as client:
        response = client.request(
            method,
            url,
            json=rendered_params.get("json"),
            headers=rendered_params.get("headers"),
            params=rendered_params.get("params")
        )
    response.raise_for_status() # Raise exception for 4xx/5xx responses
    return response

# --- Assertion Engine ---
def run_assertions(step_name: str, response: httpx.Response, assertions: List[Dict[str, Any]], context: Dict[str, Any]):
    """Runs assertions against an HTTP response."""
    results = []
    for assertion in assertions:
        assertion_result = {"description": f"{step_name}: {assertion.get('path', 'status_code')}", "status": "FAIL"}
        try:
            if assertion["type"] == "status_code":
                if response.status_code == assertion["expected_value"]:
                    assertion_result["status"] = "PASS"
            elif assertion["type"] == "json_response":
                json_body = response.json()
                actual_value = dpath_get(json_body, assertion["path"])
                if "expected_value" in assertion:
                    if str(actual_value) == str(jinja_env.from_string(assertion["expected_value"]).render(context)):
                         assertion_result["status"] = "PASS"
                elif assertion.get("is_not_empty", False):
                    if actual_value:
                        assertion_result["status"] = "PASS"
        except Exception as e:
            assertion_result["details"] = str(e)
        results.append(assertion_result)
    return results

# --- Simulation Runner ---
def run_scenario(sim_id: str, scenario: Dict[str, Any]):
    thread_logger = logger.bind(simulation_id=sim_id)
    thread_logger.info("Starting scenario", scenario_name=scenario['name'])
    simulations[sim_id]["status"] = "RUNNING"
    simulations[sim_id]["steps"] = []

    # Initialize context with services and a unique run_id
    context = {
        "services": scenario.get("services", {}),
        "run_id": jinja_env.from_string(scenario.get("run_id", "")).render(uuid=uuid),
        "steps": {}
    }

    try:
        for step in scenario.get("steps", []):
            step_name = step["name"]
            thread_logger.info(f"Executing step: {step_name}")
            step_result = {"name": step_name, "status": "PENDING"}

            response = execute_http_request(step["params"], context)
            step_result["response_code"] = response.status_code
            
            # Run assertions for the current step
            assertion_results = run_assertions(step_name, response, step.get("assertions", []), context)
            step_result["assertions"] = assertion_results

            if any(r["status"] == "FAIL" for r in assertion_results):
                step_result["status"] = "FAILED"
                simulations[sim_id]["steps"].append(step_result)
                raise Exception(f"Assertion failed in step: {step_name}")
            
            # Extract outputs to context
            if "outputs" in step:
                json_response = response.json()
                for output in step["outputs"]:
                    value = dpath_get(json_response, output["from_path"])
                    context["steps"][step["id"]] = context["steps"].get(step["id"], {})
                    context["steps"][step["id"]]["outputs"] = context["steps"][step["id"]].get("outputs", {})
                    context["steps"][step["id"]]["outputs"][output["name"]] = value

            step_result["status"] = "COMPLETED"
            simulations[sim_id]["steps"].append(step_result)

        simulations[sim_id]["status"] = "COMPLETED_SUCCESS"
    except Exception as e:
        thread_logger.error("Scenario failed", error=str(e))
        simulations[sim_id]["status"] = "COMPLETED_FAILURE"
    finally:
        thread_logger.info("Scenario finished.")


# --- API Endpoints ---
class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    details: Optional[Dict[str, Any]]

@app.post("/simulations/run/{scenario_name}", response_model=SimulationResponse)
async def run_simulation(scenario_name: str):
    scenario_path = os.path.join(os.path.dirname(__file__), '..', 'scenarios', f"{scenario_name}.yaml")
    if not os.path.exists(scenario_path):
        raise HTTPException(status_code=404, detail="Scenario not found")
    
    with open(scenario_path, 'r') as f:
        scenario = yaml.safe_load(f)
        
    sim_id = str(uuid.uuid4())
    
    simulations[sim_id] = {
        "id": sim_id,
        "scenario_name": scenario_name,
        "status": "INITIALIZING",
        "start_time": time.time(),
    }
    
    runner = threading.Thread(target=run_scenario, args=(sim_id, scenario), name=f"runner-{sim_id[:8]}")
    runner.start()
    
    return SimulationResponse(simulation_id=sim_id, status="STARTED", details=None)

@app.get("/simulations/status/{sim_id}", response_model=SimulationResponse)
async def get_simulation_status(sim_id: str):
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return SimulationResponse(
        simulation_id=sim_id,
        status=simulations[sim_id]["status"],
        details=simulations[sim_id]
    )
