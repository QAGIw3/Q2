import yaml
import pulsar
import time
import argparse
import logging
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def poll_for_workflow(manager_url: str, workflow_id: str, timeout: int = 120) -> dict:
    """Polls the managerQ API for the status of a workflow."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{manager_url}/v1/workflows/{workflow_id}", timeout=10)
            if response.status_code == 200:
                workflow = response.json()
                if workflow['status'] in ['COMPLETED', 'FAILED']:
                    logger.info(f"Workflow '{workflow_id}' finished with status: {workflow['status']}")
                    return workflow
            time.sleep(5)
        except requests.RequestException as e:
            logger.warning(f"Could not connect to managerQ: {e}")
            time.sleep(5)
    raise TimeoutError(f"Workflow '{workflow_id}' did not complete within {timeout} seconds.")

def get_workflow_by_event(manager_url: str, event_id: str, timeout: int = 30) -> dict:
    """Gets the workflow associated with a given event ID."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{manager_url}/v1/workflows/by_event/{event_id}", timeout=10)
            if response.status_code == 200:
                logger.info(f"Found workflow for event '{event_id}'.")
                return response.json()
            time.sleep(2)
        except requests.RequestException as e:
            logger.warning(f"Could not connect to managerQ to find workflow for event '{event_id}': {e}")
            time.sleep(2)
    raise TimeoutError(f"Could not find workflow for event '{event_id}' within {timeout} seconds.")

def get_workflow_history(manager_url: str, workflow_id: str) -> list:
    """Fetches the history of a workflow."""
    try:
        response = requests.get(f"{manager_url}/v1/workflows/{workflow_id}/history", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Could not fetch history for workflow '{workflow_id}': {e}")
        return []

def query_knowledge_graph(gremlin_query: str) -> list:
    """Sends a Gremlin query to the KnowledgeGraphQ service."""
    try:
        # Assuming KnowledgeGraphQ is running on the default port
        url = "http://localhost:8182/" 
        response = requests.post(url, json={"gremlin": gremlin_query}, timeout=30)
        response.raise_for_status()
        return response.json().get("result", {}).get("data", [])
    except requests.RequestException as e:
        logger.error(f"Could not query Knowledge Graph: {e}")
        return []

def run_collaborative_scenario(manager_url: str, workflow_id: str):
    """Runs assertions specific to the collaborative root cause scenario."""
    history = get_workflow_history(manager_url, workflow_id)
    
    # Check that the devops agent delegated a task to the data analyst agent
    delegation_task = next((t for t in history if "delegate_task" in t.get('prompt', '') and "data_analyst" in t.get('prompt', '')), None)
    assert delegation_task is not None, "Expected a delegation task to the data analyst agent."
    logger.info("Assertion PASSED: DevOps agent correctly delegated to Data Analyst.")
    
    # Check that the data analyst used its specific tools
    sql_task = next((t for t in history if "execute_sql_query" in t.get('prompt', '')), None)
    assert sql_task is not None, "Expected the data analyst to execute a SQL query."
    logger.info("Assertion PASSED: Data Analyst agent used the 'execute_sql_query' tool.")

def run_scenario(scenario_path: str, pulsar_url: str, topic: str, manager_url: str):
    """
    Reads a scenario file, publishes it, finds the created workflow, and asserts its successful completion and effects.
    """
    try:
        with open(scenario_path, 'r') as f:
            scenario_data = yaml.safe_load(f)
        logger.info(f"Loaded scenario from: {scenario_path}")

        client = pulsar.Client(pulsar_url)
        producer = client.create_producer(topic)
        
        message = json.dumps(scenario_data).encode('utf-8')
        producer.send(message)
        logger.info(f"Published event '{scenario_data['event_id']}' to topic '{topic}'.")
        producer.close()
        client.close()

        # Get the workflow that should have been created by the event
        event_id = scenario_data['event_id']
        workflow = get_workflow_by_event(manager_url, event_id)
        workflow_id = workflow['workflow_id']
        
        # Poll for the workflow to complete
        final_workflow = poll_for_workflow(manager_url, workflow_id)
        
        # Assert the outcome
        assert final_workflow['status'] == 'COMPLETED', f"Workflow failed with status: {final_workflow['status']}"
        logger.info("Assertion PASSED: Workflow completed successfully.")
        
        # Assert on tool usage
        history = get_workflow_history(manager_url, workflow_id)
        
        # Example assertion: Check if 'get_service_logs' was called
        # A more robust test would check the sequence and parameters
        tool_calls = [task['prompt'] for task in history if task.get('type') == 'task']
        
        # This is a simple example. Real assertions would be more specific to the scenario.
        if "collaborative" in scenario_path:
            run_collaborative_scenario(manager_url, workflow_id)
        else:
            assert any("get_service_logs" in call for call in tool_calls), "Expected 'get_service_logs' to be called."
            logger.info("Assertion PASSED: 'get_service_logs' was called.")

        # Assert that an insight was created for this workflow
        insight_query = f"g.V().has('Workflow', 'workflow_id', '{workflow_id}').out('generated_insight').count()"
        insight_count = query_knowledge_graph(insight_query)
        assert insight_count and insight_count[0] > 0, "Expected an insight to be generated for the workflow."
        logger.info("Assertion PASSED: An insight was created in the Knowledge Graph.")

    except FileNotFoundError:
        logger.error(f"Scenario file not found: {scenario_path}")
    except Exception as e:
        logger.error(f"An error occurred during scenario execution: {e}", exc_info=True)
        # Re-raise to make sure the test runner sees the failure
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AIOps test scenario.")
    parser.add_argument("scenario_file", help="Path to the scenario YAML file.")
    parser.add_argument("--pulsar-url", default="pulsar://localhost:6650", help="Pulsar service URL.")
    parser.add_argument("--topic", default="persistent://public/default/platform-events", help="Pulsar topic to publish to.")
    parser.add_argument("--manager-url", default="http://localhost:8003", help="managerQ API URL.")
    args = parser.parse_args()

    run_scenario(args.scenario_file, args.pulsar_url, args.topic, args.manager_url) 