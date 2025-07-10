# KnowledgeGraphQ/scripts/ingest_insight.py

import os
import logging
import argparse
import json
from gremlin_python.driver import client, serializer
from gremlin_python.process.anonymous_traversal import traversal
from sentence_transformers import SentenceTransformer

# --- Configuration ---
LOG_LEVEL = "INFO"
JANUSGRAPH_HOST = os.getenv("JANUSGRAPH_HOST", "localhost")
JANUSGRAPH_PORT = int(os.getenv("JANUSGRAPH_PORT", "8182"))
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # A good default for sentence similarity

# --- Logging ---
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

def run(workflow_id: str, original_prompt: str, final_status: str, lesson_learned: str):
    """Connects to the graph and ingests a single insight from a completed workflow."""
    try:
        remote_connection = client.DriverRemoteConnection(
            f"ws://{JANUSGRAPH_HOST}:{JANUSGRAPH_PORT}/gremlin", 'g',
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        g = traversal().withRemote(remote_connection)
        
        logger.info("Successfully connected to Gremlin server for insight ingestion.")
        
        # Load the embedding model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        ingest_insight(g, workflow_id, original_prompt, final_status, lesson_learned, embedding_model)

    except Exception as e:
        logger.error(f"An error occurred during the insight ingestion process: {e}", exc_info=True)
    finally:
        if 'remote_connection' in locals() and remote_connection:
            remote_connection.close()
            logger.info("Gremlin connection closed.")

def ingest_insight(g, workflow_id: str, original_prompt: str, final_status: str, lesson_learned: str, embedding_model):
    """
    Creates or updates a Workflow vertex, creates a new Insight vertex with an embedding,
    and connects them. This process is designed to be idempotent.
    """
    logger.info(f"Ingesting insight for workflow: {workflow_id}")

    # 1. Generate the vector embedding for the lesson learned
    logger.info("Generating embedding for the lesson...")
    embedding = embedding_model.encode(lesson_learned).tolist()
    logger.info(f"Generated embedding with dimension: {len(embedding)}")

    # 2. Create or find the Workflow vertex.
    # We use coalesce to either find the existing vertex or create it if it's not present.
    workflow_vertex = g.V().has('Workflow', 'workflow_id', workflow_id).fold().coalesce(
        g.V().has('Workflow', 'workflow_id', workflow_id),
        g.addV('Workflow').property('workflow_id', workflow_id)
                         .property('original_prompt', original_prompt)
                         .property('final_status', final_status)
    ).next()
    
    # 3. Create the new Insight vertex with its embedding.
    # Insights are unique to a workflow, so we don't check for existence.
    insight_vertex = g.addV('Insight').property('lesson', lesson_learned) \
                                      .property('embedding', json.dumps(embedding)) \
                                      .property('source_workflow', workflow_id).next()

    # 4. Create an edge from the Workflow to its new Insight.
    g.V(workflow_vertex).addE('generated_insight').to(insight_vertex).iterate()

    logger.info(f"Successfully created Insight and linked it to Workflow '{workflow_id}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a workflow insight into the Knowledge Graph.")
    parser.add_argument("--workflow-id", required=True, help="The unique ID of the workflow.")
    parser.add_argument("--original-prompt", required=True, help="The user's original prompt for the workflow.")
    parser.add_argument("--final-status", required=True, choices=['COMPLETED', 'FAILED'], help="The final status of the workflow.")
    parser.add_argument("--lesson-learned", required=True, help="The concise insight or lesson learned from the workflow.")
    
    args = parser.parse_args()
    
    run(
        workflow_id=args.workflow_id,
        original_prompt=args.original_prompt,
        final_status=args.final_status,
        lesson_learned=args.lesson_learned
    ) 