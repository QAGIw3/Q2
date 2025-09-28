import asyncio
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

try:
    from shared.q_auth_parser.models import UserClaims  # type: ignore
    from shared.q_auth_parser.parser import get_current_user  # type: ignore
except Exception:  # Offline/dev fallback

    class UserClaims(BaseModel):  # minimal stub
        sub: str = "dev-user"

    def get_current_user():  # pragma: no cover - fallback
        return UserClaims()


from managerQ.app.dependencies import get_kg_client, get_pulse_client, get_vector_store_client
from managerQ.app.models import KGEdge, KGNode, KnowledgeGraphResult, SearchQuery, SearchResponse, VectorStoreResult
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.q_pulse_client.client import QuantumPulseClient
from shared.q_pulse_client.models import QPChatMessage, QPChatRequest
from shared.q_vectorstore_client.client import Query as VectorQuery
from shared.q_vectorstore_client.client import VectorStoreClient

router = APIRouter()
logger = logging.getLogger(__name__)


def _build_summary_prompt(
    query: str, vector_results: List[VectorStoreResult], kg_result: KnowledgeGraphResult | None
) -> str:
    return f"""Based on the following information, provide a concise, one-paragraph summary for the query: "{query}"

Semantic Search Results:
{'- ' + '\n- '.join([res.content for res in vector_results]) if vector_results else 'None'}

Knowledge Graph Context:
Found {len(kg_result.nodes) if kg_result else 0} related entities.

Summary:"""


def _parse_vector_results(raw_result) -> List[VectorStoreResult]:
    items: List[VectorStoreResult] = []
    if raw_result and raw_result.results:
        for res in raw_result.results[0].hits:
            items.append(
                VectorStoreResult(
                    source=res.metadata.get("source", "Unknown"),
                    content=res.metadata.get("text", ""),
                    score=res.score,
                    metadata=res.metadata,
                )
            )
    return items


def _parse_kg_results(raw_result) -> KnowledgeGraphResult | None:
    raw_graph_data = raw_result.get("result", {}).get("data", [])
    nodes = []
    for item in raw_graph_data:
        if item.get("@type") == "g:Vertex":
            vertex_value = item.get("@value", {})
            node_id = vertex_value.get("id")
            if not node_id:
                continue
            properties = {}
            for key, prop_list in vertex_value.get("properties", {}).items():
                if isinstance(prop_list, list) and prop_list:
                    prop_value = prop_list[0].get("@value", {}).get("value")
                    if prop_value is not None:
                        properties[key] = prop_value
            nodes.append(KGNode(id=node_id, label=vertex_value.get("label", "Unknown"), properties=properties))
    return KnowledgeGraphResult(nodes=nodes, edges=[])


@router.post("/", response_model=SearchResponse)
async def cognitive_search(
    search_query: SearchQuery,
    user: UserClaims = Depends(get_current_user),
    vector_store_client: VectorStoreClient = Depends(get_vector_store_client),
    kg_client: KnowledgeGraphClient = Depends(get_kg_client),
    pulse_client: QuantumPulseClient = Depends(get_pulse_client),
):
    """High level multi-source cognitive search orchestration."""
    try:
        vector_query = VectorQuery(query=search_query.query, top_k=5)
        semantic_future = vector_store_client.search(collection_name="documents", queries=[vector_query])
        gremlin_query = f"g.V().has('name', textContains('{search_query.query}')).elementMap().limit(10)"
        graph_future = kg_client.execute_gremlin_query(gremlin_query)
        raw_vector, raw_graph = await asyncio.gather(semantic_future, graph_future, return_exceptions=True)

        vector_results: List[VectorStoreResult] = []
        if isinstance(raw_vector, Exception):
            logger.error("Vector store search failed", exc_info=raw_vector)
        else:
            vector_results = _parse_vector_results(raw_vector)

        kg_result: KnowledgeGraphResult | None = None
        if isinstance(raw_graph, Exception):
            logger.error("Knowledge graph search failed", exc_info=raw_graph)
        else:
            kg_result = _parse_kg_results(raw_graph)

        summary = "Could not generate a summary."
        model_version = None
        if vector_results or (kg_result and kg_result.nodes):
            prompt = _build_summary_prompt(search_query.query, vector_results, kg_result)
            try:
                summary_request = QPChatRequest(
                    messages=[QPChatMessage(role="user", content=prompt)],
                    model="q-alpha-v3-summarizer",
                )
                summary_response = await pulse_client.get_chat_completion(summary_request)
                summary = summary_response.choices[0].message.content
                model_version = summary_response.model
            except Exception:  # pragma: no cover - external service
                logger.error("Failed to generate summary from QuantumPulse", exc_info=True)
                summary = "Error generating summary."

        return SearchResponse(
            ai_summary=summary,
            vector_results=vector_results,
            knowledge_graph_result=kg_result,
            model_version=model_version,
        )
    except Exception:  # Broad catch to wrap into HTTP error
        logger.error("Unexpected error during cognitive search", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during the search."
        )


class NodeNeighborsRequest(BaseModel):
    node_id: str
    hops: int = 1


@router.post("/kg-neighbors", response_model=KnowledgeGraphResult)
async def get_node_neighbors(
    request: NodeNeighborsRequest,
    user: UserClaims = Depends(get_current_user),
    kg_client: KnowledgeGraphClient = Depends(get_kg_client),
):
    """
    Fetches the neighbors of a given node in the knowledge graph.
    """
    logger.info(f"Fetching neighbors for node '{request.node_id}'")

    # A Gremlin query to find a node and its neighbors up to N hops
    query = f"g.V('{request.node_id}').repeat(both().simplePath()).times({request.hops}).emit().path().by(elementMap())"

    try:
        raw_graph_data = await kg_client.execute_gremlin_query(query)
        # We need a function to parse this path-based result into nodes and edges
        kg_result = parse_gremlin_path_to_graph(raw_graph_data)
        return kg_result
    except Exception as e:
        logger.error(f"Failed to get node neighbors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get node neighbors.")


def parse_gremlin_path_to_graph(gremlin_response: Dict[str, Any]) -> KnowledgeGraphResult:
    """
    Parses a Gremlin path() response into a KnowledgeGraphResult.
    The response contains a list of paths, and each path is a list of elements (nodes and edges).
    """
    nodes = {}
    edges = {}

    response_data = gremlin_response.get("result", {}).get("data", [])
    if not isinstance(response_data, list):
        return KnowledgeGraphResult(nodes=[], edges=[])

    for path in response_data:
        path_objects = path.get("objects", [])
        for i, element in enumerate(path_objects):
            element_value = element.get("@value", {})
            element_type = element.get("@type")

            if element_type == "g:Vertex":
                node_id = element_value.get("id")
                if node_id and node_id not in nodes:
                    properties = {}
                    for key, prop_list in element_value.get("properties", {}).items():
                        if isinstance(prop_list, list) and prop_list:
                            prop_value = prop_list[0].get("@value", {}).get("value")
                            if prop_value is not None:
                                properties[key] = prop_value
                    nodes[node_id] = KGNode(
                        id=node_id, label=element_value.get("label", "Unknown"), properties=properties
                    )

            # Reconstruct edges from the path sequence
            if i > 0 and (i % 2) != 0:  # An edge appears at every odd index > 0
                source_vertex = path_objects[i - 1].get("@value", {})
                target_vertex = path_objects[i + 1].get("@value", {})
                edge_id = f"{source_vertex.get('id')}-{element_value.get('label')}-{target_vertex.get('id')}"

                if edge_id not in edges:
                    edges[edge_id] = KGEdge(
                        source=source_vertex.get("id"),
                        target=target_vertex.get("id"),
                        label=element_value.get("label", "related"),
                    )

    return KnowledgeGraphResult(nodes=list(nodes.values()), edges=list(edges.values()))
