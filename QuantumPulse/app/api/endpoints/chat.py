import json
import logging
import random
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import OpenAI

from shared.q_auth_parser.models import UserClaims
from shared.q_auth_parser.parser import get_current_user
from shared.q_h2m_client.client import h2m_client
from shared.vault_client import VaultClient

from ...core.model_manager import model_manager
from ...models.chat import ChatChoice, ChatMessage, ChatRequest, ChatResponse, ChatUsage

# Configure logging
logger = logging.getLogger(__name__)
router = APIRouter()

# --- OpenAI Client Initialization ---
# The client is initialized upon first request to this endpoint.
# This avoids trying to connect to Vault/OpenAI at application startup.
client: OpenAI | None = None


def get_openai_client():
    """FastAPI dependency to initialize and get the OpenAI client."""
    global client
    if client is None:
        try:
            logger.info("Initializing OpenAI client for the first time.")
            vault_client = VaultClient()
            api_key = vault_client.read_secret("secret/data/openai", "api_key")
            if not api_key:
                raise ValueError("OpenAI API key not found in Vault.")
            client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize OpenAI client from Vault: {e}", exc_info=True)
            # We don't raise here, but the client will remain None,
            # and subsequent calls will fail with a 503.
    return client


async def sse_generator(openai_stream):
    """Generator function to yield Server-Sent Events from the OpenAI stream."""
    try:
        for chunk in openai_stream:
            yield f"data: {chunk.json()}\n\n"
    except Exception as e:
        logger.error(f"Error in SSE generator: {e}", exc_info=True)
        # Yield a final error message if something goes wrong
        error_payload = {"error": "An error occurred while streaming."}
        yield f"data: {json.dumps(error_payload)}\n\n"


@router.post("/completions", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def create_chat_completion(request: ChatRequest, user: UserClaims = Depends(get_current_user)):
    """
    Provides a synchronous, request/response endpoint for chat completions.
    This acts as a centralized gateway to locally hosted, fine-tuned models.
    It performs A/B testing if multiple active models are found.
    """
    logger.info(f"Received chat completion request from user '{user.username}' for base model '{request.model}'.")

    try:
        # 1. Get active models from the registry
        active_models = await h2m_client.get_active_models(base_model=request.model)

        if not active_models:
            # Fallback to the base model if no active fine-tuned models are found
            selected_model_name = request.model
        else:
            # 2. A/B Testing: Simple random choice for now
            selected_model_name = random.choice([m["metadata"]["model_name"] for m in active_models])

        logger.info(f"Routing request to model: '{selected_model_name}'")

        # 3. Get the model and tokenizer from the manager
        model, tokenizer = model_manager.get_model_and_tokenizer(selected_model_name)

        if not model or not tokenizer:
            raise HTTPException(status_code=503, detail=f"Model '{selected_model_name}' could not be loaded.")

        # 4. Generate the completion
        # Note: This is a simplified generation process.
        # A real implementation would handle tokenization, attention masks, etc. more robustly.
        inputs = tokenizer(request.messages[-1].content, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=request.max_tokens)
        completion_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 5. Format the response to match the expected ChatResponse model
        response = ChatResponse(
            id=f"cmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=selected_model_name,
            choices=[
                ChatChoice(
                    index=0, message=ChatMessage(role="assistant", content=completion_text), finish_reason="stop"
                )
            ],
            usage=ChatUsage(
                prompt_tokens=len(inputs.input_ids[0]),
                completion_tokens=len(outputs[0]),
                total_tokens=len(inputs.input_ids[0]) + len(outputs[0]),
            ),
        )

        logger.info(f"Successfully generated chat completion from model '{selected_model_name}'.")
        return response

    except Exception as e:
        logger.error(f"Error during local model inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during inference: {e}",
        )
