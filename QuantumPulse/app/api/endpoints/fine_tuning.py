import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from datasets import Dataset
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

from shared.q_auth_parser.models import UserClaims
from shared.q_auth_parser.parser import get_current_user
from shared.vault_client import VaultClient

from ...core.model_manager import model_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Pydantic Models ---
class PreferencePair(BaseModel):
    chosen: Optional[str] = Field(None, description="The preferred (e.g., 'good' feedback) response from a prompt.")
    rejected: Optional[str] = Field(
        None, description="The rejected (e.g., 'bad' feedback) response from the same prompt."
    )
    prompt: str = Field(..., description="The original prompt that led to the chosen/rejected responses.")


class FineTuneRequest(BaseModel):
    model_to_fine_tune: str = Field(description="The ID of the base model on Hugging Face to fine-tune (e.g., 'gpt2').")
    dataset: List[PreferencePair] = Field(description="The dataset of preference pairs for training.")
    new_model_name: str = Field(description="The desired name for the new, fine-tuned model on the Hugging Face Hub.")


class FineTuneResponse(BaseModel):
    job_id: str
    status: str
    message: str
    hub_url: str | None = None


# --- Fine-Tuning Logic ---


def run_fine_tuning_job(request: FineTuneRequest, job_id: str):
    """
    This function runs the actual DPO fine-tuning process.
    It's designed to be run in a background task.
    """
    logger.info(f"Starting fine-tuning job {job_id} for model {request.model_to_fine_tune}.")
    try:
        # 1. Fetch Hugging Face token from Vault
        vault_client = VaultClient()
        hf_token = vault_client.read_secret("secret/data/huggingface", "token")
        if not hf_token:
            raise ValueError("Hugging Face token not found in Vault.")

        # 2. Prepare the dataset
        data_dict = {"prompt": [], "chosen": [], "rejected": []}
        for item in request.dataset:
            data_dict["prompt"].append(item.prompt)
            data_dict["chosen"].append(item.chosen)
            data_dict["rejected"].append(item.rejected)

        train_dataset = Dataset.from_dict(data_dict)

        # 3. Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(request.model_to_fine_tune)
        tokenizer = AutoTokenizer.from_pretrained(request.model_to_fine_tune)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 4. Set up Training Arguments
        output_dir = f"./dpo_results_{job_id}"
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            num_train_epochs=1,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            logging_steps=1,
            output_dir=output_dir,
            push_to_hub=True,
            hub_model_id=request.new_model_name,
            hub_token=hf_token,
        )

        # 5. Initialize and run the DPO Trainer
        dpo_trainer = DPOTrainer(
            model,
            args=training_args,
            beta=0.1,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
        dpo_trainer.train()

        # 6. Push the final model to the hub
        dpo_trainer.push_model()
        logger.info(f"Fine-tuning job {job_id} completed. Model pushed to {request.new_model_name}.")

        # 7. Dynamically load the new model into the ModelManager
        logger.info(f"Attempting to dynamically load new model '{request.new_model_name}'...")
        model_manager.load_model(request.new_model_name)

    except Exception as e:
        logger.error(f"Fine-tuning job {job_id} failed: {e}", exc_info=True)
        # Here you might update a database with the failure status


@router.post("", response_model=FineTuneResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_fine_tuning_job(
    request: FineTuneRequest,
    background_tasks: BackgroundTasks,
    user: UserClaims = Depends(get_current_user),  # Ensure only authorized users/services can run this
):
    """
    Accepts a dataset of preference pairs and launches an asynchronous DPO fine-tuning job.
    """
    logger.info(f"Received fine-tuning request for model '{request.model_to_fine_tune}' from user '{user.username}'.")

    job_id = f"ft-job-{uuid.uuid4()}"

    # Run the complex, long-running training process in the background
    background_tasks.add_task(run_fine_tuning_job, request, job_id)

    logger.info(f"Dispatched fine-tuning job '{job_id}' to a background worker.")

    hub_url = f"https://huggingface.co/{os.getenv('HF_USERNAME', 'your-hf-user')}/{request.new_model_name}"

    return FineTuneResponse(
        job_id=job_id,
        status="submitted",
        message="Fine-tuning job has been successfully submitted and is running in the background.",
        hub_url=hub_url,
    )
