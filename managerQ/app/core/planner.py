# managerQ/app/core/planner.py
import logging
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from managerQ.app.models import Workflow, TaskBlock, WorkflowTask, ConditionalBlock, ConditionalBranch
from shared.q_pulse_client.client import QuantumPulseClient
from shared.q_pulse_client.models import QPChatRequest, QPChatMessage, QPChatResponse
from agentQ.app.core.prompts import ANALYSIS_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Global Q Pulse client instance - will be initialized when needed
q_pulse_client: Optional[QuantumPulseClient] = None

def get_q_pulse_client() -> QuantumPulseClient:
    """Lazy initialization of the Q Pulse client."""
    global q_pulse_client
    if q_pulse_client is None:
        from managerQ.app.config import settings
        q_pulse_client = QuantumPulseClient(base_url=settings.qpulse_url)
    return q_pulse_client


class PlanAnalysis(BaseModel):
    """Result of the first phase - analyzing the user's request for clarity and high-level planning."""
    summary: str = Field(description="A concise summary of the user's intent.")
    is_ambiguous: bool = Field(description="Whether the request is vague and needs clarification.")
    clarifying_question: Optional[str] = Field(None, description="Question to ask if ambiguous.")
    high_level_steps: List[str] = Field(default_factory=list, description="High-level steps if not ambiguous.")


class AmbiguousGoalError(Exception):
    """Raised when a goal is too ambiguous to create a concrete plan."""
    
    def __init__(self, clarifying_question: str, message: str = "The goal is ambiguous and needs clarification."):
        self.clarifying_question = clarifying_question
        self.message = message
        super().__init__(message)


class Planner:
    """
    A two-phase planner that:
    1. Analyzes user prompts for clarity and intent
    2. Generates structured workflows for clear goals
    """
    
    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model
        self.retrieved_insights = ""
        self.retrieved_lessons = ""
    
    async def create_plan(self, prompt: str, insights: str = "", lessons: str = "") -> Workflow:
        """
        Create a workflow plan from a user prompt using two-phase planning.
        
        Args:
            prompt: The user's goal/request
            insights: Relevant insights from previous workflows
            lessons: Relevant lessons learned from the knowledge graph
            
        Returns:
            Workflow: A structured workflow plan
            
        Raises:
            AmbiguousGoalError: If the goal is too ambiguous
            ValueError: If LLM returns invalid JSON
        """
        self.retrieved_insights = insights
        self.retrieved_lessons = lessons
        
        # Phase 1: Analyze the prompt for clarity
        analysis = await self._analyze_prompt(prompt)
        
        if analysis.is_ambiguous:
            raise AmbiguousGoalError(
                clarifying_question=analysis.clarifying_question or "Please provide more details about your goal.",
                message="The goal is ambiguous and needs clarification."
            )
        
        # Phase 2: Generate the full workflow
        workflow = await self._generate_workflow(prompt, analysis)
        
        return workflow
    
    async def _analyze_prompt(self, prompt: str) -> PlanAnalysis:
        """
        Phase 1: Analyze the user's prompt for clarity and create high-level steps.
        """
        # Format insights section for the prompt
        insights_section = ""
        if self.retrieved_insights:
            insights_section = f"**Relevant Insights from Past Workflows:**\n{self.retrieved_insights}\n"
        
        # Format lessons section for the prompt  
        lessons_section = ""
        if self.retrieved_lessons:
            lessons_section = f"**Past Lessons:**\n{self.retrieved_lessons}\n"
        
        # Use manual replacement instead of .format() to avoid JSON brace conflicts
        analysis_prompt = ANALYSIS_SYSTEM_PROMPT.replace("{insights}", insights_section)
        analysis_prompt = analysis_prompt.replace("{lessons}", lessons_section)
        analysis_prompt += f"\n\n**User Request:**\n{prompt}"
        
        messages = [QPChatMessage(role="system", content=analysis_prompt)]
        request = QPChatRequest(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=512
        )
        
        response = await get_q_pulse_client().get_chat_completion(request)
        
        try:
            analysis_data = json.loads(response.choices[0].message.content)
            return PlanAnalysis(**analysis_data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse analysis response: {e}")
            raise ValueError(f"Invalid JSON response from analysis phase: {e}")
    
    async def _generate_workflow(self, original_prompt: str, analysis: PlanAnalysis) -> Workflow:
        """
        Phase 2: Generate a detailed workflow based on the analysis.
        """
        high_level_steps = "\n".join(f"- {step}" for step in analysis.high_level_steps)
        
        # Create the planner prompt with context information
        context_section = ""
        if self.retrieved_insights or self.retrieved_lessons:
            context_section = "\n\n**Context from Past Experience:**\n"
            if self.retrieved_insights:
                context_section += f"**Relevant Insights:**\n{self.retrieved_insights}\n\n"
            if self.retrieved_lessons:
                context_section += f"**Past Lessons:**\n{self.retrieved_lessons}\n\n"
        
        planner_prompt = (
            f"{PLANNER_SYSTEM_PROMPT}{context_section}\n\n"
            f"**Goal Summary:**\n{analysis.summary}\n\n"
            f"**High-Level Steps:**\n{high_level_steps}\n\n"
            f"**Original User Request:**\n{original_prompt}"
        )
        
        messages = [QPChatMessage(role="system", content=planner_prompt)]
        request = QPChatRequest(
            model=self.model,
            messages=messages,
            temperature=0.0
        )
        
        response = await get_q_pulse_client().get_chat_completion(request)
        
        try:
            workflow_data = json.loads(response.choices[0].message.content)
            return self._parse_workflow_data(workflow_data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse workflow response: {e}")
            raise ValueError(f"Invalid JSON response from workflow generation phase: {e}")
    
    def _parse_workflow_data(self, workflow_data: Dict[str, Any]) -> Workflow:
        """
        Parse workflow JSON data into proper Workflow model with TaskBlocks.
        """
        # Convert task dictionaries to proper TaskBlock objects
        tasks = []
        for task_data in workflow_data.get('tasks', []):
            task_block = self._create_task_block(task_data)
            tasks.append(task_block)
        
        return Workflow(
            original_prompt=workflow_data.get('original_prompt', ''),
            shared_context=workflow_data.get('shared_context', {}),
            tasks=tasks
        )
    
    def _create_task_block(self, task_data: Dict[str, Any]) -> TaskBlock:
        """
        Create the appropriate TaskBlock type from dictionary data.
        """
        task_type = task_data.get('type')
        
        if task_type == 'task':
            return WorkflowTask(**task_data)
        elif task_type == 'conditional':
            # Parse conditional branches
            branches = []
            for branch_data in task_data.get('branches', []):
                branch_tasks = [self._create_task_block(t) for t in branch_data.get('tasks', [])]
                branch = ConditionalBranch(
                    condition=branch_data.get('condition', ''),
                    tasks=branch_tasks
                )
                branches.append(branch)
            
            return ConditionalBlock(
                task_id=task_data.get('task_id', ''),
                dependencies=task_data.get('dependencies', []),
                branches=branches
            )
        else:
            # Default to WorkflowTask if type is not recognized
            logger.warning(f"Unknown task type '{task_type}', defaulting to WorkflowTask")
            return WorkflowTask(**{k: v for k, v in task_data.items() if k != 'type'})


# Global planner instance
planner = Planner()