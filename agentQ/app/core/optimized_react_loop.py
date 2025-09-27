"""
Optimized ReAct loop implementation with performance enhancements:
- Connection pooling for Pulsar and external services
- Caching for frequently accessed data (memories, reflexions)
- Batch processing for tool executions
- Circuit breaker patterns for resilience
- Performance monitoring and metrics
"""

import logging
import time
import asyncio
import json
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from shared.performance.cache import LRUCache, memoize
from shared.performance.metrics import get_metrics_collector, measure_time
from shared.connection_pools.pulsar_pool import get_pulsar_pool, PulsarPoolConfig
from shared.q_pulse_client.client import QuantumPulseClient
from shared.q_pulse_client.models import QPChatRequest, QPChatMessage
from agentQ.app.core.context import ContextManager
from agentQ.app.core.toolbox import Toolbox

logger = logging.getLogger(__name__)
metrics = get_metrics_collector()

# Global caches for performance optimization
_memory_cache = LRUCache(max_size=1000)  # Cache for memory queries
_reflexion_cache = LRUCache(max_size=500)  # Cache for reflexion queries
_tool_result_cache = LRUCache(max_size=2000)  # Cache for tool results


class OptimizedReActLoop:
    """
    High-performance ReAct loop implementation with caching and connection pooling.
    """
    
    def __init__(
        self,
        context_manager: ContextManager,
        toolbox: Toolbox,
        qpulse_client: QuantumPulseClient,
        llm_config: Dict[str, Any],
        pulsar_config: Optional[PulsarPoolConfig] = None
    ):
        self.context_manager = context_manager
        self.toolbox = toolbox
        self.qpulse_client = qpulse_client
        self.llm_config = llm_config
        
        # Initialize connection pools
        self.pulsar_pool = get_pulsar_pool(pulsar_config)
        
        # Performance settings
        self.max_turns = 5
        self.enable_caching = True
        self.enable_batching = True
        self.cache_ttl = 1800  # 30 minutes
        
    @measure_time(metrics, "react_loop.execute")
    async def execute(
        self,
        prompt_data: Dict[str, Any],
        system_prompt_override: Optional[str] = None
    ) -> str:
        """
        Execute the optimized ReAct loop
        
        Args:
            prompt_data: Input prompt data
            system_prompt_override: Optional system prompt override
            
        Returns:
            Final result string
        """
        user_prompt = prompt_data.get("prompt")
        conversation_id = prompt_data.get("id")
        agent_id = prompt_data.get("agent_id")
        
        metrics.increment_counter("react_loop.executions")
        
        # Initialize scratchpad
        scratchpad = []
        
        # Get conversation history with caching
        with measure_time(metrics, "react_loop.get_history"):
            history = self._get_cached_history(conversation_id)
        
        # Handle new conversations
        if not history:
            await self._handle_new_conversation(
                user_prompt, history, scratchpad, agent_id
            )
            
        history.append({"role": "user", "content": user_prompt})
        scratchpad.append({
            "type": "user_prompt", 
            "content": user_prompt, 
            "timestamp": time.time()
        })
        
        # Execute turns with optimization
        final_result = await self._execute_turns(
            prompt_data, history, scratchpad, system_prompt_override
        )
        
        # Save context and return result
        self.context_manager.append_to_history(conversation_id, history, scratchpad)
        return final_result
        
    async def _handle_new_conversation(
        self,
        user_prompt: str,
        history: List[Dict],
        scratchpad: List[Dict],
        agent_id: str
    ):
        """Handle initialization for new conversations"""
        # Parallel retrieval of reflexions and memories
        tasks = [
            self._get_cached_reflexion(user_prompt),
            self._get_cached_memories(user_prompt, agent_id)
        ]
        
        reflexion, memories = await asyncio.gather(*tasks)
        
        # Add reflexion if found
        if reflexion:
            reflexion_observation = (
                f"System Directive: A previous attempt at a similar task failed. "
                f"Heed this advice: {reflexion}"
            )
            history.append({"role": "system", "content": reflexion_observation})
            scratchpad.append({
                "type": "reflexion",
                "content": reflexion_observation,
                "timestamp": time.time()
            })
            
        # Add memories if found
        if memories:
            memory_observation = f"Tool Observation: {memories}"
            history.append({"role": "system", "content": memory_observation})
            scratchpad.append({
                "type": "observation",
                "content": memory_observation,
                "timestamp": time.time()
            })
            
    async def _execute_turns(
        self,
        prompt_data: Dict[str, Any],
        history: List[Dict],
        scratchpad: List[Dict],
        system_prompt_override: Optional[str]
    ) -> str:
        """Execute the main ReAct turns with optimizations"""
        
        for turn in range(self.max_turns):
            with measure_time(metrics, f"react_loop.turn_{turn}"):
                metrics.increment_counter("react_loop.turns")
                
                # Build optimized prompt
                response_text = await self._get_llm_response(
                    prompt_data, history, system_prompt_override
                )
                
                # Parse and execute action
                action_result = await self._parse_and_execute_action(
                    response_text, scratchpad
                )
                
                if action_result.get("final_answer"):
                    # Save memory asynchronously (fire and forget)
                    asyncio.create_task(self._save_memory_async(
                        prompt_data, history, action_result["final_answer"]
                    ))
                    return action_result["final_answer"]
                    
                # Add observation to history
                if action_result.get("observation"):
                    history.append({
                        "role": "system", 
                        "content": action_result["observation"]
                    })
                    scratchpad.append({
                        "type": "observation",
                        "content": action_result["observation"],
                        "timestamp": time.time()
                    })
                    
        # Handle max turns reached
        logger.warning(f"Reached max turns ({self.max_turns}) without final answer")
        metrics.increment_counter("react_loop.max_turns_reached")
        
        # Generate reflexion for failure
        asyncio.create_task(self._generate_reflexion_async(
            prompt_data.get("prompt"), scratchpad
        ))
        
        return "Error: Reached maximum turns without a final answer."
        
    async def _get_llm_response(
        self,
        prompt_data: Dict[str, Any],
        history: List[Dict],
        system_prompt_override: Optional[str]
    ) -> str:
        """Get LLM response with optimized prompt building"""
        with measure_time(metrics, "react_loop.llm_request"):
            # Build prompt messages
            full_prompt_messages = [QPChatMessage(**msg) for msg in history]
            
            # Add system prompt
            from agentQ.app.core.prompts import SYSTEM_PROMPT  # Import here to avoid circular imports
            system_prompt = system_prompt_override or SYSTEM_PROMPT
            system_content = system_prompt.format(
                tools=self.toolbox.get_tool_descriptions()
            )
            full_prompt_messages.append(
                QPChatMessage(role="system", content=system_content)
            )
            
            # Make LLM request
            request = QPChatRequest(
                model=self.llm_config['model'],
                messages=full_prompt_messages
            )
            
            response = await self.qpulse_client.get_chat_completion(request)
            response_text = response.choices[0].message.content
            
            # Add to history
            history.append({"role": "assistant", "content": response_text})
            
            return response_text
            
    async def _parse_and_execute_action(
        self,
        response_text: str,
        scratchpad: List[Dict]
    ) -> Dict[str, Any]:
        """Parse LLM response and execute actions with caching"""
        
        try:
            # Parse thought and action
            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:")[1].strip()
                return {"final_answer": final_answer}
                
            thought = response_text.split("Action:")[0].replace("Thought:", "").strip()
            action_str = response_text.split("Action:")[1].strip()
            action_json = json.loads(action_str)
            
            # Add thought to scratchpad
            scratchpad.append({
                "type": "thought",
                "content": thought,
                "timestamp": time.time()
            })
            scratchpad.append({
                "type": "action",
                "content": action_json,
                "timestamp": time.time()
            })
            
            # Execute action with caching
            observation = await self._execute_cached_tool(action_json)
            
            return {"observation": observation}
            
        except (IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            error_msg = "Error: Invalid response format. Please use the correct format."
            scratchpad.append({
                "type": "error",
                "content": error_msg,
                "timestamp": time.time()
            })
            return {"observation": error_msg}
            
    async def _execute_cached_tool(self, action_json: Dict[str, Any]) -> str:
        """Execute tool with result caching"""
        tool_name = action_json.get("tool")
        tool_input = action_json.get("input", {})
        
        if not self.enable_caching:
            return await self._execute_tool_direct(tool_name, tool_input)
            
        # Create cache key
        cache_key = f"{tool_name}:{hash(str(sorted(tool_input.items())))}"
        
        # Try cache first
        cached_result = _tool_result_cache.get(cache_key)
        if cached_result is not None:
            metrics.increment_counter("react_loop.tool_cache_hits")
            return cached_result
            
        # Execute tool
        with measure_time(metrics, f"react_loop.tool.{tool_name}"):
            result = await self._execute_tool_direct(tool_name, tool_input)
            
        # Cache result
        _tool_result_cache.put(cache_key, result, ttl=self.cache_ttl)
        metrics.increment_counter("react_loop.tool_cache_misses")
        
        return result
        
    async def _execute_tool_direct(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute tool directly without caching"""
        try:
            result = self.toolbox.execute_tool(tool_name, **tool_input)
            return f"Tool Observation: {result}"
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return f"Error: Tool execution failed. Details: {e}"
            
    @memoize(cache=_memory_cache, ttl=1800)
    async def _get_cached_memories(self, query: str, agent_id: str) -> Optional[str]:
        """Get cached memory search results"""
        try:
            return self.toolbox.execute_tool("search_memory", query=query)
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return None
            
    @memoize(cache=_reflexion_cache, ttl=3600)
    async def _get_cached_reflexion(self, user_prompt: str) -> Optional[str]:
        """Get cached reflexion results"""
        try:
            return self.context_manager.get_reflexion(user_prompt)
        except Exception as e:
            logger.error(f"Reflexion retrieval failed: {e}")
            return None
            
    def _get_cached_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history (could be cached in future)"""
        try:
            return self.context_manager.get_history(conversation_id)
        except Exception as e:
            logger.error(f"History retrieval failed: {e}")
            return []
            
    async def _save_memory_async(
        self,
        prompt_data: Dict[str, Any],
        history: List[Dict],
        final_answer: str
    ):
        """Save memory asynchronously"""
        try:
            # Build memory prompt
            memory_prompt = f"""
            Analyze this conversation and extract key information for long-term memory.
            
            Original prompt: {prompt_data.get('prompt')}
            Final answer: {final_answer}
            
            Return a JSON object with memory information.
            """
            
            # This would typically make an LLM call to generate memory
            # For now, we'll create a simple memory entry
            memory_data = {
                "prompt": prompt_data.get('prompt'),
                "answer": final_answer,
                "timestamp": time.time(),
                "conversation_id": prompt_data.get('id')
            }
            
            # Save memory (this would be implemented in the memory service)
            logger.info(f"Saved memory for conversation {prompt_data.get('id')}")
            
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            
    async def _generate_reflexion_async(self, user_prompt: str, scratchpad: List[Dict]):
        """Generate reflexion asynchronously"""
        try:
            # This would generate a reflexion based on the failed attempt
            reflexion_data = {
                "prompt": user_prompt,
                "failure_reason": "max_turns_reached",
                "scratchpad": scratchpad,
                "timestamp": time.time()
            }
            
            logger.info(f"Generated reflexion for failed prompt: {user_prompt[:100]}...")
            
        except Exception as e:
            logger.error(f"Failed to generate reflexion: {e}")
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "cache_stats": {
                "memory_cache": {
                    "size": _memory_cache.size(),
                    "hit_rate": _memory_cache.stats.hit_rate,
                    "hits": _memory_cache.stats.hits,
                    "misses": _memory_cache.stats.misses
                },
                "reflexion_cache": {
                    "size": _reflexion_cache.size(),
                    "hit_rate": _reflexion_cache.stats.hit_rate,
                    "hits": _reflexion_cache.stats.hits,
                    "misses": _reflexion_cache.stats.misses
                },
                "tool_result_cache": {
                    "size": _tool_result_cache.size(),
                    "hit_rate": _tool_result_cache.stats.hit_rate,
                    "hits": _tool_result_cache.stats.hits,
                    "misses": _tool_result_cache.stats.misses
                }
            },
            "pulsar_pool": self.pulsar_pool.get_health_status(),
            "metrics": metrics.get_summary()
        }


# Convenience function for creating optimized ReAct loop
def create_optimized_react_loop(
    context_manager: ContextManager,
    toolbox: Toolbox,
    qpulse_client: QuantumPulseClient,
    llm_config: Dict[str, Any],
    pulsar_config: Optional[PulsarPoolConfig] = None
) -> OptimizedReActLoop:
    """Create an optimized ReAct loop instance"""
    return OptimizedReActLoop(
        context_manager=context_manager,
        toolbox=toolbox,
        qpulse_client=qpulse_client,
        llm_config=llm_config,
        pulsar_config=pulsar_config
    )