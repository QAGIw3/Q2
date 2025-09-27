"""
Shared test utilities for Q2 Platform tests.
"""

import json
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Union
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone
import tempfile
import os


class TestDataFactory:
    """
    Factory class for creating test data objects.
    """
    
    @staticmethod
    def create_workflow_task(
        task_id: Optional[str] = None,
        task_type: str = "task",
        agent_personality: str = "default",
        prompt: str = "Test task prompt",
        dependencies: Optional[List[str]] = None,
        status: str = "PENDING",
        **kwargs
    ) -> Dict[str, Any]:
        """Create a test WorkflowTask."""
        if task_id is None:
            task_id = f"test_task_{uuid.uuid4().hex[:8]}"
        
        if dependencies is None:
            dependencies = []
            
        task = {
            "task_id": task_id,
            "type": task_type,
            "agent_personality": agent_personality,
            "prompt": prompt,
            "dependencies": dependencies,
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        
        return task
    
    @staticmethod
    def create_workflow(
        workflow_id: Optional[str] = None,
        original_prompt: str = "Test workflow prompt",
        status: str = "RUNNING",
        tasks: Optional[List[Dict[str, Any]]] = None,
        shared_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a test Workflow."""
        if workflow_id is None:
            workflow_id = f"test_workflow_{uuid.uuid4().hex[:8]}"
        
        if tasks is None:
            tasks = [TestDataFactory.create_workflow_task()]
        
        if shared_context is None:
            shared_context = {}
            
        workflow = {
            "workflow_id": workflow_id,
            "original_prompt": original_prompt,
            "status": status,
            "tasks": tasks,
            "shared_context": shared_context,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        
        return workflow
    
    @staticmethod
    def create_agent_message(
        message_id: Optional[str] = None,
        content: str = "Test message",
        role: str = "user",
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a test agent message."""
        if message_id is None:
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
            
        if conversation_id is None:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            
        message = {
            "id": message_id,
            "content": content,
            "role": role,
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        
        return message
    
    @staticmethod
    def create_vector_data(
        vector_id: Optional[str] = None,
        values: Optional[List[float]] = None,
        dimension: int = 384,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create test vector data."""
        if vector_id is None:
            vector_id = f"vec_{uuid.uuid4().hex[:8]}"
            
        if values is None:
            # Create a simple test vector
            values = [0.1 * i for i in range(dimension)]
            
        if metadata is None:
            metadata = {"test": "data"}
            
        vector = {
            "id": vector_id,
            "values": values,
            "metadata": metadata,
            **kwargs
        }
        
        return vector


class MockServices:
    """
    Collection of mock services for testing.
    """
    
    @staticmethod
    def create_mock_pulsar_producer():
        """Create a mock Pulsar producer."""
        producer = MagicMock()
        producer.send.return_value = MagicMock()  # Message ID
        producer.send_async.return_value = asyncio.Future()
        producer.send_async.return_value.set_result(MagicMock())
        return producer
    
    @staticmethod
    def create_mock_pulsar_consumer():
        """Create a mock Pulsar consumer."""
        consumer = MagicMock()
        consumer.receive.return_value = MagicMock()
        consumer.acknowledge.return_value = None
        consumer.negative_acknowledge.return_value = None
        return consumer
    
    @staticmethod
    def create_mock_ignite_cache():
        """Create a mock Ignite cache."""
        cache = MagicMock()
        cache.put.return_value = None
        cache.get.return_value = None
        cache.remove.return_value = True
        cache.size.return_value = 0
        return cache
    
    @staticmethod
    def create_mock_qpulse_response(content: str = "Test response"):
        """Create a mock QPulse response."""
        return {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": len(content.split()),
                "total_tokens": 10 + len(content.split())
            },
            "model": "test-model",
            "id": f"test_response_{uuid.uuid4().hex[:8]}"
        }


class AsyncTestHelper:
    """
    Helper utilities for async testing.
    """
    
    @staticmethod
    def run_async(coro):
        """Run an async coroutine in a test."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    @staticmethod
    def create_async_mock():
        """Create an AsyncMock."""
        return AsyncMock()


class FileTestHelper:
    """
    Helper utilities for file-based testing.
    """
    
    @staticmethod
    def create_temp_file(content: str, suffix: str = ".txt") -> str:
        """Create a temporary file with content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name
    
    @staticmethod
    def create_temp_json_file(data: Dict[str, Any]) -> str:
        """Create a temporary JSON file."""
        content = json.dumps(data, indent=2)
        return FileTestHelper.create_temp_file(content, suffix=".json")
    
    @staticmethod
    def cleanup_temp_file(filepath: str):
        """Clean up a temporary file."""
        try:
            os.unlink(filepath)
        except (OSError, FileNotFoundError):
            pass


class AssertionHelper:
    """
    Custom assertion helpers for Q2 Platform testing.
    """
    
    @staticmethod
    def assert_workflow_structure(workflow: Dict[str, Any]):
        """Assert that a workflow has the correct structure."""
        required_fields = ["workflow_id", "original_prompt", "status", "tasks"]
        for field in required_fields:
            assert field in workflow, f"Workflow missing required field: {field}"
        
        assert isinstance(workflow["tasks"], list), "Workflow tasks must be a list"
        for task in workflow["tasks"]:
            AssertionHelper.assert_task_structure(task)
    
    @staticmethod
    def assert_task_structure(task: Dict[str, Any]):
        """Assert that a task has the correct structure."""
        required_fields = ["task_id", "type", "prompt"]
        for field in required_fields:
            assert field in task, f"Task missing required field: {field}"
        
        if "dependencies" in task:
            assert isinstance(task["dependencies"], list), "Task dependencies must be a list"
    
    @staticmethod
    def assert_message_structure(message: Dict[str, Any]):
        """Assert that a message has the correct structure."""
        required_fields = ["id", "content", "role"]
        for field in required_fields:
            assert field in message, f"Message missing required field: {field}"
    
    @staticmethod
    def assert_config_structure(config: Dict[str, Any]):
        """Assert that a configuration has the expected structure."""
        required_sections = ["database", "pulsar", "observability"]
        for section in required_sections:
            assert section in config, f"Config missing required section: {section}"


class PerformanceTestHelper:
    """
    Helper utilities for performance testing.
    """
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """Measure execution time of a function."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    async def measure_async_execution_time(coro):
        """Measure execution time of an async coroutine."""
        import time
        start_time = time.time()
        result = await coro
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time


# Convenience functions for common test operations
def create_test_workflow(**kwargs) -> Dict[str, Any]:
    """Convenience function to create a test workflow."""
    return TestDataFactory.create_workflow(**kwargs)


def create_test_task(**kwargs) -> Dict[str, Any]:
    """Convenience function to create a test task."""
    return TestDataFactory.create_workflow_task(**kwargs)


def create_test_message(**kwargs) -> Dict[str, Any]:
    """Convenience function to create a test message."""
    return TestDataFactory.create_agent_message(**kwargs)