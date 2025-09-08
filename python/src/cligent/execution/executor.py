"""SDK-based executor implementations for different agent types."""

import asyncio
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, AsyncIterator, List
from abc import ABC, abstractmethod

from .task_models import TaskResult, TaskUpdate, TaskConfig, TaskStatus, UpdateType


class BaseExecutor(ABC):
    """Base class for agent task executors."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    @abstractmethod
    async def execute_task(self, task: str, config: TaskConfig) -> TaskResult:
        """Execute a task and return result."""
        pass

    @abstractmethod  
    async def execute_task_stream(self, task: str, config: TaskConfig) -> AsyncIterator[TaskUpdate]:
        """Execute task with streaming updates."""
        pass

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        return f"{self.agent_name}-{uuid.uuid4().hex[:8]}"


class ClaudeExecutor(BaseExecutor):
    """Executor for Claude Code using claude-code-sdk."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude Code executor.
        
        Args:
            api_key: Anthropic API key (will use ANTHROPIC_API_KEY env var if not provided)
        """
        super().__init__("claude-code")
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        """Get or create Claude Code client."""
        if self._client is None:
            try:
                from claude_code_sdk import ClaudeCodeClient
                self._client = ClaudeCodeClient(api_key=self.api_key)
            except ImportError:
                raise ImportError("claude-code-sdk package not installed. Run: pip install claude-code-sdk")
        return self._client

    async def execute_task(self, task: str, config: TaskConfig) -> TaskResult:
        """Execute task using Claude Code SDK."""
        task_id = self._generate_task_id()
        
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            created_at=datetime.now()
        )
        
        try:
            client = self._get_client()
            
            # Execute task using Claude Code
            response = await asyncio.to_thread(
                client.execute,
                task=task,
                model=config.model or "claude-3-5-sonnet-20241022",
                workspace=config.workspace
            )
            
            result.status = TaskStatus.COMPLETED
            result.output = response.output
            result.completed_at = datetime.now()
            
            # Add Claude Code specific metadata
            if hasattr(response, 'model'):
                result.logs.append(f"Model: {response.model}")
            if hasattr(response, 'usage'):
                result.logs.append(f"Usage: {response.usage}")
            if hasattr(response, 'session_id'):
                result.logs.append(f"Session: {response.session_id}")
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = f"Claude Code error: {str(e)}"
            result.completed_at = datetime.now()
        
        return result

    async def execute_task_stream(self, task: str, config: TaskConfig) -> AsyncIterator[TaskUpdate]:
        """Execute task with streaming Claude Code SDK."""
        task_id = self._generate_task_id()
        
        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.RUNNING.value})
        
        try:
            client = self._get_client()
            
            # Create streaming execution
            stream = await asyncio.to_thread(
                client.execute_stream,
                task=task,
                model=config.model or "claude-3-5-sonnet-20241022",
                workspace=config.workspace
            )
            
            async for chunk in stream:
                if hasattr(chunk, 'content') and chunk.content:
                    yield TaskUpdate(task_id, UpdateType.OUTPUT, {
                        "content": chunk.content,
                        "partial": True
                    })
                elif hasattr(chunk, 'status'):
                    if chunk.status == 'completed':
                        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.COMPLETED.value})
                    elif chunk.status == 'error':
                        yield TaskUpdate(task_id, UpdateType.ERROR, {"error": chunk.error})
            
        except Exception as e:
            yield TaskUpdate(task_id, UpdateType.ERROR, {"error": str(e)})


class GeminiExecutor(BaseExecutor):
    """Executor for Gemini using Google AI SDK."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini executor.
        
        Args:
            api_key: Google API key (will use GOOGLE_API_KEY env var if not provided)
        """
        super().__init__("gemini")
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        """Get or create Gemini client."""
        if self._client is None:
            try:
                from google import genai
                import os
                # Set API key from parameter or environment
                api_key = self.api_key or os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
                self._client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError("google-genai package not installed. Run: pip install google-genai")
        return self._client

    async def execute_task(self, task: str, config: TaskConfig) -> TaskResult:
        """Execute task using Gemini API."""
        task_id = self._generate_task_id()
        
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            created_at=datetime.now()
        )
        
        try:
            client = self._get_client()
            
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=config.model or 'gemini-2.0-flash-001',
                contents=task
            )
            
            result.status = TaskStatus.COMPLETED
            result.output = response.text
            result.completed_at = datetime.now()
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = f"Gemini API error: {str(e)}"
            result.completed_at = datetime.now()
        
        return result

    async def execute_task_stream(self, task: str, config: TaskConfig) -> AsyncIterator[TaskUpdate]:
        """Execute task with streaming Gemini API."""
        task_id = self._generate_task_id()
        
        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.RUNNING.value})
        
        try:
            client = self._get_client()
            
            def stream_content():
                return client.models.generate_content_stream(
                    model=config.model or 'gemini-2.0-flash-001',
                    contents=task
                )
            
            response = await asyncio.to_thread(stream_content)
            
            for chunk in response:
                if chunk.text:
                    yield TaskUpdate(task_id, UpdateType.OUTPUT, {
                        "content": chunk.text,
                        "partial": True
                    })
            
            yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.COMPLETED.value})
            
        except Exception as e:
            yield TaskUpdate(task_id, UpdateType.ERROR, {"error": str(e)})


class QwenExecutor(BaseExecutor):
    """Executor for Qwen using Alibaba Cloud SDK."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Qwen executor.
        
        Args:
            api_key: Qwen API key (will use DASHSCOPE_API_KEY env var if not provided)
        """
        super().__init__("qwen")
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        """Get or create Qwen client."""
        if self._client is None:
            try:
                import dashscope
                if self.api_key:
                    dashscope.api_key = self.api_key
                self._client = dashscope
            except ImportError:
                raise ImportError("dashscope package not installed. Run: pip install dashscope")
        return self._client

    async def execute_task(self, task: str, config: TaskConfig) -> TaskResult:
        """Execute task using Qwen API."""
        task_id = self._generate_task_id()
        
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            created_at=datetime.now()
        )
        
        try:
            dashscope = self._get_client()
            from dashscope import Generation
            
            response = await asyncio.to_thread(
                Generation.call,
                model=config.model or 'qwen-turbo',
                prompt=task,
                max_tokens=config.max_tokens or 4000
            )
            
            if response.status_code == 200:
                result.status = TaskStatus.COMPLETED
                result.output = response.output.text
                result.completed_at = datetime.now()
            else:
                result.status = TaskStatus.FAILED
                result.error = f"Qwen API error: {response.message}"
                result.completed_at = datetime.now()
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = f"Qwen API error: {str(e)}"
            result.completed_at = datetime.now()
        
        return result

    async def execute_task_stream(self, task: str, config: TaskConfig) -> AsyncIterator[TaskUpdate]:
        """Execute task with streaming Qwen API."""
        task_id = self._generate_task_id()
        
        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.RUNNING.value})
        
        try:
            dashscope = self._get_client()
            from dashscope import Generation
            
            responses = await asyncio.to_thread(
                Generation.call,
                model=config.model or 'qwen-turbo',
                prompt=task,
                max_tokens=config.max_tokens or 4000,
                stream=True
            )
            
            for response in responses:
                if response.status_code == 200:
                    if hasattr(response.output, 'text'):
                        yield TaskUpdate(task_id, UpdateType.OUTPUT, {
                            "content": response.output.text,
                            "partial": True
                        })
                else:
                    yield TaskUpdate(task_id, UpdateType.ERROR, {
                        "error": f"Qwen API error: {response.message}"
                    })
                    return
            
            yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.COMPLETED.value})
            
        except Exception as e:
            yield TaskUpdate(task_id, UpdateType.ERROR, {"error": str(e)})


class MockExecutor(BaseExecutor):
    """Mock executor for testing and demonstration."""
    
    async def execute_task(self, task: str, config: TaskConfig) -> TaskResult:
        """Mock task execution."""
        task_id = self._generate_task_id()
        
        # Simulate work
        await asyncio.sleep(1)
        
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            output=f"Mock execution of: {task}",
            created_at=datetime.now(),
            completed_at=datetime.now(),
            logs=[f"Executed task: {task}"]
        )

    async def execute_task_stream(self, task: str, config: TaskConfig) -> AsyncIterator[TaskUpdate]:
        """Mock streaming execution."""
        task_id = self._generate_task_id()
        
        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.RUNNING.value})
        
        # Simulate progress updates
        for i in range(3):
            await asyncio.sleep(0.5)
            yield TaskUpdate(task_id, UpdateType.PROGRESS, {"step": i + 1, "total": 3})
            yield TaskUpdate(task_id, UpdateType.OUTPUT, {"content": f"Step {i + 1} completed"})
        
        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.COMPLETED.value})