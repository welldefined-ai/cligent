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

    def map_options_to_config(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Map generic options to SDK-specific configuration.
        
        Each executor should implement this to filter and transform options
        for their specific SDK requirements.
        
        Args:
            options: Generic options dictionary from TaskConfig
            
        Returns:
            SDK-specific configuration dictionary
        """
        # Default implementation returns all options
        return options.copy()

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
    
    def map_options_to_config(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Map generic options to Claude Code SDK configuration.
        
        Args:
            options: Generic options dictionary
            
        Returns:
            Dictionary of Claude Code SDK compatible options
        """
        # Define Claude Code SDK supported options (from actual ClaudeCodeOptions)
        supported_options = {
            'system_prompt', 'append_system_prompt', 'max_turns', 'model', 
            'max_thinking_tokens', 'allowed_tools', 'disallowed_tools',
            'continue_conversation', 'resume', 'cwd', 'add_dirs', 'settings',
            'permission_mode', 'permission_prompt_tool_name', 'mcp_servers', 'extra_args'
        }
        
        # Filter and return only supported options
        claude_options = {}
        for key, value in options.items():
            if key in supported_options:
                claude_options[key] = value
        
        return claude_options

    def _get_client_class(self):
        """Get Claude Code SDK client class."""
        try:
            from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions
            return ClaudeSDKClient, ClaudeCodeOptions
        except ImportError:
            raise ImportError("claude-code-sdk package not installed. Run: pip install claude-code-sdk")

    async def execute_task(self, task: str, config: TaskConfig) -> TaskResult:
        """Execute task using Claude Code SDK."""
        task_id = self._generate_task_id()
        
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            created_at=datetime.now()
        )
        
        try:
            ClaudeSDKClient, ClaudeCodeOptions = self._get_client_class()
            
            # Get SDK-specific options from task config
            sdk_options = self.map_options_to_config(config.options)
            
            # Build ClaudeCodeOptions with only valid parameters
            options = ClaudeCodeOptions(**sdk_options)
            
            # Initialize client with api_key at client level
            client = ClaudeSDKClient(api_key=self.api_key, options=options)
            response = await client.query(task)
                
            result.status = TaskStatus.COMPLETED
            result.output = response if isinstance(response, str) else str(response)
            result.completed_at = datetime.now()
            
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
            ClaudeSDKClient, ClaudeCodeOptions = self._get_client_class()
            
            # Get SDK-specific options from task config
            sdk_options = self.map_options_to_config(config.options)
            
            # Build ClaudeCodeOptions with only valid parameters
            options = ClaudeCodeOptions(**sdk_options)
            
            # Initialize client with api_key at client level
            client = ClaudeSDKClient(api_key=self.api_key, options=options)
            async for chunk in client.query_stream(task):
                if chunk:
                    yield TaskUpdate(task_id, UpdateType.OUTPUT, {
                        "content": chunk,
                        "partial": True
                    })
            
            yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.COMPLETED.value})
            
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
            
            # Import types for GenerateContentConfig
            from google.genai import types
            
            # Build GenerateContentConfig from options
            config_options = {}
            for key, value in config.options.items():
                if key not in {'model', 'contents', 'timeout', 'stream', 'save_logs', 'workspace'}:
                    config_options[key] = value
            
            # Create GenerateContentConfig if we have options
            generate_config = types.GenerateContentConfig(**config_options) if config_options else None
            
            # Use native async interface with correct API
            if generate_config:
                response = await client.aio.models.generate_content(
                    model=config.get('model', 'gemini-2.0-flash-001'),
                    contents=task,
                    config=generate_config
                )
            else:
                response = await client.aio.models.generate_content(
                    model=config.get('model', 'gemini-2.0-flash-001'),
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
            
            # Import types for GenerateContentConfig
            from google.genai import types
            
            # Build GenerateContentConfig from options
            config_options = {}
            for key, value in config.options.items():
                if key not in {'model', 'contents', 'timeout', 'stream', 'save_logs', 'workspace'}:
                    config_options[key] = value
            
            # Create GenerateContentConfig if we have options
            generate_config = types.GenerateContentConfig(**config_options) if config_options else None
            
            # Use native async streaming interface with correct API
            if generate_config:
                stream = client.aio.models.generate_content_stream(
                    model=config.get('model', 'gemini-2.0-flash-001'),
                    contents=task,
                    config=generate_config
                )
            else:
                stream = client.aio.models.generate_content_stream(
                    model=config.get('model', 'gemini-2.0-flash-001'),
                    contents=task
                )
            
            async for chunk in stream:
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
            
            # Build Generation.call arguments from config
            generation_args = {
                'model': config.get('model', 'qwen-turbo'),
                'prompt': task,
                'max_tokens': config.get('max_tokens', 4000)
            }
            
            # Add all other options from config.options
            for key, value in config.options.items():
                if key not in generation_args and key not in {'timeout', 'stream', 'save_logs', 'workspace'}:
                    generation_args[key] = value
            
            # Try async call if available, fallback to sync
            try:
                if hasattr(Generation, 'acall'):
                    response = await Generation.acall(**generation_args)
                else:
                    # Fallback to sync wrapped in thread
                    response = await asyncio.to_thread(Generation.call, **generation_args)
            except AttributeError:
                # Fallback to sync method
                response = await asyncio.to_thread(Generation.call, **generation_args)
            
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
            
            # Build Generation.call arguments from config for streaming
            generation_args = {
                'model': config.get('model', 'qwen-turbo'),
                'prompt': task,
                'max_tokens': config.get('max_tokens', 4000),
                'stream': True
            }
            
            # Add all other options from config.options
            for key, value in config.options.items():
                if key not in generation_args and key not in {'timeout', 'save_logs', 'workspace'}:
                    generation_args[key] = value
            
            # Try async streaming if available, fallback to sync
            try:
                if hasattr(Generation, 'acall'):
                    responses = await Generation.acall(**generation_args)
                    async for response in responses:
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
                else:
                    # Fallback to sync wrapped in thread
                    def stream_sync():
                        return Generation.call(**generation_args)
                    
                    responses = await asyncio.to_thread(stream_sync)
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
            except AttributeError:
                # Fallback to sync method
                def stream_sync():
                    return Generation.call(**generation_args)
                
                responses = await asyncio.to_thread(stream_sync)
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