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
    async def execute_task(self, task: str, config: TaskConfig) -> AsyncIterator[TaskUpdate]:
        """Execute a task with streaming updates."""
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

    async def execute_task(self, task: str, config: TaskConfig) -> AsyncIterator[TaskUpdate]:
        """Execute task with streaming Claude Code SDK."""
        task_id = self._generate_task_id()
        
        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.RUNNING.value})
        
        try:
            ClaudeSDKClient, ClaudeCodeOptions = self._get_client_class()
            
            # Get SDK-specific options from task config
            sdk_options = self.map_options_to_config(config.options)
            
            # Build ClaudeCodeOptions with only valid parameters
            options = ClaudeCodeOptions(**sdk_options)
            
            # Use async context manager and correct streaming API
            async with ClaudeSDKClient(api_key=self.api_key, options=options) as client:
                # Send query
                await client.query(task)
                
                # Stream responses using receive_response()
                async for message in client.receive_response():
                    if hasattr(message, 'content'):
                        for block in message.content:
                            if hasattr(block, 'text') and block.text:
                                yield TaskUpdate(task_id, UpdateType.OUTPUT, {
                                    "content": block.text,
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

    def map_options_to_config(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Map generic options to Gemini GenerateContentConfig parameters.
        
        Args:
            options: Generic options dictionary
            
        Returns:
            Dictionary of GenerateContentConfig compatible options
        """
        # Define GenerateContentConfig supported parameters from official schema
        supported_options = {
            # HTTP and response options
            'httpOptions',
            'shouldReturnHttpResponse',
            
            # Core generation parameters
            'systemInstruction',
            'temperature',
            'topP',
            'topK', 
            'candidateCount',
            'maxOutputTokens',
            'stopSequences',
            
            # Probability and penalty settings
            'responseLogprobs',
            'logprobs',
            'presencePenalty',
            'frequencyPenalty',
            'seed',
            
            # Response format
            'responseMimeType',
            'responseSchema',
            'responseJsonSchema',
            
            # Advanced configurations
            'routingConfig',
            'modelSelectionConfig',
            'safetySettings',
            'tools',
            'toolConfig',
            'labels',
            'cachedContent',
            
            # Multimodal features
            'responseModalities',
            'mediaResolution',
            'speechConfig',
            'audioTimestamp',
            
            # Function calling and thinking
            'automaticFunctionCalling',
            'thinkingConfig'
        }
        
        # Map common parameter names to official names
        name_mapping = {
            'top_p': 'topP',
            'top_k': 'topK',
            'candidate_count': 'candidateCount',
            'max_output_tokens': 'maxOutputTokens',
            'stop_sequences': 'stopSequences',
            'response_logprobs': 'responseLogprobs',
            'presence_penalty': 'presencePenalty',
            'frequency_penalty': 'frequencyPenalty',
            'response_mime_type': 'responseMimeType',
            'response_schema': 'responseSchema',
            'response_json_schema': 'responseJsonSchema',
            'system_instruction': 'systemInstruction',
            'safety_settings': 'safetySettings',
            'tool_config': 'toolConfig',
            'cached_content': 'cachedContent',
            'response_modalities': 'responseModalities',
            'media_resolution': 'mediaResolution',
            'speech_config': 'speechConfig',
            'audio_timestamp': 'audioTimestamp',
            'automatic_function_calling': 'automaticFunctionCalling',
            'thinking_config': 'thinkingConfig'
        }
        
        # Filter and map options to only include supported parameters
        config = {}
        for key, value in options.items():
            # Map common names to official names
            official_key = name_mapping.get(key, key)
            
            if official_key in supported_options:
                config[official_key] = value
                
        return config

    async def execute_task(self, task: str, config: TaskConfig) -> AsyncIterator[TaskUpdate]:
        """Execute task with streaming Gemini API."""
        task_id = self._generate_task_id()
        
        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.RUNNING.value})
        
        try:
            client = self._get_client()
            
            # Import types for GenerateContentConfig
            from google.genai import types
            
            # Get Gemini-specific options from task config
            config_options = self.map_options_to_config(config.options)
            
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
                import os
                # Set API key from parameter or environment
                api_key = self.api_key or os.getenv('DASHSCOPE_API_KEY')
                if api_key:
                    dashscope.api_key = api_key
                # Set base URL (can be configured via options)
                dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
                self._client = dashscope
            except ImportError:
                raise ImportError("dashscope package not installed. Run: pip install dashscope")
        return self._client

    def map_options_to_config(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Map generic options to Qwen DashScope Generation.call parameters.
        
        Args:
            options: Generic options dictionary
            
        Returns:
            Dictionary of Generation.call compatible options
        """
        # Define Generation.call supported parameters from official docs
        supported_options = {
            # Core parameters
            'model',
            'messages', 
            'api_key',
            'system',

            # Format and generation controls
            'result_format',
            'incremental_output',
            
            # Generation parameters
            'max_tokens',
            'max_input_tokens',
            'temperature',
            'top_p',
            'top_k',
            'repetition_penalty',
            'presence_penalty',
            'seed',
            
            # Advanced features
            'tools',
            'tool_choice',

            # Response options
            'response_format',
        }
        
        # Map common parameter names
        name_mapping = {
            'max_output_tokens': 'max_tokens',
        }
        
        # Filter and map options to only include supported parameters
        config = {}
        for key, value in options.items():
            # Map common names to official names
            official_key = name_mapping.get(key, key)
            
            if official_key in supported_options:
                config[official_key] = value
                
        return config

    async def execute_task(self, task: str, config: TaskConfig) -> AsyncIterator[TaskUpdate]:
        """Execute task with streaming Qwen API using official DashScope pattern."""
        task_id = self._generate_task_id()
        
        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.RUNNING.value})
        
        try:
            dashscope = self._get_client()
            from dashscope import Generation
            import os
            
            # Get Qwen-specific options from task config
            qwen_options = self.map_options_to_config(config.options)
            
            # Build messages in official format
            messages = []
            
            # Add system message if provided
            system_prompt = qwen_options.get('system')
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
                
            # Add user message
            messages.append({'role': 'user', 'content': task})
            
            # Build Generation.call arguments using official pattern
            generation_args = {
                'api_key': self.api_key or os.getenv('DASHSCOPE_API_KEY'),
                'model': qwen_options.get('model', 'qwen-plus'),
                'messages': messages,
                'result_format': qwen_options.get('result_format', 'text'),
                'stream': True  # Enable streaming
            }
            
            # Add other supported options
            for key, value in qwen_options.items():
                if key not in {'system', 'model', 'result_format'} and key in generation_args:
                    generation_args[key] = value
                elif key not in generation_args:  # Add any additional valid parameters
                    generation_args[key] = value
            
            # Use streaming Generation.call
            responses = Generation.call(**generation_args)
            
            # Handle streaming response based on official API structure
            for response in responses:
                # Check status_code (200 = success)
                if hasattr(response, 'status_code') and response.status_code == 200:
                    # Extract content from response.output according to official structure
                    content = ""
                    
                    if hasattr(response, 'output') and response.output:
                        output = response.output
                        
                        # Handle result_format="text" mode
                        if hasattr(output, 'text') and output.text is not None:
                            content = output.text
                        
                        # Handle result_format="message" mode
                        elif hasattr(output, 'choices') and output.choices:
                            choice = output.choices[0]  # Get first choice
                            if hasattr(choice, 'message') and choice.message:
                                message = choice.message
                                if hasattr(message, 'content') and message.content:
                                    content = message.content
                            
                            # For streaming, also check for delta content
                            elif hasattr(choice, 'delta') and choice.delta:
                                delta = choice.delta
                                if hasattr(delta, 'content') and delta.content:
                                    content = delta.content
                    
                    # Yield content if available
                    if content:
                        yield TaskUpdate(task_id, UpdateType.OUTPUT, {
                            "content": content,
                            "partial": True
                        })
                        
                    # Check if generation is finished
                    if hasattr(response, 'output') and response.output:
                        output = response.output
                        finish_reason = None
                        
                        # Get finish_reason from output level or choice level
                        if hasattr(output, 'finish_reason'):
                            finish_reason = output.finish_reason
                        elif hasattr(output, 'choices') and output.choices:
                            choice = output.choices[0]
                            if hasattr(choice, 'finish_reason'):
                                finish_reason = choice.finish_reason
                        
                        # If finished, we can optionally yield usage info
                        if finish_reason in ['stop', 'length', 'tool_calls']:
                            if hasattr(response, 'usage') and response.usage:
                                usage = response.usage
                                yield TaskUpdate(task_id, UpdateType.STATUS, {
                                    "usage": {
                                        "input_tokens": getattr(usage, 'input_tokens', 0),
                                        "output_tokens": getattr(usage, 'output_tokens', 0),
                                        "total_tokens": getattr(usage, 'total_tokens', 0)
                                    }
                                })
                
                else:
                    # Handle error cases based on official structure
                    error_msg = "Unknown error"
                    error_code = ""
                    
                    if hasattr(response, 'code') and response.code:
                        error_code = response.code
                    if hasattr(response, 'message') and response.message:
                        error_msg = response.message
                    
                    # Format error message
                    if error_code:
                        error_msg = f"[{error_code}] {error_msg}"
                    
                    yield TaskUpdate(task_id, UpdateType.ERROR, {
                        "error": f"Qwen API error: {error_msg}",
                        "status_code": getattr(response, 'status_code', None),
                        "request_id": getattr(response, 'request_id', None)
                    })
                    return
            
            yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.COMPLETED.value})
            
        except Exception as e:
            yield TaskUpdate(task_id, UpdateType.ERROR, {"error": str(e)})


class MockExecutor(BaseExecutor):
    """Mock executor for testing and demonstration."""
    
    async def execute_task(self, task: str, config: TaskConfig) -> AsyncIterator[TaskUpdate]:
        """Mock streaming execution."""
        task_id = self._generate_task_id()
        
        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.RUNNING.value})
        
        # Simulate progress updates
        for i in range(3):
            await asyncio.sleep(0.5)
            yield TaskUpdate(task_id, UpdateType.PROGRESS, {"step": i + 1, "total": 3})
            yield TaskUpdate(task_id, UpdateType.OUTPUT, {"content": f"Step {i + 1} completed"})
        
        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.COMPLETED.value})