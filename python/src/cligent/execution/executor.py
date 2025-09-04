"""Base executor implementations for different agent types."""

import asyncio
import subprocess
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


class SubprocessExecutor(BaseExecutor):
    """Executor that runs agents as subprocesses."""
    
    def __init__(self, agent_name: str, command_template: str):
        """
        Args:
            agent_name: Name of the agent
            command_template: Command template with {task} placeholder
        """
        super().__init__(agent_name)
        self.command_template = command_template

    async def execute_task(self, task: str, config: TaskConfig) -> TaskResult:
        """Execute task via subprocess."""
        task_id = self._generate_task_id()
        
        # Build command
        command = self.command_template.format(task=task)
        if config.workspace:
            command = f"cd '{config.workspace}' && {command}"
        
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            created_at=datetime.now()
        )
        
        try:
            # Set up environment
            env = dict(config.environment) if config.environment else {}
            
            # Run subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=config.workspace
            )
            
            # Wait for completion with timeout
            timeout = config.timeout if config.timeout else 300  # 5 min default
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            # Process results
            if process.returncode == 0:
                result.status = TaskStatus.COMPLETED
                result.output = stdout.decode('utf-8') if stdout else ""
            else:
                result.status = TaskStatus.FAILED
                result.error = stderr.decode('utf-8') if stderr else "Process failed"
            
            result.completed_at = datetime.now()
            
            if stderr:
                result.logs.append(f"STDERR: {stderr.decode('utf-8')}")
            if stdout:
                result.logs.append(f"STDOUT: {stdout.decode('utf-8')}")
                
        except asyncio.TimeoutError:
            result.status = TaskStatus.FAILED
            result.error = f"Task timed out after {timeout} seconds"
            result.completed_at = datetime.now()
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = f"Execution error: {str(e)}"
            result.completed_at = datetime.now()
        
        return result

    async def execute_task_stream(self, task: str, config: TaskConfig) -> AsyncIterator[TaskUpdate]:
        """Execute task with streaming output."""
        task_id = self._generate_task_id()
        
        # Initial status update
        yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.RUNNING.value})
        
        # Build command
        command = self.command_template.format(task=task)
        if config.workspace:
            command = f"cd '{config.workspace}' && {command}"
            
        try:
            # Set up environment
            env = dict(config.environment) if config.environment else {}
            
            # Start subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=config.workspace
            )
            
            # Stream output
            async def read_stream(stream, stream_name):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode('utf-8').rstrip()
                    if text:
                        yield TaskUpdate(task_id, UpdateType.OUTPUT, {
                            "stream": stream_name,
                            "content": text
                        })
            
            # Read both streams concurrently  
            async for update in self._merge_streams([
                read_stream(process.stdout, "stdout"),
                read_stream(process.stderr, "stderr")
            ]):
                yield update
            
            # Wait for process to complete
            await process.wait()
            
            # Final status
            if process.returncode == 0:
                yield TaskUpdate(task_id, UpdateType.STATUS, {"status": TaskStatus.COMPLETED.value})
            else:
                yield TaskUpdate(task_id, UpdateType.ERROR, {
                    "error": f"Process exited with code {process.returncode}"
                })
                
        except Exception as e:
            yield TaskUpdate(task_id, UpdateType.ERROR, {"error": str(e)})

    async def _merge_streams(self, streams) -> AsyncIterator[TaskUpdate]:
        """Merge multiple async streams."""
        tasks = [asyncio.create_task(self._exhaust_stream(stream)) for stream in streams]
        
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            while tasks:
                for task in done:
                    async for item in task.result():
                        yield item
                    tasks.remove(task)
                
                if tasks:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()

    async def _exhaust_stream(self, stream) -> List[TaskUpdate]:
        """Collect all items from async stream."""
        items = []
        async for item in stream:
            items.append(item)
        return items


class ClaudeCodeExecutor(SubprocessExecutor):
    """Executor for Claude Code agent."""
    
    def __init__(self):
        # Claude Code command template - adjust based on actual CLI
        super().__init__("claude-code", "claude-code '{task}'")


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