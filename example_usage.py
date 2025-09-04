"""Example usage of the unified cligent SDK."""

import asyncio
from cligent import cligent, claude, gemini, qwen


async def basic_usage_example():
    """Basic usage examples."""
    print("=== Basic Usage Examples ===\n")
    
    # Method 1: Using factory function
    client = cligent("claude")
    print(f"Created client: {client}")
    print(f"Agent info: {client.get_agent_info()}")
    
    # Method 2: Using convenience functions
    claude_client = claude()
    gemini_client = gemini()
    qwen_client = qwen()
    
    print(f"Available agents: {claude_client.list_available_agents()}")
    
    # Switch agents dynamically
    client.switch_agent("gemini-cli")
    print(f"Switched to: {client.get_agent_info()['display_name']}")


async def task_execution_example():
    """Task execution examples."""
    print("\n=== Task Execution Examples ===\n")
    
    client = claude()
    
    # Check if agent supports execution
    if client.supports_execution():
        print("Agent supports task execution!")
        
        # Simple task execution
        print("Executing task...")
        result = await client.execute("Create a Python function to calculate fibonacci numbers")
        print(f"Task result: {result.status.value}")
        print(f"Output: {result.output[:100]}..." if result.output else "No output")
        
        # Streaming execution
        print("\nStreaming task execution:")
        async for update in client.execute_stream("Write a simple hello world script"):
            print(f"Update: {update.update_type.value} - {update.data}")
            
    else:
        print("Agent does not support task execution (using mock)")


async def log_parsing_example():
    """Log parsing examples (existing functionality)."""
    print("\n=== Log Parsing Examples ===\n")
    
    client = claude()
    
    # List available logs
    logs = client.list_logs()
    print(f"Found {len(logs)} log files")
    
    if logs:
        # Parse the latest log
        latest_log = logs[0][0]  # Get URI of first log
        chat = client.parse_logs(latest_log)
        print(f"Parsed {len(chat.messages)} messages from log: {latest_log}")
        
        # Export to YAML
        yaml_output = client.compose_yaml()
        print(f"YAML output length: {len(yaml_output)} characters")
    else:
        print("No logs found to parse")


async def combined_workflow_example():
    """Combined task execution and log parsing."""
    print("\n=== Combined Workflow Example ===\n")
    
    client = claude()
    
    if client.supports_execution():
        # Execute a task and immediately parse its logs
        result, chat = await client.task_and_parse("Write a function to sort a list")
        
        print(f"Task completed: {result.status.value}")
        print(f"Generated {len(chat.messages)} messages")
        
        # Export the conversation to YAML
        yaml_output = chat.export()
        print(f"Conversation YAML:\n{yaml_output[:300]}...")
    else:
        print("Task execution not supported, using mock data")


async def main():
    """Run all examples."""
    try:
        await basic_usage_example()
        await task_execution_example()
        await log_parsing_example()
        await combined_workflow_example()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Cligent Unified SDK - Usage Examples")
    print("=" * 40)
    
    # Run the examples
    asyncio.run(main())
    
    print("\n" + "=" * 40)
    print("Examples completed!")
    
    # Show usage patterns
    print("\nUsage Patterns:")
    print("1. cligent('claude') - Create client for specific agent")
    print("2. claude() - Direct Claude client")
    print("3. await client.execute('task') - Execute task")
    print("4. client.parse_logs() - Parse logs")
    print("5. await client.task_and_parse('task') - Execute + parse")