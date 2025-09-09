"""Example usage of the simplified cligent chat parser."""

from cligent import claude, gemini, qwen, cligent


def basic_usage_example():
    """Basic usage examples for chat parsers."""
    print("=== Basic Usage Examples ===\n")
    
    # Method 1: Using specific functions
    claude_client = claude()
    print(f"Created Claude agent: {claude_client}")
    print(f"Agent info: {claude_client.get_agent_info()}")
    
    # Method 2: Using generic function
    parser = cligent("claude")
    print(f"Created agent via cligent: {parser}")
    
    # Different agents
    gemini_client = gemini()
    qwen_client = qwen()
    
    print(f"Available parsers: Claude, Gemini, Qwen")


def log_parsing_example():
    """Log parsing examples."""
    print("\n=== Log Parsing Examples ===\n")
    
    parser = claude()
    
    # List available logs
    logs = parser.list_logs()
    print(f"Found {len(logs)} log files")
    
    if logs:
        # Parse the latest log
        latest_log = logs[0][0]  # Get URI of first log
        chat = parser.parse(latest_log)
        print(f"Parsed {len(chat.messages)} messages from log: {latest_log}")
        
        # Select some messages
        parser.select(latest_log, [0, 1])  # Select first two messages
        
        # Export to YAML
        yaml_output = parser.compose()
        print(f"YAML output length: {len(yaml_output)} characters")
        print(f"YAML preview:\n{yaml_output[:200]}...")
    else:
        print("No logs found to parse")


def message_selection_example():
    """Message selection and composition examples."""
    print("\n=== Message Selection Examples ===\n")
    
    parser = claude()
    logs = parser.list_logs()
    
    if logs:
        log_uri = logs[0][0]
        chat = parser.parse(log_uri)
        print(f"Loaded chat with {len(chat.messages)} messages")
        
        # Select specific messages
        parser.select(log_uri, [0, 2])  # First and third messages
        print(f"Selected {len(parser.selected_messages)} messages")
        
        # Compose YAML output
        yaml_output = parser.compose()
        print("Composed YAML from selected messages")
        
        # Clear selection and select all
        parser.clear_selection()
        parser.select(log_uri)  # All messages
        print(f"Selected all {len(parser.selected_messages)} messages")
        
        # Export full chat
        full_yaml = parser.compose()
        print(f"Full YAML length: {len(full_yaml)} characters")
    else:
        print("No logs available for selection example")


def multi_parser_example():
    """Examples using different parsers for different agents."""
    print("\n=== Multi-Parser Example ===\n")
    
    parsers = {
        'claude': claude(),
        'gemini': gemini(),
        'qwen': qwen()
    }
    
    for agent_name, parser in parsers.items():
        logs = parser.list_logs()
        print(f"{agent_name}: Found {len(logs)} logs")
        
        if logs:
            # Parse the most recent log
            latest_chat = parser.parse()  # Live/most recent log
            if latest_chat:
                print(f"  Latest chat has {len(latest_chat.messages)} messages")
            else:
                print("  No recent chat found")


def main():
    """Run all examples."""
    try:
        basic_usage_example()
        log_parsing_example()
        message_selection_example()
        multi_parser_example()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Cligent Chat Parser - Usage Examples")
    print("=" * 40)
    
    # Run the examples
    main()
    
    print("\n" + "=" * 40)
    print("Examples completed!")
    
    # Show usage patterns
    print("\nUsage Patterns:")
    print("1. claude() - Create Claude Code agent")
    print("2. gemini() - Create Gemini CLI agent") 
    print("3. qwen() - Create Qwen Code agent")
    print("4. cligent('claude') - Generic agent factory")
    print("5. parser.list_logs() - List available logs")
    print("6. parser.parse(log_uri) - Parse specific log")
    print("7. parser.select(log_uri, indices) - Select messages")
    print("8. parser.compose() - Export to YAML")