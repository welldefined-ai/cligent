#!/usr/bin/env python3
"""
Parse all Claude Code logs and export to single Tigs YAML file.
"""

from pathlib import Path
from cligent import ChatParser, Chat


def main():
    # Initialize parser with default Claude location
    parser = ChatParser("claude-code")
    
    # List all available logs
    logs = parser.list_logs()
    
    if not logs:
        print("No Claude Code logs found")
        return
    
    print(f"Found {len(logs)} log files")
    
    # Collect all messages from all logs
    all_messages = []
    
    for uri, metadata in logs:
        try:
            chat = parser.parse(uri)
            if chat and chat.messages:
                all_messages.extend(chat.messages)
                print(f"  ✓ {Path(uri).name}: {len(chat.messages)} messages")
        except Exception as e:
            print(f"  ✗ {Path(uri).name}: {e}")
    
    # Export to single YAML file
    if all_messages:
        combined_chat = Chat(messages=all_messages)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"all_claude_logs_{timestamp}.yaml"
        
        yaml_content = combined_chat.export()
        Path(output_file).write_text(yaml_content)
        
        print(f"\nExported {len(all_messages)} messages to {output_file}")
        
        # Verify YAML syntax by parsing it back
        import yaml
        try:
            parsed = yaml.safe_load(yaml_content)
            assert parsed["schema"] == "tigs.chat/v1"
            assert len(parsed["messages"]) == len(all_messages)
            print(f"✓ YAML validation passed")
        except Exception as e:
            print(f"✗ YAML validation failed: {e}")
    else:
        print("No messages found")


if __name__ == "__main__":
    main()
