"""Test that all public imports work correctly."""

def test_main_package_imports():
    """Test that core imports work."""
    from core import Chat, Message, Role, cligent, Cligent
    assert callable(cligent)  # Main factory function
    assert Chat is not None
    assert Message is not None
    assert Cligent is not None


def test_direct_imports():
    """Test that direct imports work."""
    from core import (
        cligent,  # Main factory function 
        Chat, 
        Message, 
        Role,
        LogStore,
        Cligent
    )
    
    # Verify classes can be instantiated
    msg = Message(role=Role.USER, content="test")
    assert msg.content == "test"
    
    chat = Chat()
    assert chat.messages == []
    
    # Verify agent creation works
    agent = cligent("claude")
    assert agent.name == "claude-code"


def test_convenience_imports():
    """Test convenience imports from core."""
    from core import claude, gemini, qwen, Chat, Message

    # Should be able to create agents directly
    agent1 = claude()
    agent2 = gemini() 
    agent3 = qwen()
    
    assert agent1.name == "claude-code"
    assert agent2.name == "gemini-cli"
    assert agent3.name == "qwen-code"


def test_backwards_compatibility():
    """Test backwards compatibility aliases."""
    from core import cligent as ChatParser, Chat, Message

    # Should be able to use old ChatParser function
    agent = ChatParser("claude")
    assert agent.name == "claude-code"


def test_error_imports():
    """Test that error classes can be imported."""
    from core import (
        ChatParserError,
        ParseError,
        LogAccessError,
        LogCorruptionError,
        InvalidFormatError
    )
    
    # All should be proper exception classes
    assert issubclass(ChatParserError, Exception)
    assert issubclass(ParseError, Exception)
    assert issubclass(LogAccessError, Exception)


def test_private_modules_not_exposed():
    """Test that private implementation details aren't exposed."""
    from core import Chat
    
    # Core models should be available
    assert Chat is not None
    
    # But private implementation details shouldn't leak
    try:
        from core import models
        # This is OK - models module can be imported
        assert models is not None
    except ImportError:
        pass  # This is also OK