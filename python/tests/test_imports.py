"""Test that all public imports work correctly."""

def test_main_package_imports():
    """Test that main package imports work."""
    import cligent
    assert hasattr(cligent, '__version__')
    assert hasattr(cligent, 'cligent')  # Main factory function
    assert hasattr(cligent, 'Chat')
    assert hasattr(cligent, 'Message')
    assert hasattr(cligent, 'AgentBackend')


def test_cligent_direct_imports():
    """Test that cligent direct imports work."""
    from cligent import (
        cligent,  # Main factory function 
        Chat, 
        Message, 
        Role,
        LogStore,
        AgentBackend
    )
    
    # Verify classes can be instantiated
    msg = Message(role=Role.USER, content="test")
    assert msg.content == "test"
    
    chat = Chat()
    assert chat.messages == []
    
    # Verify agent creation works
    agent = cligent("claude-code")
    assert isinstance(agent, AgentBackend)


def test_convenience_imports():
    """Test convenience imports from main package."""
    from cligent import cligent, claude, gemini, qwen, Chat, Message
    
    # Should be able to create agents directly
    agent1 = cligent("claude-code")
    agent2 = claude()
    agent3 = gemini()
    agent4 = qwen()
    
    assert agent1.config.name == "claude-code"
    assert agent2.config.name == "claude-code"
    assert agent3.config.name == "gemini-cli"
    assert agent4.config.name == "qwen-code"


def test_error_imports():
    """Test error classes can be imported."""
    from cligent import (
        ChatParserError,
        ParseError,
        LogAccessError,
        LogCorruptionError,
        InvalidFormatError
    )
    
    # Should be able to create exceptions
    error = ChatParserError("test error")
    assert str(error) == "test error"


def test_private_modules_not_exposed():
    """Test that internal modules are not exposed in public API."""
    import cligent
    
    # These should not be in __all__
    public_api = cligent.__all__
    
    # Internal Claude implementation should not be public
    assert "ClaudeStore" not in public_api
    assert "Record" not in public_api
    assert "Session" not in public_api