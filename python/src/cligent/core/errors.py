"""Error handling for the chat parser."""

from typing import Optional, List, Any


class ChatParserError(Exception):
    """Base exception for chat parser errors."""
    pass


class ParseError(ChatParserError):
    """Error during parsing of logs or records."""
    
    def __init__(self, message: str, log_path: str = None, 
                 line_number: int = None, recoverable: bool = True):
        """Initialize parse error.
        
        Args:
            message: Error description
            log_path: Path to the problematic log file
            line_number: Line number where error occurred
            recoverable: Whether parsing can continue
        """
        self.message = message
        self.log_path = log_path
        self.line_number = line_number
        self.recoverable = recoverable
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with context."""
        msg = self.message
        if self.log_path:
            msg = f"{self.log_path}: {msg}"
        if self.line_number:
            msg = f"{msg} (line {self.line_number})"
        return msg


class LogAccessError(ChatParserError):
    """Error accessing log files."""
    
    def __init__(self, path: str, reason: str = "Permission denied"):
        """Initialize access error.
        
        Args:
            path: Path that couldn't be accessed
            reason: Reason for access failure
        """
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot access {path}: {reason}")


class LogCorruptionError(ParseError):
    """Error when log file is corrupted."""
    
    def __init__(self, log_path: str, details: str = None):
        """Initialize corruption error.
        
        Args:
            log_path: Path to corrupted log
            details: Additional details about corruption
        """
        message = f"Log file corrupted"
        if details:
            message += f": {details}"
        super().__init__(message, log_path=log_path, recoverable=False)


class InvalidFormatError(ParseError):
    """Error when log format is not recognized."""
    
    def __init__(self, log_path: str, expected: str = None, found: str = None):
        """Initialize format error.
        
        Args:
            log_path: Path to log with invalid format
            expected: Expected format description
            found: What was actually found
        """
        message = "Invalid log format"
        if expected and found:
            message = f"Expected {expected}, found {found}"
        elif expected:
            message = f"Expected {expected}"
        super().__init__(message, log_path=log_path)


class ErrorCollector:
    """Collects errors during parsing for reporting."""
    
    def __init__(self):
        """Initialize error collector."""
        self.errors: List[ParseError] = []
        self.recovered_count = 0
        self.lost_count = 0
    
    def add_error(self, error: ParseError) -> None:
        """Add an error to the collection.
        
        Args:
            error: Error to add
        """
        self.errors.append(error)
        if error.recoverable:
            self.recovered_count += 1
        else:
            self.lost_count += 1
    
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0
    
    def has_fatal_errors(self) -> bool:
        """Check if any non-recoverable errors occurred."""
        return any(not e.recoverable for e in self.errors)
    
    def generate_report(self) -> str:
        """Generate a summary report of all errors.
        
        Returns:
            Formatted error report
        """
        if not self.errors:
            return "No errors occurred."
        
        lines = []
        lines.append(f"Error Summary: {len(self.errors)} total errors")
        lines.append(f"  - Recovered: {self.recovered_count}")
        lines.append(f"  - Fatal: {self.lost_count}")
        lines.append("")
        
        for i, error in enumerate(self.errors, 1):
            status = "✓ Recovered" if error.recoverable else "✗ Fatal"
            lines.append(f"{i}. {status}: {error.message}")
            if error.location:
                lines.append(f"   Location: {error.location}")
            if error.context:
                lines.append(f"   Context: {error.context}")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert error collection to dictionary.
        
        Returns:
            Dictionary with error statistics and details
        """
        return {
            "total_errors": len(self.errors),
            "recovered_count": self.recovered_count,
            "fatal_count": self.lost_count,
            "has_fatal_errors": self.has_fatal_errors(),
            "errors": [
                {
                    "message": error.message,
                    "location": error.location,
                    "context": error.context,
                    "recoverable": error.recoverable
                }
                for error in self.errors
            ]
        }