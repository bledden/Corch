"""
Input Sanitization Module

Provides security-focused sanitization for all user inputs to prevent:
- Injection attacks (SQL, NoSQL, Command, etc.)
- XSS attempts
- Path traversal attacks
- Malicious payloads
- Excessive resource consumption
"""

import re
import html
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class InputSanitizer:
    """
    Sanitizes user inputs for security

    This class provides multiple sanitization methods for different input types.
    Use the appropriate method based on how the input will be used.
    """

    # Dangerous patterns that could indicate attacks
    INJECTION_PATTERNS = [
        r'(\bOR\b|\bAND\b).*=.*',  # SQL injection patterns
        r'(union|select|insert|update|delete|drop|create|alter)\s',  # SQL keywords
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'on\w+\s*=',  # Event handlers (onclick, onerror, etc.)
        r'\$\{.*\}',  # Template injection
        r'`.*`',  # Command injection
        r'\|\||&&',  # Command chaining
        r'\.\./|\.\.',  # Path traversal
    ]

    # Maximum lengths for different input types
    MAX_TASK_LENGTH = 10000
    MAX_AGENT_NAME_LENGTH = 50
    MAX_LIST_SIZE = 100

    @staticmethod
    def sanitize_task_description(task: str) -> str:
        """
        Sanitize task description for use in LLM prompts

        Args:
            task: Raw task description from user

        Returns:
            Sanitized task description

        Raises:
            ValueError: If task contains dangerous patterns
        """
        if not task:
            raise ValueError("Task cannot be empty")

        # Strip whitespace
        task = task.strip()

        # Check if empty after stripping
        if not task:
            raise ValueError("Task cannot be empty")

        # Check length
        if len(task) > InputSanitizer.MAX_TASK_LENGTH:
            raise ValueError(
                f"Task too long. Maximum {InputSanitizer.MAX_TASK_LENGTH} characters, "
                f"got {len(task)}"
            )

        # Check for injection patterns
        for pattern in InputSanitizer.INJECTION_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                logger.warning(f"Potential injection attempt detected: {pattern}")
                raise ValueError(
                    "Task contains potentially unsafe content. "
                    "Please remove special characters or code patterns."
                )

        # Check for excessive control characters (exclude newline, tab, carriage return)
        control_chars = sum(1 for c in task if ord(c) < 32 and c not in '\n\r\t')
        if control_chars > 5:
            raise ValueError("Task contains excessive control characters")

        # Remove null bytes
        task = task.replace('\x00', '')

        # Normalize whitespace (but keep newlines for multi-line tasks)
        task = re.sub(r'[ \t]+', ' ', task)
        task = re.sub(r'\n\n+', '\n\n', task)

        return task

    @staticmethod
    def sanitize_agent_name(agent_name: str) -> str:
        """
        Sanitize agent name

        Args:
            agent_name: Agent name from user input

        Returns:
            Sanitized agent name (lowercase, alphanumeric + underscore only)

        Raises:
            ValueError: If agent name is invalid
        """
        if not agent_name:
            raise ValueError("Agent name cannot be empty")

        # Convert to lowercase
        agent_name = agent_name.lower().strip()

        # Check length
        if len(agent_name) > InputSanitizer.MAX_AGENT_NAME_LENGTH:
            raise ValueError(
                f"Agent name too long. Maximum {InputSanitizer.MAX_AGENT_NAME_LENGTH} "
                f"characters, got {len(agent_name)}"
            )

        # Only allow alphanumeric and underscore
        if not re.match(r'^[a-z0-9_]+$', agent_name):
            raise ValueError(
                "Agent name must contain only lowercase letters, numbers, and underscores"
            )

        return agent_name

    @staticmethod
    def sanitize_agent_list(agents: List[str]) -> List[str]:
        """
        Sanitize list of agent names

        Args:
            agents: List of agent names

        Returns:
            List of sanitized agent names

        Raises:
            ValueError: If list is too large or contains invalid names
        """
        if not agents:
            return []

        # Check list size
        if len(agents) > InputSanitizer.MAX_LIST_SIZE:
            raise ValueError(
                f"Agent list too large. Maximum {InputSanitizer.MAX_LIST_SIZE} items, "
                f"got {len(agents)}"
            )

        # Check for duplicates
        if len(agents) != len(set(agents)):
            logger.warning("Agent list contains duplicates, removing them")
            agents = list(set(agents))

        # Sanitize each agent name
        return [InputSanitizer.sanitize_agent_name(agent) for agent in agents]

    @staticmethod
    def sanitize_for_html(text: str) -> str:
        """
        Sanitize text for safe display in HTML

        Args:
            text: Text to sanitize

        Returns:
            HTML-escaped text
        """
        if not text:
            return ""

        # Remove script tags BEFORE HTML escaping (so we can match actual tags)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)

        # HTML escape
        text = html.escape(text)

        return text

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal

        Args:
            filename: Filename from user input

        Returns:
            Sanitized filename (safe for filesystem operations)

        Raises:
            ValueError: If filename is unsafe
        """
        if not filename:
            raise ValueError("Filename cannot be empty")

        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')

        # Remove parent directory references
        filename = filename.replace('..', '')

        # Remove null bytes
        filename = filename.replace('\x00', '')

        # Only allow alphanumeric, dots, dashes, underscores
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

        # Prevent hidden files
        if filename.startswith('.'):
            filename = '_' + filename

        # Check length
        if len(filename) > 255:
            raise ValueError("Filename too long (max 255 characters)")

        if not filename or filename == '_':
            raise ValueError("Invalid filename")

        return filename

    @staticmethod
    def sanitize_json_key(key: str) -> str:
        """
        Sanitize JSON object key

        Args:
            key: JSON key from user input

        Returns:
            Sanitized key

        Raises:
            ValueError: If key is invalid
        """
        if not key:
            raise ValueError("JSON key cannot be empty")

        # Strip whitespace
        key = key.strip()

        # Check length
        if len(key) > 100:
            raise ValueError("JSON key too long (max 100 characters)")

        # Prevent __proto__ pollution
        if key in ['__proto__', 'constructor', 'prototype']:
            raise ValueError(f"Reserved key name: {key}")

        return key

    @staticmethod
    def validate_numeric_range(
        value: float,
        min_value: float,
        max_value: float,
        name: str = "value"
    ):
        """
        Validate numeric value is within acceptable range

        Args:
            value: Numeric value to validate
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value
            name: Name of the parameter (for error messages)

        Raises:
            ValueError: If value is out of range
        """
        if value < min_value or value > max_value:
            raise ValueError(
                f"{name} must be between {min_value} and {max_value}, "
                f"got {value}"
            )

    @staticmethod
    def sanitize_url(url: str, allowed_schemes: Optional[List[str]] = None) -> str:
        """
        Sanitize and validate URL

        Args:
            url: URL from user input
            allowed_schemes: List of allowed URL schemes (default: ['http', 'https'])

        Returns:
            Sanitized URL

        Raises:
            ValueError: If URL is invalid or uses disallowed scheme
        """
        if not url:
            raise ValueError("URL cannot be empty")

        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']

        # Remove whitespace
        url = url.strip()

        # Check length
        if len(url) > 2048:
            raise ValueError("URL too long (max 2048 characters)")

        # Prevent javascript: and data: URLs FIRST (before checking for ://)
        if url.lower().startswith(('javascript:', 'data:', 'vbscript:')):
            raise ValueError("Unsafe URL scheme detected")

        # Parse scheme
        if '://' in url:
            scheme = url.split('://')[0].lower()
            if scheme not in allowed_schemes:
                raise ValueError(
                    f"URL scheme '{scheme}' not allowed. "
                    f"Allowed schemes: {allowed_schemes}"
                )
        else:
            raise ValueError("URL must include scheme (http:// or https://)")

        return url


# Convenience functions for common sanitization tasks

def sanitize_task(task: str) -> str:
    """Sanitize task description (convenience function)"""
    return InputSanitizer.sanitize_task_description(task)


def sanitize_agents(agents: Optional[List[str]]) -> List[str]:
    """Sanitize agent list (convenience function)"""
    if agents is None:
        return []
    return InputSanitizer.sanitize_agent_list(agents)


def sanitize_for_display(text: str) -> str:
    """Sanitize text for HTML display (convenience function)"""
    return InputSanitizer.sanitize_for_html(text)
