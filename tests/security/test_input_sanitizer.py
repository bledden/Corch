"""
Tests for input sanitization module
"""

import pytest
from src.security.input_sanitizer import InputSanitizer


class TestTaskSanitization:
    """Test task description sanitization"""

    def test_valid_task(self):
        """Test that valid tasks pass through"""
        task = "Write a Python function to calculate factorial"
        result = InputSanitizer.sanitize_task_description(task)
        assert result == task

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped"""
        task = "  Write a function  "
        result = InputSanitizer.sanitize_task_description(task)
        assert result == "Write a function"

    def test_empty_task_raises(self):
        """Test that empty task raises ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            InputSanitizer.sanitize_task_description("")

    def test_whitespace_only_raises(self):
        """Test that whitespace-only task raises ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            InputSanitizer.sanitize_task_description("   ")

    def test_too_long_task_raises(self):
        """Test that overly long tasks are rejected"""
        task = "x" * 10001
        with pytest.raises(ValueError, match="too long"):
            InputSanitizer.sanitize_task_description(task)

    def test_sql_injection_detected(self):
        """Test that SQL injection attempts are caught"""
        malicious_tasks = [
            "Write function OR 1=1",
            "SELECT * FROM users",
            "'; DROP TABLE users; --",
            "task UNION SELECT password FROM users"
        ]

        for task in malicious_tasks:
            with pytest.raises(ValueError, match="unsafe content"):
                InputSanitizer.sanitize_task_description(task)

    def test_script_injection_detected(self):
        """Test that script injection attempts are caught"""
        malicious_tasks = [
            "<script>alert('xss')</script>",
            "Write function <script>fetch('evil.com')</script>",
            "javascript:alert(1)"
        ]

        for task in malicious_tasks:
            with pytest.raises(ValueError, match="unsafe content"):
                InputSanitizer.sanitize_task_description(task)

    def test_command_injection_detected(self):
        """Test that command injection attempts are caught"""
        malicious_tasks = [
            "Write function `rm -rf /`",
            "task && cat /etc/passwd",
            "task || whoami"
        ]

        for task in malicious_tasks:
            with pytest.raises(ValueError, match="unsafe content"):
                InputSanitizer.sanitize_task_description(task)

    def test_path_traversal_detected(self):
        """Test that path traversal attempts are caught"""
        malicious_tasks = [
            "Read file ../../etc/passwd",
            "Write function ../../../secrets",
            "task with ../ in it"
        ]

        for task in malicious_tasks:
            with pytest.raises(ValueError, match="unsafe content"):
                InputSanitizer.sanitize_task_description(task)

    def test_null_bytes_removed(self):
        """Test that null bytes are removed"""
        task = "Write function\x00with null"
        result = InputSanitizer.sanitize_task_description(task)
        assert '\x00' not in result

    def test_excessive_control_chars_rejected(self):
        """Test that excessive control characters are rejected"""
        task = "Write" + "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b" + "function"
        with pytest.raises(ValueError, match="control characters"):
            InputSanitizer.sanitize_task_description(task)

    def test_normalizes_whitespace(self):
        """Test that multiple spaces are normalized"""
        task = "Write    a    function"
        result = InputSanitizer.sanitize_task_description(task)
        assert result == "Write a function"

    def test_preserves_newlines(self):
        """Test that newlines are preserved (for multi-line tasks)"""
        task = "Write a function:\n1. First step\n2. Second step"
        result = InputSanitizer.sanitize_task_description(task)
        assert "\n" in result


class TestAgentNameSanitization:
    """Test agent name sanitization"""

    def test_valid_agent_name(self):
        """Test valid agent names"""
        assert InputSanitizer.sanitize_agent_name("architect") == "architect"
        assert InputSanitizer.sanitize_agent_name("coder") == "coder"
        assert InputSanitizer.sanitize_agent_name("test_agent") == "test_agent"

    def test_uppercase_normalized(self):
        """Test that uppercase is normalized to lowercase"""
        assert InputSanitizer.sanitize_agent_name("ARCHITECT") == "architect"
        assert InputSanitizer.sanitize_agent_name("Coder") == "coder"

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped"""
        assert InputSanitizer.sanitize_agent_name("  architect  ") == "architect"

    def test_empty_name_raises(self):
        """Test that empty name raises ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            InputSanitizer.sanitize_agent_name("")

    def test_too_long_name_raises(self):
        """Test that overly long names are rejected"""
        name = "a" * 51
        with pytest.raises(ValueError, match="too long"):
            InputSanitizer.sanitize_agent_name(name)

    def test_special_chars_rejected(self):
        """Test that special characters are rejected"""
        invalid_names = [
            "agent-name",  # hyphen not allowed
            "agent.name",  # dot not allowed
            "agent name",  # space not allowed
            "agent@name",  # special char not allowed
            "../agent",    # path traversal
        ]

        for name in invalid_names:
            with pytest.raises(ValueError, match="lowercase letters, numbers, and underscores"):
                InputSanitizer.sanitize_agent_name(name)


class TestAgentListSanitization:
    """Test agent list sanitization"""

    def test_valid_list(self):
        """Test valid agent list"""
        agents = ["architect", "coder", "reviewer"]
        result = InputSanitizer.sanitize_agent_list(agents)
        assert result == agents

    def test_empty_list(self):
        """Test empty list returns empty"""
        assert InputSanitizer.sanitize_agent_list([]) == []

    def test_removes_duplicates(self):
        """Test that duplicates are removed"""
        agents = ["architect", "coder", "architect"]
        result = InputSanitizer.sanitize_agent_list(agents)
        assert len(result) == 2
        assert set(result) == {"architect", "coder"}

    def test_too_large_list_raises(self):
        """Test that overly large lists are rejected"""
        agents = [f"agent_{i}" for i in range(101)]
        with pytest.raises(ValueError, match="too large"):
            InputSanitizer.sanitize_agent_list(agents)

    def test_sanitizes_each_name(self):
        """Test that each name is sanitized"""
        agents = ["ARCHITECT", " coder ", "reviewer"]
        result = InputSanitizer.sanitize_agent_list(agents)
        assert result == ["architect", "coder", "reviewer"]


class TestHTMLSanitization:
    """Test HTML sanitization"""

    def test_escapes_html(self):
        """Test that HTML is escaped"""
        text = "<p>Hello</p>"
        result = InputSanitizer.sanitize_for_html(text)
        assert result == "&lt;p&gt;Hello&lt;/p&gt;"

    def test_removes_script_tags(self):
        """Test that script tags are removed"""
        text = "Hello <script>alert('xss')</script> World"
        result = InputSanitizer.sanitize_for_html(text)
        assert "<script>" not in result
        assert "alert" not in result

    def test_empty_string(self):
        """Test empty string"""
        assert InputSanitizer.sanitize_for_html("") == ""


class TestFilenameSanitization:
    """Test filename sanitization"""

    def test_valid_filename(self):
        """Test valid filenames"""
        assert InputSanitizer.sanitize_filename("test.txt") == "test.txt"
        assert InputSanitizer.sanitize_filename("my_file-2024.pdf") == "my_file-2024.pdf"

    def test_removes_path_separators(self):
        """Test that path separators are replaced"""
        assert InputSanitizer.sanitize_filename("path/to/file.txt") == "path_to_file.txt"
        assert InputSanitizer.sanitize_filename("path\\to\\file.txt") == "path_to_file.txt"

    def test_removes_parent_refs(self):
        """Test that parent directory references are removed"""
        assert ".." not in InputSanitizer.sanitize_filename("../../../etc/passwd")

    def test_prevents_hidden_files(self):
        """Test that hidden files are prevented"""
        result = InputSanitizer.sanitize_filename(".hidden")
        assert result.startswith("_")

    def test_too_long_filename_raises(self):
        """Test that overly long filenames are rejected"""
        filename = "a" * 256
        with pytest.raises(ValueError, match="too long"):
            InputSanitizer.sanitize_filename(filename)


class TestNumericValidation:
    """Test numeric range validation"""

    def test_valid_range(self):
        """Test that valid values pass"""
        # Should not raise
        InputSanitizer.validate_numeric_range(5.0, 0.0, 10.0)
        InputSanitizer.validate_numeric_range(0.0, 0.0, 1.0)
        InputSanitizer.validate_numeric_range(1.0, 0.0, 1.0)

    def test_out_of_range_raises(self):
        """Test that out of range values raise ValueError"""
        with pytest.raises(ValueError, match="must be between"):
            InputSanitizer.validate_numeric_range(11.0, 0.0, 10.0)

        with pytest.raises(ValueError, match="must be between"):
            InputSanitizer.validate_numeric_range(-1.0, 0.0, 10.0)


class TestURLSanitization:
    """Test URL sanitization"""

    def test_valid_http_url(self):
        """Test valid HTTP URL"""
        url = "http://example.com/path"
        result = InputSanitizer.sanitize_url(url)
        assert result == url

    def test_valid_https_url(self):
        """Test valid HTTPS URL"""
        url = "https://example.com/path"
        result = InputSanitizer.sanitize_url(url)
        assert result == url

    def test_javascript_url_rejected(self):
        """Test that javascript: URLs are rejected"""
        with pytest.raises(ValueError, match="Unsafe URL scheme"):
            InputSanitizer.sanitize_url("javascript:alert(1)")

    def test_data_url_rejected(self):
        """Test that data: URLs are rejected"""
        with pytest.raises(ValueError, match="Unsafe URL scheme"):
            InputSanitizer.sanitize_url("data:text/html,<script>alert(1)</script>")

    def test_no_scheme_rejected(self):
        """Test that URLs without scheme are rejected"""
        with pytest.raises(ValueError, match="must include scheme"):
            InputSanitizer.sanitize_url("example.com/path")

    def test_disallowed_scheme_rejected(self):
        """Test that disallowed schemes are rejected"""
        with pytest.raises(ValueError, match="not allowed"):
            InputSanitizer.sanitize_url("ftp://example.com", allowed_schemes=['http', 'https'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
