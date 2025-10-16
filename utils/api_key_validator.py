"""
API Key Validation Module
Validates all required API keys on startup to fail fast
"""

import os
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class KeyValidationResult:
    """Result of validating an API key"""
    key_name: str
    is_valid: bool
    error_message: str = ""
    is_required: bool = True


class APIKeyValidator:
    """Validates API keys according to their expected formats"""

    # Expected formats for each API key
    KEY_PATTERNS = {
        "OPENAI_API_KEY": r"^sk-[A-Za-z0-9]{20,}$",
        "ANTHROPIC_API_KEY": r"^sk-ant-[A-Za-z0-9\-_]{20,}$",
        "GOOGLE_API_KEY": r"^[A-Za-z0-9\-_]{20,}$",
        "WANDB_API_KEY": r"^[a-f0-9]{40}$",
        "OPENROUTER_API_KEY": r"^sk-or-v1-[a-f0-9]{64}$",
        "TAVILY_API_KEY": r"^tvly-[a-zA-Z0-9\-]{20,}$",
    }

    # Keys that are required for basic functionality
    REQUIRED_KEYS = {
        "WANDB_API_KEY",  # Required for WeaveHacks
    }

    # Keys where at least one from the group must be present
    OPTIONAL_GROUPS = {
        "llm": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"],
    }

    def __init__(self):
        self.results: List[KeyValidationResult] = []

    def validate_all(self) -> Tuple[bool, List[KeyValidationResult]]:
        """
        Validate all API keys

        Returns:
            Tuple of (all_valid, list_of_results)
        """
        self.results = []

        # Check required keys
        for key_name in self.REQUIRED_KEYS:
            result = self._validate_key(key_name, required=True)
            self.results.append(result)

        # Check optional groups (at least one from each group)
        for group_name, key_names in self.OPTIONAL_GROUPS.items():
            group_results = [self._validate_key(key_name, required=False) for key_name in key_names]
            self.results.extend(group_results)

            # Check if at least one key in the group is valid
            if not any(r.is_valid for r in group_results):
                self.results.append(KeyValidationResult(
                    key_name=f"{group_name}_group",
                    is_valid=False,
                    error_message=f"At least one {group_name} API key is required: {', '.join(key_names)}",
                    is_required=True
                ))

        # Check if all required validations passed
        all_valid = all(
            r.is_valid for r in self.results
            if r.is_required
        )

        return all_valid, self.results

    def _validate_key(self, key_name: str, required: bool = True) -> KeyValidationResult:
        """
        Validate a single API key

        Args:
            key_name: Name of the environment variable
            required: Whether this key is required

        Returns:
            KeyValidationResult
        """
        # Get key from environment
        key_value = os.getenv(key_name)

        # Check if key exists
        if not key_value:
            return KeyValidationResult(
                key_name=key_name,
                is_valid=False,
                error_message=f"{key_name} is not set in environment",
                is_required=required
            )

        # Check for demo/placeholder values
        if key_value.startswith("demo_mode") or key_value == "your_key_here":
            return KeyValidationResult(
                key_name=key_name,
                is_valid=False,
                error_message=f"{key_name} is set to a placeholder value",
                is_required=required
            )

        # Validate format
        pattern = self.KEY_PATTERNS.get(key_name)
        if pattern and not re.match(pattern, key_value):
            return KeyValidationResult(
                key_name=key_name,
                is_valid=False,
                error_message=f"{key_name} does not match expected format",
                is_required=required
            )

        # Key is valid
        return KeyValidationResult(
            key_name=key_name,
            is_valid=True,
            is_required=required
        )

    def print_results(self, results: List[KeyValidationResult]) -> None:
        """Print validation results in a readable format"""
        print("\n" + "="*60)
        print(" API Key Validation Results")
        print("="*60)

        required_valid = []
        required_invalid = []
        optional_valid = []
        optional_invalid = []

        for result in results:
            if result.is_required:
                if result.is_valid:
                    required_valid.append(result)
                else:
                    required_invalid.append(result)
            else:
                if result.is_valid:
                    optional_valid.append(result)
                else:
                    optional_invalid.append(result)

        # Print required keys
        print("\n[OK] Required Keys:")
        if required_valid:
            for result in required_valid:
                print(f"   [OK] {result.key_name}")
        else:
            print("   (none)")

        if required_invalid:
            print("\n[FAIL] Missing/Invalid Required Keys:")
            for result in required_invalid:
                print(f"   [X] {result.key_name}: {result.error_message}")

        # Print optional keys
        print("\n[LIST] Optional Keys:")
        if optional_valid:
            for result in optional_valid:
                print(f"   [OK] {result.key_name}")

        if optional_invalid:
            for result in optional_invalid:
                print(f"    {result.key_name}: {result.error_message}")

        print("\n" + "="*60)


def validate_on_startup() -> bool:
    """
    Validate API keys on startup
    Returns True if all required keys are valid, False otherwise
    """
    validator = APIKeyValidator()
    all_valid, results = validator.validate_all()

    validator.print_results(results)

    if not all_valid:
        print("\n[FAIL] VALIDATION FAILED: Please fix the above errors before continuing")
        print("[IDEA] Tip: Check your .env file and ensure all required API keys are set correctly\n")
        return False

    print("\n[OK] All required API keys are valid! Starting application...\n")
    return True


if __name__ == "__main__":
    # Test validation
    import sys
    if not validate_on_startup():
        sys.exit(1)
