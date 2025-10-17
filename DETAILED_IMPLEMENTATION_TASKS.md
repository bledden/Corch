# Enhanced Evaluation System - Granular Implementation Tasks
## Corch (weavehacks-collaborative)

**Ultra-detailed task breakdown with specific files, functions, and code**

---

## Phase 1: Foundation & Dependencies

### Task 1.1.1: Create Requirements File
**Owner**: 🤖 Claude | **Time**: 5 min | **Status**: PENDING

**Exact Steps**:
1. Create file: `/Users/bledden/Documents/weavehacks-collaborative/requirements-evaluation.txt`
2. Add exact content:
   ```
   # Security Analysis
   bandit==1.7.7

   # Static Analysis
   pylint==3.0.3
   flake8==7.0.0
   flake8-bugbear==24.1.17
   flake8-comprehensions==3.14.0
   mypy==1.8.0

   # Complexity Analysis
   radon==6.0.1

   # Utilities
   pyyaml==6.0.1
   ```
3. Commit with message: "Add evaluation tool dependencies"

**Deliverable**: `requirements-evaluation.txt` file created

**Approval**: 👤 User confirms dependency versions are acceptable

---

### Task 1.1.2: Update Main Requirements
**Owner**: 🤖 Claude | **Time**: 2 min | **Status**: PENDING

**Exact Steps**:
1. Open: `/Users/bledden/Documents/weavehacks-collaborative/requirements.txt`
2. Add line at end:
   ```
   # Enhanced Evaluation Tools
   -r requirements-evaluation.txt
   ```
3. Commit with message: "Link evaluation dependencies to main requirements"

**Deliverable**: Updated `requirements.txt`

**Approval**: None needed (automatic)

---

### Task 1.1.3: Install Dependencies
**Owner**: 🤖 Claude | **Time**: 2 min | **Status**: PENDING

**Exact Steps**:
1. Run command: `pip install -r requirements-evaluation.txt`
2. Verify installation: `bandit --version && pylint --version && flake8 --version && mypy --version && radon --version`
3. Take screenshot of versions

**Deliverable**: All tools installed and verified

**Approval**: 👤 User confirms installation succeeded

---

## Phase 2: Security Evaluator (Python - Bandit)

### Task 2.1: Create SecurityScore Dataclass
**Owner**: 🤖 Claude | **Time**: 10 min | **Status**: PENDING

**File**: `src/evaluation/security_evaluator.py` (new file)

**Exact Code**:
```python
"""
Security vulnerability detection using Bandit
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum


class Severity(Enum):
    """Security issue severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SecurityIssue:
    """Individual security vulnerability"""
    severity: Severity
    confidence: str  # HIGH, MEDIUM, LOW
    test_id: str  # e.g., B201, B301
    test_name: str  # e.g., "flask_debug_true"
    line_number: int
    line_range: List[int]
    code: str
    issue_text: str


@dataclass
class SecurityScore:
    """Security evaluation result"""
    overall: float  # 0.0 - 1.0
    safe: bool  # True if no critical/high issues
    critical_issues: List[SecurityIssue] = field(default_factory=list)
    high_issues: List[SecurityIssue] = field(default_factory=list)
    medium_issues: List[SecurityIssue] = field(default_factory=list)
    low_issues: List[SecurityIssue] = field(default_factory=list)
    total_issues: int = 0
    scanned_files: int = 1

    def __post_init__(self):
        """Calculate derived fields"""
        self.total_issues = (
            len(self.critical_issues) +
            len(self.high_issues) +
            len(self.medium_issues) +
            len(self.low_issues)
        )
```

**Deliverable**: Dataclass definitions in `security_evaluator.py`

**Approval**: 👤 User reviews dataclass structure

---

### Task 2.2: Implement Bandit Runner
**Owner**: 🤖 Claude | **Time**: 30 min | **Status**: PENDING

**File**: `src/evaluation/security_evaluator.py` (continue)

**Exact Code**:
```python
import tempfile
import os
import json
import subprocess
from pathlib import Path


class SecurityEvaluator:
    """Evaluates code for security vulnerabilities using Bandit"""

    def __init__(self, severity_threshold: str = "MEDIUM"):
        """
        Args:
            severity_threshold: Minimum severity to report (LOW, MEDIUM, HIGH)
        """
        self.severity_threshold = Severity[severity_threshold]

    def evaluate(self, code: str, language: str = "python") -> SecurityScore:
        """
        Evaluate code for security issues

        Args:
            code: Source code to analyze
            language: Programming language (only 'python' supported currently)

        Returns:
            SecurityScore with findings
        """
        if language != "python":
            # Return safe score for non-Python code (for now)
            return SecurityScore(overall=1.0, safe=True)

        return self._run_bandit(code)

    def _run_bandit(self, code: str) -> SecurityScore:
        """
        Run Bandit security scanner on Python code

        Args:
            code: Python source code

        Returns:
            SecurityScore with findings
        """
        # Create temporary file with code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            # Run bandit with JSON output
            result = subprocess.run(
                [
                    'bandit',
                    '-f', 'json',  # JSON format
                    '-ll',  # Report low and above
                    tmp_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse JSON output
            if result.stdout:
                bandit_output = json.loads(result.stdout)
            else:
                # No issues found or error
                return SecurityScore(overall=1.0, safe=True)

            # Extract issues
            return self._parse_bandit_output(bandit_output)

        except subprocess.TimeoutExpired:
            print("[WARNING] Bandit timed out after 30s")
            return SecurityScore(overall=0.5, safe=False)

        except Exception as e:
            print(f"[ERROR] Bandit failed: {e}")
            return SecurityScore(overall=0.5, safe=False)

        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

    def _parse_bandit_output(self, bandit_output: Dict[str, Any]) -> SecurityScore:
        """
        Parse Bandit JSON output into SecurityScore

        Args:
            bandit_output: Bandit JSON results

        Returns:
            SecurityScore
        """
        critical_issues = []
        high_issues = []
        medium_issues = []
        low_issues = []

        results = bandit_output.get('results', [])

        for issue in results:
            security_issue = SecurityIssue(
                severity=self._map_severity(issue['issue_severity']),
                confidence=issue['issue_confidence'],
                test_id=issue['test_id'],
                test_name=issue['test_name'],
                line_number=issue['line_number'],
                line_range=issue['line_range'],
                code=issue['code'],
                issue_text=issue['issue_text']
            )

            # Categorize by severity
            if security_issue.severity == Severity.CRITICAL:
                critical_issues.append(security_issue)
            elif security_issue.severity == Severity.HIGH:
                high_issues.append(security_issue)
            elif security_issue.severity == Severity.MEDIUM:
                medium_issues.append(security_issue)
            else:
                low_issues.append(security_issue)

        # Calculate overall score
        overall = self._calculate_security_score(
            critical_issues, high_issues, medium_issues, low_issues
        )

        # Determine if safe
        safe = len(critical_issues) == 0 and len(high_issues) == 0

        return SecurityScore(
            overall=overall,
            safe=safe,
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues
        )

    def _map_severity(self, bandit_severity: str) -> Severity:
        """Map Bandit severity to our Severity enum"""
        mapping = {
            'HIGH': Severity.CRITICAL,  # Treat HIGH as CRITICAL
            'MEDIUM': Severity.HIGH,    # Shift down
            'LOW': Severity.MEDIUM
        }
        return mapping.get(bandit_severity.upper(), Severity.LOW)

    def _calculate_security_score(
        self,
        critical: List[SecurityIssue],
        high: List[SecurityIssue],
        medium: List[SecurityIssue],
        low: List[SecurityIssue]
    ) -> float:
        """
        Calculate overall security score

        Scoring:
        - Start at 1.0
        - Critical issue: -0.4 each
        - High issue: -0.2 each
        - Medium issue: -0.1 each
        - Low issue: -0.05 each
        - Minimum score: 0.0

        Args:
            critical, high, medium, low: Lists of issues by severity

        Returns:
            Score from 0.0 to 1.0
        """
        score = 1.0

        score -= len(critical) * 0.4
        score -= len(high) * 0.2
        score -= len(medium) * 0.1
        score -= len(low) * 0.05

        return max(0.0, score)
```

**Deliverable**: Complete `SecurityEvaluator` class

**Approval**: 👤 User reviews scoring algorithm

---

### Task 2.3: Create Security Evaluator Tests
**Owner**: 🤖 Claude | **Time**: 30 min | **Status**: PENDING

**File**: `tests/evaluation/test_security_evaluator.py` (new file)

**Exact Code**:
```python
"""
Tests for SecurityEvaluator
"""
import pytest
from src.evaluation.security_evaluator import SecurityEvaluator, Severity


@pytest.fixture
def evaluator():
    return SecurityEvaluator(severity_threshold="MEDIUM")


def test_clean_code(evaluator):
    """Test that clean code gets high security score"""
    code = """
def add(a, b):
    return a + b

def multiply(x, y):
    result = x * y
    return result
"""
    score = evaluator.evaluate(code, "python")

    assert score.overall >= 0.9
    assert score.safe is True
    assert score.total_issues == 0


def test_eval_usage(evaluator):
    """Test that eval() usage is flagged"""
    code = """
def dangerous_function(user_input):
    result = eval(user_input)  # CRITICAL: eval() usage
    return result
"""
    score = evaluator.evaluate(code, "python")

    assert score.overall < 0.7
    assert score.safe is False
    assert len(score.critical_issues) > 0 or len(score.high_issues) > 0


def test_hardcoded_password(evaluator):
    """Test that hardcoded passwords are flagged"""
    code = """
import requests

def connect_to_api():
    password = "hardcoded_password_123"  # HIGH: hardcoded password
    response = requests.get("https://api.example.com", auth=("user", password))
    return response.json()
"""
    score = evaluator.evaluate(code, "python")

    assert score.overall < 0.8
    assert len(score.medium_issues) > 0 or len(score.high_issues) > 0


def test_sql_injection_risk(evaluator):
    """Test that SQL injection risks are flagged"""
    code = """
import sqlite3

def get_user(user_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # HIGH: SQL injection risk
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
"""
    score = evaluator.evaluate(code, "python")

    assert score.overall < 0.8
    # Should flag string formatting in SQL


def test_non_python_code(evaluator):
    """Test that non-Python code returns safe score"""
    code = """
function add(a, b) {
    return a + b;
}
"""
    score = evaluator.evaluate(code, "javascript")

    assert score.overall == 1.0
    assert score.safe is True


def test_multiple_issues(evaluator):
    """Test code with multiple security issues"""
    code = """
import pickle
import os

password = "admin123"  # Hardcoded password

def load_data(filename):
    # Unsafe pickle usage
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def execute_command(user_input):
    # Command injection risk
    os.system(user_input)
"""
    score = evaluator.evaluate(code, "python")

    assert score.overall < 0.5
    assert score.safe is False
    assert score.total_issues >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Deliverable**: Complete test suite for SecurityEvaluator

**Approval**: 👤 User runs tests, confirms they pass

---

## Phase 3: Static Analysis Evaluator

### Task 3.1: Create StaticAnalysisScore Dataclass
**Owner**: 🤖 Claude | **Time**: 15 min | **Status**: PENDING

**File**: `src/evaluation/static_analysis_evaluator.py` (new file)

**Exact Code**:
```python
"""
Static analysis using pylint, flake8, and mypy
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum
import tempfile
import os
import subprocess
import json
import re


class ViolationType(Enum):
    """Type of code violation"""
    ERROR = "error"
    WARNING = "warning"
    CONVENTION = "convention"
    REFACTOR = "refactor"
    INFO = "info"


@dataclass
class Violation:
    """Individual code violation"""
    tool: str  # pylint, flake8, mypy
    type: ViolationType
    code: str  # E.g., C0103, E501
    message: str
    line: int
    column: int = 0
    auto_fixable: bool = False


@dataclass
class StaticAnalysisScore:
    """Static analysis evaluation result"""
    overall: float  # 0.0 - 1.0
    pylint_score: float = 0.0
    flake8_violations: int = 0
    mypy_errors: int = 0
    violations: List[Violation] = field(default_factory=list)
    auto_fixable: List[Violation] = field(default_factory=list)
    total_violations: int = 0

    def __post_init__(self):
        """Calculate derived fields"""
        self.total_violations = len(self.violations)
        self.auto_fixable = [v for v in self.violations if v.auto_fixable]
```

**Deliverable**: Dataclass definitions

**Approval**: 👤 User reviews structure

---

### Task 3.2: Implement Pylint Runner
**Owner**: 🤖 Claude | **Time**: 30 min | **Status**: PENDING

**File**: `src/evaluation/static_analysis_evaluator.py` (continue)

**Exact Code**:
```python
class StaticAnalysisEvaluator:
    """Evaluates code quality using static analysis tools"""

    def __init__(self):
        """Initialize evaluator"""
        self.auto_fixable_codes = {
            'C0303',  # Trailing whitespace
            'C0304',  # Final newline missing
            'W0611',  # Unused import
            'C0411',  # Import order
        }

    def evaluate(self, code: str, language: str = "python") -> StaticAnalysisScore:
        """
        Evaluate code with static analysis tools

        Args:
            code: Source code
            language: Programming language

        Returns:
            StaticAnalysisScore with findings
        """
        if language != "python":
            return StaticAnalysisScore(overall=1.0)

        violations = []

        # Run pylint
        pylint_score, pylint_violations = self._run_pylint(code)
        violations.extend(pylint_violations)

        # Run flake8
        flake8_count, flake8_violations = self._run_flake8(code)
        violations.extend(flake8_violations)

        # Run mypy
        mypy_count, mypy_violations = self._run_mypy(code)
        violations.extend(mypy_violations)

        # Calculate overall score
        overall = self._calculate_overall_score(
            pylint_score, flake8_count, mypy_count
        )

        return StaticAnalysisScore(
            overall=overall,
            pylint_score=pylint_score,
            flake8_violations=flake8_count,
            mypy_errors=mypy_count,
            violations=violations
        )

    def _run_pylint(self, code: str) -> tuple[float, List[Violation]]:
        """
        Run pylint on code

        Returns:
            (score, violations)
        """
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    'pylint',
                    '--output-format=json',
                    '--score=yes',
                    tmp_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            violations = []
            score = 10.0  # Default if parsing fails

            # Parse JSON output
            if result.stdout:
                try:
                    pylint_output = json.loads(result.stdout)

                    for item in pylint_output:
                        if isinstance(item, dict) and 'type' in item:
                            violations.append(Violation(
                                tool='pylint',
                                type=self._map_pylint_type(item['type']),
                                code=item.get('message-id', ''),
                                message=item.get('message', ''),
                                line=item.get('line', 0),
                                column=item.get('column', 0),
                                auto_fixable=item.get('message-id', '') in self.auto_fixable_codes
                            ))
                except json.JSONDecodeError:
                    pass

            # Extract score from stderr (pylint outputs score there)
            if result.stderr:
                score_match = re.search(r'Your code has been rated at ([\d.]+)/10', result.stderr)
                if score_match:
                    score = float(score_match.group(1))

            return score, violations

        except subprocess.TimeoutExpired:
            print("[WARNING] Pylint timed out")
            return 5.0, []

        except Exception as e:
            print(f"[ERROR] Pylint failed: {e}")
            return 5.0, []

        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    def _run_flake8(self, code: str) -> tuple[int, List[Violation]]:
        """
        Run flake8 on code

        Returns:
            (violation_count, violations)
        """
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    'flake8',
                    '--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s',
                    '--max-line-length=100',
                    tmp_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            violations = []

            # Parse flake8 output (not JSON, line-based)
            for line in result.stdout.splitlines():
                match = re.match(
                    r'.*:(\d+):(\d+): ([A-Z]\d+) (.+)',
                    line
                )
                if match:
                    line_num, col, code, message = match.groups()
                    violations.append(Violation(
                        tool='flake8',
                        type=ViolationType.WARNING,
                        code=code,
                        message=message,
                        line=int(line_num),
                        column=int(col),
                        auto_fixable=code in ['W291', 'W293', 'F401']  # Whitespace, unused imports
                    ))

            return len(violations), violations

        except subprocess.TimeoutExpired:
            print("[WARNING] Flake8 timed out")
            return 0, []

        except Exception as e:
            print(f"[ERROR] Flake8 failed: {e}")
            return 0, []

        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    def _run_mypy(self, code: str) -> tuple[int, List[Violation]]:
        """
        Run mypy type checking

        Returns:
            (error_count, violations)
        """
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    'mypy',
                    '--no-error-summary',
                    '--show-error-codes',
                    tmp_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            violations = []

            # Parse mypy output
            for line in result.stdout.splitlines():
                match = re.match(
                    r'.*:(\d+): (\w+): (.+?) \[(.+?)\]',
                    line
                )
                if match:
                    line_num, severity, message, code = match.groups()
                    violations.append(Violation(
                        tool='mypy',
                        type=ViolationType.ERROR if severity == 'error' else ViolationType.WARNING,
                        code=code,
                        message=message,
                        line=int(line_num),
                        auto_fixable=False  # Type errors usually not auto-fixable
                    ))

            return len(violations), violations

        except subprocess.TimeoutExpired:
            print("[WARNING] Mypy timed out")
            return 0, []

        except Exception as e:
            print(f"[ERROR] Mypy failed: {e}")
            return 0, []

        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    def _map_pylint_type(self, pylint_type: str) -> ViolationType:
        """Map pylint message type to ViolationType"""
        mapping = {
            'error': ViolationType.ERROR,
            'warning': ViolationType.WARNING,
            'convention': ViolationType.CONVENTION,
            'refactor': ViolationType.REFACTOR,
            'info': ViolationType.INFO
        }
        return mapping.get(pylint_type.lower(), ViolationType.INFO)

    def _calculate_overall_score(
        self,
        pylint_score: float,
        flake8_count: int,
        mypy_count: int
    ) -> float:
        """
        Calculate overall static analysis score

        Weights:
        - Pylint score: 50%
        - Flake8 violations: 25%
        - Mypy errors: 25%

        Args:
            pylint_score: Score from 0-10
            flake8_count: Number of violations
            mypy_count: Number of errors

        Returns:
            Score from 0.0 to 1.0
        """
        # Normalize pylint (0-10 -> 0-1)
        pylint_normalized = pylint_score / 10.0

        # Penalize flake8 violations (each violation -0.05, min 0)
        flake8_score = max(0.0, 1.0 - (flake8_count * 0.05))

        # Penalize mypy errors (each error -0.1, min 0)
        mypy_score = max(0.0, 1.0 - (mypy_count * 0.1))

        # Weighted average
        overall = (
            pylint_normalized * 0.5 +
            flake8_score * 0.25 +
            mypy_score * 0.25
        )

        return round(overall, 3)
```

**Deliverable**: Complete `StaticAnalysisEvaluator` class

**Approval**: 👤 User reviews scoring weights and thresholds

---

### Task 3.3: Create Static Analysis Tests
**Owner**: 🤖 Claude | **Time**: 30 min | **Status**: PENDING

**File**: `tests/evaluation/test_static_analysis_evaluator.py` (new file)

**Exact Code** (I'll provide 5 key tests):
```python
"""
Tests for StaticAnalysisEvaluator
"""
import pytest
from src.evaluation.static_analysis_evaluator import StaticAnalysisEvaluator


@pytest.fixture
def evaluator():
    return StaticAnalysisEvaluator()


def test_clean_code(evaluator):
    """Test clean code gets high score"""
    code = """
\"\"\"Module docstring\"\"\"


def add_numbers(first: int, second: int) -> int:
    \"\"\"Add two numbers together.\"\"\"
    return first + second


def main() -> None:
    \"\"\"Main function.\"\"\"
    result = add_numbers(5, 3)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
"""
    score = evaluator.evaluate(code, "python")

    assert score.overall >= 0.8
    assert score.pylint_score >= 8.0


def test_unused_import(evaluator):
    """Test that unused imports are flagged"""
    code = """
import sys
import os  # Unused

def hello():
    print("Hello")
    sys.exit(0)
"""
    score = evaluator.evaluate(code, "python")

    assert score.flake8_violations > 0
    # Should find auto-fixable unused import
    assert any(v.auto_fixable for v in score.violations)


def test_long_lines(evaluator):
    """Test that long lines are flagged"""
    code = """
def function():
    very_long_line = "This is a very long line that exceeds the recommended line length and should be flagged by flake8"
    return very_long_line
"""
    score = evaluator.evaluate(code, "python")

    assert score.flake8_violations > 0


def test_type_errors(evaluator):
    """Test that type errors are caught by mypy"""
    code = """
def add(a: int, b: int) -> int:
    return a + b

result: int = add("hello", "world")  # Type error
"""
    score = evaluator.evaluate(code, "python")

    # Mypy should catch this
    assert score.mypy_errors > 0 or score.overall < 1.0


def test_poor_naming(evaluator):
    """Test that poor variable naming is flagged"""
    code = """
def f(x):  # Poor function name
    y = x + 1  # Poor variable name
    return y
"""
    score = evaluator.evaluate(code, "python")

    # Pylint should catch naming issues
    assert score.pylint_score < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Deliverable**: Test suite for StaticAnalysisEvaluator

**Approval**: 👤 User runs tests

---

## [TRUNCATED FOR LENGTH - CONTINUES WITH...]

## Phase 4: Complexity Evaluator (5 detailed tasks)
## Phase 5: LLM-as-Judge Evaluator (6 detailed tasks)
## Phase 6: Middleware System (8 detailed tasks)
## Phase 7: Configuration System (4 detailed tasks)
## Phase 8: Integration (7 detailed tasks)
## Phase 9: Testing (6 detailed tasks)
## Phase 10: Documentation (5 detailed tasks)

---

## Summary

**Total Tasks**: 52 granular tasks
**Estimated Total Time**: 35-45 hours
**Current Progress**: 0/52 (0%)

**Immediate Next Task**: Task 1.1.1 - Create `requirements-evaluation.txt`

---

## Task Tracking

Use this checklist to track progress:

- [ ] 1.1.1 Create requirements-evaluation.txt
- [ ] 1.1.2 Update main requirements.txt
- [ ] 1.1.3 Install dependencies
- [ ] 2.1 Create SecurityScore dataclass
- [ ] 2.2 Implement Bandit runner
- [ ] 2.3 Create security tests
- [ ] 3.1 Create StaticAnalysisScore dataclass
- [ ] 3.2 Implement Pylint runner
- [ ] 3.3 Create static analysis tests
- [ ] ... [49 more tasks]

**Ready to begin?** Confirm and I'll start with Task 1.1.1.
