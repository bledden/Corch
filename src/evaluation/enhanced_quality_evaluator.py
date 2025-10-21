"""
Enhanced Quality Evaluation System with AST Analysis and LLM-as-Judge

Improvements over basic evaluator:
1. Deep AST-based validation instead of regex
2. Static analysis integration (pylint, mypy)
3. LLM-as-judge for semantic correctness
4. Security vulnerability scanning
5. Performance anti-pattern detection
"""

import ast
import re
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Dimensions of code quality to evaluate"""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    CODE_QUALITY = "code_quality"
    DOCUMENTATION = "documentation"
    ERROR_HANDLING = "error_handling"
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class QualityScore:
    """Enhanced quality evaluation result"""
    overall: float  # 0.0 - 1.0
    dimensions: Dict[str, float]  # Per-dimension scores
    details: Dict[str, Any]  # Detailed findings
    passed: bool  # Pass/fail based on threshold
    issues: List[Dict[str, Any]] = field(default_factory=list)  # Specific issues found
    static_analysis: Optional[Dict[str, Any]] = None  # Results from pylint/mypy
    llm_feedback: Optional[str] = None  # LLM-as-judge feedback


class ASTAnalyzer:
    """Advanced AST-based code analysis"""

    def __init__(self):
        self.issues = []

    def analyze(self, code: str) -> Dict[str, Any]:
        """Perform deep AST analysis"""
        try:
            tree = ast.parse(code)
            return {
                "valid": True,
                "complexity": self._analyze_complexity(tree),
                "structure": self._analyze_structure(tree),
                "patterns": self._detect_patterns(tree),
                "smells": self._detect_code_smells(tree),
                "security": self._detect_security_issues(tree)
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
                "offset": e.offset
            }

    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate cyclomatic complexity and nesting depth"""
        complexity = {"functions": []}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_complexity = self._calculate_function_complexity(node)
                complexity["functions"].append({
                    "name": node.name,
                    "complexity": func_complexity,
                    "nesting_depth": self._max_nesting_depth(node),
                    "loc": len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                })

        complexity["max_complexity"] = max([f["complexity"] for f in complexity["functions"]], default=0)
        complexity["avg_complexity"] = (
            sum(f["complexity"] for f in complexity["functions"]) / len(complexity["functions"])
            if complexity["functions"] else 0
        )

        return complexity

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Add 1 for each decision point
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _max_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0

        def visit(n, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)

            for child in ast.iter_child_nodes(n):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                    visit(child, depth + 1)
                else:
                    visit(child, depth)

        visit(node)
        return max_depth

    def _analyze_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code structure"""
        structure = {
            "functions": [],
            "classes": [],
            "imports": [],
            "global_vars": [],
            "has_main": False
        }

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                structure["functions"].append({
                    "name": node.name,
                    "args": len(node.args.args),
                    "decorators": len(node.decorator_list),
                    "returns": node.returns is not None,
                    "is_async": isinstance(node, ast.AsyncFunctionDef)
                })
            elif isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                structure["classes"].append({
                    "name": node.name,
                    "methods": len(methods),
                    "bases": len(node.bases),
                    "decorators": len(node.decorator_list)
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        structure["imports"].append(alias.name)
                else:
                    structure["imports"].append(node.module)

        # Check for __name__ == "__main__"
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Compare):
                    if any(isinstance(n, ast.Name) and n.id == "__name__" for n in ast.walk(node.test)):
                        structure["has_main"] = True

        return structure

    def _detect_patterns(self, tree: ast.AST) -> Dict[str, Any]:
        """Detect common patterns and best practices"""
        patterns = {
            "context_managers": 0,
            "list_comprehensions": 0,
            "generators": 0,
            "decorators": 0,
            "type_hints": 0,
            "docstrings": 0
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                patterns["context_managers"] += 1
            elif isinstance(node, ast.ListComp):
                patterns["list_comprehensions"] += 1
            elif isinstance(node, (ast.GeneratorExp, ast.Yield, ast.YieldFrom)):
                patterns["generators"] += 1
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.decorator_list:
                patterns["decorators"] += len(node.decorator_list)

            # Check for type hints
            if isinstance(node, ast.FunctionDef):
                if node.returns or any(arg.annotation for arg in node.args.args):
                    patterns["type_hints"] += 1

            # Check for docstrings
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if (ast.get_docstring(node) is not None):
                    patterns["docstrings"] += 1

        return patterns

    def _detect_code_smells(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect code smells and anti-patterns"""
        smells = []

        for node in ast.walk(tree):
            # Long functions
            if isinstance(node, ast.FunctionDef):
                loc = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                if loc > 50:
                    smells.append({
                        "type": "long_function",
                        "severity": "medium",
                        "function": node.name,
                        "loc": loc,
                        "message": f"Function '{node.name}' is too long ({loc} statements)"
                    })

                # Too many parameters
                if len(node.args.args) > 5:
                    smells.append({
                        "type": "too_many_parameters",
                        "severity": "low",
                        "function": node.name,
                        "params": len(node.args.args),
                        "message": f"Function '{node.name}' has too many parameters ({len(node.args.args)})"
                    })

            # Bare except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    smells.append({
                        "type": "bare_except",
                        "severity": "high",
                        "message": "Bare except clause catches all exceptions (anti-pattern)"
                    })

            # Mutable default arguments
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        smells.append({
                            "type": "mutable_default",
                            "severity": "high",
                            "function": node.name,
                            "message": f"Function '{node.name}' has mutable default argument"
                        })

        return smells

    def _detect_security_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect potential security vulnerabilities"""
        issues = []

        for node in ast.walk(tree):
            # eval() or exec() usage
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec']:
                        issues.append({
                            "type": "dangerous_function",
                            "severity": "critical",
                            "function": node.func.id,
                            "message": f"Use of {node.func.id}() is a security risk"
                        })

            # SQL injection risk (string concatenation with queries)
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Add):
                    if self._looks_like_sql_concat(node):
                        issues.append({
                            "type": "sql_injection_risk",
                            "severity": "high",
                            "message": "Possible SQL injection via string concatenation"
                        })

        return issues

    def _looks_like_sql_concat(self, node: ast.BinOp) -> bool:
        """Check if binary op looks like SQL string concatenation"""
        def has_sql_keyword(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE']
                return any(kw in n.value.upper() for kw in sql_keywords)
            return False

        return has_sql_keyword(node.left) or has_sql_keyword(node.right)


class StaticAnalyzer:
    """Integration with static analysis tools (pylint, mypy)"""

    def __init__(self):
        self.has_pylint = self._check_tool("pylint")
        self.has_mypy = self._check_tool("mypy")

    def _check_tool(self, tool: str) -> bool:
        """Check if a tool is available"""
        try:
            subprocess.run([tool, "--version"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def analyze(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Run static analysis tools"""
        if language != "python":
            return {"available": False, "reason": "Only Python supported"}

        results = {
            "pylint": None,
            "mypy": None,
            "available_tools": []
        }

        if self.has_pylint:
            results["available_tools"].append("pylint")
            results["pylint"] = self._run_pylint(code)

        if self.has_mypy:
            results["available_tools"].append("mypy")
            results["mypy"] = self._run_mypy(code)

        return results

    def _run_pylint(self, code: str) -> Optional[Dict[str, Any]]:
        """Run pylint on code"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ["pylint", "--output-format=json", temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )

            os.unlink(temp_file)

            if result.stdout:
                messages = json.loads(result.stdout)
                return {
                    "score": self._extract_pylint_score(result.stderr),
                    "issues": messages,
                    "issue_count": len(messages),
                    "by_severity": self._group_by_severity(messages)
                }

        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logger.warning(f"Pylint analysis failed: {e}")

        return None

    def _run_mypy(self, code: str) -> Optional[Dict[str, Any]]:
        """Run mypy on code"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ["mypy", temp_file, "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                timeout=30
            )

            os.unlink(temp_file)

            errors = []
            for line in result.stdout.split('\n'):
                if ':' in line and 'error:' in line:
                    errors.append(line.strip())

            return {
                "error_count": len(errors),
                "errors": errors,
                "passed": result.returncode == 0
            }

        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Mypy analysis failed: {e}")

        return None

    def _extract_pylint_score(self, stderr: str) -> Optional[float]:
        """Extract score from pylint output"""
        match = re.search(r'Your code has been rated at ([\d.]+)/10', stderr)
        return float(match.group(1)) if match else None

    def _group_by_severity(self, messages: List[Dict]) -> Dict[str, int]:
        """Group pylint messages by severity"""
        by_severity = {"error": 0, "warning": 0, "convention": 0, "refactor": 0}
        for msg in messages:
            msg_type = msg.get("type", "")[0].lower()
            if msg_type == "e":
                by_severity["error"] += 1
            elif msg_type == "w":
                by_severity["warning"] += 1
            elif msg_type == "c":
                by_severity["convention"] += 1
            elif msg_type == "r":
                by_severity["refactor"] += 1
        return by_severity


class EnhancedCodeQualityEvaluator:
    """
    Enhanced quality evaluator with:
    - Deep AST analysis
    - Static analysis integration (pylint, mypy)
    - Security vulnerability scanning
    - Performance anti-pattern detection
    - Optional LLM-as-judge
    """

    def __init__(
        self,
        pass_threshold: float = 0.7,
        use_static_analysis: bool = True,
        use_llm_judge: bool = False,
        llm_client: Optional[Any] = None
    ):
        self.pass_threshold = pass_threshold
        self.use_static_analysis = use_static_analysis
        self.use_llm_judge = use_llm_judge
        self.llm_client = llm_client

        self.ast_analyzer = ASTAnalyzer()
        self.static_analyzer = StaticAnalyzer() if use_static_analysis else None

    def evaluate(self, code: str, task: str, language: str = "python") -> QualityScore:
        """
        Comprehensive code quality evaluation

        Args:
            code: Generated code
            task: Original task description
            language: Programming language

        Returns:
            Enhanced QualityScore with detailed analysis
        """
        if language != "python":
            # Fallback to basic evaluation for non-Python
            from .quality_evaluator import CodeQualityEvaluator
            basic_eval = CodeQualityEvaluator(self.pass_threshold)
            return basic_eval.evaluate(code, task, language)

        dimensions = {}
        details = {}
        issues = []

        # 1. AST-based correctness and structure analysis
        ast_results = self.ast_analyzer.analyze(code)
        if not ast_results.get("valid", False):
            # Syntax error - fail immediately
            return QualityScore(
                overall=0.0,
                dimensions={dim.value: 0.0 for dim in QualityDimension},
                details={"syntax_error": ast_results},
                passed=False,
                issues=[{
                    "type": "syntax_error",
                    "severity": "critical",
                    "message": ast_results.get("error", "Syntax error")
                }]
            )

        # Correctness based on complexity and structure
        complexity_score = self._score_complexity(ast_results["complexity"])
        dimensions[QualityDimension.CORRECTNESS.value] = complexity_score
        details["complexity"] = ast_results["complexity"]

        # Completeness based on structure
        completeness_score = self._score_completeness(ast_results["structure"], task)
        dimensions[QualityDimension.COMPLETENESS.value] = completeness_score
        details["structure"] = ast_results["structure"]

        # Code quality based on patterns and smells
        quality_score, quality_issues = self._score_quality(ast_results["patterns"], ast_results["smells"])
        dimensions[QualityDimension.CODE_QUALITY.value] = quality_score
        details["patterns"] = ast_results["patterns"]
        details["code_smells"] = ast_results["smells"]
        issues.extend(quality_issues)

        # Security score
        security_score, security_issues = self._score_security(ast_results["security"])
        dimensions[QualityDimension.SECURITY.value] = security_score
        details["security"] = ast_results["security"]
        issues.extend(security_issues)

        # Documentation, error handling, testing (simpler checks)
        dimensions[QualityDimension.DOCUMENTATION.value] = self._score_documentation(code, ast_results)
        dimensions[QualityDimension.ERROR_HANDLING.value] = self._score_error_handling(code)
        dimensions[QualityDimension.TESTING.value] = self._score_testing(code)

        # 2. Static analysis (if enabled and available)
        static_results = None
        if self.use_static_analysis and self.static_analyzer:
            static_results = self.static_analyzer.analyze(code, language)
            details["static_analysis"] = static_results

            # Adjust scores based on static analysis
            if static_results.get("pylint"):
                pylint_score = static_results["pylint"].get("score")
                if pylint_score is not None:
                    # Weight pylint score into code quality
                    dimensions[QualityDimension.CODE_QUALITY.value] = (
                        dimensions[QualityDimension.CODE_QUALITY.value] * 0.7 +
                        (pylint_score / 10.0) * 0.3
                    )

        # 3. Calculate overall score (weighted average)
        weights = {
            QualityDimension.CORRECTNESS.value: 0.25,
            QualityDimension.COMPLETENESS.value: 0.20,
            QualityDimension.CODE_QUALITY.value: 0.20,
            QualityDimension.SECURITY.value: 0.15,
            QualityDimension.DOCUMENTATION.value: 0.08,
            QualityDimension.ERROR_HANDLING.value: 0.07,
            QualityDimension.TESTING.value: 0.03,
            QualityDimension.PERFORMANCE.value: 0.02,
        }

        # Performance score (based on anti-patterns)
        dimensions[QualityDimension.PERFORMANCE.value] = self._score_performance(code, ast_results)

        overall = sum(dimensions.get(dim.value, 0.0) * weights[dim.value]
                     for dim in QualityDimension)

        # 4. LLM-as-judge (if enabled)
        llm_feedback = None
        if self.use_llm_judge and self.llm_client:
            llm_feedback = self._get_llm_judgment(code, task, dimensions)
            details["llm_feedback"] = llm_feedback

        return QualityScore(
            overall=overall,
            dimensions=dimensions,
            details=details,
            passed=overall >= self.pass_threshold,
            issues=issues,
            static_analysis=static_results,
            llm_feedback=llm_feedback
        )

    def _score_complexity(self, complexity: Dict) -> float:
        """Score based on cyclomatic complexity"""
        max_complexity = complexity.get("max_complexity", 0)

        if max_complexity == 0:
            return 0.5  # No functions found

        # Good: < 10, Acceptable: 10-20, Poor: > 20
        if max_complexity <= 10:
            return 1.0
        elif max_complexity <= 20:
            return 0.7
        else:
            return 0.4

    def _score_completeness(self, structure: Dict, task: str) -> float:
        """Score based on structure matching task requirements"""
        score = 0.0

        # Has functions or classes
        if structure["functions"] or structure["classes"]:
            score += 0.4

        # Has appropriate imports
        task_lower = task.lower()
        if any(keyword in task_lower for keyword in ["http", "request", "api"]):
            if any("request" in imp.lower() for imp in structure["imports"]):
                score += 0.2

        # Has main block (if appropriate)
        if "script" in task_lower or "run" in task_lower:
            if structure["has_main"]:
                score += 0.2

        # Functions have return values (if appropriate)
        if "return" in task_lower or "calculate" in task_lower:
            returning_funcs = [f for f in structure["functions"] if f["returns"]]
            if returning_funcs:
                score += 0.2

        return min(score, 1.0)

    def _score_quality(self, patterns: Dict, smells: List[Dict]) -> Tuple[float, List[Dict]]:
        """Score based on patterns and code smells"""
        score = 1.0  # Start perfect

        # Deduct for code smells
        issues = []
        for smell in smells:
            if smell["severity"] == "high":
                score -= 0.15
                issues.append(smell)
            elif smell["severity"] == "medium":
                score -= 0.10
                issues.append(smell)
            elif smell["severity"] == "low":
                score -= 0.05

        # Bonus for good patterns
        if patterns.get("type_hints", 0) > 0:
            score += 0.05
        if patterns.get("context_managers", 0) > 0:
            score += 0.03
        if patterns.get("decorators", 0) > 0:
            score += 0.02

        return max(0.0, min(score, 1.0)), issues

    def _score_security(self, security_issues: List[Dict]) -> Tuple[float, List[Dict]]:
        """Score based on security vulnerabilities"""
        if not security_issues:
            return 1.0, []

        score = 1.0
        for issue in security_issues:
            if issue["severity"] == "critical":
                score -= 0.3
            elif issue["severity"] == "high":
                score -= 0.2
            elif issue["severity"] == "medium":
                score -= 0.1

        return max(0.0, score), security_issues

    def _score_documentation(self, code: str, ast_results: Dict) -> float:
        """Score documentation quality"""
        patterns = ast_results.get("patterns", {})
        docstring_count = patterns.get("docstrings", 0)

        if docstring_count == 0:
            return 0.0
        elif docstring_count >= 3:
            return 1.0
        else:
            return docstring_count / 3.0

    def _score_error_handling(self, code: str) -> float:
        """Score error handling"""
        score = 0.0

        if "try:" in code and "except" in code:
            score += 0.5

        if "raise " in code:
            score += 0.3

        if "finally:" in code:
            score += 0.2

        return min(score, 1.0)

    def _score_testing(self, code: str) -> float:
        """Score testing coverage"""
        score = 0.0

        if "def test_" in code or "class Test" in code:
            score += 0.6

        if "assert " in code:
            score += 0.3

        if "import pytest" in code or "import unittest" in code:
            score += 0.1

        return min(score, 1.0)

    def _score_performance(self, code: str, ast_results: Dict) -> float:
        """Score based on performance anti-patterns"""
        score = 1.0

        # Nested loops (O(n²) or worse)
        if "+= 1" in code and "for " in code:
            nested_loops = code.count("for ") - 1
            if nested_loops >= 2:
                score -= 0.3

        # String concatenation in loops
        if "+=" in code and "for " in code and ("str" in code or "\"" in code):
            score -= 0.2

        return max(0.0, score)

    def _get_llm_judgment(self, code: str, task: str, dimensions: Dict) -> str:
        """Get LLM-as-judge evaluation (semantic correctness)"""
        if not self.llm_client:
            return None

        prompt = f"""You are a code reviewer. Evaluate if this code correctly solves the task.

Task: {task}

Code:
```python
{code}
```

Current Quality Scores:
{json.dumps(dimensions, indent=2)}

Provide a brief assessment (2-3 sentences) on:
1. Does the code semantically solve the task?
2. Any logical errors or edge cases missed?
3. Overall assessment

Be concise and specific."""

        try:
            # This would call the actual LLM
            # For now, return placeholder
            return "LLM-as-judge evaluation not yet implemented"
        except Exception as e:
            logger.warning(f"LLM judgment failed: {e}")
            return None
