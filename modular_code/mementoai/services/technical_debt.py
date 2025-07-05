# mementoai/services/technical_debt.py

import ast
from typing import List, Dict
import radon
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
import lizard
from core.models import TechnicalDebtMetrics

def detect_code_smells(code: str) -> List[str]:
    """
    Detects common code smells using AST.
    Handles parsing errors gracefully.
    """
    smells = []
    try:
        tree = ast.parse(code)

        # Long Method/Function detection
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count lines in function
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > 30:
                    smells.append(f"Long function: {node.name} ({func_lines} lines)")

                # Count parameters
                if len(node.args.args) > 5:
                    smells.append(f"Many parameters in {node.name} ({len(node.args.args)})")

        # Duplicate code detection (simplified)
        # In production, you'd use a more sophisticated approach
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        function_bodies = [ast.unparse(f.body) for f in functions]

        for i, body1 in enumerate(function_bodies):
            for j, body2 in enumerate(function_bodies[i+1:], i+1):
                if body1 == body2:
                    smells.append(f"Duplicate code between {functions[i].name} and {functions[j].name}")
                    break

    except SyntaxError as e: # Catch specific syntax errors from AST parsing
        print(f"Error detecting code smells (SyntaxError): {e}")
    except Exception as e:
        print(f"Error detecting code smells: {e}")

    return smells

def calculate_debt_score(metrics: TechnicalDebtMetrics) -> float:
    """
    Calculates a composite technical debt score (0-100).
    """
    score = 0

    # Weighted components
    if metrics.cyclomatic_complexity:
        avg_complexity = sum(metrics.cyclomatic_complexity.values()) / len(metrics.cyclomatic_complexity)
        score += min(avg_complexity * 2, 30)  # max 30 points for complexity

    score += (100 - metrics.maintainability_index) * 0.3  # max 30 points for MI

    score += metrics.duplication * 0.4  # max 40 points for duplication

    # Add points for each code smell
    score += min(len(metrics.code_smells) * 2, 20)  # max 20 points for smells

    return min(score, 100)

def analyze_technical_debt(code: str, language: str = 'python') -> TechnicalDebtMetrics:
    """
    Analyzes code for technical debt indicators using Radon and Lizard.
    Always returns a TechnicalDebtMetrics object, even if analysis fails.
    """
    metrics = TechnicalDebtMetrics() # Initialize an empty metrics object

    if not code:
        return metrics # Return empty metrics if no code

    try:
        # Radon analysis
        try:
            # Cyclomatic complexity
            # cc_visit expects valid Python code
            complexity_results = cc_visit(code)
            metrics.cyclomatic_complexity = {
                func.name: func.complexity
                for func in complexity_results
            }

            # Maintainability index
            # mi_visit expects valid Python code
            metrics.maintainability_index = mi_visit(code, multi=True)

            # Raw metrics (lines of code)
            raw_metrics = analyze(code)
            metrics.lines_of_code = raw_metrics.loc

        except SyntaxError as e: # Catch specific syntax errors from Radon
            print(f"Radon analysis error (SyntaxError): {e}")
            # Do not re-raise, let it continue with partial metrics
        except Exception as e:
            print(f"Radon analysis error: {e}")
            # Do not re-raise, let it continue with partial metrics

        # Lizard analysis (for duplication)
        try:
            lizard_analysis = lizard.analyze_file.analyze_source_code(
                "temp.py" if language == 'python' else "temp.file",
                code
            )
            if hasattr(lizard_analysis, 'average_duplication'):
                metrics.duplication = lizard_analysis.average_duplication * 100
            elif hasattr(lizard_analysis, 'global_stats') and hasattr(lizard_analysis.global_stats, 'duplicate_rate'):
                metrics.duplication = lizard_analysis.global_stats.duplicate_rate * 100

        except Exception as e:
            print(f"Lizard analysis error: {e}")
            # Do not re-raise, let it continue with partial metrics

        # Code smell detection (simplified)
        metrics.code_smells = detect_code_smells(code)

        # Calculate composite technical debt score
        metrics.technical_debt_score = calculate_debt_score(metrics)

    except Exception as e:
        print(f"Error in technical debt analysis: {e}")
        # If a top-level error occurs, metrics will remain the initialized empty object

    return metrics # Always return the metrics object

