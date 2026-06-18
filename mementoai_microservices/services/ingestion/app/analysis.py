from __future__ import annotations

import ast
from typing import Dict, List, Tuple

from shared.models import FunctionChange, TechnicalDebtSnapshot


def analyze_code_changes(diff_text: str) -> List[FunctionChange]:
    old_code, new_code = extract_code_from_diff(diff_text)
    old_functions = _parse_functions(old_code)
    new_functions = _parse_functions(new_code)
    changes: List[FunctionChange] = []

    old_names = set(old_functions.keys())
    new_names = set(new_functions.keys())

    for name in sorted(new_names - old_names):
        changes.append(FunctionChange(name=name, change_type="added"))

    for name in sorted(old_names - new_names):
        changes.append(FunctionChange(name=name, change_type="removed"))

    for name in sorted(old_names & new_names):
        if old_functions[name] != new_functions[name]:
            old_complexity = _complexity(old_functions[name])
            new_complexity = _complexity(new_functions[name])
            changes.append(
                FunctionChange(
                    name=name,
                    change_type="modified",
                    complexity_change=new_complexity - old_complexity,
                )
            )

    return changes


def analyze_technical_debt(code: str) -> TechnicalDebtSnapshot:
    if not code:
        return TechnicalDebtSnapshot()

    functions = _parse_functions(code)
    complexities = {name: _complexity(body) for name, body in functions.items()}
    smells = _detect_code_smells(code)
    avg_complexity = sum(complexities.values()) / len(complexities) if complexities else 0.0
    lines_of_code = len(code.splitlines())
    maintainability = max(0.0, 100.0 - (avg_complexity * 4) - (len(smells) * 2))
    duplication = _duplicate_function_ratio(functions)
    score = min(
        100.0,
        (avg_complexity * 2)
        + ((100.0 - maintainability) * 0.3)
        + (duplication * 0.4)
        + (len(smells) * 2),
    )

    return TechnicalDebtSnapshot(
        code_smells=smells,
        cyclomatic_complexity=complexities,
        maintainability_index=maintainability,
        duplication=duplication,
        lines_of_code=lines_of_code,
        technical_debt_score=score,
        avg_complexity=avg_complexity,
    )


def extract_code_from_diff(diff_text: str) -> Tuple[str, str]:
    old_lines: List[str] = []
    new_lines: List[str] = []

    for line in diff_text.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith("+"):
            new_lines.append(line[1:])
        elif not line.startswith("@"):
            old_lines.append(line)
            new_lines.append(line)

    return "\n".join(old_lines), "\n".join(new_lines)


def _parse_functions(code: str) -> Dict[str, str]:
    if not code.strip():
        return {}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    functions: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = ast.unparse(node)

    return functions


def _complexity(code: str) -> int:
    if not code.strip():
        return 0

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0

    score = 1
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.Try,
                ast.With,
                ast.BoolOp,
                ast.comprehension,
            ),
        ):
            score += 1

    return score


def _detect_code_smells(code: str) -> List[str]:
    smells: List[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return smells

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_lines = (node.end_lineno or node.lineno) - node.lineno if hasattr(node, "end_lineno") else 0
            if func_lines > 30:
                smells.append(f"Long function: {node.name} ({func_lines} lines)")
            if len(node.args.args) > 5:
                smells.append(f"Many parameters in {node.name} ({len(node.args.args)})")

    return smells


def _duplicate_function_ratio(functions: Dict[str, str]) -> float:
    if len(functions) < 2:
        return 0.0

    bodies = list(functions.values())
    duplicates = 0
    seen = set()

    for body in bodies:
        if body in seen:
            duplicates += 1
        seen.add(body)

    return (duplicates / len(bodies)) * 100.0