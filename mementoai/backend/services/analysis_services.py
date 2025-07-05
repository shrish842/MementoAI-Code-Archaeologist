import ast
import json
import logging
from venv import logger
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
import lizard
from typing import Dict, List, Optional
from backend.models.schemas import CodeChangeAnalysis, TechnicalDebtMetrics

def analyze_code_changes(diff_text: str) -> CodeChangeAnalysis:
    """Analyze code changes using AST to detect function modifications."""
    analysis = CodeChangeAnalysis()
    
    if not diff_text or "Error" in diff_text:
        return analysis
    
    try:
        # Parse the diff to get old and new code sections
        old_code = []
        new_code = []
        
        for line in diff_text.split('\n'):
            if line.startswith('-') and not line.startswith('---'):
                old_code.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                new_code.append(line[1:])
        
        # Parse functions from old and new code
        old_functions = parse_functions('\n'.join(old_code))
        new_functions = parse_functions('\n'.join(new_code))
        
        # Compare functions
        old_func_names = {f['name'] for f in old_functions}
        new_func_names = {f['name'] for f in new_functions}
        
        analysis.functions_added = list(new_func_names - old_func_names)
        analysis.functions_removed = list(old_func_names - new_func_names)
        
        # Find modified functions
        modified = []
        complexity_changes = {}
        
        for new_func in new_functions:
            if new_func['name'] in old_func_names:
                old_func = next(f for f in old_functions if f['name'] == new_func['name'])
                if new_func['body'] != old_func['body']:
                    modified.append(new_func['name'])
                    # Calculate complexity change
                    old_complexity = calculate_complexity(old_func['body'])
                    new_complexity = calculate_complexity(new_func['body'])
                    if old_complexity != new_complexity:
                        complexity_changes[new_func['name']] = new_complexity - old_complexity
        
        analysis.functions_modified = modified
        analysis.complexity_changes = complexity_changes
        
    except Exception as e:
        print(f"Error during AST analysis: {e}")
    
    return analysis

def parse_functions(code: str) -> List[Dict]:
    """Parse functions from code using AST."""
    if not code:
        return []
    
    try:
        if "<?php" in code or "<html" in code or "package " in code:
            return []
        tree = ast.parse(code)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'body': ast.unparse(node.body),
                    'args': [arg.arg for arg in node.args.args]
                })
        
        return functions
    except SyntaxError as e:
        logger.warning(f"Skipping file due to syntax error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing functions: {e}")
        return []

def calculate_complexity(code: str) -> int:
    """Calculate simple complexity metric (number of control flow statements)."""
    try:
        tree = ast.parse(code)
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.Call):
                complexity += 0.5  # Weight function calls less than control flow
        
        return int(complexity)
    except Exception as e:
        print(f"Error calculating complexity: {e}")
        return 0


def analyze_technical_debt(code: str, language: str = 'python') -> TechnicalDebtMetrics:
    """Analyze code for technical debt indicators."""
    metrics = TechnicalDebtMetrics()
    
    if not code:
        return metrics
    
    try:
        # Radon analysis
        try:
            # Cyclomatic complexity
            complexity_results = cc_visit(code)
            metrics.cyclomatic_complexity = {
                func.name: func.complexity 
                for func in complexity_results
            }
            
            # Maintainability index
            metrics.maintainability_index = mi_visit(code, multi=True)
            
            # Raw metrics (lines of code)
            raw_metrics = analyze(code)
            metrics.lines_of_code = raw_metrics.loc
            
        except Exception as e:
            print(f"Radon analysis error: {e}")
        
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
                
        # Code smell detection (simplified)
        metrics.code_smells = detect_code_smells(code)
        
        # Calculate composite technical debt score
        metrics.technical_debt_score = calculate_debt_score(metrics)
        
    except Exception as e:
        print(f"Error in technical debt analysis: {e}")
    
    return metrics

def detect_code_smells(code: str) -> List[str]:
    """Detect common code smells."""
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
        
    except Exception as e:
        print(f"Error detecting code smells: {e}")
    
    
    return smells

def calculate_debt_score(metrics: TechnicalDebtMetrics) -> float:
    """Calculate composite technical debt score (0-100)."""
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
