import ast
from typing import Dict, List
from core.models import CodeChangeAnalysis

def parse_functions(code: str) -> List[Dict]:
    """
    Parses functions from code using AST.
    Returns a list of dictionaries, each containing 'name', 'body', and 'args'.
    """
    if not code:
        return []

    try:
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
        # More specific error handling
        print(f"Syntax error parsing functions: {e} at line {e.lineno}")
        return []
    except Exception as e:
        print(f"Unexpected parsing error: {e}")
        return []
    
def calculate_complexity(code: str) -> int:
    """
    Calculates a simple complexity metric (number of control flow statements).
    """
    try:
        tree = ast.parse(code)
        complexity = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.Call):
                complexity += 0.5  # Weight function calls less than control flow

        return int(complexity)
    except SyntaxError as e: # Catch specific syntax errors
        print(f"Error calculating complexity (SyntaxError): {e}")
        return 0
    except Exception as e:
        print(f"Error calculating complexity: {e}")
        return 0

def analyze_code_changes(diff_text: str) -> CodeChangeAnalysis:
    """
    Analyzes code changes using AST to detect function modifications.
    """
    analysis = CodeChangeAnalysis()

    if not diff_text or "Error" in diff_text:
        return analysis

    try:
        # Parse the diff to get old and new code sections
        old_code_lines = []
        new_code_lines = []

        for line in diff_text.split('\n'):
            if line.startswith('-') and not line.startswith('---'):
                old_code_lines.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                new_code_lines.append(line[1:])
            elif not line.startswith('@'): # Context lines
                old_code_lines.append(line)
                new_code_lines.append(line)

        # Parse functions from old and new code
        old_functions = parse_functions('\n'.join(old_code_lines))
        new_functions = parse_functions('\n'.join(new_code_lines))

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

