# ast_utils.py
from tree_sitter import Node

def parse_code_to_ast(code_content: str, parser) -> Node | None:
    """Parses code string into a tree-sitter AST root node."""
    if not parser:
        return None
    try:
        tree = parser.parse(bytes(code_content, "utf8"))
        return tree.root_node
    except Exception as e:
        print(f"ERROR: AST parsing failed: {e}")
        return None

def extract_python_definitions(root_node: Node) -> set:
    """Extracts function and class names from a Python AST."""
    definitions = set()
    if not root_node:
        return definitions

    # Tree-sitter queries are powerful for this.
    # This is a simplified direct traversal for common cases.
    query_str_functions = "(function_definition name: (identifier) @func_name)"
    query_str_classes = "(class_definition name: (identifier) @class_name)"

    try:
        ts_lang = root_node.tree.language # Get the language from the tree
        
        function_query = ts_lang.query(query_str_functions)
        class_query = ts_lang.query(query_str_classes)

        for capture, name in function_query.captures(root_node):
            if name == "func_name":
                definitions.add(f"Function: {capture.text.decode('utf8')}")
        
        for capture, name in class_query.captures(root_node):
            if name == "class_name":
                definitions.add(f"Class: {capture.text.decode('utf8')}")
    except Exception as e:
        print(f"ERROR extracting Python definitions: {e}")
    return definitions

def extract_javascript_definitions(root_node: Node) -> set:
    """Extracts function and class names from a JavaScript AST."""
    definitions = set()
    if not root_node:
        return definitions

    # Queries for JavaScript (may need adjustment based on JS flavor/grammar specifics)
    query_str_functions = """
    [
      (function_declaration name: (identifier) @func_name)
      (arrow_function (identifier) @func_name) ;; common for const myFunc = () => ...
      (method_definition name: (property_identifier) @func_name)
    ]
    """
    query_str_classes = "(class_declaration name: (identifier) @class_name)"
    
    try:
        ts_lang = root_node.tree.language
        function_query = ts_lang.query(query_str_functions)
        class_query = ts_lang.query(query_str_classes)

        for capture, name in function_query.captures(root_node):
            if name == "func_name":
                definitions.add(f"Function: {capture.text.decode('utf8')}")
        
        for capture, name in class_query.captures(root_node):
            if name == "class_name":
                definitions.add(f"Class: {capture.text.decode('utf8')}")
    except Exception as e:
        print(f"ERROR extracting JavaScript definitions: {e}")

    return definitions


def compare_asts_summary(old_content: str | None, new_content: str | None, language: str, parser) -> list[str]:
    """
    Compares two versions of code content and returns a summary of AST changes.
    Returns a list of strings describing changes.
    """
    if not parser:
        return ["AST parser not available for this language."]

    old_ast_root = parse_code_to_ast(old_content, parser) if old_content else None
    new_ast_root = parse_code_to_ast(new_content, parser) if new_content else None

    if not old_ast_root and not new_ast_root:
        return ["No parsable content provided."]
    if not new_ast_root and old_ast_root:
        return ["Content deleted or became unparsable."]
    if not old_ast_root and new_ast_root:
        return ["Content added and parsed."] # Could list all new defs here

    old_defs = set()
    new_defs = set()

    if language == "python":
        if old_ast_root: old_defs = extract_python_definitions(old_ast_root)
        if new_ast_root: new_defs = extract_python_definitions(new_ast_root)
    elif language == "javascript":
        if old_ast_root: old_defs = extract_javascript_definitions(old_ast_root)
        if new_ast_root: new_defs = extract_javascript_definitions(new_ast_root)
    else:
        return [f"AST comparison not implemented for language: {language}"]

    changes_summary = []
    added = new_defs - old_defs
    removed = old_defs - new_defs
    
    for item in added: changes_summary.append(f"Added: {item}")
    for item in removed: changes_summary.append(f"Removed: {item}")

    # For potentially modified items (same name, content might differ)
    # This is a simplification; true modification detection is much harder.
    # We just check if a definition with the same "name" exists in both.
    # A more robust way would involve hashing node content or deeper structural comparison.
    potentially_modified = old_defs.intersection(new_defs)
    if potentially_modified:
        # Crude check: if the string representation of the whole file changed, assume modified elements are possible
        if old_content != new_content:
            for item in potentially_modified:
                # We can't easily tell *if* it was modified without deeper diffing
                # So, we just note it was present in both and the file changed.
                changes_summary.append(f"Present in both versions (may be modified): {item}")


    if not changes_summary and old_content != new_content:
        changes_summary.append("Code changed, but no top-level structural definition changes detected (e.g., comments, internal logic).")
    elif not changes_summary and old_content == new_content:
        changes_summary.append("No textual or structural changes detected.")

    return changes_summary if changes_summary else ["No significant structural changes detected."]