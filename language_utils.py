# language_utils.py
import os
from tree_sitter import Language, Parser
import platform # To potentially adjust .so/.dll naming if needed

# --- Configuration ---
# Absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GRAMMAR_BASE_PATH = os.path.join(PROJECT_ROOT, "grammars")

# So file for the combined library - adjust extension for OS
if platform.system() == "Windows":
    LANG_LIB_FILENAME = "languages.dll"
else:
    LANG_LIB_FILENAME = "languages.so"
LANG_SO_FILE = os.path.join(GRAMMAR_BASE_PATH, LANG_LIB_FILENAME)


# Mapping from our internal language name to the tree-sitter grammar directory and language name
# The second element of the tuple is the name you'd use in tree-sitter queries if needed.
LANGUAGE_DEFINITIONS = {
    "python": (os.path.join(GRAMMAR_BASE_PATH, "python"), "python"),
    "javascript": (os.path.join(GRAMMAR_BASE_PATH, "javascript"), "javascript"),
    # "java": (os.path.join(GRAMMAR_BASE_PATH, "java"), "java"), # Example
    # "go": (os.path.join(GRAMMAR_BASE_PATH, "go"), "go"),       # Example
}

# Mapping from file extensions to our internal language names
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    # ".java": "java",
    # ".go": "go",
}

# --- Global Loaded Languages ---
LOADED_LANGUAGES = {} # Stores { 'python': Language_object, ... }

def build_languages_library():
    """Builds a shared library for all specified languages."""
    print(f"INFO (language_utils): Attempting to build tree-sitter library at {LANG_SO_FILE}")
    grammar_paths = []
    for lang, (grammar_dir, _) in LANGUAGE_DEFINITIONS.items():
        if os.path.isdir(grammar_dir):
            grammar_paths.append(grammar_dir)
            print(f"INFO (language_utils): Found grammar directory for {lang}: {grammar_dir}")
        else:
            print(f"WARNING (language_utils): Grammar directory not found for {lang}: {grammar_dir}")

    if not grammar_paths:
        print("ERROR (language_utils): No valid grammar paths found in LANGUAGE_DEFINITIONS. Cannot build library.")
        return False

    try:
        # Ensure the target directory for the .so/.dll file exists
        os.makedirs(GRAMMAR_BASE_PATH, exist_ok=True)
        
        print(f"INFO (language_utils): Calling Language.build_library with target '{LANG_SO_FILE}' and sources {grammar_paths}")
        Language.build_library(
            LANG_SO_FILE, # Output path for the shared library
            grammar_paths # List of paths to grammar source directories
        )
        print(f"INFO (language_utils): Tree-sitter languages library built successfully at {LANG_SO_FILE}")
        return True
    except AttributeError as ae:
        print(f"FATAL ERROR (language_utils): 'Language.build_library' attribute not found: {ae}")
        print("     This likely means your 'tree-sitter' Python package is outdated or not installed correctly.")
        print("     Try: pip install --upgrade tree-sitter")
        return False
    except Exception as e:
        import traceback
        print(f"ERROR (language_utils): Failed to build tree-sitter languages library: {e}")
        print(traceback.format_exc())
        print("     Ensure you have a C/C++ compiler installed and in your PATH.")
        print("     Also, check that grammar source folders are correctly cloned in the 'grammars' directory.")
        return False

def load_grammars():
    """Loads all defined languages from the pre-built library."""
    global LOADED_LANGUAGES
    LOADED_LANGUAGES.clear() # Clear any previously loaded grammars

    if not os.path.exists(LANG_SO_FILE):
        print(f"INFO (language_utils): Languages library {LANG_SO_FILE} not found. Attempting to build it now...")
        if not build_languages_library():
            print("ERROR (language_utils): Could not build or find languages library. AST features will be disabled for this session.")
            return # Exit if build fails

    if not os.path.exists(LANG_SO_FILE): # Double check after build attempt
        print(f"ERROR (language_utils): Language library {LANG_SO_FILE} still not found after build attempt. AST features disabled.")
        return

    print(f"INFO (language_utils): Loading grammars from library: {LANG_SO_FILE}")
    for lang_name, (_, ts_lang_name_in_lib) in LANGUAGE_DEFINITIONS.items():
        # Check if the corresponding grammar directory was found (and thus likely included in the build)
        grammar_dir_check = LANGUAGE_DEFINITIONS[lang_name][0]
        if not os.path.isdir(grammar_dir_check):
            print(f"INFO (language_utils): Skipping grammar for {lang_name} as its source directory was not found.")
            continue

        try:
            lang_object = Language(LANG_SO_FILE, ts_lang_name_in_lib)
            LOADED_LANGUAGES[lang_name] = lang_object
            print(f"INFO (language_utils): Successfully loaded grammar for {lang_name} (as '{ts_lang_name_in_lib}')")
        except Exception as e:
            print(f"ERROR (language_utils): Could not load grammar for {lang_name} (as '{ts_lang_name_in_lib}') from {LANG_SO_FILE}: {e}")
            print(f"      Is '{ts_lang_name_in_lib}' the correct symbol name for the language in the compiled library?")
    
    if not LOADED_LANGUAGES:
        print("WARNING (language_utils): No languages were successfully loaded after attempting to use the library.")
    else:
        print(f"INFO (language_utils): Total languages loaded into LOADED_LANGUAGES: {len(LOADED_LANGUAGES)}")


def get_language_from_filename(filename: str) -> str | None:
    """Detects language from filename extension."""
    if not filename: return None
    _, ext = os.path.splitext(filename)
    return EXTENSION_TO_LANGUAGE.get(ext.lower())

def get_parser(language_name: str) -> Parser | None:
    """Gets a tree-sitter parser for the given language."""
    if language_name not in LOADED_LANGUAGES:
        # This warning is now a bit redundant if load_grammars already logged extensively
        # print(f"WARNING (language_utils): No loaded grammar for language: {language_name}")
        return None
    try:
        parser = Parser()
        parser.set_language(LOADED_LANGUAGES[language_name])
        return parser
    except Exception as e:
        print(f"ERROR (language_utils): Failed to create parser for language {language_name}: {e}")
        return None