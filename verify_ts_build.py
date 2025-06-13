from tree_sitter import Language
print(hasattr(Language, 'build_library'))  # Should return True
import tree_sitter
print(tree_sitter.__file__)  # Should point to the GitHub version