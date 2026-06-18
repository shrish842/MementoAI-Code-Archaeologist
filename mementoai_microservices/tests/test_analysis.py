from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.ingestion.app.analysis import analyze_code_changes, analyze_technical_debt


class AnalysisTests(unittest.TestCase):
    def test_analyze_code_changes_detects_modified_function(self) -> None:
        diff_text = """@@
-def greet(name):
-    return "hi"
+def greet(name):
+    if name:
+        return "hi " + name
+    return "hi"
"""

        changes = analyze_code_changes(diff_text)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].name, "greet")
        self.assertEqual(changes[0].change_type, "modified")

    def test_analyze_technical_debt_returns_expected_shape(self) -> None:
        code = """
def complicated(a, b, c, d, e, f):
    if a:
        return b
    if c:
        return d
    if e:
        return f
    return None
"""
        debt = analyze_technical_debt(code)
        self.assertIn("complicated", debt.cyclomatic_complexity)
        self.assertGreaterEqual(debt.technical_debt_score, 0.0)
        self.assertTrue(any("Many parameters" in smell for smell in debt.code_smells))


if __name__ == "__main__":
    unittest.main()