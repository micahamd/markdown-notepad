#!/usr/bin/env python3
"""
test_spreadsheet_ai.py - Unit and Integration Tests for Spreadsheet AI Integration
===================================================================================
Tests:
1. SpreadsheetPatchParser with row-by-row grading, colon-bearing feedback, and range syntax.
2. Prevention of false-positive 2D range expansion from feedback text like "D3: Q1: ...".
3. Prioritization of specific spreadsheet-edits blocks over conversational prose tables.
4. Markdown table parsing with composite column headers (e.g., "Score (C)").
5. Multi-line participant answer sanitization in SpreadsheetTab.get_ai_context.
6. Expanded selection bounds in get_ai_context when output columns are empty.
7. AIChatSidebar._show_apply_button availability for spreadsheet patches without pre-selection.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spreadsheet_tab import (
    SpreadsheetPatchParser,
    SpreadsheetDiffDialog,
    SpreadsheetTab,
    FormulaEngine,
    cell_to_ref,
    parse_cell_ref,
)


class TestSpreadsheetPatchParser(unittest.TestCase):
    """Test AI patch extraction across various formatting conventions and edge cases."""

    def test_row_evaluation_with_colons_in_feedback(self):
        """Verify that feedback containing colons (e.g., 'D3: Q1: Good answer') does NOT trigger a 2D range flood."""
        ai_response = """
Here is my evaluation for the participant responses:

```spreadsheet-edits
C3: 0.8
D3: Q1: Participant accurately defined the Big Five traits but omitted facet descriptions.
C4: 0.5
D4: Note: Participant only partially answered the question; missing section 2.
```
"""
        patches = SpreadsheetPatchParser.parse_patches(ai_response)
        self.assertIn("C3", patches)
        self.assertEqual(patches["C3"], "0.8")
        self.assertIn("D3", patches)
        self.assertEqual(
            patches["D3"],
            "Q1: Participant accurately defined the Big Five traits but omitted facet descriptions."
        )
        self.assertIn("C4", patches)
        self.assertEqual(patches["C4"], "0.5")
        self.assertIn("D4", patches)
        self.assertEqual(
            patches["D4"],
            "Note: Participant only partially answered the question; missing section 2."
        )

        # Critical: Must NOT expand into spurious range cells like Q1, Q2, Q3, E3..P3
        self.assertNotIn("Q1", patches)
        self.assertNotIn("E3", patches)
        self.assertNotIn("P3", patches)
        self.assertEqual(len(patches), 4)

    def test_genuine_range_expansion(self):
        """Verify that genuine range syntax (e.g. C3:C6: =...) correctly expands across rows."""
        ai_response = """
```spreadsheet-edits
C3:C6: =ROUND(A3*B3, 2)
D1:D3 = 1.0
```
"""
        patches = SpreadsheetPatchParser.parse_patches(ai_response)
        self.assertIn("C3", patches)
        self.assertIn("C4", patches)
        self.assertIn("C5", patches)
        self.assertIn("C6", patches)
        self.assertEqual(patches["C3"], "=ROUND(A3*B3, 2)")
        self.assertEqual(patches["C4"], "=ROUND(A4*B4, 2)")
        self.assertEqual(patches["C5"], "=ROUND(A5*B5, 2)")
        self.assertEqual(patches["C6"], "=ROUND(A6*B6, 2)")

        self.assertIn("D1", patches)
        self.assertIn("D2", patches)
        self.assertIn("D3", patches)
        self.assertEqual(patches["D1"], "1.0")

    def test_prioritizes_spreadsheet_edits_over_prose_tables(self):
        """Verify that prose summary tables outside the code block do not corrupt column A and B."""
        ai_response = """
I evaluated the participants as requested:

| Row | Question | Answer | Score | Feedback |
|---|---|---|---|---|
| 3 | Define X | X is Y | 1.0 | Correct definition |
| 4 | Define Z | Z is W | 0.5 | Incomplete |

Here are the edits to apply:
```spreadsheet-edits
C3: 1.0
D3: Correct definition
C4: 0.5
D4: Incomplete
```
"""
        patches = SpreadsheetPatchParser.parse_patches(ai_response)
        # Should contain ONLY C3, D3, C4, D4
        self.assertEqual(set(patches.keys()), {"C3", "D3", "C4", "D4"})
        self.assertEqual(patches["C3"], "1.0")
        self.assertEqual(patches["D3"], "Correct definition")
        self.assertEqual(patches["C4"], "0.5")
        self.assertEqual(patches["D4"], "Incomplete")
        # Ensure Question/Answer columns A & B are NOT overwritten
        self.assertNotIn("A1", patches)
        self.assertNotIn("B1", patches)
        self.assertNotIn("A3", patches)
        self.assertNotIn("B3", patches)

    def test_markdown_table_with_composite_column_headers(self):
        """Verify parsing of markdown tables when the AI supplies headers like 'Score (C)' and 'Feedback (D)'."""
        ai_response = """
```markdown
| Row | Score (C) | Feedback (D) |
|---|---|---|
| 3 | 0.9 | Excellent explanation |
| 4 | 0.4 | Failed to cite primary source |
```
"""
        patches = SpreadsheetPatchParser.parse_patches(ai_response)
        self.assertEqual(patches.get("C3"), "0.9")
        self.assertEqual(patches.get("D3"), "Excellent explanation")
        self.assertEqual(patches.get("C4"), "0.4")
        self.assertEqual(patches.get("D4"), "Failed to cite primary source")
        self.assertNotIn("A1", patches)
        self.assertNotIn("B1", patches)


class TestSpreadsheetTabAIContext(unittest.TestCase):
    """Test get_ai_context with multi-line cell answers and empty target ranges."""

    def test_multiline_answer_sanitization(self):
        """Verify that embedded newlines in participant answers do not break Markdown table row structure."""
        engine = FormulaEngine()
        # Row 1: Headers
        engine.set_cell(0, 0, "Question")
        engine.set_cell(0, 1, "Participant Response")
        engine.set_cell(0, 2, "Score")
        engine.set_cell(0, 3, "Feedback")

        # Row 2: Sample grade
        engine.set_cell(1, 0, "Q1: Explain introversion")
        engine.set_cell(1, 1, "Introversion relates to internal focus.")
        engine.set_cell(1, 2, "1.0")
        engine.set_cell(1, 3, "Clear and concise explanation.")

        # Row 3: Ungraded participant answer with multi-line text
        multiline_answer = "First paragraph of participant response.\n\nSecond paragraph with details.\r\nThird line."
        engine.set_cell(2, 0, "Q2: Compare traits")
        engine.set_cell(2, 1, multiline_answer)
        # C3 and D3 are left empty for the AI to fill

        # Test formatting without requiring full Tk display loop
        tab = SpreadsheetTab.__new__(SpreadsheetTab)
        tab.engine = engine
        tab.file_path = "participants.xlsx"

        class DummyGrid:
            def get_selected_range(self):
                # User selected A1:D3
                return ((0, 0), (2, 3))
        tab.grid_widget = DummyGrid()

        context = tab.get_ai_context()

        # Check that context has unbroken table lines
        table_lines = [l for l in context.splitlines() if l.startswith("|") and l.endswith("|")]
        # Headers + separator + 3 data rows = 5 table lines
        self.assertEqual(len(table_lines), 5)

        # Ensure the multi-line text is sanitized onto a single row line
        row3_line = table_lines[4]
        self.assertTrue(row3_line.startswith("| 3 |"))
        self.assertIn("First paragraph of participant response.", row3_line)
        self.assertIn("Second paragraph with details.", row3_line)
        self.assertIn("Third line.", row3_line)

        # Verify instructions for row-by-row grading are included
        self.assertIn("Instructions for Editing or Populating the Spreadsheet", context)
        self.assertIn("Use sample grades and feedback already present in previous rows", context)
        self.assertIn("```spreadsheet-edits", context)

    def test_extreme_cell_sanitization(self):
        """Verify that null bytes, tabs, newlines, triple backticks, and unescaped pipes are safely neutralized."""
        raw_evil_input = "Answer with null\x00byte\x07bell\tTAB\r\nNewline and ```code``` and a | pipe symbol."
        clean = SpreadsheetTab._sanitize_cell_for_markdown(raw_evil_input)
        self.assertNotIn("\x00", clean)
        self.assertNotIn("\x07", clean)
        self.assertNotIn("\t", clean)
        self.assertNotIn("\r", clean)
        self.assertNotIn("\n", clean)
        self.assertNotIn("```", clean)
        self.assertIn("` ` `", clean)
        self.assertIn("\\|", clean)


class TestHardenedSpreadsheetParserAndFormulas(unittest.TestCase):
    """Test multi-line continuation, quote stripping, prose leading equals, and formula evaluation."""

    def test_multiline_continuation_and_quotes(self):
        """Verify that wrapped feedback lines are merged and outer quotes are stripped."""
        ai_response = """
```spreadsheet-edits
C3: 0.85
D3: "Participant correctly explained the central dogma of biology,
and accurately noted the role of mRNA transcription."
C4: 0.40
D4: 'Partial credit awarded:
#1 omission was the translation step.'
E3: =C3*100
```
"""
        patches = SpreadsheetPatchParser.parse_patches(ai_response)
        self.assertEqual(patches.get("C3"), "0.85")
        self.assertEqual(
            patches.get("D3"),
            "Participant correctly explained the central dogma of biology, and accurately noted the role of mRNA transcription."
        )
        self.assertEqual(patches.get("C4"), "0.40")
        self.assertEqual(
            patches.get("D4"),
            "Partial credit awarded: #1 omission was the translation step."
        )
        self.assertEqual(patches.get("E3"), "=C3*100")

    def test_prose_leading_equals_cleaned_and_formulas_preserved(self):
        """Verify that prose starting with '=' is sanitized to avoid #VALUE! / #NAME?, while real formulas evaluate."""
        ai_response = """
```spreadsheet-edits
D3: = 0.5 points awarded because the hypothesis was omitted.
D4: == missing control condition.
E3: =C3*100
E4: =AVERAGE(C3:C4)
```
"""
        patches = SpreadsheetPatchParser.parse_patches(ai_response)
        self.assertEqual(patches.get("D3"), "0.5 points awarded because the hypothesis was omitted.")
        self.assertEqual(patches.get("D4"), "missing control condition.")
        self.assertEqual(patches.get("E3"), "=C3*100")
        self.assertEqual(patches.get("E4"), "=AVERAGE(C3:C4)")

        # Verify execution in FormulaEngine
        engine = FormulaEngine()
        engine.set_cell(2, 2, "0.8")   # C3
        engine.set_cell(3, 2, "0.6")   # C4
        engine.set_cell(2, 3, patches["D3"])  # D3 (prose)
        engine.set_cell(3, 3, patches["D4"])  # D4 (prose)
        engine.set_cell(2, 4, patches["E3"])  # E3 (=C3*100)
        engine.set_cell(3, 4, patches["E4"])  # E4 (=AVERAGE(C3:C4))

        # Check prose doesn't evaluate to #VALUE! or #ERROR!
        self.assertEqual(engine.evaluate_cell(2, 3), "0.5 points awarded because the hypothesis was omitted.")
        self.assertEqual(engine.evaluate_cell(3, 3), "missing control condition.")

        # Check formulas compute properly
        self.assertEqual(engine.evaluate_cell(2, 4), 80.0)
        self.assertEqual(engine.evaluate_cell(3, 4), 0.7)

    def test_apostrophe_literal_escape_in_formula_engine(self):
        """Verify that leading apostrophe in FormulaEngine evaluates to literal text without #VALUE!."""
        engine = FormulaEngine()
        engine.set_cell(0, 0, "'= literal note starting with equals")
        self.assertEqual(engine.evaluate_cell(0, 0), "= literal note starting with equals")

    def test_escaped_pipe_in_markdown_table_fallback(self):
        """Verify fallback markdown table parsing when cells contain escaped pipe characters."""
        ai_response = """
| Row | C | D |
|---|---|---|
| 3 | 0.9 | High performance \\| excellent analysis |
| 4 | 0.3 | Low performance \\| failed all checks |
"""
        patches = SpreadsheetPatchParser.parse_patches(ai_response)
        self.assertEqual(patches.get("C3"), "0.9")
        self.assertEqual(patches.get("D3"), "High performance | excellent analysis")
        self.assertEqual(patches.get("C4"), "0.3")
        self.assertEqual(patches.get("D4"), "Low performance | failed all checks")


class TestSpreadsheetDiffDialog(unittest.TestCase):
    """Test SpreadsheetDiffDialog empty-cell filtering and unchanged-cell detection."""

    def test_diff_dialog_empty_cell_filtering_and_unchanged_detection(self):
        """Verify that unchanged cells are unchecked by default, and selecting empty only isolates new rows."""
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        try:
            engine = FormulaEngine()
            # C1 and D1 already populated (sample grade & feedback)
            engine.set_cell(0, 2, "1.0")
            engine.set_cell(0, 3, "Sample feedback.")
            # C2 and D2 already populated
            engine.set_cell(1, 2, "0.5")
            engine.set_cell(1, 3, "Sample partial feedback.")
            # C3 and D3 are empty in the sheet

            tab = SpreadsheetTab.__new__(SpreadsheetTab)
            tab.engine = engine

            # AI patches echo C1 (identical), update D1 (different), and provide C3 & D3 (new empty cells)
            patches = {
                "C1": "1.0",                  # Identical to existing -> unchecked by default
                "D1": "Rephrased feedback.",  # Populated with change -> checked by default
                "C3": "0.8",                  # Empty cell -> checked by default
                "D3": "Good answer."          # Empty cell -> checked by default
            }

            dialog = SpreadsheetDiffDialog(root, tab, patches)
            # Find item IDs by cell coordinate
            cell_items = {}
            for item_id in dialog.tree.get_children():
                vals = dialog.tree.item(item_id, "values")
                cell_items[vals[1]] = (item_id, vals[0], vals[2], vals[3])

            # Initial state: C1 (identical) is unchecked "☐", D1/C3/D3 are checked "☑"
            self.assertFalse(dialog.row_checked[cell_items["C1"][0]])
            self.assertEqual(cell_items["C1"][1], "☐")

            self.assertTrue(dialog.row_checked[cell_items["D1"][0]])
            self.assertEqual(cell_items["D1"][1], "☑")

            self.assertTrue(dialog.row_checked[cell_items["C3"][0]])
            self.assertEqual(cell_items["C3"][1], "☑")

            # Now test "Select Empty Only"
            dialog._select_empty_only()

            # Populated cells C1 and D1 must now be unchecked "☐"
            self.assertFalse(dialog.row_checked[cell_items["C1"][0]])
            self.assertFalse(dialog.row_checked[cell_items["D1"][0]])
            # Empty cells C3 and D3 must remain checked "☑"
            self.assertTrue(dialog.row_checked[cell_items["C3"][0]])
            self.assertTrue(dialog.row_checked[cell_items["D3"][0]])

            dialog.destroy()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
