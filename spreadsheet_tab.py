"""
Spreadsheet Tab and Engine for MarkItDown Notepad

Provides an Excel-like interactive spreadsheet processor with virtualized 2D grid,
safe AST formula engine (SUM, AVERAGE, ROUND, MIN, MAX, COUNT, IF, etc.),
bidirectional AI Chat integration, and Excel/TSV/CSV/Markdown table interoperability.
"""

import ast
import csv
import io
import json
import math
import os
import re
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Callable
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


# =============================================================================
# Coordinate & Reference Utilities
# =============================================================================

def col_index_to_letter(col_idx: int) -> str:
    """Convert a 0-based column index to an Excel column letter (e.g. 0 -> A, 25 -> Z, 26 -> AA)."""
    result = []
    col = col_idx + 1
    while col > 0:
        col, remainder = divmod(col - 1, 26)
        result.append(chr(65 + remainder))
    return "".join(reversed(result))


def col_letter_to_index(col_str: str) -> int:
    """Convert an Excel column letter to a 0-based index (e.g. A -> 0, Z -> 25, AA -> 26)."""
    col_str = col_str.strip().upper()
    idx = 0
    for char in col_str:
        if not ("A" <= char <= "Z"):
            raise ValueError(f"Invalid column character: {char}")
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


_CELL_REF_RE = re.compile(r"^\$?([A-Za-z]+)\$?([0-9]+)$")
_RANGE_REF_RE = re.compile(r"^\$?([A-Za-z]+)\$?([0-9]+):\$?([A-Za-z]+)\$?([0-9]+)$")


def parse_cell_ref(ref_str: str) -> Optional[Tuple[int, int]]:
    """Parse a single cell reference string (e.g. 'A1', '$B$5') into (row, col) 0-based indices."""
    m = _CELL_REF_RE.match(ref_str.strip())
    if not m:
        return None
    col_str, row_str = m.groups()
    try:
        col = col_letter_to_index(col_str)
        row = int(row_str) - 1
        return (row, col)
    except Exception:
        return None


def cell_to_ref(row: int, col: int) -> str:
    """Convert (row, col) 0-based indices to Excel coordinate (e.g. (0, 0) -> 'A1')."""
    return f"{col_index_to_letter(col)}{row + 1}"


def parse_range_ref(range_str: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Parse a range reference (e.g. 'A1:C5') into ((r1, c1), (r2, c2))."""
    m = _RANGE_REF_RE.match(range_str.strip())
    if not m:
        return None
    c1_str, r1_str, c2_str, r2_str = m.groups()
    try:
        c1 = col_letter_to_index(c1_str)
        r1 = int(r1_str) - 1
        c2 = col_letter_to_index(c2_str)
        r2 = int(r2_str) - 1
        return ((min(r1, r2), min(c1, c2)), (max(r1, r2), max(c1, c2)))
    except Exception:
        return None


def expand_range(r1: int, c1: int, r2: int, c2: int) -> List[Tuple[int, int]]:
    """Return all (row, col) coordinates within the bounding box."""
    cells = []
    min_r, max_r = min(r1, r2), max(r1, r2)
    min_c, max_c = min(c1, c2), max(c1, c2)
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            cells.append((r, c))
    return cells


def shift_formula_references(formula: str, dr: int, dc: int) -> str:
    """Shift relative cell coordinates in an Excel formula by (dr, dc), preserving absolute ($) references."""
    if not formula or not formula.startswith("="):
        return formula

    def _repl(m):
        abs_c, col_str, abs_r, row_str = m.groups()
        try:
            c = col_letter_to_index(col_str)
            r = int(row_str) - 1
            new_c = c if abs_c else max(0, c + dc)
            new_r = r if abs_r else max(0, r + dr)
            c_str = f"${col_index_to_letter(new_c)}" if abs_c else col_index_to_letter(new_c)
            r_str = f"${new_r + 1}" if abs_r else str(new_r + 1)
            return f"{c_str}{r_str}"
        except Exception:
            return m.group(0)

    # Process outside string literals
    parts = []
    last_end = 0
    pattern = re.compile(r'(\$?)([A-Za-z]+)(\$?)([0-9]+)')
    for m in re.finditer(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')', formula):
        chunk = formula[last_end:m.start()]
        parts.append(pattern.sub(_repl, chunk))
        parts.append(m.group(0))
        last_end = m.end()
    parts.append(pattern.sub(_repl, formula[last_end:]))
    return "".join(parts)


def preprocess_excel_formula(expr: str) -> str:
    """Transform Excel formula syntax ($A$1, <>, =, ^, ranges) to safe Python-evaluable AST format."""
    cleaned = expr.strip()
    if not cleaned:
        return ""

    def _process_chunk(chunk: str) -> str:
        # 1. Range syntax $A$1:$B$10 -> _RANGE_('A1', 'B10')
        range_pat = re.compile(r"(\$?[A-Za-z]+\$?[0-9]+):(\$?[A-Za-z]+\$?[0-9]+)")
        def _range_repl(m):
            c1 = m.group(1).replace('$', '')
            c2 = m.group(2).replace('$', '')
            return f"_RANGE_('{c1}', '{c2}')"
        chunk = range_pat.sub(_range_repl, chunk)

        # 2. Caret ^ -> **
        chunk = chunk.replace("^", "**")

        # 3. Not equal <> -> !=
        chunk = chunk.replace("<>", "!=")

        # 4. Standalone = -> == (while preserving <=, >=, !=, ==)
        chunk = re.sub(r'(?<![<>!=])=(?![=])', '==', chunk)

        # 5. Strip $ from remaining single cell references $A$1, A$1, $A1 -> A1
        single_ref_pat = re.compile(r'(?<![A-Za-z0-9_])\$?([A-Za-z]+)\$?([0-9]+)(?![A-Za-z0-9_])')
        def _strip_dollar(m):
            return f"{m.group(1)}{m.group(2)}"
        chunk = single_ref_pat.sub(_strip_dollar, chunk)

        return chunk

    # Tokenize to avoid altering inside quotes
    parts = []
    last_end = 0
    for m in re.finditer(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')', cleaned):
        parts.append(_process_chunk(cleaned[last_end:m.start()]))
        parts.append(m.group(0))
        last_end = m.end()
    parts.append(_process_chunk(cleaned[last_end:]))
    return "".join(parts)


class ToolTip:
    """Screen-edge-clamped tooltip for interactive spreadsheet buttons."""

    def __init__(self, widget, text, delay=350):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.scheduled_id = None
        self.label = None
        widget.bind('<Enter>', self._on_enter, add=True)
        widget.bind('<Leave>', self._on_leave, add=True)
        widget.bind('<Button>', self._on_leave, add=True)

    def _on_enter(self, event=None):
        self._cancel_scheduled()
        self.scheduled_id = self.widget.after(self.delay, self._show_tooltip)

    def _on_leave(self, event=None):
        self._cancel_scheduled()
        self._hide_tooltip()

    def _cancel_scheduled(self):
        if self.scheduled_id:
            try:
                self.widget.after_cancel(self.scheduled_id)
            except Exception:
                pass
            self.scheduled_id = None

    def _show_tooltip(self):
        if self.tooltip_window or not self.widget.winfo_exists():
            return
        try:
            wx = self.widget.winfo_rootx()
            wy = self.widget.winfo_rooty()
            ww = self.widget.winfo_width()
            wh = self.widget.winfo_height()
        except Exception:
            return
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        self.label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#2e1065", foreground="#faf5ff",
            relief=tk.SOLID, borderwidth=1,
            font=("Segoe UI", 9), padx=8, pady=5,
            wraplength=320
        )
        self.label.pack()
        tw.update_idletasks()
        tip_w = tw.winfo_reqwidth()
        tip_h = tw.winfo_reqheight()
        x = wx + (ww - tip_w) // 2
        y = wy + wh + 6
        screen_w = tw.winfo_screenwidth()
        screen_h = tw.winfo_screenheight()
        if x + tip_w > screen_w - 12:
            x = screen_w - tip_w - 12
        if x < 12:
            x = 12
        if y + tip_h > screen_h - 12:
            y = wy - tip_h - 6
            if y < 12:
                y = 12
        tw.wm_geometry(f"+{x}+{y}")

    def _hide_tooltip(self):
        if self.tooltip_window:
            try:
                self.tooltip_window.destroy()
            except Exception:
                pass
            self.tooltip_window = None
            self.label = None


# =============================================================================
# Formula Engine & Evaluator
# =============================================================================

class FormulaEngine:
    """Safe, AST-based Excel formula evaluator with circular dependency detection."""

    ERROR_DIV_ZERO = "#DIV/0!"
    ERROR_VALUE = "#VALUE!"
    ERROR_NAME = "#NAME?"
    ERROR_REF = "#REF!"
    ERROR_CIRC = "#CIRC!"
    ERROR_GENERIC = "#ERROR!"

    def __init__(self):
        self.raw_cells: Dict[Tuple[int, int], str] = {}
        self.computed_values: Dict[Tuple[int, int], Any] = {}
        self.cell_styles: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._eval_stack: Set[Tuple[int, int]] = set()

        # Excel Built-in Functions Registry
        self.functions: Dict[str, Callable] = {
            "SUM": self._fn_sum,
            "AVERAGE": self._fn_average,
            "AVG": self._fn_average,
            "ROUND": self._fn_round,
            "ROUNDUP": self._fn_roundup,
            "ROUNDDOWN": self._fn_rounddown,
            "MIN": self._fn_min,
            "MAX": self._fn_max,
            "COUNT": self._fn_count,
            "COUNTA": self._fn_counta,
            "ABS": self._fn_abs,
            "SQRT": self._fn_sqrt,
            "POWER": self._fn_power,
            "MOD": self._fn_mod,
            "INT": self._fn_int,
            "PRODUCT": self._fn_product,
            "MEDIAN": self._fn_median,
            "STDEV": self._fn_stdev,
            "VAR": self._fn_var,
            "IF": self._fn_if,
            "IFERROR": self._fn_iferror,
            "SUMIF": self._fn_sumif,
            "COUNTIF": self._fn_countif,
            "AND": self._fn_and,
            "OR": self._fn_or,
            "NOT": self._fn_not,
            "CONCAT": self._fn_concat,
            "CONCATENATE": self._fn_concat,
            "LEN": self._fn_len,
            "UPPER": self._fn_upper,
            "LOWER": self._fn_lower,
            "TRIM": self._fn_trim,
            "LEFT": self._fn_left,
            "RIGHT": self._fn_right,
        }

    # -------------------------------------------------------------------------
    # Value Flattening & Extraction Helpers
    # -------------------------------------------------------------------------

    def _flatten_values(self, args: Any) -> List[Any]:
        """Flatten nested iterables (tuples, lists) and strip cell containers."""
        result = []
        if isinstance(args, (list, tuple)):
            for item in args:
                result.extend(self._flatten_values(item))
        else:
            result.append(args)
        return result

    def _to_numbers(self, args: Any) -> List[float]:
        """Extract all numeric values from args, converting string numbers if valid."""
        flat = self._flatten_values(args)
        nums = []
        for x in flat:
            if x is None or x == "":
                continue
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                nums.append(float(x))
            elif isinstance(x, str):
                s = x.strip().replace("$", "").replace(",", "")
                try:
                    nums.append(float(s))
                except ValueError:
                    pass
        return nums

    # -------------------------------------------------------------------------
    # Built-in Formula Functions
    # -------------------------------------------------------------------------

    def _fn_sum(self, *args) -> float:
        nums = self._to_numbers(args)
        return sum(nums)

    def _fn_average(self, *args) -> Union[float, str]:
        nums = self._to_numbers(args)
        if not nums:
            return self.ERROR_DIV_ZERO
        return sum(nums) / len(nums)

    def _fn_round(self, val: Any, digits: Any = 0) -> Union[float, str]:
        try:
            num = float(self._to_numbers(val)[0])
            d = int(float(self._to_numbers(digits)[0])) if self._to_numbers(digits) else 0
            rounded = round(num, d)
            return int(rounded) if d == 0 else rounded
        except Exception:
            return self.ERROR_VALUE

    def _fn_roundup(self, val: Any, digits: Any = 0) -> Union[float, str]:
        try:
            num = float(self._to_numbers(val)[0])
            d = int(float(self._to_numbers(digits)[0])) if self._to_numbers(digits) else 0
            factor = 10 ** d
            return math.ceil(num * factor) / factor
        except Exception:
            return self.ERROR_VALUE

    def _fn_rounddown(self, val: Any, digits: Any = 0) -> Union[float, str]:
        try:
            num = float(self._to_numbers(val)[0])
            d = int(float(self._to_numbers(digits)[0])) if self._to_numbers(digits) else 0
            factor = 10 ** d
            return math.floor(num * factor) / factor
        except Exception:
            return self.ERROR_VALUE

    def _fn_min(self, *args) -> Union[float, str]:
        nums = self._to_numbers(args)
        return min(nums) if nums else 0.0

    def _fn_max(self, *args) -> Union[float, str]:
        nums = self._to_numbers(args)
        return max(nums) if nums else 0.0

    def _fn_count(self, *args) -> int:
        return len(self._to_numbers(args))

    def _fn_counta(self, *args) -> int:
        flat = self._flatten_values(args)
        return sum(1 for x in flat if x is not None and str(x).strip() != "")

    def _fn_abs(self, val: Any) -> Union[float, str]:
        try:
            return abs(float(self._to_numbers(val)[0]))
        except Exception:
            return self.ERROR_VALUE

    def _fn_sqrt(self, val: Any) -> Union[float, str]:
        try:
            num = float(self._to_numbers(val)[0])
            if num < 0:
                return self.ERROR_VALUE
            return math.sqrt(num)
        except Exception:
            return self.ERROR_VALUE

    def _fn_power(self, base: Any, exp: Any) -> Union[float, str]:
        try:
            b = float(self._to_numbers(base)[0])
            e = float(self._to_numbers(exp)[0])
            return b ** e
        except Exception:
            return self.ERROR_VALUE

    def _fn_mod(self, x: Any, y: Any) -> Union[float, str]:
        try:
            a = float(self._to_numbers(x)[0])
            b = float(self._to_numbers(y)[0])
            if b == 0:
                return self.ERROR_DIV_ZERO
            return a % b
        except Exception:
            return self.ERROR_VALUE

    def _fn_int(self, val: Any) -> Union[int, str]:
        try:
            return int(float(self._to_numbers(val)[0]))
        except Exception:
            return self.ERROR_VALUE

    def _fn_product(self, *args) -> float:
        nums = self._to_numbers(args)
        if not nums:
            return 0.0
        p = 1.0
        for n in nums:
            p *= n
        return p

    def _fn_median(self, *args) -> Union[float, str]:
        nums = sorted(self._to_numbers(args))
        n = len(nums)
        if n == 0:
            return self.ERROR_DIV_ZERO
        mid = n // 2
        return (nums[mid - 1] + nums[mid]) / 2.0 if n % 2 == 0 else nums[mid]

    def _fn_stdev(self, *args) -> Union[float, str]:
        nums = self._to_numbers(args)
        if len(nums) < 2:
            return self.ERROR_DIV_ZERO
        mean = sum(nums) / len(nums)
        variance = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)
        return math.sqrt(variance)

    def _fn_var(self, *args) -> Union[float, str]:
        nums = self._to_numbers(args)
        if len(nums) < 2:
            return self.ERROR_DIV_ZERO
        mean = sum(nums) / len(nums)
        return sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)

    def _fn_if(self, cond: Any, t_val: Any, f_val: Any = 0) -> Any:
        return t_val if bool(cond) else f_val

    def _fn_iferror(self, val: Any, alt_val: Any) -> Any:
        if isinstance(val, str) and val.startswith("#"):
            return alt_val
        return val

    def _fn_sumif(self, criteria_range: Any, criteria: Any, sum_range: Any = None) -> float:
        c_vals = self._flatten_values(criteria_range)
        s_vals = self._flatten_values(sum_range) if sum_range is not None else c_vals
        crit_str = str(criteria).strip()
        total = 0.0
        for cv, sv in zip(c_vals, s_vals):
            match = False
            try:
                if crit_str.startswith(">="):
                    match = float(cv) >= float(crit_str[2:])
                elif crit_str.startswith("<="):
                    match = float(cv) <= float(crit_str[2:])
                elif crit_str.startswith(">"):
                    match = float(cv) > float(crit_str[1:])
                elif crit_str.startswith("<"):
                    match = float(cv) < float(crit_str[1:])
                elif crit_str.startswith("="):
                    match = str(cv) == crit_str[1:]
                else:
                    match = str(cv).lower() == crit_str.lower()
            except Exception:
                match = str(cv).lower() == crit_str.lower()
            if match:
                try:
                    total += float(sv)
                except (ValueError, TypeError):
                    pass
        return total

    def _fn_countif(self, criteria_range: Any, criteria: Any) -> int:
        c_vals = self._flatten_values(criteria_range)
        crit_str = str(criteria).strip()
        count = 0
        for cv in c_vals:
            match = False
            try:
                if crit_str.startswith(">="):
                    match = float(cv) >= float(crit_str[2:])
                elif crit_str.startswith("<="):
                    match = float(cv) <= float(crit_str[2:])
                elif crit_str.startswith(">"):
                    match = float(cv) > float(crit_str[1:])
                elif crit_str.startswith("<"):
                    match = float(cv) < float(crit_str[1:])
                elif crit_str.startswith("="):
                    match = str(cv) == crit_str[1:]
                else:
                    match = str(cv).lower() == crit_str.lower()
            except Exception:
                match = str(cv).lower() == crit_str.lower()
            if match:
                count += 1
        return count

    def _fn_and(self, *args) -> bool:
        flat = self._flatten_values(args)
        return all(bool(x) for x in flat)

    def _fn_or(self, *args) -> bool:
        flat = self._flatten_values(args)
        return any(bool(x) for x in flat)

    def _fn_not(self, val: Any) -> bool:
        return not bool(val)

    def _fn_concat(self, *args) -> str:
        flat = self._flatten_values(args)
        return "".join(str(x) for x in flat if x is not None)

    def _fn_len(self, s: Any) -> int:
        return len(str(s)) if s is not None else 0

    def _fn_upper(self, s: Any) -> str:
        return str(s).upper() if s is not None else ""

    def _fn_lower(self, s: Any) -> str:
        return str(s).lower() if s is not None else ""

    def _fn_trim(self, s: Any) -> str:
        return str(s).strip() if s is not None else ""

    def _fn_left(self, s: Any, n: Any = 1) -> str:
        try:
            num = int(float(self._to_numbers(n)[0]))
            return str(s)[:num]
        except Exception:
            return str(s)[:1]

    def _fn_right(self, s: Any, n: Any = 1) -> str:
        try:
            num = int(float(self._to_numbers(n)[0]))
            return str(s)[-num:] if num > 0 else ""
        except Exception:
            return str(s)[-1:] if str(s) else ""

    # -------------------------------------------------------------------------
    # Evaluation Core
    # -------------------------------------------------------------------------

    def set_cell(self, row: int, col: int, raw_value: str):
        """Set raw cell value or formula."""
        raw_str = str(raw_value) if raw_value is not None else ""
        if raw_str == "":
            self.raw_cells.pop((row, col), None)
            self.computed_values.pop((row, col), None)
        else:
            self.raw_cells[(row, col)] = raw_str

    def get_raw(self, row: int, col: int) -> str:
        """Get raw formula or text for cell."""
        return self.raw_cells.get((row, col), "")

    def get_value(self, row: int, col: int) -> Any:
        """Get computed value for cell."""
        if (row, col) in self.computed_values:
            return self.computed_values[(row, col)]
        return self.evaluate_cell(row, col)

    def evaluate_cell(self, row: int, col: int) -> Any:
        """Recursively evaluate cell, detecting circular references."""
        raw = self.raw_cells.get((row, col), "")
        if not raw or not raw.startswith("="):
            # Literal constant
            if raw == "":
                self.computed_values[(row, col)] = ""
                return ""
            try:
                # If numeric string, store as float/int
                if "." in raw:
                    val = float(raw)
                else:
                    val = int(raw)
            except ValueError:
                val = raw
            self.computed_values[(row, col)] = val
            return val

        # Circular reference check
        if (row, col) in self._eval_stack:
            return self.ERROR_CIRC

        self._eval_stack.add((row, col))
        try:
            computed = self._eval_formula(raw[1:], row, col)
        except Exception as e:
            computed = self.ERROR_GENERIC
        finally:
            self._eval_stack.discard((row, col))

        self.computed_values[(row, col)] = computed
        return computed

    def recalculate_all(self):
        """Recalculate all formulas across the entire sheet."""
        self._eval_stack.clear()
        self.computed_values.clear()
        for coord in list(self.raw_cells.keys()):
            self.evaluate_cell(coord[0], coord[1])

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Return (min_r, min_c, max_r, max_c) of all populated cells, or (0, 0, 0, 0) if empty."""
        populated = list(self.raw_cells.keys())
        if not populated:
            return (0, 0, 0, 0)
        min_r = min(r for r, c in populated)
        max_r = max(r for r, c in populated)
        min_c = min(c for r, c in populated)
        max_c = max(c for r, c in populated)
        return (min_r, min_c, max_r, max_c)

    # -------------------------------------------------------------------------
    # AST Transformation and Parsing
    # -------------------------------------------------------------------------

    def _eval_formula(self, expr: str, cur_r: int, cur_c: int) -> Any:
        """Parse formula expression and execute via AST."""
        cleaned_expr = expr.strip()
        if not cleaned_expr:
            return ""

        processed_expr = preprocess_excel_formula(cleaned_expr)

        # Parse with AST
        try:
            tree = ast.parse(processed_expr, mode="eval")
        except SyntaxError:
            return self.ERROR_VALUE

        return self._eval_ast_node(tree.body, cur_r, cur_c)

    def _eval_ast_node(self, node: ast.AST, cur_r: int, cur_c: int) -> Any:
        """Safely evaluate an AST node."""
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):
            name_upper = node.id.upper()
            # Check if this name is a single cell reference like A1 or B12
            cell_pos = parse_cell_ref(node.id)
            if cell_pos is not None:
                val = self.evaluate_cell(cell_pos[0], cell_pos[1])
                return 0.0 if val == "" else val
            # Check for boolean literals
            if name_upper == "TRUE":
                return True
            if name_upper == "FALSE":
                return False
            return self.ERROR_NAME

        elif isinstance(node, ast.BinOp):
            left = self._eval_ast_node(node.left, cur_r, cur_c)
            right = self._eval_ast_node(node.right, cur_r, cur_c)

            if isinstance(left, str) and left.startswith("#"):
                return left
            if isinstance(right, str) and right.startswith("#"):
                return right

            # Excel string concatenation operator (&)
            if isinstance(node.op, ast.BitAnd):
                l_str = "" if left is None else str(left)
                r_str = "" if right is None else str(right)
                return l_str + r_str

            try:
                l_num = float(left) if left != "" else 0.0
                r_num = float(right) if right != "" else 0.0
            except (ValueError, TypeError):
                if isinstance(node.op, ast.Add):
                    return str(left) + str(right)
                return self.ERROR_VALUE

            if isinstance(node.op, ast.Add):
                return l_num + r_num
            elif isinstance(node.op, ast.Sub):
                return l_num - r_num
            elif isinstance(node.op, ast.Mult):
                return l_num * r_num
            elif isinstance(node.op, ast.Div):
                if r_num == 0:
                    return self.ERROR_DIV_ZERO
                return l_num / r_num
            elif isinstance(node.op, ast.Pow):
                return l_num ** r_num
            elif isinstance(node.op, ast.Mod):
                if r_num == 0:
                    return self.ERROR_DIV_ZERO
                return l_num % r_num
            return self.ERROR_VALUE

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_ast_node(node.operand, cur_r, cur_c)
            if isinstance(operand, str) and operand.startswith("#"):
                return operand
            try:
                num = float(operand)
            except (ValueError, TypeError):
                return self.ERROR_VALUE
            if isinstance(node.op, ast.USub):
                return -num
            elif isinstance(node.op, ast.UAdd):
                return num
            elif isinstance(node.op, ast.Not):
                return not bool(operand)
            return self.ERROR_VALUE

        elif isinstance(node, ast.Compare):
            left = self._eval_ast_node(node.left, cur_r, cur_c)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_ast_node(comparator, cur_r, cur_c)
                try:
                    # Try numeric comparison first
                    l_val = float(left) if left != "" else 0.0
                    r_val = float(right) if right != "" else 0.0
                except (ValueError, TypeError):
                    l_val = str(left)
                    r_val = str(right)

                if isinstance(op, ast.Eq):
                    match = (l_val == r_val)
                elif isinstance(op, ast.NotEq):
                    match = (l_val != r_val)
                elif isinstance(op, ast.Lt):
                    match = (l_val < r_val)
                elif isinstance(op, ast.LtE):
                    match = (l_val <= r_val)
                elif isinstance(op, ast.Gt):
                    match = (l_val > r_val)
                elif isinstance(op, ast.GtE):
                    match = (l_val >= r_val)
                else:
                    return self.ERROR_VALUE
                if not match:
                    return False
                left = right
            return True

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id.upper()
                # Internal Range Constructor: _RANGE_('A1', 'C5')
                if func_name == "_RANGE_":
                    r1_str = self._eval_ast_node(node.args[0], cur_r, cur_c)
                    r2_str = self._eval_ast_node(node.args[1], cur_r, cur_c)
                    p1 = parse_cell_ref(str(r1_str))
                    p2 = parse_cell_ref(str(r2_str))
                    if p1 is None or p2 is None:
                        return self.ERROR_REF
                    coords = expand_range(p1[0], p1[1], p2[0], p2[1])
                    return [self.evaluate_cell(r, c) for r, c in coords]

                # Evaluate Arguments
                eval_args = [self._eval_ast_node(arg, cur_r, cur_c) for arg in node.args]

                if func_name in self.functions:
                    try:
                        return self.functions[func_name](*eval_args)
                    except Exception as e:
                        return self.ERROR_GENERIC
                return self.ERROR_NAME

            return self.ERROR_NAME

        return self.ERROR_VALUE


# =============================================================================
# Virtualized 2D Canvas Grid Widget
# =============================================================================

class SpreadsheetGrid(tk.Canvas):
    """High-performance virtualized 2D canvas grid with Royal Purple accents."""

    DEFAULT_COL_WIDTH = 88
    DEFAULT_ROW_HEIGHT = 24
    HEADER_ROW_HEIGHT = 26
    HEADER_COL_WIDTH = 48
    MIN_ROWS = 100
    MIN_COLS = 26

    def __init__(self, parent, engine: FormulaEngine, on_cell_select: Optional[Callable] = None,
                 on_modified: Optional[Callable] = None, theme: Optional[Dict] = None):
        super().__init__(parent, bg="#ffffff", highlightthickness=0)
        self.parent = parent
        self.engine = engine
        self.on_cell_select_cb = on_cell_select
        self.on_modified_cb = on_modified
        self.theme = theme or {}

        # Sizing and Dimensions
        self.col_widths: Dict[int, int] = {}
        self.row_heights: Dict[int, int] = {}
        self.total_rows = self.MIN_ROWS
        self.total_cols = self.MIN_COLS

        # Selection state: (row, col)
        self.active_cell = (0, 0)
        self.sel_start = (0, 0)
        self.sel_end = (0, 0)
        self.is_selecting = False

        # Undo/Redo Stacks
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []

        # Inline Cell Editor Widget (Overlay Entry)
        self.editor_entry: Optional[ttk.Entry] = None
        self.is_editing = False

        # Resizing column dividers
        self.resizing_col: Optional[int] = None
        self.resize_start_x = 0
        self.resize_start_w = 0

        # Colors & Styling
        self._accent_color = self.theme.get("accent_purple", "#5b21b6")
        self._grid_line_color = "#e2e8f0"
        self._header_bg = "#f3effa"
        self._header_fg = "#475569"
        self._header_active_bg = "#ede9fe"
        self._header_active_fg = "#5b21b6"
        self._selection_bg = "#f5f3ff"
        self._selection_border = "#7c3aed"

        # Zoom scaling factor (1.0 = 100%)
        self.zoom_factor: float = 1.0

        # Point-and-click formula referencing state
        self.is_dragging_ref = False
        self.ref_start_cell: Optional[Tuple[int, int]] = None
        self._ref_token_start_idx: Optional[int] = None

        # Fill-handle dragging state
        self.is_dragging_fill = False
        self.fill_start_range: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

        self._setup_bindings()
        self.recompute_bounds()

    # -------------------------------------------------------------------------
    # Layout & Metrics Calculations
    # -------------------------------------------------------------------------

    @property
    def header_row_height(self) -> int:
        return max(18, int(self.HEADER_ROW_HEIGHT * self.zoom_factor))

    @property
    def header_col_width(self) -> int:
        return max(32, int(self.HEADER_COL_WIDTH * self.zoom_factor))

    def get_col_width(self, col: int) -> int:
        base = self.col_widths.get(col, self.DEFAULT_COL_WIDTH)
        return max(24, int(base * self.zoom_factor))

    def set_col_width(self, col: int, width: int):
        self.col_widths[col] = max(30, int(width / self.zoom_factor))
        self.recompute_bounds()
        self.redraw()

    def get_row_height(self, row: int) -> int:
        base = self.row_heights.get(row, self.DEFAULT_ROW_HEIGHT)
        return max(16, int(base * self.zoom_factor))

    def set_row_height(self, row: int, height: int):
        self.row_heights[row] = max(16, int(height / self.zoom_factor))
        self.recompute_bounds()
        self.redraw()

    def col_to_x(self, target_col: int) -> int:
        """Return the canvas x coordinate of the left edge of target_col."""
        x = self.header_col_width
        for c in range(target_col):
            x += self.get_col_width(c)
        return x

    def row_to_y(self, target_row: int) -> int:
        """Return the canvas y coordinate of the top edge of target_row."""
        y = self.header_row_height
        for r in range(target_row):
            y += self.get_row_height(r)
        return y

    def x_to_col(self, x: int) -> int:
        """Convert a canvas x coordinate to a column index."""
        if x < self.header_col_width:
            return -1
        cur_x = self.header_col_width
        for c in range(self.total_cols + 50):
            w = self.get_col_width(c)
            if cur_x <= x < cur_x + w:
                return c
            cur_x += w
        return self.total_cols - 1

    def y_to_row(self, y: int) -> int:
        """Convert a canvas y coordinate to a row index."""
        if y < self.header_row_height:
            return -1
        cur_y = self.header_row_height
        for r in range(self.total_rows + 50):
            h = self.get_row_height(r)
            if cur_y <= y < cur_y + h:
                return r
            cur_y += h
        return self.total_rows - 1

    def xview(self, *args):
        """Override canvas xview to guarantee virtualized viewport redraw."""
        res = super().xview(*args)
        self.redraw()
        return res

    def yview(self, *args):
        """Override canvas yview to guarantee virtualized viewport redraw."""
        res = super().yview(*args)
        self.redraw()
        return res

    def recompute_bounds(self):
        """Update canvas scrollable bounding box based on total populated cells."""
        # Find maximum populated row and column
        max_r = self.MIN_ROWS
        max_c = self.MIN_COLS
        for r, c in self.engine.raw_cells.keys():
            if r + 10 > max_r:
                max_r = r + 10
            if c + 5 > max_c:
                max_c = c + 5

        self.total_rows = max(max_r, self.total_rows)
        self.total_cols = max(max_c, self.total_cols)

        total_w = self.col_to_x(self.total_cols) + 100
        total_h = self.row_to_y(self.total_rows) + 100
        self.config(scrollregion=(0, 0, total_w, total_h))

    # -------------------------------------------------------------------------
    # Virtualized Canvas Rendering
    # -------------------------------------------------------------------------

    def redraw(self):
        """Render visible cells, headers, grid lines, selection rectangle, and fill handle."""
        self.delete("all")
        vx0 = self.canvasx(0)
        vy0 = self.canvasy(0)
        vx1 = vx0 + self.winfo_width()
        vy1 = vy0 + self.winfo_height()

        start_col = max(0, self.x_to_col(int(vx0)))
        end_col = min(self.total_cols - 1, self.x_to_col(int(vx1)) + 1)
        start_row = max(0, self.y_to_row(int(vy0)))
        end_row = min(self.total_rows - 1, self.y_to_row(int(vy1)) + 1)

        # 1. Selection Fill & Border
        min_sr = min(self.sel_start[0], self.sel_end[0])
        max_sr = max(self.sel_start[0], self.sel_end[0])
        min_sc = min(self.sel_start[1], self.sel_end[1])
        max_sc = max(self.sel_start[1], self.sel_end[1])

        sel_x1 = self.col_to_x(min_sc)
        sel_y1 = self.row_to_y(min_sr)
        sel_x2 = self.col_to_x(max_sc) + self.get_col_width(max_sc)
        sel_y2 = self.row_to_y(max_sr) + self.get_row_height(max_sr)

        if max_sr > min_sr or max_sc > min_sc:
            self.create_rectangle(
                sel_x1, sel_y1, sel_x2, sel_y2,
                fill=self._selection_bg, outline=self._selection_border, width=1.5,
                tags="selection"
            )

        # Draw fill-drag preview rectangle if dragging fill handle
        if self.is_dragging_fill and self.fill_start_range and getattr(self, 'fill_current_cell', None):
            (sr1, sc1), (sr2, sc2) = self.fill_start_range
            fr, fc = self.fill_current_cell
            preview_r1 = min(sr1, fr)
            preview_r2 = max(sr2, fr)
            preview_c1 = min(sc1, fc)
            preview_c2 = max(sc2, fc)
            px1 = self.col_to_x(preview_c1)
            py1 = self.row_to_y(preview_r1)
            px2 = self.col_to_x(preview_c2) + self.get_col_width(preview_c2)
            py2 = self.row_to_y(preview_r2) + self.get_row_height(preview_r2)
            self.create_rectangle(
                px1, py1, px2, py2,
                outline=self._accent_color, width=1.5, dash=(4, 3), tags="fill_preview"
            )

        # 2. Draw Cell Grid Lines & Content
        f_size = max(7, min(24, int(9 * self.zoom_factor)))
        for r in range(start_row, end_row + 1):
            y = self.row_to_y(r)
            h = self.get_row_height(r)
            for c in range(start_col, end_col + 1):
                x = self.col_to_x(c)
                w = self.get_col_width(c)

                # Cell grid rectangle
                self.create_rectangle(
                    x, y, x + w, y + h,
                    outline=self._grid_line_color, fill="", tags="grid"
                )

                # Cell Value Display
                val = self.engine.get_value(r, c)
                if val != "" and val is not None:
                    txt = self._format_display_value(val, r, c)
                    align = self.engine.cell_styles.get((r, c), {}).get("align", "auto")
                    is_bold = self.engine.cell_styles.get((r, c), {}).get("bold", False)

                    # Determine text anchor and offset
                    f_family = "Segoe UI"
                    f_weight = "bold" if is_bold else "normal"
                    font_spec = (f_family, f_size, f_weight)

                    fg_color = "#000000"
                    if isinstance(val, str) and val.startswith("#"):
                        fg_color = "#dc2626"  # Red for formula errors

                    # Text Alignment & boundary clipping against populated neighbor
                    has_right_neighbor = (r, c + 1) in self.engine.raw_cells and self.engine.raw_cells[(r, c + 1)] != ""
                    clip_w = max(10, w - 8) if has_right_neighbor else None

                    if align == "right" or (align == "auto" and isinstance(val, (int, float)) and not isinstance(val, bool)):
                        tx = x + w - 6
                        anchor = "e"
                    elif align == "center":
                        tx = x + w // 2
                        anchor = "center"
                    else:
                        tx = x + 6
                        anchor = "w"

                    if clip_w is not None and anchor == "w":
                        self.create_text(
                            tx, y + h // 2, text=txt, anchor=anchor,
                            font=font_spec, fill=fg_color, width=clip_w, tags="cell_text"
                        )
                    else:
                        self.create_text(
                            tx, y + h // 2, text=txt, anchor=anchor,
                            font=font_spec, fill=fg_color, tags="cell_text"
                        )

        # 3. Active Single Cell Highlight
        ac_r, ac_c = self.active_cell
        ac_x = self.col_to_x(ac_c)
        ac_y = self.row_to_y(ac_r)
        ac_w = self.get_col_width(ac_c)
        ac_h = self.get_row_height(ac_r)

        self.create_rectangle(
            ac_x, ac_y, ac_x + ac_w, ac_y + ac_h,
            outline=self._accent_color, width=2, tags="active_cell"
        )
        # Fill Handle Square at bottom-right corner of selection
        fh_x = sel_x2
        fh_y = sel_y2
        self.create_rectangle(
            fh_x - 4, fh_y - 4, fh_x + 2, fh_y + 2,
            fill=self._accent_color, outline="#ffffff", width=1, tags="fill_handle"
        )

        # 4. Floating Header Row (Columns A, B, C...)
        hdr_y0 = vy0
        hdr_y1 = vy0 + self.header_row_height

        # Header background banner
        self.create_rectangle(
            vx0, hdr_y0, vx1, hdr_y1,
            fill=self._header_bg, outline=self._grid_line_color, width=1, tags="header_row"
        )

        for c in range(start_col, end_col + 1):
            x = self.col_to_x(c)
            w = self.get_col_width(c)
            is_active_col = (min_sc <= c <= max_sc)
            bg_col = self._header_active_bg if is_active_col else self._header_bg
            fg_col = self._header_active_fg if is_active_col else self._header_fg

            self.create_rectangle(
                x, hdr_y0, x + w, hdr_y1,
                fill=bg_col, outline=self._grid_line_color, width=1, tags="header_row"
            )
            self.create_text(
                x + w // 2, hdr_y0 + self.header_row_height // 2,
                text=col_index_to_letter(c), font=("Segoe UI", f_size, "bold" if is_active_col else "normal"),
                fill=fg_col, tags="header_row"
            )

        # 5. Floating Header Column (Rows 1, 2, 3...)
        hdr_x0 = vx0
        hdr_x1 = vx0 + self.header_col_width

        self.create_rectangle(
            hdr_x0, vy0, hdr_x1, vy1,
            fill=self._header_bg, outline=self._grid_line_color, width=1, tags="header_col"
        )

        for r in range(start_row, end_row + 1):
            y = self.row_to_y(r)
            h = self.get_row_height(r)
            is_active_row = (min_sr <= r <= max_sr)
            bg_row = self._header_active_bg if is_active_row else self._header_bg
            fg_row = self._header_active_fg if is_active_row else self._header_fg

            self.create_rectangle(
                hdr_x0, y, hdr_x1, y + h,
                fill=bg_row, outline=self._grid_line_color, width=1, tags="header_col"
            )
            self.create_text(
                hdr_x0 + self.header_col_width // 2, y + h // 2,
                text=str(r + 1), font=("Segoe UI", f_size, "bold" if is_active_row else "normal"),
                fill=fg_row, tags="header_col"
            )

        # 6. Top-Left Corner Box
        self.create_rectangle(
            vx0, vy0, vx0 + self.header_col_width, vy0 + self.header_row_height,
            fill="#e9e5f5", outline=self._grid_line_color, width=1, tags="corner"
        )
        self.create_polygon(
            vx0 + self.header_col_width - 12, vy0 + self.header_row_height - 4,
            vx0 + self.header_col_width - 4, vy0 + self.header_row_height - 12,
            vx0 + self.header_col_width - 4, vy0 + self.header_row_height - 4,
            fill="#a78bfa", outline="", tags="corner"
        )

    def _format_display_value(self, val: Any, row: int, col: int) -> str:
        """Format number according to cell styling."""
        if val is None:
            return ""
        fmt = self.engine.cell_styles.get((row, col), {}).get("format", "general")
        if isinstance(val, float):
            if fmt == "fixed2":
                return f"{val:,.2f}"
            elif fmt == "currency":
                return f"${val:,.2f}"
            elif fmt == "percent":
                return f"{val * 100:.1f}%"
            elif val.is_integer():
                return str(int(val))
            return f"{val:.4f}".rstrip("0").rstrip(".")
        elif isinstance(val, int) and not isinstance(val, bool):
            if fmt == "currency":
                return f"${val:,.2f}"
            elif fmt == "fixed2":
                return f"{val:,.2f}"
            return str(val)
        return str(val)

    # -------------------------------------------------------------------------
    # Event Handlers & Interactions
    # -------------------------------------------------------------------------

    def _setup_bindings(self):
        """Bind keyboard, mouse, mousewheel, and resize events."""
        self.bind("<Button-1>", self._on_button_1)
        self.bind("<B1-Motion>", self._on_b1_motion)
        self.bind("<ButtonRelease-1>", self._on_b1_release)
        self.bind("<Double-Button-1>", self._on_double_click)
        self.bind("<Motion>", self._on_mouse_motion)
        self.bind("<Configure>", lambda e: self.redraw())

        # Mousewheel Scrolling & Zooming
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)

        # Keyboard Navigation
        self.bind("<Up>", lambda e: self._move_active(-1, 0, e.state & 0x0001))
        self.bind("<Down>", lambda e: self._move_active(1, 0, e.state & 0x0001))
        self.bind("<Left>", lambda e: self._move_active(0, -1, e.state & 0x0001))
        self.bind("<Right>", lambda e: self._move_active(0, 1, e.state & 0x0001))
        self.bind("<Tab>", self._on_tab_key)
        self.bind("<ISO_Left_Tab>", self._on_shift_tab_key)
        self.bind("<Return>", self._on_return_key)
        self.bind("<Delete>", lambda e: self.clear_selected_cells())
        self.bind("<BackSpace>", lambda e: self.clear_selected_cells())
        self.bind("<F2>", lambda e: self.start_inline_edit())

        # Typing alphanumeric starts editing immediately
        self.bind("<Key>", self._on_key_press)

    def _on_mousewheel(self, event):
        delta = int(-1 * (event.delta / 120)) if event.delta else 0
        self.yview_scroll(delta * 3, "units")
        return "break"

    def _on_shift_mousewheel(self, event):
        delta = int(-1 * (event.delta / 120)) if event.delta else 0
        self.xview_scroll(delta * 3, "units")
        return "break"

    def _on_ctrl_mousewheel(self, event):
        if event.delta and event.delta > 0:
            self.set_zoom(self.zoom_factor + 0.1)
        elif event.delta and event.delta < 0:
            self.set_zoom(self.zoom_factor - 0.1)
        return "break"

    def set_zoom(self, zoom_factor: float):
        """Set spreadsheet zoom scale (0.5 to 2.5) and refresh layout."""
        self.zoom_factor = max(0.5, min(2.5, float(zoom_factor)))
        self.recompute_bounds()
        self.redraw()

    def _on_mouse_motion(self, event):
        """Update cursor for column resize divider or fill-handle hover detection."""
        if self.resizing_col is not None or self.is_dragging_fill:
            return
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)

        # Check fill-handle hover
        (min_r, min_c), (max_r, max_c) = self.get_selected_range()
        fh_x = self.col_to_x(max_c) + self.get_col_width(max_c)
        fh_y = self.row_to_y(max_r) + self.get_row_height(max_r)
        if abs(cx - fh_x) <= 6 and abs(cy - fh_y) <= 6:
            self.config(cursor="crosshair")
            return

        # Check column divider hover
        if cy <= self.canvasy(0) + self.header_row_height:
            col = self._get_divider_col_at_x(cx)
            if col is not None:
                self.config(cursor="sb_h_double_arrow")
                return
        self.config(cursor="")

    def _get_divider_col_at_x(self, cx: float) -> Optional[int]:
        """Detect if cursor is within 4 pixels of a column divider."""
        cur_x = self.header_col_width
        for c in range(self.total_cols):
            cur_x += self.get_col_width(c)
            if abs(cx - cur_x) <= 4:
                return c
        return None

    def _is_expecting_formula_ref(self) -> bool:
        """Check if active inline editor or formula bar expects a cell reference insertion."""
        if self.is_dragging_ref:
            return True
        if not self.is_editing or not self.editor_entry:
            return False
        txt = self.editor_entry.get()
        if not txt.startswith("="):
            return False
        pos = self.editor_entry.index(tk.INSERT)
        prefix = txt[:pos].rstrip()
        if not prefix:
            return False
        return prefix[-1] in ('=', '(', '+', '-', '*', '/', ',', '^', '<', '>', '&', ':')

    def _on_button_1(self, event):
        self.focus_set()
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)

        # 1. Check for fill handle click
        (min_r, min_c), (max_r, max_c) = self.get_selected_range()
        fh_x = self.col_to_x(max_c) + self.get_col_width(max_c)
        fh_y = self.row_to_y(max_r) + self.get_row_height(max_r)
        if abs(cx - fh_x) <= 6 and abs(cy - fh_y) <= 6:
            self.is_dragging_fill = True
            self.fill_start_range = ((min_r, min_c), (max_r, max_c))
            self.fill_current_cell = (self.y_to_row(int(cy)), self.x_to_col(int(cx)))
            return

        # 2. Check for column divider drag
        divider_col = self._get_divider_col_at_x(cx)
        if cy <= self.canvasy(0) + self.header_row_height and divider_col is not None:
            self.resizing_col = divider_col
            self.resize_start_x = event.x
            self.resize_start_w = self.get_col_width(divider_col)
            return

        # 3. Check for point-and-click formula reference insertion
        row = self.y_to_row(int(cy))
        col = self.x_to_col(int(cx))
        if self._is_expecting_formula_ref() and row >= 0 and col >= 0:
            self.is_dragging_ref = True
            self.ref_start_cell = (row, col)
            pos = self.editor_entry.index(tk.INSERT)
            self._ref_token_start_idx = pos
            ref_str = cell_to_ref(row, col)
            self.editor_entry.insert(pos, ref_str)
            self.editor_entry.icursor(pos + len(ref_str))
            self._on_editor_change()
            return

        # Commit existing inline edit if open
        if self.is_editing:
            self.commit_inline_edit()

        # Click on Top-Left Corner -> Select All
        if cx < self.header_col_width and cy < self.canvasy(0) + self.header_row_height:
            self.select_all_cells()
            return

        # Click on Column Header -> Select Entire Column
        if cy < self.canvasy(0) + self.header_row_height:
            col = self.x_to_col(int(cx))
            if col >= 0:
                self.active_cell = (0, col)
                self.sel_start = (0, col)
                self.sel_end = (self.total_rows - 1, col)
                self._notify_select()
                self.redraw()
            return

        # Click on Row Header -> Select Entire Row
        if cx < self.header_col_width:
            row = self.y_to_row(int(cy))
            if row >= 0:
                self.active_cell = (row, 0)
                self.sel_start = (row, 0)
                self.sel_end = (row, self.total_cols - 1)
                self._notify_select()
                self.redraw()
            return

        # Normal Cell Click
        if row >= 0 and col >= 0:
            if event.state & 0x0001:  # Shift-click
                self.sel_end = (row, col)
            else:
                self.active_cell = (row, col)
                self.sel_start = (row, col)
                self.sel_end = (row, col)
            self.is_selecting = True
            self._notify_select()
            self.redraw()

    def _on_b1_motion(self, event):
        if self.resizing_col is not None:
            dx = event.x - self.resize_start_x
            new_w = max(35, self.resize_start_w + dx)
            self.col_widths[self.resizing_col] = new_w
            self.recompute_bounds()
            self.redraw()
            return

        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)
        row = self.y_to_row(int(cy))
        col = self.x_to_col(int(cx))

        if self.is_dragging_fill:
            if row >= 0 and col >= 0:
                self.fill_current_cell = (row, col)
                self.redraw()
            return

        if self.is_dragging_ref and self.ref_start_cell:
            if row >= 0 and col >= 0 and self.editor_entry:
                r1, c1 = self.ref_start_cell
                min_r, max_r = min(r1, row), max(r1, row)
                min_c, max_c = min(c1, col), max(c1, col)
                if min_r == max_r and min_c == max_c:
                    token = cell_to_ref(min_r, min_c)
                else:
                    token = f"{cell_to_ref(min_r, min_c)}:{cell_to_ref(max_r, max_c)}"

                cur_txt = self.editor_entry.get()
                start_idx = self._ref_token_start_idx if self._ref_token_start_idx is not None else 0
                end_idx = start_idx
                while end_idx < len(cur_txt) and (cur_txt[end_idx].isalnum() or cur_txt[end_idx] in ('$', ':')):
                    end_idx += 1
                new_txt = cur_txt[:start_idx] + token + cur_txt[end_idx:]
                self.editor_entry.delete(0, tk.END)
                self.editor_entry.insert(0, new_txt)
                self.editor_entry.icursor(start_idx + len(token))
                self._on_editor_change()
            return

        if self.is_selecting:
            if row >= 0 and col >= 0:
                self.sel_end = (row, col)
                self._notify_select()
                self.redraw()

    def _on_b1_release(self, event):
        self.resizing_col = None
        self.is_selecting = False

        if self.is_dragging_fill:
            self.is_dragging_fill = False
            if self.fill_start_range and getattr(self, 'fill_current_cell', None):
                self._execute_fill(self.fill_start_range, self.fill_current_cell)
            self.fill_start_range = None
            self.fill_current_cell = None
            self.redraw()
            return

        if self.is_dragging_ref:
            self.is_dragging_ref = False
            self.ref_start_cell = None
            self._ref_token_start_idx = None
            if self.editor_entry:
                self.editor_entry.focus_set()
            return

    def _execute_fill(self, start_range: Tuple[Tuple[int, int], Tuple[int, int]], target_cell: Tuple[int, int]):
        """Perform smart auto-fill or formula extrapolation when drag-filling."""
        (min_r, min_c), (max_r, max_c) = start_range
        tgt_r, tgt_c = target_cell

        old_cells = {}
        new_cells = {}

        # Dragging Down
        if tgt_r > max_r:
            h = max_r - min_r + 1
            for c in range(min_c, max_c + 1):
                vals = [self.engine.get_value(r, c) for r in range(min_r, max_r + 1)]
                num_vals = []
                for v in vals:
                    try:
                        num_vals.append(float(v) if v != "" and v is not None else None)
                    except (ValueError, TypeError):
                        num_vals.append(None)

                is_numeric_series = len(num_vals) >= 2 and all(v is not None for v in num_vals)
                if is_numeric_series:
                    step = (num_vals[-1] - num_vals[0]) / (len(num_vals) - 1)
                    for r in range(max_r + 1, tgt_r + 1):
                        delta_steps = r - max_r
                        new_num = num_vals[-1] + step * delta_steps
                        val_str = str(int(new_num)) if new_num.is_integer() else f"{new_num:.2f}"
                        old_cells[(r, c)] = self.engine.get_raw(r, c)
                        new_cells[(r, c)] = val_str
                        self.engine.set_cell(r, c, val_str)
                else:
                    for r in range(max_r + 1, tgt_r + 1):
                        src_r = min_r + ((r - max_r - 1) % h)
                        raw_src = self.engine.get_raw(src_r, c)
                        dr = r - src_r
                        shifted = shift_formula_references(raw_src, dr, 0) if raw_src.startswith("=") else raw_src
                        old_cells[(r, c)] = self.engine.get_raw(r, c)
                        new_cells[(r, c)] = shifted
                        self.engine.set_cell(r, c, shifted)

            self.sel_end = (tgt_r, max_c)

        # Dragging Right
        elif tgt_c > max_c:
            w = max_c - min_c + 1
            for r in range(min_r, max_r + 1):
                vals = [self.engine.get_value(r, c) for c in range(min_c, max_c + 1)]
                num_vals = []
                for v in vals:
                    try:
                        num_vals.append(float(v) if v != "" and v is not None else None)
                    except (ValueError, TypeError):
                        num_vals.append(None)

                is_numeric_series = len(num_vals) >= 2 and all(v is not None for v in num_vals)
                if is_numeric_series:
                    step = (num_vals[-1] - num_vals[0]) / (len(num_vals) - 1)
                    for c in range(max_c + 1, tgt_c + 1):
                        delta_steps = c - max_c
                        new_num = num_vals[-1] + step * delta_steps
                        val_str = str(int(new_num)) if new_num.is_integer() else f"{new_num:.2f}"
                        old_cells[(r, c)] = self.engine.get_raw(r, c)
                        new_cells[(r, c)] = val_str
                        self.engine.set_cell(r, c, val_str)
                else:
                    for c in range(max_c + 1, tgt_c + 1):
                        src_c = min_c + ((c - max_c - 1) % w)
                        raw_src = self.engine.get_raw(r, src_c)
                        dc = c - src_c
                        shifted = shift_formula_references(raw_src, 0, dc) if raw_src.startswith("=") else raw_src
                        old_cells[(r, c)] = self.engine.get_raw(r, c)
                        new_cells[(r, c)] = shifted
                        self.engine.set_cell(r, c, shifted)

            self.sel_end = (max_r, tgt_c)

        if old_cells:
            self._save_undo_state({"type": "batch_cells", "old_cells": old_cells, "new_cells": new_cells})
            self.engine.recalculate_all()
            self.recompute_bounds()
            self.redraw()
            self._notify_select()
            if self.on_modified_cb:
                self.on_modified_cb()

    def toggle_bold_selection(self):
        """Toggle bold font on selected cells."""
        (min_r, min_c), (max_r, max_c) = self.get_selected_range()
        old_styles = dict(self.engine.cell_styles)
        first_bold = self.engine.cell_styles.get((min_r, min_c), {}).get("bold", False)
        new_bold = not first_bold
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                st = dict(self.engine.cell_styles.get((r, c), {}))
                st["bold"] = new_bold
                self.engine.cell_styles[(r, c)] = st
        self._save_undo_state({"type": "styles", "old_styles": old_styles, "new_styles": dict(self.engine.cell_styles)})
        self.redraw()
        if self.on_modified_cb:
            self.on_modified_cb()

    def format_selection(self, fmt: str):
        """Apply number format (currency, percent, fixed2, general) to selection."""
        (min_r, min_c), (max_r, max_c) = self.get_selected_range()
        old_styles = dict(self.engine.cell_styles)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                st = dict(self.engine.cell_styles.get((r, c), {}))
                st["format"] = fmt
                self.engine.cell_styles[(r, c)] = st
        self._save_undo_state({"type": "styles", "old_styles": old_styles, "new_styles": dict(self.engine.cell_styles)})
        self.redraw()
        if self.on_modified_cb:
            self.on_modified_cb()

    def set_alignment(self, align: str):
        """Apply alignment (left, center, right) to selection."""
        (min_r, min_c), (max_r, max_c) = self.get_selected_range()
        old_styles = dict(self.engine.cell_styles)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                st = dict(self.engine.cell_styles.get((r, c), {}))
                st["align"] = align
                self.engine.cell_styles[(r, c)] = st
        self._save_undo_state({"type": "styles", "old_styles": old_styles, "new_styles": dict(self.engine.cell_styles)})
        self.redraw()
        if self.on_modified_cb:
            self.on_modified_cb()

    def _on_double_click(self, event):
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)

        # Double click divider -> auto-fit column width
        divider_col = self._get_divider_col_at_x(cx)
        if cy <= self.canvasy(0) + self.HEADER_ROW_HEIGHT and divider_col is not None:
            self._autofit_column_width(divider_col)
            return

        row = self.y_to_row(int(cy))
        col = self.x_to_col(int(cx))
        if row >= 0 and col >= 0:
            self.active_cell = (row, col)
            self.sel_start = (row, col)
            self.sel_end = (row, col)
            self.start_inline_edit()

    def _autofit_column_width(self, col: int):
        """Calculate optimal column width based on content."""
        max_chars = len(col_index_to_letter(col))
        for r in range(self.total_rows):
            val = self.engine.get_value(r, col)
            if val != "" and val is not None:
                max_chars = max(max_chars, len(str(val)))
        ideal_w = max(60, min(350, max_chars * 9 + 24))
        self.set_col_width(col, ideal_w)

    def _move_active(self, dr: int, dc: int, extend_selection: bool = False):
        if self.is_editing:
            self.commit_inline_edit()
        r, c = self.active_cell
        new_r = max(0, min(self.total_rows - 1, r + dr))
        new_c = max(0, min(self.total_cols - 1, c + dc))

        if extend_selection:
            self.sel_end = (new_r, new_c)
        else:
            self.active_cell = (new_r, new_c)
            self.sel_start = (new_r, new_c)
            self.sel_end = (new_r, new_c)

        self._ensure_cell_visible(new_r, new_c)
        self._notify_select()
        self.redraw()

    def _on_tab_key(self, event):
        self._move_active(0, 1)
        return "break"

    def _on_shift_tab_key(self, event):
        self._move_active(0, -1)
        return "break"

    def _on_return_key(self, event):
        if self.is_editing:
            self.commit_inline_edit()
        self._move_active(1, 0)
        return "break"

    def _on_key_press(self, event):
        """Typing a printable character begins editing active cell."""
        if self.is_editing:
            return
        if event.char and event.char.isprintable() and ord(event.char) >= 32:
            self.start_inline_edit(initial_text=event.char)
            return "break"

    def _ensure_cell_visible(self, row: int, col: int):
        """Scroll canvas viewport if necessary to keep cell in view."""
        vx0 = self.canvasx(0)
        vy0 = self.canvasy(0)
        vw = self.winfo_width()
        vh = self.winfo_height()

        cx = self.col_to_x(col)
        cw = self.get_col_width(col)
        cy = self.row_to_y(row)
        ch = self.get_row_height(row)

        if cx + cw > vx0 + vw:
            self.xview_moveto((cx + cw - vw + 40) / (self.col_to_x(self.total_cols) + 100))
        elif cx < vx0 + self.header_col_width:
            self.xview_moveto(max(0, cx - self.header_col_width - 20) / (self.col_to_x(self.total_cols) + 100))

        if cy + ch > vy0 + vh:
            self.yview_moveto((cy + ch - vh + 40) / (self.row_to_y(self.total_rows) + 100))
        elif cy < vy0 + self.header_row_height:
            self.yview_moveto(max(0, cy - self.header_row_height - 20) / (self.row_to_y(self.total_rows) + 100))

    # -------------------------------------------------------------------------
    # Inline Cell Editor Management
    # -------------------------------------------------------------------------

    def start_inline_edit(self, initial_text: Optional[str] = None):
        """Launch the overlay Entry widget on active cell."""
        if self.is_editing:
            return
        r, c = self.active_cell
        raw_val = self.engine.get_raw(r, c)
        edit_text = initial_text if initial_text is not None else raw_val

        x = self.col_to_x(c)
        y = self.row_to_y(r)
        w = self.get_col_width(c)
        h = self.get_row_height(r)

        self.editor_entry = ttk.Entry(self, font=("Segoe UI", max(8, int(9 * self.zoom_factor))))
        self.editor_entry.insert(0, edit_text)
        if initial_text is None:
            self.editor_entry.select_range(0, tk.END)
        else:
            self.editor_entry.icursor(tk.END)

        self.create_window(x, y, window=self.editor_entry, width=w + 1, height=h + 1, anchor="nw", tags="editor_win")
        self.editor_entry.focus_set()
        self.is_editing = True

        # Editor Bindings
        self.editor_entry.bind("<Return>", lambda e: self.commit_inline_edit(move_down=True))
        self.editor_entry.bind("<Tab>", lambda e: self.commit_inline_edit(move_right=True))
        self.editor_entry.bind("<Escape>", lambda e: self.cancel_inline_edit())
        self.editor_entry.bind("<KeyRelease>", lambda e: self._on_editor_change())

    def _on_editor_change(self):
        """Sync live typing to parent formula bar."""
        if self.editor_entry and self.on_cell_select_cb:
            text = self.editor_entry.get()
            self.on_cell_select_cb(self.active_cell, text, in_editor=True)

    def commit_inline_edit(self, move_down: bool = False, move_right: bool = False):
        """Commit edited text to FormulaEngine and recalculate."""
        if not self.is_editing or not self.editor_entry:
            return
        new_val = self.editor_entry.get()
        r, c = self.active_cell
        old_val = self.engine.get_raw(r, c)

        if new_val != old_val:
            self._save_undo_state({"type": "set_cell", "r": r, "c": c, "old": old_val, "new": new_val})
            self.engine.set_cell(r, c, new_val)
            self.engine.recalculate_all()
            if self.on_modified_cb:
                self.on_modified_cb()

        self._cleanup_editor()
        if move_down:
            self._move_active(1, 0)
        elif move_right:
            self._move_active(0, 1)
        else:
            self.redraw()

    def cancel_inline_edit(self):
        """Cancel inline edit without saving."""
        if not self.is_editing:
            return
        self._cleanup_editor()
        self.redraw()

    def _cleanup_editor(self):
        self.is_editing = False
        if self.editor_entry:
            self.editor_entry.destroy()
            self.editor_entry = None
        self.delete("editor_win")
        self.focus_set()
        self._notify_select()

    def _notify_select(self):
        """Notify parent tab of active cell change."""
        if self.on_cell_select_cb:
            r, c = self.active_cell
            raw_text = self.engine.get_raw(r, c)
            self.on_cell_select_cb((r, c), raw_text, in_editor=False)

    # -------------------------------------------------------------------------
    # Undo / Redo Mechanics
    # -------------------------------------------------------------------------

    def _save_undo_state(self, action: Dict[str, Any]):
        self._undo_stack.append(action)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        action = self._undo_stack.pop()
        self._redo_stack.append(action)
        if action["type"] == "set_cell":
            self.engine.set_cell(action["r"], action["c"], action["old"])
        elif action["type"] == "batch_cells":
            for (r, c), old_val in action["old_cells"].items():
                self.engine.set_cell(r, c, old_val)
        elif action["type"] == "styles":
            self.engine.cell_styles = dict(action["old_styles"])
        elif action["type"] == "replace_sheet":
            self.engine.raw_cells = dict(action["old_raw"])
            self.engine.cell_styles = dict(action["old_styles"])
            self.recompute_bounds()
        self.engine.recalculate_all()
        self.redraw()
        self._notify_select()
        if self.on_modified_cb:
            self.on_modified_cb()

    def redo(self):
        if not self._redo_stack:
            return
        action = self._redo_stack.pop()
        self._undo_stack.append(action)
        if action["type"] == "set_cell":
            self.engine.set_cell(action["r"], action["c"], action["new"])
        elif action["type"] == "batch_cells":
            for (r, c), new_val in action["new_cells"].items():
                self.engine.set_cell(r, c, new_val)
        elif action["type"] == "styles":
            self.engine.cell_styles = dict(action["new_styles"])
        elif action["type"] == "replace_sheet":
            self.engine.raw_cells = dict(action["new_raw"])
            self.engine.cell_styles = dict(action["new_styles"])
            self.recompute_bounds()
        self.engine.recalculate_all()
        self.redraw()
        self._notify_select()
        if self.on_modified_cb:
            self.on_modified_cb()

    # -------------------------------------------------------------------------
    # Selection, Clipboard & Batch Operations
    # -------------------------------------------------------------------------

    def get_selected_range(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        min_r = min(self.sel_start[0], self.sel_end[0])
        max_r = max(self.sel_start[0], self.sel_end[0])
        min_c = min(self.sel_start[1], self.sel_end[1])
        max_c = max(self.sel_start[1], self.sel_end[1])
        return ((min_r, min_c), (max_r, max_c))

    def select_all_cells(self):
        self.sel_start = (0, 0)
        self.sel_end = (self.total_rows - 1, self.total_cols - 1)
        self.active_cell = (0, 0)
        self._notify_select()
        self.redraw()

    def clear_selected_cells(self):
        """Clear raw content from all cells in the selected bounding box."""
        (min_r, min_c), (max_r, max_c) = self.get_selected_range()
        old_cells = {}
        new_cells = {}
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) in self.engine.raw_cells:
                    old_cells[(r, c)] = self.engine.raw_cells[(r, c)]
                    new_cells[(r, c)] = ""
                    self.engine.set_cell(r, c, "")
        if old_cells:
            self._save_undo_state({"type": "batch_cells", "old_cells": old_cells, "new_cells": new_cells})
            self.engine.recalculate_all()
            self.redraw()
            self._notify_select()
            if self.on_modified_cb:
                self.on_modified_cb()

    def copy_selection(self) -> str:
        """Copy selected range to system clipboard formatted as standard TSV."""
        (min_r, min_c), (max_r, max_c) = self.get_selected_range()
        rows_data = []
        for r in range(min_r, max_r + 1):
            row_vals = []
            for c in range(min_c, max_c + 1):
                # Prefer raw formula if present, else computed value
                val = self.engine.get_raw(r, c)
                row_vals.append(val)
            rows_data.append("\t".join(row_vals))
        tsv_text = "\n".join(rows_data)
        self.clipboard_clear()
        self.clipboard_append(tsv_text)
        return tsv_text

    def cut_selection(self) -> str:
        tsv = self.copy_selection()
        self.clear_selected_cells()
        return tsv

    def paste_clipboard(self):
        """Paste TSV, CSV, or Markdown table from clipboard starting at active cell."""
        try:
            raw_text = self.clipboard_get()
        except tk.TclError:
            return

        if not raw_text.strip():
            return

        lines = [line for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return

        # Determine delimiter: Tab, Pipe, or Comma
        first_line = lines[0]
        if "\t" in first_line:
            delimiter = "\t"
            parsed_rows = [l.split("\t") for l in lines]
        elif "|" in first_line:
            # Markdown table
            parsed_rows = []
            for l in lines:
                parts = [p.strip() for p in l.split("|")[1:-1]]
                if parts and not all(re.match(r"^:?-+:?$", p) for p in parts):
                    parsed_rows.append(parts)
        else:
            # Default CSV sniffer
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(raw_text[:2048])
                reader = csv.reader(io.StringIO(raw_text), dialect)
                parsed_rows = list(reader)
            except Exception:
                reader = csv.reader(io.StringIO(raw_text))
                parsed_rows = list(reader)

        if not parsed_rows:
            return

        start_r, start_c = self.active_cell
        old_cells = {}
        new_cells = {}

        for dr, row_data in enumerate(parsed_rows):
            r = start_r + dr
            for dc, cell_text in enumerate(row_data):
                c = start_c + dc
                old_cells[(r, c)] = self.engine.get_raw(r, c)
                new_cells[(r, c)] = cell_text
                self.engine.set_cell(r, c, cell_text)

        self._save_undo_state({"type": "batch_cells", "old_cells": old_cells, "new_cells": new_cells})
        self.engine.recalculate_all()
        self.recompute_bounds()
        self.sel_end = (start_r + len(parsed_rows) - 1, start_c + max(len(row) for row in parsed_rows) - 1)
        self.redraw()
        self._notify_select()
        if self.on_modified_cb:
            self.on_modified_cb()


# =============================================================================
# SpreadsheetTab Container (Notebook Page)
# =============================================================================

class SpreadsheetTab(ttk.Frame):
    """Complete Excel-like spreadsheet tab with formula bar, grid, and AI bridge."""

    def __init__(self, parent, theme: Optional[Dict] = None, on_modified_callback: Optional[Callable] = None,
                 on_selection_change: Optional[Callable] = None):
        super().__init__(parent)
        self.parent = parent
        self.theme = theme or {}
        self.on_modified_callback = on_modified_callback
        self.on_selection_change = on_selection_change

        # Polymorphic properties matching DocumentTab
        self.tab_type = "spreadsheet"
        self.file_path: Optional[str] = None
        self.is_modified = False
        self.current_mode = "spreadsheet"
        self.snapshots: List[Dict[str, Any]] = []

        # Engines
        self.engine = FormulaEngine()

        self._setup_ui()

    def _setup_ui(self):
        """Construct the formula bar, quick functions ribbon, and 2D grid."""
        # Top Formula Ribbon Bar
        self.toolbar_frame = ttk.Frame(self, padding=(4, 4, 4, 4))
        self.toolbar_frame.pack(fill=tk.X, side=tk.TOP)

        # Coordinate box (e.g. "A1")
        self.coord_var = tk.StringVar(value="A1")
        self.coord_entry = ttk.Entry(self.toolbar_frame, textvariable=self.coord_var, width=8, font=("Segoe UI", 9, "bold"))
        self.coord_entry.pack(side=tk.LEFT, padx=(2, 6))
        self.coord_entry.bind("<Return>", self._on_coord_entry_submit)
        ToolTip(self.coord_entry, "Active cell or range coordinate (type to jump, e.g. B4 or A1:C10)")

        # fx Label
        fx_label = ttk.Label(self.toolbar_frame, text="fx", font=("Segoe UI", 10, "italic bold"), foreground="#5b21b6")
        fx_label.pack(side=tk.LEFT, padx=(2, 4))

        # Formula input bar
        self.formula_var = tk.StringVar()
        self.formula_entry = ttk.Entry(self.toolbar_frame, textvariable=self.formula_var, font=("Segoe UI", 9))
        self.formula_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 8))
        self.formula_entry.bind("<Return>", self._on_formula_bar_submit)
        self.formula_entry.bind("<KeyRelease>", self._on_formula_bar_typing)
        ToolTip(self.formula_entry, "Formula or cell value editor (e.g. =SUM(A1:A10) or text)")

        # Quick Math, Formatting & Action Buttons
        btn_frame = ttk.Frame(self.toolbar_frame)
        btn_frame.pack(side=tk.RIGHT)

        b_sum = ttk.Button(btn_frame, text="∑ Sum", command=self._insert_sum_formula, style="Toolbar.TButton")
        b_sum.pack(side=tk.LEFT, padx=1)
        ToolTip(b_sum, "Auto-insert SUM formula for column or selected range")

        b_avg = ttk.Button(btn_frame, text="x̄ Avg", command=self._insert_avg_formula, style="Toolbar.TButton")
        b_avg.pack(side=tk.LEFT, padx=1)
        ToolTip(b_avg, "Auto-insert AVERAGE formula")

        b_rnd = ttk.Button(btn_frame, text="Round", command=self._insert_round_formula, style="Toolbar.TButton")
        b_rnd.pack(side=tk.LEFT, padx=1)
        ToolTip(b_rnd, "Round active cell formula or value to 2 decimal places")

        b_bold = ttk.Button(btn_frame, text="𝐁", command=self.toggle_bold_selection, width=3, style="Toolbar.TButton")
        b_bold.pack(side=tk.LEFT, padx=1)
        ToolTip(b_bold, "Toggle Bold font on selected cells (Ctrl+B)")

        b_curr = ttk.Button(btn_frame, text="$", command=lambda: self.format_selection("currency"), width=3, style="Toolbar.TButton")
        b_curr.pack(side=tk.LEFT, padx=1)
        ToolTip(b_curr, "Format selection as Currency ($X.XX)")

        b_pct = ttk.Button(btn_frame, text="%", command=lambda: self.format_selection("percent"), width=3, style="Toolbar.TButton")
        b_pct.pack(side=tk.LEFT, padx=1)
        ToolTip(b_pct, "Format selection as Percentage (XX.X%)")

        b_dec = ttk.Button(btn_frame, text=".00", command=lambda: self.format_selection("fixed2"), width=4, style="Toolbar.TButton")
        b_dec.pack(side=tk.LEFT, padx=1)
        ToolTip(b_dec, "Format selection with 2 decimal places")

        b_row = ttk.Button(btn_frame, text="+ Row", command=self._insert_row_above, style="Toolbar.TButton")
        b_row.pack(side=tk.LEFT, padx=1)
        ToolTip(b_row, "Insert empty row above active cell")

        b_col = ttk.Button(btn_frame, text="+ Col", command=self._insert_col_left, style="Toolbar.TButton")
        b_col.pack(side=tk.LEFT, padx=1)
        ToolTip(b_col, "Insert empty column to the left of active cell")

        b_md = ttk.Button(btn_frame, text="📋 MD Table", command=self._copy_as_markdown_table, style="Toolbar.TButton")
        b_md.pack(side=tk.LEFT, padx=1)
        ToolTip(b_md, "Copy sheet or selected range as a GitHub-flavored Markdown table")

        # Central 2D Grid with Scrollbars
        grid_container = ttk.Frame(self)
        grid_container.pack(fill=tk.BOTH, expand=True)

        self.v_scroll = ttk.Scrollbar(grid_container, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.h_scroll = ttk.Scrollbar(grid_container, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.grid_widget = SpreadsheetGrid(
            grid_container, self.engine,
            on_cell_select=self._on_cell_selected,
            on_modified=self._on_grid_modified,
            theme=self.theme
        )
        self.grid_widget.pack(fill=tk.BOTH, expand=True)

        self.grid_widget.config(
            xscrollcommand=self.h_scroll.set,
            yscrollcommand=self.v_scroll.set
        )
        self.h_scroll.config(command=self.grid_widget.xview)
        self.v_scroll.config(command=self.grid_widget.yview)

        # Bind Context Menu
        self.grid_widget.bind("<Button-3>", self._show_context_menu)

    # -------------------------------------------------------------------------
    # Formula Bar & UI Synchronization
    # -------------------------------------------------------------------------

    def _on_cell_selected(self, coord: Tuple[int, int], raw_text: str, in_editor: bool = False):
        """Update coordinate box and formula bar when user selects or edits a cell."""
        r, c = coord
        ref_name = cell_to_ref(r, c)
        (min_r, min_c), (max_r, max_c) = self.grid_widget.get_selected_range()
        if max_r > min_r or max_c > min_c:
            ref_name = f"{cell_to_ref(min_r, min_c)}:{cell_to_ref(max_r, max_c)}"

        self.coord_var.set(ref_name)
        if not in_editor:
            self.formula_var.set(raw_text)

        if self.on_selection_change:
            self.on_selection_change()

    def _on_coord_entry_submit(self, event):
        """Jump to coordinate or range typed in coordinate entry."""
        target = self.coord_var.get().strip().upper()
        rng = parse_range_ref(target)
        if rng:
            (r1, c1), (r2, c2) = rng
            self.grid_widget.active_cell = (r1, c1)
            self.grid_widget.sel_start = (r1, c1)
            self.grid_widget.sel_end = (r2, c2)
            self.grid_widget._ensure_cell_visible(r1, c1)
            self.grid_widget.redraw()
            return

        single = parse_cell_ref(target)
        if single:
            r, c = single
            self.grid_widget.active_cell = (r, c)
            self.grid_widget.sel_start = (r, c)
            self.grid_widget.sel_end = (r, c)
            self.grid_widget._ensure_cell_visible(r, c)
            self.grid_widget.redraw()

    def _on_formula_bar_typing(self, event):
        """Sync text from top formula bar directly to inline editor if open."""
        if self.grid_widget.is_editing and self.grid_widget.editor_entry:
            self.grid_widget.editor_entry.delete(0, tk.END)
            self.grid_widget.editor_entry.insert(0, self.formula_var.get())

    def _on_formula_bar_submit(self, event):
        """Commit formula bar text to active cell."""
        new_val = self.formula_var.get()
        r, c = self.grid_widget.active_cell
        old_val = self.engine.get_raw(r, c)
        if new_val != old_val:
            self.grid_widget._save_undo_state({"type": "set_cell", "r": r, "c": c, "old": old_val, "new": new_val})
            self.engine.set_cell(r, c, new_val)
            self.engine.recalculate_all()
            self._on_grid_modified()
            self.grid_widget.redraw()
        self.grid_widget.focus_set()

    def _on_grid_modified(self):
        self.is_modified = True
        if self.on_modified_callback:
            self.on_modified_callback()
        if self.on_selection_change:
            self.on_selection_change()

    # -------------------------------------------------------------------------
    # Quick Function Helpers
    # -------------------------------------------------------------------------

    def _insert_sum_formula(self):
        """Automatically detect column above active cell and insert =SUM(X1:XN)."""
        r, c = self.grid_widget.active_cell
        (min_r, min_c), (max_r, max_c) = self.grid_widget.get_selected_range()
        if max_r > min_r:
            formula = f"=SUM({cell_to_ref(min_r, min_c)}:{cell_to_ref(max_r, min_c)})"
            target_r = max_r + 1
            target_c = min_c
        elif r > 0:
            formula = f"=SUM({cell_to_ref(0, c)}:{cell_to_ref(r - 1, c)})"
            target_r, target_c = r, c
        else:
            formula = "=SUM()"
            target_r, target_c = r, c

        self.engine.set_cell(target_r, target_c, formula)
        self.engine.recalculate_all()
        self.grid_widget.active_cell = (target_r, target_c)
        self.grid_widget.sel_start = (target_r, target_c)
        self.grid_widget.sel_end = (target_r, target_c)
        self.formula_var.set(formula)
        self._on_grid_modified()
        self.grid_widget.redraw()

    def _insert_avg_formula(self):
        r, c = self.grid_widget.active_cell
        if r > 0:
            formula = f"=AVERAGE({cell_to_ref(0, c)}:{cell_to_ref(r - 1, c)})"
        else:
            formula = "=AVERAGE()"
        self.formula_var.set(formula)
        self._on_formula_bar_submit(None)

    def _insert_round_formula(self):
        r, c = self.grid_widget.active_cell
        raw = self.engine.get_raw(r, c)
        if raw.startswith("="):
            inner = raw[1:]
            formula = f"=ROUND({inner}, 2)"
        elif raw:
            formula = f"=ROUND({raw}, 2)"
        else:
            formula = "=ROUND(A1, 2)"
        self.formula_var.set(formula)
        self._on_formula_bar_submit(None)

    def _insert_row_above(self):
        r, _ = self.grid_widget.active_cell
        old_raw = dict(self.engine.raw_cells)
        old_styles = dict(self.engine.cell_styles)
        new_cells = {}
        new_styles = {}
        for (row, col), val in list(self.engine.raw_cells.items()):
            if row >= r:
                shifted_val = shift_formula_references(val, 1, 0) if isinstance(val, str) and val.startswith("=") else val
                new_cells[(row + 1, col)] = shifted_val
            else:
                new_cells[(row, col)] = val
        for (row, col), st in list(self.engine.cell_styles.items()):
            if row >= r:
                new_styles[(row + 1, col)] = st
            else:
                new_styles[(row, col)] = st
        self.grid_widget._save_undo_state({"type": "replace_sheet", "old_raw": old_raw, "old_styles": old_styles, "new_raw": new_cells, "new_styles": new_styles})
        self.engine.raw_cells = new_cells
        self.engine.cell_styles = new_styles
        self.engine.recalculate_all()
        self.grid_widget.recompute_bounds()
        self._on_grid_modified()
        self.grid_widget.redraw()

    def _insert_row_below(self):
        r, _ = self.grid_widget.active_cell
        target_r = r + 1
        old_raw = dict(self.engine.raw_cells)
        old_styles = dict(self.engine.cell_styles)
        new_cells = {}
        new_styles = {}
        for (row, col), val in list(self.engine.raw_cells.items()):
            if row >= target_r:
                shifted_val = shift_formula_references(val, 1, 0) if isinstance(val, str) and val.startswith("=") else val
                new_cells[(row + 1, col)] = shifted_val
            else:
                new_cells[(row, col)] = val
        for (row, col), st in list(self.engine.cell_styles.items()):
            if row >= target_r:
                new_styles[(row + 1, col)] = st
            else:
                new_styles[(row, col)] = st
        self.grid_widget._save_undo_state({"type": "replace_sheet", "old_raw": old_raw, "old_styles": old_styles, "new_raw": new_cells, "new_styles": new_styles})
        self.engine.raw_cells = new_cells
        self.engine.cell_styles = new_styles
        self.engine.recalculate_all()
        self.grid_widget.recompute_bounds()
        self._on_grid_modified()
        self.grid_widget.redraw()

    def _delete_selected_rows(self):
        (min_r, _), (max_r, _) = self.grid_widget.get_selected_range()
        del_count = max_r - min_r + 1
        old_raw = dict(self.engine.raw_cells)
        old_styles = dict(self.engine.cell_styles)
        new_cells = {}
        new_styles = {}
        for (row, col), val in list(self.engine.raw_cells.items()):
            if min_r <= row <= max_r:
                continue
            elif row > max_r:
                shifted_val = shift_formula_references(val, -del_count, 0) if isinstance(val, str) and val.startswith("=") else val
                new_cells[(row - del_count, col)] = shifted_val
            else:
                new_cells[(row, col)] = val
        for (row, col), st in list(self.engine.cell_styles.items()):
            if min_r <= row <= max_r:
                continue
            elif row > max_r:
                new_styles[(row - del_count, col)] = st
            else:
                new_styles[(row, col)] = st
        self.grid_widget._save_undo_state({"type": "replace_sheet", "old_raw": old_raw, "old_styles": old_styles, "new_raw": new_cells, "new_styles": new_styles})
        self.engine.raw_cells = new_cells
        self.engine.cell_styles = new_styles
        self.engine.recalculate_all()
        self.grid_widget.active_cell = (max(0, min_r), self.grid_widget.active_cell[1])
        self.grid_widget.sel_start = self.grid_widget.active_cell
        self.grid_widget.sel_end = self.grid_widget.active_cell
        self.grid_widget.recompute_bounds()
        self._on_grid_modified()
        self.grid_widget.redraw()

    def _insert_col_left(self):
        _, c = self.grid_widget.active_cell
        old_raw = dict(self.engine.raw_cells)
        old_styles = dict(self.engine.cell_styles)
        new_cells = {}
        new_styles = {}
        for (row, col), val in list(self.engine.raw_cells.items()):
            if col >= c:
                shifted_val = shift_formula_references(val, 0, 1) if isinstance(val, str) and val.startswith("=") else val
                new_cells[(row, col + 1)] = shifted_val
            else:
                new_cells[(row, col)] = val
        for (row, col), st in list(self.engine.cell_styles.items()):
            if col >= c:
                new_styles[(row, col + 1)] = st
            else:
                new_styles[(row, col)] = st
        self.grid_widget._save_undo_state({"type": "replace_sheet", "old_raw": old_raw, "old_styles": old_styles, "new_raw": new_cells, "new_styles": new_styles})
        self.engine.raw_cells = new_cells
        self.engine.cell_styles = new_styles
        self.engine.recalculate_all()
        self.grid_widget.recompute_bounds()
        self._on_grid_modified()
        self.grid_widget.redraw()

    def _insert_col_right(self):
        _, c = self.grid_widget.active_cell
        target_c = c + 1
        old_raw = dict(self.engine.raw_cells)
        old_styles = dict(self.engine.cell_styles)
        new_cells = {}
        new_styles = {}
        for (row, col), val in list(self.engine.raw_cells.items()):
            if col >= target_c:
                shifted_val = shift_formula_references(val, 0, 1) if isinstance(val, str) and val.startswith("=") else val
                new_cells[(row, col + 1)] = shifted_val
            else:
                new_cells[(row, col)] = val
        for (row, col), st in list(self.engine.cell_styles.items()):
            if col >= target_c:
                new_styles[(row, col + 1)] = st
            else:
                new_styles[(row, col)] = st
        self.grid_widget._save_undo_state({"type": "replace_sheet", "old_raw": old_raw, "old_styles": old_styles, "new_raw": new_cells, "new_styles": new_styles})
        self.engine.raw_cells = new_cells
        self.engine.cell_styles = new_styles
        self.engine.recalculate_all()
        self.grid_widget.recompute_bounds()
        self._on_grid_modified()
        self.grid_widget.redraw()

    def _delete_selected_cols(self):
        (_, min_c), (_, max_c) = self.grid_widget.get_selected_range()
        del_count = max_c - min_c + 1
        old_raw = dict(self.engine.raw_cells)
        old_styles = dict(self.engine.cell_styles)
        new_cells = {}
        new_styles = {}
        for (row, col), val in list(self.engine.raw_cells.items()):
            if min_c <= col <= max_c:
                continue
            elif col > max_c:
                shifted_val = shift_formula_references(val, 0, -del_count) if isinstance(val, str) and val.startswith("=") else val
                new_cells[(row, col - del_count)] = shifted_val
            else:
                new_cells[(row, col)] = val
        for (row, col), st in list(self.engine.cell_styles.items()):
            if min_c <= col <= max_c:
                continue
            elif col > max_c:
                new_styles[(row, col - del_count)] = st
            else:
                new_styles[(row, col)] = st
        self.grid_widget._save_undo_state({"type": "replace_sheet", "old_raw": old_raw, "old_styles": old_styles, "new_raw": new_cells, "new_styles": new_styles})
        self.engine.raw_cells = new_cells
        self.engine.cell_styles = new_styles
        self.engine.recalculate_all()
        self.grid_widget.active_cell = (self.grid_widget.active_cell[0], max(0, min_c))
        self.grid_widget.sel_start = self.grid_widget.active_cell
        self.grid_widget.sel_end = self.grid_widget.active_cell
        self.grid_widget.recompute_bounds()
        self._on_grid_modified()
        self.grid_widget.redraw()

    def toggle_bold_selection(self):
        """Toggle bold formatting on selected cells (Ctrl+B)."""
        self.grid_widget.toggle_bold_selection()

    def format_selection(self, fmt: str):
        """Format selected cells (currency, percent, fixed2, general)."""
        self.grid_widget.format_selection(fmt)

    def set_alignment(self, align: str):
        """Set text alignment on selected cells (left, center, right)."""
        self.grid_widget.set_alignment(align)

    def _copy_as_markdown_table(self):
        """Format sheet or selection as GitHub Flavored Markdown table and copy."""
        (min_r, min_c), (max_r, max_c) = self.grid_widget.get_selected_range()
        if min_r == max_r and min_c == max_c:
            populated = list(self.engine.raw_cells.keys())
            if populated:
                min_r = min(r for r, c in populated)
                max_r = max(r for r, c in populated)
                min_c = min(c for r, c in populated)
                max_c = max(c for r, c in populated)

        rows = []
        for r in range(min_r, max_r + 1):
            row_items = []
            for c in range(min_c, max_c + 1):
                v = self.engine.get_value(r, c)
                row_items.append(str(v).replace("|", "\\|") if v is not None else "")
            rows.append(row_items)

        if not rows:
            return

        headers = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []

        md_lines = []
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for dr in data_rows:
            dr = dr + [""] * (len(headers) - len(dr))
            md_lines.append("| " + " | ".join(dr[:len(headers)]) + " |")

        table_str = "\n".join(md_lines)
        self.clipboard_clear()
        self.clipboard_append(table_str)
        messagebox.showinfo("Copied", f"Copied {len(rows)}x{len(headers)} table to clipboard as Markdown.", parent=self)

    def to_markdown_document(self) -> str:
        """Render spreadsheet as a clean markdown document suitable for PDF/DOCX/HTML export."""
        populated = list(self.engine.raw_cells.keys())
        name = self.get_display_name()
        if not populated:
            return f"# {name}\n\n*Empty Spreadsheet*\n"

        min_r = min(r for r, c in populated)
        max_r = max(r for r, c in populated)
        min_c = min(c for r, c in populated)
        max_c = max(c for r, c in populated)

        lines = [f"# {name}", ""]
        headers = []
        for c in range(min_c, max_c + 1):
            val = self.engine.get_value(min_r, c)
            h_str = str(val).replace("|", "\\|") if val is not None and val != "" else col_index_to_letter(c)
            headers.append(h_str)

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for r in range(min_r + 1, max_r + 1):
            row_items = []
            for c in range(min_c, max_c + 1):
                val = self.engine.get_value(r, c)
                v_str = str(val).replace("|", "\\|") if val is not None and val != "" else ""
                row_items.append(v_str)
            lines.append("| " + " | ".join(row_items) + " |")

        lines.append("")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Context Menu & AI Integration
    # -------------------------------------------------------------------------

    def _show_context_menu(self, event):
        """Open right-click context menu on grid with selection & AI actions."""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Cut", command=self.grid_widget.cut_selection, accelerator="Ctrl+X")
        menu.add_command(label="Copy", command=self.grid_widget.copy_selection, accelerator="Ctrl+C")
        menu.add_command(label="Paste", command=self.grid_widget.paste_clipboard, accelerator="Ctrl+V")
        menu.add_command(label="Clear", command=self.grid_widget.clear_selected_cells, accelerator="Delete")
        menu.add_separator()
        menu.add_command(label="Insert Row Above", command=self._insert_row_above)
        menu.add_command(label="Insert Row Below", command=self._insert_row_below)
        menu.add_command(label="Delete Selected Row(s)", command=self._delete_selected_rows)
        menu.add_separator()
        menu.add_command(label="Insert Column Left", command=self._insert_col_left)
        menu.add_command(label="Insert Column Right", command=self._insert_col_right)
        menu.add_command(label="Delete Selected Column(s)", command=self._delete_selected_cols)
        menu.add_separator()
        menu.add_command(label="Toggle Bold", command=self.toggle_bold_selection, accelerator="Ctrl+B")
        menu.add_command(label="Format as Currency ($)", command=lambda: self.format_selection("currency"))
        menu.add_command(label="Format as Percent (%)", command=lambda: self.format_selection("percent"))
        menu.add_command(label="Format as Decimal (.00)", command=lambda: self.format_selection("fixed2"))
        menu.add_separator()
        menu.add_command(label="💬 Transfer Selection to AI Chat", command=self._transfer_selection_to_chat)
        menu.add_command(label="🤖 Ask AI About Selection", command=self._ask_ai_about_selection)
        menu.add_command(label="📋 Copy as Markdown Table", command=self._copy_as_markdown_table)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _transfer_selection_to_chat(self):
        """Transfer selected cell coordinates, values, and formulas to AI Sidebar."""
        (min_r, min_c), (max_r, max_c) = self.grid_widget.get_selected_range()
        notepad = self.winfo_toplevel()
        ai_sidebar = getattr(notepad, "ai_sidebar", None)
        if not ai_sidebar:
            messagebox.showwarning("AI Chat", "AI Chat sidebar is not active.", parent=self)
            return

        if not getattr(notepad, "ai_sidebar_visible", False):
            notepad._toggle_ai_sidebar()

        r_name = f"{cell_to_ref(min_r, min_c)}:{cell_to_ref(max_r, max_c)}"
        lines = [f"[Spreadsheet Range: {r_name}]"]
        for r in range(min_r, max_r + 1):
            row_parts = []
            for c in range(min_c, max_c + 1):
                raw = self.engine.get_raw(r, c)
                val = self.engine.get_value(r, c)
                coord = cell_to_ref(r, c)
                clean_raw = str(raw).replace("\r\n", " ").replace("\n", " ").strip()
                clean_val = str(val).replace("\r\n", " ").replace("\n", " ").strip()
                if clean_raw.startswith("="):
                    row_parts.append(f"{coord}: {clean_raw} => {clean_val}")
                elif clean_raw:
                    row_parts.append(f"{coord}: {clean_raw}")
                else:
                    row_parts.append(f"{coord}: [empty]")
            lines.append(" | ".join(row_parts))
        lines.append("[End Range]")

        snippet = "\n".join(lines)
        ai_sidebar.set_pending_selection(snippet, f"{cell_to_ref(min_r, min_c)}", f"{cell_to_ref(max_r, max_c)}")
        notepad._set_status(f"Transferred spreadsheet range {r_name} to AI chat")

    def _ask_ai_about_selection(self):
        self._transfer_selection_to_chat()
        notepad = self.winfo_toplevel()
        ai_sidebar = getattr(notepad, "ai_sidebar", None)
        if ai_sidebar and hasattr(ai_sidebar, "input_text"):
            ai_sidebar.input_text.insert(tk.END, "Please analyze the selected rows/columns and summarize key insights or anomalies:")
            ai_sidebar.input_text.focus_set()

    # -------------------------------------------------------------------------
    # AI Context Generation (Include Document)
    # -------------------------------------------------------------------------

    def get_ai_context(self) -> str:
        """Serialize complete spreadsheet state, structure, and formulas for AI LLM."""
        populated = list(self.engine.raw_cells.keys())
        lines = []
        name = self.get_display_name()
        lines.append(f"# Spreadsheet: {name}")

        (sel_min_r, sel_min_c), (sel_max_r, sel_max_c) = self.grid_widget.get_selected_range()
        sel_coord = f"{cell_to_ref(sel_min_r, sel_min_c)}:{cell_to_ref(sel_max_r, sel_max_c)}"
        lines.append(f"Active Selection: {sel_coord}")

        if not populated and (sel_min_r, sel_min_c) == (0, 0) and (sel_max_r, sel_max_c) == (0, 0):
            lines.append("[Spreadsheet is currently empty]")
        else:
            if populated:
                pop_min_r = min(r for r, c in populated)
                pop_max_r = max(r for r, c in populated)
                pop_min_c = min(c for r, c in populated)
                pop_max_c = max(c for r, c in populated)
                if sel_max_r > sel_min_r or sel_max_c > sel_min_c:
                    min_r = min(pop_min_r, sel_min_r)
                    max_r = max(pop_max_r, sel_max_r)
                    min_c = min(pop_min_c, sel_min_c)
                    max_c = max(pop_max_c, sel_max_c)
                else:
                    min_r, max_r, min_c, max_c = pop_min_r, pop_max_r, pop_min_c, pop_max_c
            else:
                min_r, max_r, min_c, max_c = sel_min_r, sel_max_r, sel_min_c, sel_max_c

            lines.append(f"Dimensions: Rows {min_r + 1} to {max_r + 1}, Columns {col_index_to_letter(min_c)} to {col_index_to_letter(max_c)}")
            lines.append("")

            headers = ["Row"] + [col_index_to_letter(c) for c in range(min_c, max_c + 1)]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

            formula_map = []
            for r in range(min_r, max_r + 1):
                row_items = [str(r + 1)]
                for c in range(min_c, max_c + 1):
                    raw = self.engine.get_raw(r, c)
                    val = self.engine.get_value(r, c)
                    if str(raw).startswith("="):
                        clean_raw = str(raw).replace("\r\n", " ").replace("\n", " ").strip()
                        clean_val = str(val).replace("\r\n", " ").replace("\n", " ").strip()
                        formula_map.append(f"- {cell_to_ref(r, c)}: `{clean_raw}` => {clean_val}")
                    val_str = (
                        str(val)
                        .replace("\r\n", " ")
                        .replace("\n", " ")
                        .replace("|", "\\|")
                        .strip()
                        if val != "" and val is not None
                        else ""
                    )
                    row_items.append(val_str)
                lines.append("| " + " | ".join(row_items) + " |")

            if formula_map:
                lines.append("")
                lines.append("### Active Formulas:")
                lines.extend(formula_map)

        lines.append("")
        lines.append("### Instructions for Editing or Populating the Spreadsheet:")
        lines.append("To insert, update, evaluate, or calculate cells, provide a fenced `spreadsheet-edits` block.")
        lines.append("Formatting rules:")
        lines.append("1. Write each cell update on its own line: `CELL: VALUE_OR_FORMULA`")
        lines.append("2. For row-by-row grading, evaluation, or data filling:")
        lines.append("   - Use sample grades and feedback already present in previous rows to guide your scoring criteria, scale, and feedback cadence.")
        lines.append("   - Emit updates ONLY for the cells that need to be populated or updated (e.g., target score and feedback columns like C3 and D3).")
        lines.append("   - Do NOT re-emit or overwrite existing question, response, or sample evaluation cells unless explicitly asked.")
        lines.append("   - Keep each cell value on a single line.")
        lines.append("3. Examples:")
        lines.append("```spreadsheet-edits")
        lines.append("C3: 0.8")
        lines.append("D3: Accurate response addressing all key criteria.")
        lines.append("C4: 0.5")
        lines.append("D4: Missing the secondary explanation required by the rubric.")
        lines.append("D11: =AVERAGE(C2:C10)")
        lines.append("```")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Polymorphic Interface with DocumentTab
    # -------------------------------------------------------------------------

    def get_display_name(self) -> str:
        if self.file_path:
            return os.path.basename(self.file_path)
        return "Untitled Sheet"

    def get_content(self) -> str:
        """Serialize sheet state to JSON string for session caching."""
        data = {
            "type": "spreadsheet",
            "cells": {f"{r},{c}": val for (r, c), val in self.engine.raw_cells.items()},
            "col_widths": self.grid_widget.col_widths,
            "row_heights": self.grid_widget.row_heights,
            "styles": {f"{r},{c}": s for (r, c), s in self.engine.cell_styles.items()},
        }
        return json.dumps(data, indent=2)

    def set_content(self, content_str: str):
        """Restore sheet state from JSON or CSV string."""
        if not content_str.strip():
            return
        if content_str.strip().startswith("{"):
            try:
                data = json.loads(content_str)
                if isinstance(data, dict) and data.get("type") == "spreadsheet":
                    self.engine.raw_cells.clear()
                    self.engine.cell_styles.clear()
                    for k, val in data.get("cells", {}).items():
                        r, c = map(int, k.split(","))
                        self.engine.set_cell(r, c, val)
                    self.grid_widget.col_widths = {int(k): v for k, v in data.get("col_widths", {}).items()}
                    self.grid_widget.row_heights = {int(k): v for k, v in data.get("row_heights", {}).items()}
                    for k, s in data.get("styles", {}).items():
                        r, c = map(int, k.split(","))
                        self.engine.cell_styles[(r, c)] = s
                    self.engine.recalculate_all()
                    self.grid_widget.recompute_bounds()
                    self.grid_widget.redraw()
                    return
            except Exception:
                pass

        self.engine.raw_cells.clear()
        delim = "\t" if "\t" in content_str and (content_str.count("\t") >= content_str.count(",")) else ","
        reader = csv.reader(io.StringIO(content_str), delimiter=delim)
        for r, row in enumerate(reader):
            for c, val in enumerate(row):
                if val != "":
                    self.engine.set_cell(r, c, val)
        self.engine.recalculate_all()
        self.grid_widget.recompute_bounds()
        self.grid_widget.redraw()

    def set_zoom(self, zoom_factor: float):
        """Polymorphic hook for main notepad zoom."""
        self.grid_widget.set_zoom(zoom_factor)

    def apply_page_settings(self, *args, **kwargs):
        """Polymorphic no-op for DocumentTab page layout."""
        pass

    def set_wrap(self, wrap: bool):
        """Polymorphic no-op for DocumentTab word wrap."""
        pass

    def apply_desk_color(self, color: str):
        """Polymorphic no-op for DocumentTab desk color in focus mode."""
        pass

    def center_cursor_vertically(self):
        """Polymorphic no-op for typewriter scrolling."""
        pass

    def _apply_syntax_highlighting(self):
        """Polymorphic no-op for syntax highlighting debouncer."""
        pass

    def switch_mode(self, mode: str, theme: Optional[Dict] = None):
        """Polymorphic no-op for Source/Visual mode toggling."""
        pass

    def apply_theme(self, theme: Dict):
        self.theme = theme
        self.grid_widget.theme = theme
        self.grid_widget._accent_color = theme.get("accent_purple", "#5b21b6")
        self.grid_widget.redraw()

    def undo(self):
        self.grid_widget.undo()

    def redo(self):
        self.grid_widget.redo()

    def cut(self):
        self.grid_widget.cut_selection()

    def copy(self):
        self.grid_widget.copy_selection()

    def paste(self):
        self.grid_widget.paste_clipboard()

    def select_all(self):
        self.grid_widget.select_all_cells()


# =============================================================================
# Spreadsheet Patch Parser & Diff Dialog
# =============================================================================

class SpreadsheetPatchParser:
    """Extract cell modifications and formulas from AI messages."""

    _SPECIFIC_BLOCK_RE = re.compile(r"```(?:spreadsheet-edits|spreadsheet|excel)\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    _GENERIC_BLOCK_RE = re.compile(r"```(?:csv|tsv|markdown|table)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    _RANGE_LINE_RE = re.compile(r"^\s*(?:[-*•]\s+)?(?:SET\s+)?([A-Za-z]+[0-9]+):([A-Za-z]+[0-9]+)\s*[:=]\s*(.+)$", re.IGNORECASE)
    _LINE_RE = re.compile(r"^\s*(?:[-*•]\s+)?(?:SET\s+)?([A-Za-z]+[0-9]+)\s*[:=]\s*(.+)$", re.IGNORECASE)

    @staticmethod
    def _extract_col_letter(header: str) -> Optional[str]:
        h = header.strip()
        if not h:
            return None
        # Exact column letter e.g. "C" or "AA"
        if re.match(r"^[A-Za-z]{1,2}$", h):
            return h.upper()
        # Look for (C) or [C]
        m = re.search(r"[\(\[]\s*([A-Za-z]{1,2})\s*[\)\]]", h)
        if m:
            return m.group(1).upper()
        # Look for "Col C", "Column C", "Col. C"
        m = re.search(r"\bcol(?:umn)?\.?\s*([A-Za-z]{1,2})\b", h, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    @classmethod
    def parse_patches(cls, ai_text: str) -> Dict[str, str]:
        """Extract mapping of 'A1' -> 'new_value_or_formula' from AI text."""
        patches = {}
        specific_blocks = cls._SPECIFIC_BLOCK_RE.findall(ai_text)
        if specific_blocks:
            sources = specific_blocks
            is_specific = True
        else:
            generic_blocks = cls._GENERIC_BLOCK_RE.findall(ai_text)
            sources = generic_blocks if generic_blocks else [ai_text]
            is_specific = False

        for block in sources:
            lines = [l.strip() for l in block.splitlines() if l.strip() and not l.strip().startswith("#")]

            # 1. Parse individual cell lines (CELL: VALUE or RANGE: VALUE)
            block_line_patches = {}
            for line in lines:
                m_range = cls._RANGE_LINE_RE.match(line)
                if m_range:
                    c1_str = m_range.group(1).upper()
                    c2_str = m_range.group(2).upper()
                    val = m_range.group(3).strip()
                    p1 = parse_cell_ref(c1_str)
                    p2 = parse_cell_ref(c2_str)
                    if p1 and p2:
                        min_r = min(p1[0], p2[0])
                        max_r = max(p1[0], p2[0])
                        min_c = min(p1[1], p2[1])
                        max_c = max(p1[1], p2[1])
                        for r in range(min_r, max_r + 1):
                            for c in range(min_c, max_c + 1):
                                dr = r - p1[0]
                                dc = c - p1[1]
                                shifted = shift_formula_references(val, dr, dc) if val.startswith("=") else val
                                block_line_patches[cell_to_ref(r, c)] = shifted
                    continue

                m = cls._LINE_RE.match(line)
                if m:
                    coord = m.group(1).upper()
                    val = m.group(2).strip()
                    if parse_cell_ref(coord):
                        block_line_patches[coord] = val

            # If explicit cell lines were found, prioritize them and avoid table fallbacks
            if block_line_patches:
                patches.update(block_line_patches)
                continue

            # 2. Table parsing (only if no explicit cell lines found in this block)
            table_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]
            if len(table_lines) >= 2:
                hdr_parts = [p.strip() for p in table_lines[0].split("|")[1:-1]]
                data_rows = []
                for tl in table_lines[1:]:
                    parts = [p.strip() for p in tl.split("|")[1:-1]]
                    if parts and not all(re.match(r"^:?-+:?$", p) for p in parts):
                        data_rows.append(parts)

                # Format A: | Cell | Value |
                if len(hdr_parts) == 2 and hdr_parts[0].lower() in ("cell", "coord", "coordinate", "ref"):
                    for row in data_rows:
                        if len(row) >= 2 and parse_cell_ref(row[0].upper()):
                            patches[row[0].upper()] = row[1]
                # Format B: | Row | C | D | or | Row | Score (C) | Feedback (D) |
                elif len(hdr_parts) > 1 and hdr_parts[0].lower() in ("row", "r", "#"):
                    col_letters = [cls._extract_col_letter(h) for h in hdr_parts[1:]]
                    for row in data_rows:
                        if not row:
                            continue
                        try:
                            r_num = int(row[0])
                        except ValueError:
                            continue
                        for col_idx, c_letter in enumerate(col_letters):
                            if c_letter and col_idx + 1 < len(row):
                                val = row[col_idx + 1]
                                patches[f"{c_letter}{r_num}"] = val
                # Format C: Full grid replacement only if inside a specific spreadsheet block
                elif data_rows and is_specific:
                    for c_offset, h_val in enumerate(hdr_parts):
                        c_letter = col_index_to_letter(c_offset)
                        patches[f"{c_letter}1"] = h_val
                    for r_offset, row in enumerate(data_rows):
                        for c_offset, val in enumerate(row):
                            c_letter = col_index_to_letter(c_offset)
                            patches[f"{c_letter}{r_offset + 2}"] = val

        return patches


class SpreadsheetDiffDialog(tk.Toplevel):
    """Side-by-side or tabular preview dialog for reviewing proposed AI spreadsheet changes."""

    def __init__(self, parent, tab: SpreadsheetTab, patches: Dict[str, str], on_apply: Optional[Callable] = None):
        super().__init__(parent)
        self.parent = parent
        self.tab = tab
        self.patches = patches
        self.on_apply = on_apply
        self.result = False

        self.title("Review AI Spreadsheet Changes")
        self.geometry("680x500")
        self.resizable(True, True)
        self.minsize(580, 420)
        self.transient(parent)
        self.grab_set()

        self._setup_ui()

        # Center on parent with auto-fitting geometry
        self.update_idletasks()
        try:
            req_w = max(680, self.winfo_reqwidth() + 24)
            req_h = max(500, self.winfo_reqheight() + 24)
            x = parent.winfo_rootx() + (parent.winfo_width() - req_w) // 2
            y = parent.winfo_rooty() + (parent.winfo_height() - req_h) // 2
            self.geometry(f"{req_w}x{req_h}+{max(10, x)}+{max(10, y)}")
        except Exception:
            pass

    def _setup_ui(self):
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header_lbl = ttk.Label(
            main_frame,
            text=f"The AI proposed {len(self.patches)} cell edit(s). Select the changes to apply:",
            font=("Segoe UI", 10, "bold")
        )
        header_lbl.pack(anchor=tk.W, pady=(0, 8))

        # Treeview table of proposed changes
        columns = ("apply", "cell", "old_val", "new_val", "computed")
        self.tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=12)
        self.tree.heading("apply", text="Apply")
        self.tree.heading("cell", text="Cell")
        self.tree.heading("old_val", text="Current Formula / Value")
        self.tree.heading("new_val", text="Proposed Formula / Value")
        self.tree.heading("computed", text="Computed Preview")

        self.tree.column("apply", width=60, anchor="center")
        self.tree.column("cell", width=70, anchor="center")
        self.tree.column("old_val", width=140, anchor="w")
        self.tree.column("new_val", width=160, anchor="w")
        self.tree.column("computed", width=140, anchor="w")

        scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Temporary engine to calculate computed preview
        temp_engine = FormulaEngine()
        temp_engine.raw_cells = dict(self.tab.engine.raw_cells)
        for coord_str, val in self.patches.items():
            pos = parse_cell_ref(coord_str)
            if pos:
                temp_engine.set_cell(pos[0], pos[1], val)
        temp_engine.recalculate_all()

        def _coord_sort_key(item: Tuple[str, Any]) -> Tuple[int, int]:
            coord = item[0]
            pos = parse_cell_ref(coord)
            return pos if pos else (999999, 999999)

        self.row_checked: Dict[str, bool] = {}
        for coord_str, new_val in sorted(self.patches.items(), key=_coord_sort_key):
            pos = parse_cell_ref(coord_str)
            old_val = self.tab.engine.get_raw(pos[0], pos[1]) if pos else ""
            comp_val = str(temp_engine.get_value(pos[0], pos[1])) if pos else ""

            item_id = self.tree.insert("", tk.END, values=("☑", coord_str, old_val, new_val, comp_val))
            self.row_checked[item_id] = True

        self.tree.bind("<Button-1>", self._on_tree_click)

        # Action Buttons
        btn_frame = ttk.Frame(main_frame, padding=(0, 10, 0, 0))
        btn_frame.pack(fill=tk.X)

        b_cancel = ttk.Button(btn_frame, text="Cancel", command=self.destroy)
        b_cancel.pack(side=tk.RIGHT, padx=4)
        ToolTip(b_cancel, "Discard all proposed AI changes")

        b_all = ttk.Button(btn_frame, text="Accept All", command=self._accept_all)
        b_all.pack(side=tk.RIGHT, padx=4)
        ToolTip(b_all, "Check all proposed changes and apply immediately")

        b_apply = ttk.Button(btn_frame, text="⚡ Apply Selected", command=self._apply_selected, style="Toolbar.TButton")
        b_apply.pack(side=tk.RIGHT, padx=4)
        ToolTip(b_apply, "Apply checked changes to the active spreadsheet")

    def _on_tree_click(self, event):
        item_id = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if item_id and col == "#1":  # Clicked 'apply' column
            is_checked = self.row_checked.get(item_id, True)
            self.row_checked[item_id] = not is_checked
            mark = "☑" if not is_checked else "☐"
            vals = list(self.tree.item(item_id, "values"))
            vals[0] = mark
            self.tree.item(item_id, values=vals)

    def _accept_all(self):
        for item_id in self.row_checked:
            self.row_checked[item_id] = True
        self._apply_selected()

    def _apply_selected(self):
        selected_patches = {}
        for item_id, checked in self.row_checked.items():
            if checked:
                vals = self.tree.item(item_id, "values")
                coord = vals[1]
                new_val = vals[3]
                selected_patches[coord] = new_val

        if self.on_apply:
            self.on_apply(selected_patches)
        elif self.tab:
            for coord, val in selected_patches.items():
                pos = parse_cell_ref(coord)
                if pos:
                    self.tab.engine.set_cell(pos[0], pos[1], val)
            self.tab.engine.recalculate_all()
            self.tab.grid_widget.recompute_bounds()
            self.tab.grid_widget.redraw()

        self.destroy()

    def _on_tree_click(self, event):
        item_id = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if item_id and col == "#1":  # Clicked 'apply' column
            is_checked = self.row_checked.get(item_id, True)
            self.row_checked[item_id] = not is_checked
            mark = "☑" if not is_checked else "☐"
            vals = list(self.tree.item(item_id, "values"))
            vals[0] = mark
            self.tree.item(item_id, values=vals)

    def _accept_all(self):
        for item_id in self.row_checked:
            self.row_checked[item_id] = True
        self._apply_selected()

    def _apply_selected(self):
        selected_patches = {}
        for item_id, checked in self.row_checked.items():
            if checked:
                vals = self.tree.item(item_id, "values")
                coord = vals[1]
                new_val = vals[3]
                selected_patches[coord] = new_val

        if self.on_apply:
            self.on_apply(selected_patches)
        elif self.tab:
            for coord, val in selected_patches.items():
                pos = parse_cell_ref(coord)
                if pos:
                    self.tab.engine.set_cell(pos[0], pos[1], val)
            self.tab.engine.recalculate_all()
            self.tab.grid_widget.recompute_bounds()
            self.tab.grid_widget.redraw()

        self.destroy()


# =============================================================================
# Spreadsheet File I/O Helpers
# =============================================================================

def open_spreadsheet_file(filepath: str, tab: SpreadsheetTab) -> bool:
    """Load an existing file (.csv, .tsv, .xlsx, .mdsheet) into SpreadsheetTab."""
    if not os.path.exists(filepath):
        return False

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".xlsx" and OPENPYXL_AVAILABLE:
        try:
            wb = openpyxl.load_workbook(filepath, data_only=False)
            sheet = wb.active
            tab.engine.raw_cells.clear()
            for r_idx, row in enumerate(sheet.iter_rows(values_only=False)):
                for c_idx, cell in enumerate(row):
                    if cell.value is not None:
                        tab.engine.set_cell(r_idx, c_idx, str(cell.value))
            tab.engine.recalculate_all()
            tab.grid_widget.recompute_bounds()
            tab.grid_widget.redraw()
            tab.file_path = filepath
            tab.is_modified = False
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Excel file:\n{e}", parent=tab)
            return False

    elif ext in (".csv", ".tsv"):
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            tab.set_content(content)
            tab.file_path = filepath
            tab.is_modified = False
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV/TSV file:\n{e}", parent=tab)
            return False

    elif ext == ".mdsheet":
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            tab.set_content(content)
            tab.file_path = filepath
            tab.is_modified = False
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read spreadsheet file:\n{e}", parent=tab)
            return False

    return False


def save_spreadsheet_file(tab: SpreadsheetTab, filepath: str) -> bool:
    """Save SpreadsheetTab contents to .csv, .tsv, .xlsx, or .mdsheet."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".xlsx" and OPENPYXL_AVAILABLE:
        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            for (r, c), raw in tab.engine.raw_cells.items():
                ws.cell(row=r + 1, column=c + 1, value=raw)
            wb.save(filepath)
            tab.file_path = filepath
            tab.is_modified = False
            return True
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save XLSX file:\n{e}", parent=tab)
            return False

    elif ext in (".csv", ".tsv"):
        try:
            delimiter = "\t" if ext == ".tsv" else ","
            populated = list(tab.engine.raw_cells.keys())
            max_r = max((r for r, c in populated), default=0)
            max_c = max((c for r, c in populated), default=0)

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=delimiter)
                for r in range(max_r + 1):
                    row_vals = [tab.engine.get_raw(r, c) for c in range(max_c + 1)]
                    writer.writerow(row_vals)
            tab.file_path = filepath
            tab.is_modified = False
            return True
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save CSV file:\n{e}", parent=tab)
            return False

    elif ext == ".mdsheet":
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(tab.get_content())
            tab.file_path = filepath
            tab.is_modified = False
            return True
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save mdsheet file:\n{e}", parent=tab)
            return False

    return False
