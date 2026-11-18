"""
MarkItDown Notepad - A lightweight markdown notepad using Microsoft's MarkItDown package
Supports opening any file type and converting it to markdown for viewing/editing.

Features:
- Source Mode: Edit raw markdown text
- Visual Mode: Rendered markdown view (also editable)
- Open any file type via MarkItDown conversion
- Standard notepad operations (new, open, save, save as)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, font, simpledialog, colorchooser
from tkinter import font as tkfont
import re
import os
import io
import base64
import hashlib
import tempfile
import zipfile
import threading
import queue
import time
import json
import shutil
import webbrowser
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Callable, Set, Union


def _normalize_anchor_key(label: str) -> str:
    """Normalize an anchor or reference label for cross-reference matching."""
    if not label:
        return ''
    return re.sub(r'\s+', ' ', label.strip().lower())

# Try to import PIL for image handling
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Install with: pip install Pillow")

# Try to import markitdown
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    print("Warning: markitdown not installed. Install with: pip install 'markitdown[all]'")

# File extensions supported for MarkItDown document conversion
MARKITDOWN_DOCUMENT_EXTENSIONS = {
    '.pdf', '.epub', '.docx', '.pptx', '.xlsx', '.xls',
    '.html', '.htm', '.msg', '.jpg', '.jpeg', '.png',
    '.gif', '.bmp', '.wav', '.mp3'
}

# Try to import tkhtmlview for visual rendering
try:
    from tkhtmlview import HTMLLabel
    HTMLVIEW_AVAILABLE = True
except ImportError:
    HTMLVIEW_AVAILABLE = False

# Try to import markdown for HTML conversion
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# Import session manager for document caching
try:
    from session_manager import get_session_manager, TabState
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False
    print("Warning: session_manager not found. Session persistence disabled.")

# Import AI chat module
try:
    from ai_chat import ChatSidebar, AISettingsDialog, get_ai_settings
    AI_CHAT_AVAILABLE = True
except ImportError:
    AI_CHAT_AVAILABLE = False
    print("Warning: ai_chat not found. AI features disabled.")

# Import text cleaner for algorithmic cleanup & reflow
try:
    from text_cleaner import clean_document, TextCleanerOptions
    TEXT_CLEANER_AVAILABLE = True
except ImportError:
    TEXT_CLEANER_AVAILABLE = False

# Import document image extractor
try:
    from document_image_extractor import DocumentImageExtractor, extract_document_images
    DOC_IMAGE_EXTRACTOR_AVAILABLE = True
except ImportError:
    DOC_IMAGE_EXTRACTOR_AVAILABLE = False

# Import diff panel
try:
    from diff_panel import DiffPanel
    DIFF_PANEL_AVAILABLE = True
except ImportError:
    DIFF_PANEL_AVAILABLE = False
    print("Warning: diff_panel module not found. Diff panel disabled.")

# Import export manager
try:
    from export_manager import (
        export_to_pdf, export_to_html, export_to_docx,
        get_available_export_formats, WEASYPRINT_AVAILABLE, DOCX_AVAILABLE,
        MARKDOWN_AVAILABLE as EXPORT_MARKDOWN_AVAILABLE
    )
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False
    WEASYPRINT_AVAILABLE = False
    DOCX_AVAILABLE = False
    EXPORT_MARKDOWN_AVAILABLE = False
    print("Warning: export_manager module not found. Export disabled.")

# Import spreadsheet module
try:
    from spreadsheet_tab import (
        SpreadsheetTab, SpreadsheetDiffDialog, open_spreadsheet_file,
        save_spreadsheet_file, SpreadsheetPatchParser, parse_cell_ref, cell_to_ref
    )
    SPREADSHEET_AVAILABLE = True
except ImportError as e:
    SPREADSHEET_AVAILABLE = False
    print(f"Warning: spreadsheet_tab module not found ({e}). Spreadsheet tabs disabled.")


def _fit_and_center_dialog(dialog: tk.Toplevel, parent: tk.Misc, min_w: int = 400, min_h: int = 300):
    """Ensure dialog is resizable in both X and Y, auto-fits without clipping, and is centered."""
    dialog.resizable(True, True)
    dialog.update_idletasks()
    req_w = max(min_w, dialog.winfo_reqwidth() + 24)
    req_h = max(min_h, dialog.winfo_reqheight() + 24)
    dialog.minsize(req_w, req_h)
    try:
        px = parent.winfo_rootx() + max(0, (parent.winfo_width() - req_w) // 2)
        py = parent.winfo_rooty() + max(0, (parent.winfo_height() - req_h) // 2)
        dialog.geometry(f"{req_w}x{req_h}+{max(10, px)}+{max(10, py)}")
    except Exception:
        dialog.geometry(f"{req_w}x{req_h}")


# =============================================================================
# Font Detection Utilities
# =============================================================================

def get_available_fonts():
    """Get list of available system fonts for tkinter"""
    try:
        return sorted(font.families())
    except Exception as e:
        # This can fail if called before Tkinter root window exists
        return []

def select_best_font(preferred_fonts, fallback="TkDefaultFont"):
    """
    Select the first available font from a list of preferences.
    
    Args:
        preferred_fonts: List of font names in order of preference
        fallback: Default font to use if none are available
    
    Returns:
        Name of the first available font, or fallback
    """
    available = get_available_fonts()
    if not available:  # If no fonts available, return fallback
        return fallback
    
    available_lower = [f.lower() for f in available]
    
    for font_name in preferred_fonts:
        if font_name.lower() in available_lower:
            # Return the properly cased version from the system
            idx = available_lower.index(font_name.lower())
            return available[idx]
    
    return fallback

# Define cross-platform font preferences
SANS_SERIF_FONTS = ["Segoe UI", "SF Pro Display", "Ubuntu", "DejaVu Sans", "Arial", "Helvetica"]
MONOSPACE_FONTS = ["Consolas", "SF Mono", "Monaco", "Ubuntu Mono", "DejaVu Sans Mono", "Courier New"]
VISUAL_FONTS = ["Calibri", "Segoe UI", "SF Pro Display", "Ubuntu", "DejaVu Sans", "Arial", "Helvetica"]

# Font selection will be done after Tkinter root window is created
SANS_FONT = None
MONO_FONT = None
VISUAL_FONT = None


def _shade_color(hex_color, amount):
    """Darken (amount < 0) or lighten (amount > 0) a #rrggbb colour by a 0..1 fraction."""
    c = (hex_color or '').lstrip('#')
    if len(c) != 6:
        return hex_color
    try:
        rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return hex_color
    if amount >= 0:
        rgb = [int(v + (255 - v) * amount) for v in rgb]
    else:
        rgb = [int(v * (1 + amount)) for v in rgb]
    return '#%02x%02x%02x' % tuple(max(0, min(255, v)) for v in rgb)


def _is_dark_color(hex_color):
    """Return True if a #rrggbb colour is perceptually dark."""
    c = (hex_color or '').lstrip('#')
    if len(c) == 6:
        try:
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0 < 0.5
        except ValueError:
            pass
    return False


# =============================================================================
# UI Components
# =============================================================================

class ToolTip:
    """Rich, screen-edge-clamped tooltip widget with multi-line wrap and dynamic text updates."""
    
    def __init__(self, widget, text, delay=400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.scheduled_id = None
        self.label = None
        
        widget.bind('<Enter>', self._on_enter, add=True)
        widget.bind('<Leave>', self._on_leave, add=True)
        widget.bind('<Button>', self._on_leave, add=True)
    
    def set_text(self, text):
        """Dynamically update tooltip text while widget is alive."""
        self.text = text
        if self.label and self.tooltip_window and self.tooltip_window.winfo_exists():
            self.label.config(text=text)
    
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
        
        # Word 2010 Royal Purple / Plum themed tooltip with high contrast
        self.label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#2e1065", foreground="#faf5ff",
            relief=tk.SOLID, borderwidth=1,
            font=(SANS_FONT, 9), padx=8, pady=5,
            wraplength=340
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


class FloatingMiniToolbar:
    """
    Floating Word-style mini selection toolbar that appears above selected text.
    Provides fast inline formatting (Bold, Italic, Highlight, Code, Link, Ask AI).
    """
    def __init__(self, tab):
        self.tab = tab
        self.editor = tab.source_editor
        self.window = None
        self._check_job = None
        
        self.editor.bind("<ButtonRelease-1>", self._on_mouse_release, add=True)
        self.editor.bind("<KeyPress>", self._on_key_press, add=True)
        self.editor.bind("<FocusOut>", lambda e: self.hide(), add=True)

    def _on_mouse_release(self, event=None):
        if self._check_job:
            self.editor.after_cancel(self._check_job)
        self._check_job = self.editor.after(120, self._check_and_show)

    def _on_key_press(self, event=None):
        self.hide()

    def _check_and_show(self):
        self._check_job = None
        try:
            if not self.editor.tag_ranges(tk.SEL):
                self.hide()
                return
            
            sel_text = self.editor.get(tk.SEL_FIRST, tk.SEL_LAST)
            if not sel_text or len(sel_text.strip()) == 0:
                self.hide()
                return
            
            bbox = self.editor.bbox(tk.SEL_FIRST)
            if not bbox:
                return
            
            bx, by, bw, bh = bbox
            root_x = self.editor.winfo_rootx() + bx
            root_y = self.editor.winfo_rooty() + by - 36
            
            if self.window and self.window.winfo_exists():
                self.window.geometry(f"+{max(10, root_x)}+{max(10, root_y)}")
                return
            
            self.window = win = tk.Toplevel(self.editor)
            win.wm_overrideredirect(True)
            win.configure(bg="#3b1d6e", padx=2, pady=2)
            
            frame = tk.Frame(win, bg="#f3effa", bd=1, relief=tk.SOLID)
            frame.pack(fill=tk.BOTH, expand=True)
            
            def _wrap_fmt(prefix, suffix):
                try:
                    notepad = self.editor.winfo_toplevel()
                    if hasattr(notepad, '_insert_format'):
                        notepad._insert_format(prefix, suffix)
                finally:
                    self.hide()

            def _ask_ai():
                try:
                    notepad = self.editor.winfo_toplevel()
                    if hasattr(notepad, '_on_ask_ai_selection'):
                        notepad._on_ask_ai_selection()
                    elif hasattr(notepad, 'ai_sidebar') and notepad.ai_sidebar:
                        if not getattr(notepad, 'ai_sidebar_visible', False):
                            notepad._toggle_ai_sidebar()
                        notepad.ai_sidebar.set_input_text(f"Explain or improve this text:\n\n{sel_text}")
                finally:
                    self.hide()

            btn_defs = [
                ("𝐁", lambda: _wrap_fmt("**", "**"), "Bold (Ctrl+B)"),
                ("𝐼", lambda: _wrap_fmt("*", "*"), "Italic (Ctrl+I)"),
                ("🖍", lambda: _wrap_fmt("==", "=="), "Highlight (Ctrl+Shift+H)"),
                ("`", lambda: _wrap_fmt("`", "`"), "Inline Code (Ctrl+`)"),
                ("🔗", lambda: _wrap_fmt("[", "](url)"), "Insert Link (Ctrl+K)"),
                ("💬 AI", _ask_ai, "Ask AI about this selection"),
            ]
            
            for text, cmd, tip in btn_defs:
                b = tk.Button(
                    frame, text=text, command=cmd, font=(SANS_FONT, 9, "bold" if text in ("𝐁", "💬 AI") else "normal"),
                    bg="#f3effa", fg="#2e1065", activebackground="#ede9fe", activeforeground="#4c1d95",
                    relief=tk.FLAT, bd=0, padx=5, pady=2, cursor="hand2"
                )
                b.pack(side=tk.LEFT, padx=1, pady=1)
                ToolTip(b, tip, delay=250)
            
            win.update_idletasks()
            win_w = win.winfo_reqwidth()
            target_x = max(10, root_x - (win_w // 4))
            target_y = max(10, root_y)
            win.geometry(f"+{target_x}+{target_y}")
        except Exception:
            pass

    def hide(self):
        if self._check_job:
            try:
                self.editor.after_cancel(self._check_job)
            except Exception:
                pass
            self._check_job = None
        if self.window:
            try:
                self.window.destroy()
            except Exception:
                pass
            self.window = None


class DocumentStatisticsDialog(tk.Toplevel):
    """Comprehensive Document Statistics dialog displaying counts, structure, and reading times."""
    
    def __init__(self, parent, tab):
        super().__init__(parent)
        self.parent = parent
        self.tab = tab
        
        self.title("Document Statistics")
        self.transient(parent)
        self.grab_set()
        self.configure(bg="#f8f7fc")
        
        self._build_ui()
        _fit_and_center_dialog(self, parent, 480, 560)
        self.bind("<Escape>", lambda e: self.destroy())

    def _build_ui(self):
        if getattr(self.tab, 'tab_type', 'document') == 'spreadsheet':
            # Header banner
            header = tk.Frame(self, bg="#3b1d6e", padx=16, pady=14)
            header.pack(fill=tk.X)
            tk.Label(
                header, text="Spreadsheet Statistics", font=(SANS_FONT, 13, "bold"),
                bg="#3b1d6e", fg="#ffffff"
            ).pack(anchor=tk.W)
            doc_name = self.tab.get_display_name() if self.tab else "Current Spreadsheet"
            tk.Label(
                header, text=f"Grid overview and aggregates for {doc_name}", font=(SANS_FONT, 9),
                bg="#3b1d6e", fg="#ddd6fe"
            ).pack(anchor=tk.W, pady=(2, 0))

            body = ttk.Frame(self, padding=16)
            body.pack(fill=tk.BOTH, expand=True)

            engine = self.tab.engine
            populated_cells = len(engine.raw_cells)
            numeric_cells = 0
            text_cells = 0
            formula_cells = 0
            for (r, c), raw in engine.raw_cells.items():
                if raw.startswith("="):
                    formula_cells += 1
                v = engine.get_value(r, c)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric_cells += 1
                elif raw and not raw.startswith("="):
                    text_cells += 1

            (min_r, min_c), (max_r, max_c) = self.tab.grid_widget.get_selected_range()
            sel_count = (max_r - min_r + 1) * (max_c - min_c + 1)
            sel_nums = []
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    v = engine.get_value(r, c)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        sel_nums.append(float(v))

            sel_sum = f"{sum(sel_nums):,.2f}" if sel_nums else "N/A"
            sel_avg = f"{sum(sel_nums)/len(sel_nums):,.2f}" if sel_nums else "N/A"
            sel_min = f"{min(sel_nums):,.2f}" if sel_nums else "N/A"
            sel_max = f"{max(sel_nums):,.2f}" if sel_nums else "N/A"

            cell_frame = ttk.LabelFrame(body, text="Cell Composition", padding=10)
            cell_frame.pack(fill=tk.X, pady=(0, 10))

            rows = [
                ("Total Populated Cells:", f"{populated_cells:,}"),
                ("Formula Cells:", f"{formula_cells:,}"),
                ("Numeric Cells:", f"{numeric_cells:,}"),
                ("Text / Label Cells:", f"{text_cells:,}"),
                ("Grid Extents:", f"{self.tab.grid_widget.total_rows} Rows × {self.tab.grid_widget.total_cols} Columns"),
            ]
            for idx, (lbl, val) in enumerate(rows):
                ttk.Label(cell_frame, text=lbl).grid(row=idx, column=0, sticky=tk.W, pady=2)
                ttk.Label(cell_frame, text=val, font=(SANS_FONT, 9, "bold")).grid(row=idx, column=1, sticky=tk.E, padx=20, pady=2)

            agg_frame = ttk.LabelFrame(body, text="Active Selection Aggregates", padding=10)
            agg_frame.pack(fill=tk.X, pady=(0, 10))

            agg_rows = [
                ("Selected Cells Range:", f"{cell_to_ref(min_r, min_c)}:{cell_to_ref(max_r, max_c)} ({sel_count} cells)"),
                ("Sum (numeric cells):", sel_sum),
                ("Average:", sel_avg),
                ("Min Value:", sel_min),
                ("Max Value:", sel_max),
            ]
            for idx, (lbl, val) in enumerate(agg_rows):
                ttk.Label(agg_frame, text=lbl).grid(row=idx, column=0, sticky=tk.W, pady=2)
                ttk.Label(agg_frame, text=val, font=(SANS_FONT, 9, "bold")).grid(row=idx, column=1, sticky=tk.E, padx=20, pady=2)
            return

        content = self.tab.get_content() if self.tab else ""
        sel_text = ""
        try:
            if self.tab.source_editor.tag_ranges(tk.SEL):
                sel_text = self.tab.source_editor.get(tk.SEL_FIRST, tk.SEL_LAST)
        except Exception:
            pass
        
        # Header banner in Royal Purple Word style
        header = tk.Frame(self, bg="#3b1d6e", padx=16, pady=14)
        header.pack(fill=tk.X)
        tk.Label(
            header, text="Document Statistics", font=(SANS_FONT, 13, "bold"),
            bg="#3b1d6e", fg="#ffffff"
        ).pack(anchor=tk.W)
        doc_name = self.tab.get_display_name() if self.tab else "Current Document"
        tk.Label(
            header, text=f"Overview metrics for {doc_name}", font=(SANS_FONT, 9),
            bg="#3b1d6e", fg="#ddd6fe"
        ).pack(anchor=tk.W, pady=(2, 0))

        body = ttk.Frame(self, padding=16)
        body.pack(fill=tk.BOTH, expand=True)

        def calc_metrics(txt):
            words = len(txt.split())
            chars_with_spaces = len(txt)
            chars_no_spaces = len(re.sub(r'\s+', '', txt))
            lines = len(txt.splitlines()) if txt else 0
            paras = len([p for p in txt.split('\n\n') if p.strip()]) if txt else 0
            headings = len(re.findall(r'(?m)^#{1,6}\s+.+$', txt))
            tables = len(re.findall(r'(?m)^\|(?:\s*:?-+:?\s*\|)+$', txt))
            figures = len(re.findall(r'!\[[^\]]*\]\([^)]+\)', txt))
            read_mins = max(1, round(words / 200)) if words > 100 else (1 if words > 0 else 0)
            speak_mins = max(1, round(words / 130)) if words > 60 else (1 if words > 0 else 0)
            return {
                'words': words, 'chars_ws': chars_with_spaces, 'chars_ns': chars_no_spaces,
                'lines': lines, 'paras': paras, 'headings': headings,
                'tables': tables, 'figures': figures,
                'read_time': f"~{read_mins} min" if read_mins else "0 min",
                'speak_time': f"~{speak_mins} min" if speak_mins else "0 min"
            }

        full_m = calc_metrics(content)
        has_sel = bool(sel_text.strip())
        sel_m = calc_metrics(sel_text) if has_sel else None

        stats_frame = ttk.LabelFrame(body, text="Counts & Proportions", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))

        rows = [
            ("Words:", full_m['words'], sel_m['words'] if has_sel else None),
            ("Characters (no spaces):", full_m['chars_ns'], sel_m['chars_ns'] if has_sel else None),
            ("Characters (with spaces):", full_m['chars_ws'], sel_m['chars_ws'] if has_sel else None),
            ("Paragraphs:", full_m['paras'], sel_m['paras'] if has_sel else None),
            ("Lines:", full_m['lines'], sel_m['lines'] if has_sel else None),
        ]

        ttk.Label(stats_frame, text="Metric", font=(SANS_FONT, 9, "bold")).grid(row=0, column=0, sticky=tk.W, pady=2)
        if has_sel:
            ttk.Label(stats_frame, text="Selection", font=(SANS_FONT, 9, "bold")).grid(row=0, column=1, sticky=tk.E, padx=12, pady=2)
            ttk.Label(stats_frame, text="Full Document", font=(SANS_FONT, 9, "bold")).grid(row=0, column=2, sticky=tk.E, pady=2)
        else:
            ttk.Label(stats_frame, text="Full Document", font=(SANS_FONT, 9, "bold")).grid(row=0, column=1, sticky=tk.E, padx=20, pady=2)

        for idx, (label, full_val, sel_val) in enumerate(rows, start=1):
            ttk.Label(stats_frame, text=label).grid(row=idx, column=0, sticky=tk.W, pady=2)
            if has_sel:
                ttk.Label(stats_frame, text=f"{sel_val:,}").grid(row=idx, column=1, sticky=tk.E, padx=12, pady=2)
                ttk.Label(stats_frame, text=f"{full_val:,}").grid(row=idx, column=2, sticky=tk.E, pady=2)
            else:
                ttk.Label(stats_frame, text=f"{full_val:,}").grid(row=idx, column=1, sticky=tk.E, padx=20, pady=2)

        # Structure & Reading Time Frame
        more_frame = ttk.LabelFrame(body, text="Structure & Reading Estimates", padding=10)
        more_frame.pack(fill=tk.X, pady=(0, 10))

        struct_rows = [
            ("Section Headings:", f"{full_m['headings']}"),
            ("Embedded Tables:", f"{full_m['tables']}"),
            ("Figures & Images:", f"{full_m['figures']}"),
            ("Estimated Reading Time (200 wpm):", f"{full_m['read_time']}"),
            ("Estimated Speaking Time (130 wpm):", f"{full_m['speak_time']}"),
        ]

        for idx, (label, val_str) in enumerate(struct_rows):
            ttk.Label(more_frame, text=label).grid(row=idx, column=0, sticky=tk.W, pady=2)
            ttk.Label(more_frame, text=val_str, font=(SANS_FONT, 9, "bold")).grid(row=idx, column=1, sticky=tk.E, padx=20, pady=2)

        btn_row = ttk.Frame(body)
        btn_row.pack(fill=tk.X, side=tk.BOTTOM)
        close_btn = ttk.Button(btn_row, text="Close", command=self.destroy, width=10)
        close_btn.pack(side=tk.RIGHT)
        close_btn.focus_set()


class ImageExtractor:
    """
    Extracts images from documents alongside MarkItDown conversion.
    Supports: PDF, EPUB, DOCX, PPTX, XLSX.
    """
    
    @staticmethod
    def can_extract(filepath):
        """Check if we can extract images from this file type"""
        if DOC_IMAGE_EXTRACTOR_AVAILABLE:
            return DocumentImageExtractor.can_extract(filepath)
        ext = Path(filepath).suffix.lower()
        return ext in {'.pdf', '.epub', '.docx', '.pptx', '.xlsx'}
    
    @staticmethod
    def extract_images(filepath, output_dir, raw_markdown=""):
        """Extract images from document and return dict of extracted image paths."""
        if DOC_IMAGE_EXTRACTOR_AVAILABLE:
            _, extracted, _ = DocumentImageExtractor.extract_and_integrate(
                filepath, output_dir=output_dir, raw_markdown=raw_markdown
            )
            return extracted
        return {}
    
    @staticmethod
    def insert_image_references(markdown_text, extracted_images, output_dir):
        """Append extracted image references if not already present."""
        if not extracted_images:
            return markdown_text
        
        # If references already exist in markdown_text, return as is
        if re.search(r'!\[.*?\]\(.*?\)', markdown_text):
            return markdown_text

        # Otherwise add a section for extracted images
        image_section = "\n\n---\n\n## Extracted Images\n\n"
        for orig_path, local_path in extracted_images.items():
            filename = os.path.basename(local_path)
            image_section += f"![{filename}]({filename})\n\n"
        
        return markdown_text + image_section


class ImageHandler:
    """
    Handles safe image detection, validation, and rendering in markdown.
    
    Supports:
    - Base64 data URIs with strict validation
    - Local file paths (relative and absolute)
    - Thumbnail generation for memory efficiency
    - Click-to-expand functionality
    """
    
    # Supported image MIME types
    SUPPORTED_MIMES = {
        'image/png': 'png',
        'image/jpeg': 'jpeg', 
        'image/jpg': 'jpg',
        'image/gif': 'gif',
        'image/webp': 'webp',
        'image/bmp': 'bmp',
    }
    
    # Regex for valid markdown image syntax with data URI
    # Must match: ![alt](data:image/TYPE;base64,DATA)
    BASE64_IMAGE_PATTERN = re.compile(
        r'!\[([^\]]*)\]\(data:(image/(?:png|jpeg|jpg|gif|webp|bmp));base64,([A-Za-z0-9+/]+=*)\)',
        re.IGNORECASE
    )
    
    # Regex for file path images: ![alt](path/to/image.ext)
    # Also captures optional HTML-style attributes or size suffix
    FILE_IMAGE_PATTERN = re.compile(
        r'!\[([^\]]*)\]\(([^)\s]+\.(?:png|jpe?g|gif|webp|bmp))(?:\s*["\']([^"\']*)["\'])?\)',
        re.IGNORECASE
    )
    
    # Regex for HTML img tags (commonly used for sizing)
    HTML_IMG_PATTERN = re.compile(
        r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>',
        re.IGNORECASE
    )
    
    # Regex to extract width/height from HTML attributes
    WIDTH_PATTERN = re.compile(r'width\s*[=:]\s*["\']?(\d+)(?:px)?["\']?', re.IGNORECASE)
    HEIGHT_PATTERN = re.compile(r'height\s*[=:]\s*["\']?(\d+)(?:px)?["\']?', re.IGNORECASE)
    
    # Size limits for safety
    MAX_BASE64_LENGTH = 10 * 1024 * 1024  # 10MB max decoded size
    MIN_BASE64_LENGTH = 100  # Minimum viable image size
    
    # Thumbnail settings
    THUMBNAIL_SIZE = (150, 150)
    
    def __init__(self, base_path=None):
        """
        Initialize the image handler.
        
        Args:
            base_path: Base directory for resolving relative image paths
        """
        self.base_path = base_path or os.getcwd()
        self.image_cache = {}  # Cache for loaded images (keyed by hash)
        self.thumbnail_cache = {}  # Cache for thumbnails
        self._photo_refs = []  # Keep references to prevent garbage collection
    
    def clear_caches(self):
        """Clear image caches to free memory"""
        self.image_cache.clear()
        self.thumbnail_cache.clear()
        self._photo_refs.clear()
    
    def find_images(self, markdown_text):
        """
        Find all valid images in markdown text.
        Supports:
        - Standard markdown: ![alt](path)
        - HTML img tags: <img src="path" width="200">
        - Size hints in alt: ![alt|200x100](path)
        
        Returns a list of dicts with image info including optional width/height.
        """
        images = []
        
        # Find base64 images
        for match in self.BASE64_IMAGE_PATTERN.finditer(markdown_text):
            alt_text = match.group(1)
            mime_type = match.group(2)
            base64_data = match.group(3)
            
            # Parse size from alt text (format: "alt|200x100" or "alt|200")
            alt_text, width, height = self._parse_alt_size(alt_text)
            
            # Validate the base64 data
            if self._validate_base64(base64_data):
                images.append({
                    'start': match.start(),
                    'end': match.end(),
                    'full_match': match.group(0),
                    'alt': alt_text,
                    'type': 'base64',
                    'mime': mime_type,
                    'data': base64_data,
                    'width': width,
                    'height': height
                })
        
        # Find file path images (markdown syntax)
        for match in self.FILE_IMAGE_PATTERN.finditer(markdown_text):
            # Skip if this overlaps with a base64 match
            if any(img['start'] <= match.start() < img['end'] for img in images):
                continue
                
            alt_text = match.group(1)
            file_path = match.group(2)
            title = match.group(3) if match.lastindex >= 3 else None
            
            # Skip data URIs that might have been partially matched
            if file_path.startswith('data:'):
                continue
            
            # Parse extended parameters from alt text
            alt_info = self._parse_extended_alt(alt_text)
            alt_text = alt_info['alt']
            width = alt_info['width']
            height = alt_info['height']
            align = alt_info['align']
            caption = alt_info['caption']
            
            # Also check title for size hints if not already found
            if title and not width:
                w, h = self._parse_size_string(title)
                if w:
                    width, height = w, h
            
            images.append({
                'start': match.start(),
                'end': match.end(),
                'full_match': match.group(0),
                'alt': alt_text,
                'type': 'file',
                'path': file_path,
                'width': width,
                'height': height,
                'align': align,
                'caption': caption
            })
        
        # Find HTML img tags
        for match in self.HTML_IMG_PATTERN.finditer(markdown_text):
            # Skip if overlaps with existing matches
            if any(img['start'] <= match.start() < img['end'] for img in images):
                continue
            
            full_tag = match.group(0)
            src = match.group(1)
            
            # Skip data URIs
            if src.startswith('data:'):
                continue
            
            # Extract width and height from tag
            width_match = self.WIDTH_PATTERN.search(full_tag)
            height_match = self.HEIGHT_PATTERN.search(full_tag)
            
            width = int(width_match.group(1)) if width_match else None
            height = int(height_match.group(1)) if height_match else None
            
            # Try to find alt attribute
            alt_match = re.search(r'alt\s*=\s*["\']([^"\']*)["\']', full_tag, re.IGNORECASE)
            alt_text = alt_match.group(1) if alt_match else ''
            
            images.append({
                'start': match.start(),
                'end': match.end(),
                'full_match': full_tag,
                'alt': alt_text,
                'type': 'file',
                'path': src,
                'width': width,
                'height': height,
                'align': 'center',
                'caption': None,
                'is_html': True
            })
        
        return sorted(images, key=lambda x: x['start'])
    
    def _parse_extended_alt(self, alt_text):
        """
        Parse extended image syntax from alt text:
        Supports:
        - "description|width=450,align=center,caption='Figure 1'"
        - "description|400x300|center"
        - "description|400"
        - "description"
        Returns dict with: alt, width, height, align, caption
        """
        result = {
            'alt': alt_text,
            'width': None,
            'height': None,
            'align': 'center',
            'caption': None
        }
        if not alt_text or '|' not in alt_text:
            return result
        
        parts = alt_text.split('|')
        result['alt'] = parts[0].strip()
        param_str = '|'.join(parts[1:]).strip()

        # Check for key=value parameters
        kv_pattern = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,;|]+))')
        kv_matches = list(kv_pattern.finditer(param_str))
        remaining_str = param_str

        if kv_matches:
            for m in kv_matches:
                k = m.group(1).lower()
                v = (m.group(2) or m.group(3) or m.group(4) or '').strip()
                if k == 'width':
                    dim = re.match(r'(\d+)', v)
                    if dim:
                        result['width'] = int(dim.group(1))
                elif k == 'height':
                    dim = re.match(r'(\d+)', v)
                    if dim:
                        result['height'] = int(dim.group(1))
                elif k == 'align':
                    if v.lower() in ('left', 'right', 'center'):
                        result['align'] = v.lower()
                elif k == 'caption':
                    result['caption'] = v
            remaining_str = kv_pattern.sub('', param_str)

        # Check for positional shorthand in remaining tokens
        tokens = [t.strip() for t in re.split(r'[,|]', remaining_str) if t.strip()]
        for token in tokens:
            m_dim = re.match(r'^(\d+)\s*[xX×]\s*(\d+)$', token)
            if m_dim:
                if result['width'] is None:
                    result['width'] = int(m_dim.group(1))
                if result['height'] is None:
                    result['height'] = int(m_dim.group(2))
                continue
            m_w = re.match(r'^(\d+)(?:px)?$', token)
            if m_w:
                if result['width'] is None:
                    result['width'] = int(m_w.group(1))
                continue
            if token.lower() in ('left', 'right', 'center'):
                result['align'] = token.lower()
                continue
            if not result['caption']:
                result['caption'] = token

        return result

    def _parse_alt_size(self, alt_text):
        """Backward compatibility helper returning (alt, width, height)"""
        info = self._parse_extended_alt(alt_text)
        return info['alt'], info['width'], info['height']
    
    def _parse_size_string(self, size_str):
        """Parse a size string like '200x100' or '200'"""
        if not size_str:
            return None, None
        
        # Try "WIDTHxHEIGHT" format
        match = re.match(r'(\d+)\s*[xX×]\s*(\d+)', size_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        # Try just width
        match = re.match(r'(\d+)', size_str)
        if match:
            return int(match.group(1)), None
        
        return None, None
    
    def _validate_base64(self, data):
        """
        Validate base64 data for safety and correctness.
        
        Returns True if the data is valid and within safe limits.
        """
        if not data:
            return False
        
        # Check length bounds
        if len(data) < self.MIN_BASE64_LENGTH:
            return False
        
        # Estimate decoded size (base64 is ~4/3 of original)
        estimated_size = len(data) * 3 // 4
        if estimated_size > self.MAX_BASE64_LENGTH:
            return False
        
        # Check for valid base64 characters only
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', data):
            return False
        
        # Check proper padding
        if len(data) % 4 != 0:
            return False
        
        return True
    
    def load_image(self, image_info):
        """
        Load an image from the given image info dict.
        
        Returns a PIL Image object or None if loading fails.
        """
        if not PIL_AVAILABLE:
            return None
        
        try:
            if image_info['type'] == 'base64':
                return self._load_base64_image(image_info['data'])
            elif image_info['type'] == 'file':
                return self._load_file_image(image_info['path'])
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def _load_base64_image(self, base64_data):
        """Load image from base64 data"""
        # Check cache first
        data_hash = hashlib.md5(base64_data.encode()).hexdigest()
        if data_hash in self.image_cache:
            return self.image_cache[data_hash]
        
        try:
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            
            # Verify it's a valid image by accessing properties
            image.verify()
            
            # Reopen after verify (verify() leaves file in unusable state)
            image = Image.open(io.BytesIO(image_data))
            
            self.image_cache[data_hash] = image
            return image
        except Exception:
            return None
    
    def _load_file_image(self, file_path):
        """Load image from file path with robust fallback resolution."""
        resolved_path = None
        
        # Check direct path
        if os.path.isabs(file_path) and os.path.exists(file_path):
            resolved_path = file_path
        else:
            # Try relative to base_path
            candidate = os.path.normpath(os.path.join(self.base_path, file_path))
            if os.path.exists(candidate):
                resolved_path = candidate
            else:
                # Try just filename in base_path
                base_candidate = os.path.normpath(os.path.join(self.base_path, os.path.basename(file_path)))
                if os.path.exists(base_candidate):
                    resolved_path = base_candidate
                elif os.path.isdir(self.base_path):
                    # Search inside any _assets or images subfolders in base_path
                    filename = os.path.basename(file_path)
                    try:
                        for entry in os.scandir(self.base_path):
                            if entry.is_dir() and (entry.name.endswith('_assets') or entry.name in ('assets', 'images', 'Images', 'media')):
                                sub_candidate = os.path.join(entry.path, filename)
                                if os.path.exists(sub_candidate):
                                    resolved_path = sub_candidate
                                    break
                    except Exception:
                        pass

        if not resolved_path or not os.path.exists(resolved_path):
            return None

        file_path = os.path.normpath(resolved_path)

        # Check cache
        if file_path in self.image_cache:
            return self.image_cache[file_path]
        
        try:
            image = Image.open(file_path)
            image.load()  # Force load to catch errors early
            self.image_cache[file_path] = image
            return image
        except Exception:
            return None
    
    def create_thumbnail(self, image, max_size=None):
        """
        Create a thumbnail from a PIL Image.
        
        Returns a new PIL Image sized to fit within max_size.
        """
        if not PIL_AVAILABLE or image is None:
            return None
        
        max_size = max_size or self.THUMBNAIL_SIZE
        
        try:
            # Create a copy to avoid modifying original
            thumb = image.copy()
            thumb.thumbnail(max_size, Image.Resampling.LANCZOS)
            return thumb
        except Exception:
            return None
    
    def get_photo_image(self, image, thumbnail=True, max_size=None):
        """
        Convert PIL Image to Tkinter PhotoImage.
        
        Args:
            image: PIL Image object
            thumbnail: If True, create thumbnail; if False, use original size
            max_size: Maximum size tuple (width, height)
        
        Returns a PhotoImage suitable for use in Tkinter widgets.
        """
        if not PIL_AVAILABLE or image is None:
            return None
        
        try:
            if thumbnail:
                display_image = self.create_thumbnail(image, max_size)
            else:
                display_image = image
            
            if display_image is None:
                return None
            
            # Convert to RGB if necessary (for PNG with transparency, etc.)
            if display_image.mode in ('RGBA', 'P'):
                # Create white background
                background = Image.new('RGB', display_image.size, (255, 255, 255))
                if display_image.mode == 'P':
                    display_image = display_image.convert('RGBA')
                background.paste(display_image, mask=display_image.split()[-1] if display_image.mode == 'RGBA' else None)
                display_image = background
            elif display_image.mode != 'RGB':
                display_image = display_image.convert('RGB')
            
            photo = ImageTk.PhotoImage(display_image)
            self._photo_refs.append(photo)  # Keep reference
            return photo
        except Exception as e:
            print(f"Error creating PhotoImage: {e}")
            return None


class ImageViewerWindow(tk.Toplevel):
    """Popup window for viewing full-size images"""
    
    def __init__(self, parent, image, title="Image Viewer"):
        super().__init__(parent)
        self.title(title)
        self.image = image
        self.photo = None
        
        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Calculate appropriate window size (max 80% of screen)
        max_width = int(screen_width * 0.8)
        max_height = int(screen_height * 0.8)
        
        if PIL_AVAILABLE and image:
            # Resize image if too large
            img_width, img_height = image.size
            
            scale = min(max_width / img_width, max_height / img_height, 1.0)
            
            if scale < 1.0:
                new_size = (int(img_width * scale), int(img_height * scale))
                display_image = image.copy()
                display_image.thumbnail(new_size, Image.Resampling.LANCZOS)
            else:
                display_image = image
            
            # Convert to PhotoImage
            if display_image.mode in ('RGBA', 'P'):
                background = Image.new('RGB', display_image.size, (255, 255, 255))
                if display_image.mode == 'P':
                    display_image = display_image.convert('RGBA')
                background.paste(display_image, mask=display_image.split()[-1] if display_image.mode == 'RGBA' else None)
                display_image = background
            elif display_image.mode != 'RGB':
                display_image = display_image.convert('RGB')
            
            self.photo = ImageTk.PhotoImage(display_image)
            
            # Create canvas with scrollbars
            canvas_frame = ttk.Frame(self)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            
            canvas = tk.Canvas(canvas_frame, bg='#2d2d2d')
            canvas.pack(fill=tk.BOTH, expand=True)
            
            canvas.create_image(
                display_image.size[0] // 2,
                display_image.size[1] // 2,
                image=self.photo,
                anchor=tk.CENTER
            )
            
            # Set window size
            win_width = min(display_image.size[0] + 20, max_width)
            win_height = min(display_image.size[1] + 50, max_height)
            self.geometry(f"{win_width}x{win_height}")
        else:
            ttk.Label(self, text="Image could not be displayed").pack(pady=20)
            self.geometry("300x100")
        
        # Close button
        ttk.Button(self, text="Close", command=self.destroy).pack(pady=5)
        
        # Center window
        self.update_idletasks()
        x = (screen_width - self.winfo_width()) // 2
        y = (screen_height - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
        
        # Bind Escape to close
        self.bind("<Escape>", lambda e: self.destroy())


class PageView(tk.Frame):
    """
    A 'desk' with a fixed-width, centred 'sheet' (Word-style page) on top of it.

    The caller creates a tk.Text whose parent is `self.sheet` and registers it with
    `attach_text`. Page width, margins and the scrollbar are managed here; zoom scales
    the page width and margins so wrapping stays identical at every zoom level.
    """

    PAGE_INCHES = {'Letter': 8.5, 'A4': 8.27, 'Narrow': 7.0, 'Wide': 10.0}
    MARGIN_INCHES = {'Normal': 1.0, 'Narrow': 0.5, 'Wide': 1.5}
    GUTTER = 24    # minimum desk space left/right of the sheet (px at 96 dpi)
    TOP_GAP = 16   # desk space above/below the sheet (px at 96 dpi)

    def __init__(self, parent, desk_bg, page_bg, on_zoom_wheel=None, max_margin_in=None, **kwargs):
        super().__init__(parent, bg=desk_bg, **kwargs)
        self.on_zoom_wheel = on_zoom_wheel
        self.max_margin_in = max_margin_in
        self.on_resized = None        # callback fired when the text column width changes
        self.zoom = 1.0
        self.page_size = 'Letter'
        self.margin = 'Normal'
        self.enabled = True
        self.text = None
        self.content_width = 600
        self._last_layout = None

        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Shadow is created before the sheet so the sheet stacks above it
        self.shadow = tk.Frame(self, bg=_shade_color(desk_bg, -0.18))
        self.sheet = tk.Frame(self, bg=page_bg, highlightthickness=1,
                              highlightbackground=_shade_color(page_bg, -0.12))
        self.hbar = ttk.Scrollbar(self.sheet, orient=tk.HORIZONTAL)

        self.bind('<Configure>', self._relayout)
        for w in (self, self.sheet, self.shadow):
            w.bind('<MouseWheel>', self._wheel)
            w.bind('<Control-MouseWheel>', self._ctrl_wheel)

    def attach_text(self, text_widget):
        """Place an already-created Text widget (child of self.sheet) on the page."""
        self.text = text_widget
        text_widget.config(relief=tk.FLAT, borderwidth=0, highlightthickness=0,
                           yscrollcommand=self.scrollbar.set)
        text_widget.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=text_widget.yview)
        self.hbar.config(command=text_widget.xview)

    def dpi(self):
        try:
            return self.winfo_fpixels('1i')
        except tk.TclError:
            return 96.0

    def set_hscroll(self, enabled):
        """Show a horizontal scrollbar inside the sheet (used when word wrap is off)."""
        if not self.text:
            return
        if enabled:
            self.text.config(xscrollcommand=self.hbar.set)
            self.hbar.pack(side=tk.BOTTOM, fill=tk.X, before=self.text)
        else:
            self.text.config(xscrollcommand='')
            self.hbar.pack_forget()

    def set_zoom(self, zoom):
        self.zoom = zoom
        self._relayout()

    def set_page(self, size=None, margin=None, enabled=None):
        if size is not None:
            self.page_size = size
        if margin is not None:
            self.margin = margin
        if enabled is not None:
            self.enabled = enabled
        self._relayout()

    def set_colors(self, desk_bg, page_bg):
        self.config(bg=desk_bg)
        self.shadow.config(bg=_shade_color(desk_bg, -0.4 if _is_dark_color(desk_bg) else -0.18))
        border = _shade_color(page_bg, 0.15 if _is_dark_color(page_bg) else -0.12)
        self.sheet.config(bg=page_bg, highlightbackground=border)

    def max_page_width(self):
        """Pixel width the sheet would have at the current zoom if the window were unlimited."""
        return int(self.PAGE_INCHES.get(self.page_size, 8.5) * self.dpi() * self.zoom)

    def fit_zoom(self):
        """Zoom factor at which the sheet exactly fills the desk (minus gutters)."""
        dpi = self.dpi()
        avail = self.winfo_width() - self.scrollbar.winfo_reqwidth() - 2 * int(self.GUTTER * dpi / 96)
        base = self.PAGE_INCHES.get(self.page_size, 8.5) * dpi
        return max(0.25, avail / base) if avail > 0 else 1.0

    def _wheel(self, event):
        if self.text is not None:
            delta = int(-event.delta / 40) if event.delta else 0
            if delta:
                self.text.yview_scroll(delta, 'units')
        return 'break'

    def _ctrl_wheel(self, event):
        if self.on_zoom_wheel:
            return self.on_zoom_wheel(event)
        return 'break'

    def _relayout(self, event=None):
        dw, dh = self.winfo_width(), self.winfo_height()
        if dw < 60 or dh < 60:
            return
        dpi = self.dpi()
        avail = max(100, dw - self.scrollbar.winfo_reqwidth())
        if self.enabled:
            gutter = int(self.GUTTER * dpi / 96)
            gap = int(self.TOP_GAP * dpi / 96)
            page_w = max(int(2 * dpi), min(self.max_page_width(), avail - 2 * gutter))
            x = (avail - page_w) // 2
            h = max(50, dh - 2 * gap)
            self.shadow.place(x=x + 3, y=gap + 3, width=page_w, height=h)
            self.sheet.place(x=x, y=gap, width=page_w, height=h)
            margin_in = self.MARGIN_INCHES.get(self.margin, 1.0)
            if self.max_margin_in is not None:
                margin_in = min(margin_in, self.max_margin_in)
            pad = max(int(0.15 * dpi), min(int(margin_in * dpi * self.zoom), page_w // 4))
        else:
            page_w = avail
            self.shadow.place_forget()
            self.sheet.place(x=0, y=0, width=avail, height=dh)
            pad = int(0.25 * dpi)
        self.content_width = max(100, page_w - 2 * pad)
        if self.text is not None:
            self.text.config(padx=pad, pady=int(pad * 0.6))
        layout = (page_w, pad)
        if layout != self._last_layout:
            self._last_layout = layout
            if self.on_resized:
                self.on_resized()


_LATEX_SYMBOLS = {
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\varepsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η',
    r'\theta': 'θ', r'\vartheta': 'ϑ', r'\iota': 'ι', r'\kappa': 'κ',
    r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ',
    r'\pi': 'π', r'\rho': 'ρ', r'\sigma': 'σ', r'\tau': 'τ',
    r'\upsilon': 'υ', r'\phi': 'φ', r'\varphi': 'ϕ', r'\chi': 'χ',
    r'\psi': 'ψ', r'\omega': 'ω',
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ',
    r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
    r'\pm': '±', r'\mp': '∓', r'\times': '×', r'\div': '÷',
    r'\cdot': '⋅', r'\ast': '∗', r'\star': '⋆', r'\circ': '∘',
    r'\bullet': '∙', r'\oplus': '⊕', r'\otimes': '⊗',
    r'\leq': '≤', r'\le': '≤', r'\geq': '≥', r'\ge': '≥',
    r'\neq': '≠', r'\ne': '≠', r'\approx': '≈', r'\equiv': '≡',
    r'\sim': '∼', r'\simeq': '≃', r'\cong': '≅', r'\propto': '∝',
    r'\ll': '≪', r'\gg': '≫', r'\infty': '∞', r'\partial': '∂',
    r'\nabla': '∇', r'\sum': '∑', r'\prod': '∏', r'\coprod': '∐',
    r'\int': '∫', r'\iint': '∬', r'\iiint': '∭', r'\oint': '∮',
    r'\in': '∈', r'\notin': '∉', r'\ni': '∋',
    r'\subset': '⊂', r'\supset': '⊃', r'\subseteq': '⊆', r'\supseteq': '⊇',
    r'\cup': '∪', r'\cap': '∩', r'\emptyset': '∅', r'\varnothing': '∅',
    r'\forall': '∀', r'\exists': '∃', r'\nexists': '∄',
    r'\neg': '¬', r'\lnot': '¬', r'\land': '∧', r'\lor': '∨',
    r'\Rightarrow': '⇒', r'\Leftarrow': '⇐', r'\Leftrightarrow': '⇔',
    r'\iff': '⇔', r'\implies': '⇒',
    r'\rightarrow': '→', r'\to': '→', r'\leftarrow': '←',
    r'\leftrightarrow': '↔', r'\mapsto': '↦',
    r'\uparrow': '↑', r'\downarrow': '↓',
    r'\therefore': '∴', r'\because': '∵', r'\angle': '∠',
    r'\perp': '⊥', r'\parallel': '∥', r'\degree': '°', r'\prime': '′',
}

_SUPERSCRIPT_MAP = str.maketrans("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
_SUBSCRIPT_MAP = str.maketrans("0123456789+-=()ijknm", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ᵢⱼₖₙₘ")


def _latex_to_unicode(expr: str) -> str:
    """Convert common LaTeX math commands, fractions, roots, and sub/superscripts into readable Unicode."""
    if not expr:
        return expr
    out = expr.strip()
    # Replace \frac{a}{b} -> (a)/(b) or a/b
    out = re.sub(
        r'\\frac\{([^{}]+)\}\{([^{}]+)\}',
        lambda m: f"{m.group(1)}/{m.group(2)}" if len(m.group(1)) == 1 and len(m.group(2)) == 1 else f"({m.group(1)})/({m.group(2)})",
        out
    )
    # Replace \sqrt{x} -> √(x) or √x
    out = re.sub(
        r'\\sqrt\{([^{}]+)\}',
        lambda m: f"√{m.group(1)}" if len(m.group(1)) == 1 else f"√({m.group(1)})",
        out
    )
    # Replace known control words (longest first so \Leftrightarrow beats \Leftarrow)
    for cmd in sorted(_LATEX_SYMBOLS.keys(), key=len, reverse=True):
        out = re.sub(re.escape(cmd) + r'(?![a-zA-Z])', _LATEX_SYMBOLS[cmd], out)
    # Convert ^{...} and ^x when all characters have Unicode superscript equivalents
    def _sup_repl(m):
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        if all(ch in "0123456789+-=()ni" for ch in inner):
            return inner.translate(_SUPERSCRIPT_MAP)
        return f"^({inner})" if len(inner) > 1 else f"^{inner}"

    def _sub_repl(m):
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        if all(ch in "0123456789+-=()ijknm" for ch in inner):
            return inner.translate(_SUBSCRIPT_MAP)
        return f"_({inner})" if len(inner) > 1 else f"_{inner}"

    out = re.sub(r'\^\{([^{}]+)\}|\^([0-9+\-=ni])', _sup_repl, out)
    out = re.sub(r'_\{([^{}]+)\}|_([0-9+\-=ijknm])', _sub_repl, out)
    # Strip leftover braces from simple groupings
    out = out.replace('{', '').replace('}', '')
    return out


# =============================================================================
# Markdown Table & Spreadsheet Conversion Helpers
# =============================================================================

def _split_md_table_row(row_str: str) -> List[str]:
    """Split a Markdown table row into cells, respecting escaped pipes (\\|)."""
    s = row_str.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|') and not s.endswith(r'\|'):
        s = s[:-1]
    cells = [c.strip() for c in re.split(r'(?<!\\)\|', s)]
    return cells


def format_markdown_table_lines(lines: List[str]) -> Tuple[List[str], List[str]]:
    """Format, pad, and align Markdown pipe table lines. Returns (formatted_lines, col_alignments)."""
    parsed_rows = []
    sep_indices = set()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        cells = _split_md_table_row(s)
        if cells and all(re.match(r'^:?-+:?$', c) for c in cells if c):
            sep_indices.add(len(parsed_rows))
        parsed_rows.append(cells)

    if not parsed_rows:
        return lines, []

    num_cols = max(len(r) for r in parsed_rows)
    for r in parsed_rows:
        while len(r) < num_cols:
            r.append('')

    col_alignments = ['left'] * num_cols
    for sep_idx in sep_indices:
        for c_idx, cell in enumerate(parsed_rows[sep_idx]):
            c_str = cell.strip()
            if c_str.startswith(':') and c_str.endswith(':'):
                col_alignments[c_idx] = 'center'
            elif c_str.endswith(':'):
                col_alignments[c_idx] = 'right'
            else:
                col_alignments[c_idx] = 'left'

    # If no separator row was in the input, synthesize one after row 0
    if not sep_indices:
        sep_row = []
        for al in col_alignments:
            if al == 'center':
                sep_row.append(':---:')
            elif al == 'right':
                sep_row.append('---:')
            else:
                sep_row.append(':---')
        parsed_rows.insert(1, sep_row)
        sep_indices.add(1)

    # Calculate column widths across all non-separator rows
    col_widths = [3] * num_cols
    for r_idx, row in enumerate(parsed_rows):
        if r_idx in sep_indices:
            continue
        for c_idx in range(num_cols):
            col_widths[c_idx] = max(col_widths[c_idx], len(row[c_idx]))

    formatted_lines = []
    for r_idx, row in enumerate(parsed_rows):
        cells_out = []
        if r_idx in sep_indices:
            for c_idx in range(num_cols):
                w = max(3, col_widths[c_idx])
                al = col_alignments[c_idx]
                if al == 'center':
                    cells_out.append(':' + '-' * max(1, w - 2) + ':')
                elif al == 'right':
                    cells_out.append('-' * max(1, w - 1) + ':')
                else:
                    cells_out.append(':' + '-' * max(1, w - 1) if row[c_idx].startswith(':') else '-' * w)
        else:
            for c_idx in range(num_cols):
                w = col_widths[c_idx]
                val = row[c_idx]
                al = col_alignments[c_idx]
                if al == 'center':
                    cells_out.append(val.center(w))
                elif al == 'right':
                    cells_out.append(val.rjust(w))
                else:
                    cells_out.append(val.ljust(w))
        formatted_lines.append("| " + " | ".join(cells_out) + " |")

    return formatted_lines, col_alignments


def parse_spreadsheet_tsv_to_markdown(tsv_text: str) -> Optional[str]:
    """Convert spreadsheet clipboard text (Excel, Google Sheets, Calc TSV) into a formatted Markdown table."""
    if not tsv_text or '\t' not in tsv_text:
        return None
    raw_lines = [ln for ln in tsv_text.splitlines() if ln.strip()]
    if not raw_lines:
        return None
    tab_count = sum(1 for ln in raw_lines if '\t' in ln)
    if tab_count < 2 and not (len(raw_lines) == 1 and raw_lines[0].count('\t') >= 2):
        return None

    import csv
    import io
    try:
        use_csv_reader = (tsv_text.count('"') % 2 == 0) and ('"' in tsv_text)
        rows = []
        if use_csv_reader:
            try:
                reader = csv.reader(io.StringIO(tsv_text.strip()), delimiter='\t', quotechar='"')
                for raw_row in reader:
                    cleaned_row = [re.sub(r'[\r\n]+', ' ', c).replace('|', '\\|').strip() for c in raw_row]
                    if any(cleaned_row):
                        rows.append(cleaned_row)
            except Exception:
                rows = []

        if not rows:
            for ln in raw_lines:
                cleaned_row = [c.replace('|', '\\|').strip() for c in ln.split('\t')]
                if any(cleaned_row):
                    rows.append(cleaned_row)

        if not rows:
            return None

        max_cols = max(len(r) for r in rows)
        while max_cols > 0 and all(len(r) <= max_cols - 1 or not r[max_cols - 1] for r in rows):
            max_cols -= 1
        if max_cols < 1:
            return None

        for r in rows:
            while len(r) < max_cols:
                r.append('')
            if len(r) > max_cols:
                del r[max_cols:]

        col_align = []
        num_pattern = re.compile(r'^[\$\€\£\¥]?\s*[-+]?\d[\d,]*(\.\d+)?\s*%?$')
        for c in range(max_cols):
            vals = [rows[r][c] for r in range(1 if len(rows) > 1 else 0, len(rows)) if rows[r][c]]
            if vals and all(num_pattern.match(v) for v in vals):
                col_align.append('---:')
            else:
                col_align.append(':---')

        header = rows[0]
        data = rows[1:] if len(rows) > 1 else []
        table_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(col_align) + " |"
        ]
        for d in data:
            table_lines.append("| " + " | ".join(d) + " |")

        formatted, _ = format_markdown_table_lines(table_lines)
        return "\n" + "\n".join(formatted) + "\n"
    except Exception:
        return None


def parse_html_table_to_markdown(html_text: str) -> Optional[str]:
    """Convert HTML <table> markup from clipboard into a clean Markdown table."""
    if not html_text or '<table' not in html_text.lower() or '</table>' not in html_text.lower():
        return None
    try:
        row_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', html_text, re.IGNORECASE | re.DOTALL)
        if not row_matches:
            return None
        rows = []
        for r_html in row_matches:
            cells = re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', r_html, re.IGNORECASE | re.DOTALL)
            cleaned_cells = []
            for c in cells:
                c = re.sub(r'<(?:b|strong)[^>]*>(.*?)</(?:b|strong)>', r'**\1**', c, flags=re.IGNORECASE | re.DOTALL)
                c = re.sub(r'<(?:i|em)[^>]*>(.*?)</(?:i|em)>', r'*\1*', c, flags=re.IGNORECASE | re.DOTALL)
                c = re.sub(r'<(?:code)[^>]*>(.*?)</(?:code)>', r'`\1`', c, flags=re.IGNORECASE | re.DOTALL)
                c = re.sub(r'<[^>]+>', '', c)
                c = re.sub(r'[\r\n]+', ' ', c).replace('|', '\\|').strip()
                cleaned_cells.append(c)
            if any(cleaned_cells):
                rows.append(cleaned_cells)
        if not rows:
            return None
        num_cols = max(len(r) for r in rows)
        if num_cols < 1:
            return None
        for r in rows:
            while len(r) < num_cols:
                r.append('')

        num_pattern = re.compile(r'^[\$\€\£\¥]?\s*[-+]?\d[\d,]*(\.\d+)?\s*%?$')
        col_align = []
        for c in range(num_cols):
            vals = [rows[r][c] for r in range(1 if len(rows) > 1 else 0, len(rows)) if rows[r][c]]
            if vals and all(num_pattern.match(v) for v in vals):
                col_align.append('---:')
            else:
                col_align.append(':---')

        header = rows[0]
        data = rows[1:] if len(rows) > 1 else []
        table_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(col_align) + " |"
        ]
        for d in data:
            table_lines.append("| " + " | ".join(d) + " |")
        formatted, _ = format_markdown_table_lines(table_lines)
        return "\n" + "\n".join(formatted) + "\n"
    except Exception:
        return None


def format_existing_pipe_table_on_paste(text: str) -> Optional[str]:
    """If clipboard text is already a Markdown pipe table, align and pad it cleanly."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2 and all(ln.startswith('|') and ln.endswith('|') for ln in lines):
        if any(re.match(r'^\|(\s*:?-+:?\s*\|)+$', ln) for ln in lines):
            formatted, _ = format_markdown_table_lines(lines)
            return "\n" + "\n".join(formatted) + "\n"
    return None


class VisualTableWidget(tk.Frame):
    """
    Interactive, styled MS Word-like grid widget embedded into the document canvas.
    Supports in-place cell editing, Tab/Shift-Tab cell navigation, auto-row appending on last Tab,
    context menu for inserting/deleting rows & columns, column alignment, and copying as TSV.
    """
    def __init__(self, parent, on_change_callback, table_lines: List[str], is_dark: bool = False,
                 visual_font: str = None, visual_font_size: int = 11, content_width: int = 600, **kwargs):
        super().__init__(parent, bd=1, relief=tk.SOLID, **kwargs)
        self.on_change_callback = on_change_callback
        self.raw_table_lines = list(table_lines)
        self.is_dark = is_dark
        self.visual_font = visual_font or VISUAL_FONT or SANS_FONT
        self.visual_font_size = visual_font_size
        self.content_width = content_width

        # Styling
        self.tbl_border = "#475569" if is_dark else "#cbd5e1"
        self.header_bg = "#334155" if is_dark else "#f1f5f9"
        self.header_fg = "#f8fafc" if is_dark else "#0f172a"
        self.row_bg = "#1e293b" if is_dark else "#ffffff"
        self.row_alt_bg = "#0f172a" if is_dark else "#f8fafc"
        self.cell_fg = "#e2e8f0" if is_dark else "#1e293b"
        self.config(bg=self.tbl_border)

        self._build_table_ui()

    def _build_table_ui(self):
        for w in self.winfo_children():
            w.destroy()

        parsed_rows = []
        sep_indices = set()
        for i, row in enumerate(self.raw_table_lines):
            cells = _split_md_table_row(row)
            if cells and all(re.match(r'^:?-+:?$', c) for c in cells if c):
                sep_indices.add(len(parsed_rows))
            parsed_rows.append(cells)

        if not parsed_rows:
            return

        num_cols = max(len(row) for row in parsed_rows)
        for r in parsed_rows:
            while len(r) < num_cols:
                r.append('')

        self.col_alignments = ['left'] * num_cols
        for sep_idx in sep_indices:
            for c_idx, cell in enumerate(parsed_rows[sep_idx]):
                c_str = cell.strip()
                if c_str.startswith(':') and c_str.endswith(':'):
                    self.col_alignments[c_idx] = 'center'
                elif c_str.endswith(':'):
                    self.col_alignments[c_idx] = 'right'
                else:
                    self.col_alignments[c_idx] = 'left'

        non_sep_rows = [r for idx, r in enumerate(parsed_rows) if idx not in sep_indices]
        if not non_sep_rows:
            return

        self.headers = non_sep_rows[0]
        self.data_rows = non_sep_rows[1:] if len(non_sep_rows) > 1 else []

        # Action strip atop the table
        self.top_strip = tk.Frame(self, bg=self.header_bg)
        self.top_strip.pack(side=tk.TOP, fill=tk.X)

        strip_lbl = tk.Label(
            self.top_strip, text=f" ⊞ Table ({len(self.headers)} cols × {len(self.data_rows)} rows)",
            font=(self.visual_font, max(8, int(self.visual_font_size * 0.8)), "bold"),
            bg=self.header_bg, fg=self.header_fg
        )
        strip_lbl.pack(side=tk.LEFT, padx=4, pady=2)

        def _make_strip_btn(txt, cmd, tip):
            b = tk.Button(
                self.top_strip, text=txt, command=cmd,
                font=(self.visual_font, max(8, int(self.visual_font_size * 0.8))),
                bg=self.header_bg, fg=self.header_fg, activebackground=self.row_alt_bg,
                relief=tk.FLAT, bd=0, padx=4, pady=1, cursor="hand2"
            )
            b.pack(side=tk.RIGHT, padx=2, pady=1)
            ToolTip(b, tip, delay=300)
            return b

        _make_strip_btn("📋 TSV", self.copy_table_tsv, "Copy table data as TSV (tab-separated values) to clipboard for Excel or Sheets.")
        _make_strip_btn("⇄ Align", self.cycle_all_alignment, "Cycle table column alignment between left, center, and right.")
        _make_strip_btn("+ Col", lambda: self.insert_column(len(self.headers)), "Insert a new blank column to the right side of the table.")
        _make_strip_btn("+ Row", lambda: self.insert_row(len(self.data_rows)), "Append a new blank row at the end of the table.")

        self.table_body = tk.Frame(self, bg=self.tbl_border)
        self.table_body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        cell_font = (self.visual_font, max(8, int(self.visual_font_size * 0.95)))
        head_font = (self.visual_font, max(8, int(self.visual_font_size * 0.95)), "bold")

        col_weights = []
        for c in range(num_cols):
            w = len(self.headers[c]) if c < len(self.headers) else 3
            for r in self.data_rows:
                if c < len(r):
                    w = max(w, len(r[c]))
            col_weights.append(max(4, min(28, w)))

        for col_idx, weight in enumerate(col_weights):
            self.table_body.grid_columnconfigure(col_idx, weight=weight)

        justify_map = {'left': tk.LEFT, 'center': tk.CENTER, 'right': tk.RIGHT}

        def _clean_for_display(val: str) -> str:
            v = val.strip().replace(r'\|', '|')
            v = re.sub(r'^\*\*(.*?)\*\*$', r'\1', v)
            v = re.sub(r'^\*(.*?)\*$', r'\1', v)
            v = re.sub(r'^`(.*?)`$', r'\1', v)
            return v

        self.cell_entries = []

        # Header row
        header_row_entries = []
        for col_idx, h in enumerate(self.headers):
            h_var = tk.StringVar(value=_clean_for_display(h))
            entry = tk.Entry(
                self.table_body, textvariable=h_var, font=head_font,
                bg=self.header_bg, fg=self.header_fg, relief=tk.FLAT, bd=0,
                justify=justify_map.get(self.col_alignments[col_idx], tk.LEFT),
                width=max(4, min(28, col_weights[col_idx]))
            )
            entry.grid(row=0, column=col_idx, sticky="nsew", padx=1, pady=1, ipady=3)
            entry.bind("<FocusOut>", lambda e, c=col_idx, v=h_var: self._on_cell_commit(0, c, v.get()))
            entry.bind("<Return>", lambda e, c=col_idx, v=h_var: self._on_cell_commit(0, c, v.get()))
            entry.bind("<Button-3>", lambda e, c=col_idx: self._show_context_menu(e, 0, c))
            header_row_entries.append(entry)
        self.cell_entries.append(header_row_entries)

        # Data rows
        for row_idx, row in enumerate(self.data_rows):
            while len(row) < num_cols:
                row.append('')
            c_bg = self.row_alt_bg if (row_idx % 2 == 1) else self.row_bg
            row_entries = []
            for col_idx, cell in enumerate(row):
                cell_var = tk.StringVar(value=_clean_for_display(cell))
                entry = tk.Entry(
                    self.table_body, textvariable=cell_var, font=cell_font,
                    bg=c_bg, fg=self.cell_fg, relief=tk.FLAT, bd=0,
                    justify=justify_map.get(self.col_alignments[col_idx], tk.LEFT),
                    width=max(4, min(28, col_weights[col_idx]))
                )
                entry.grid(row=row_idx + 1, column=col_idx, sticky="nsew", padx=1, pady=1, ipady=3)
                entry.bind("<FocusOut>", lambda e, r=row_idx + 1, c=col_idx, v=cell_var: self._on_cell_commit(r, c, v.get()))
                entry.bind("<Return>", lambda e, r=row_idx + 1, c=col_idx, v=cell_var: self._on_cell_commit(r, c, v.get()))
                entry.bind("<Button-3>", lambda e, r=row_idx + 1, c=col_idx: self._show_context_menu(e, r, c))
                row_entries.append(entry)
            self.cell_entries.append(row_entries)

        total_r = len(self.cell_entries)
        for r in range(total_r):
            for c in range(num_cols):
                def _make_nav(curr_r, curr_c):
                    def _nav_tab(e):
                        if curr_c + 1 < num_cols:
                            self.cell_entries[curr_r][curr_c + 1].focus_set()
                        elif curr_r + 1 < total_r:
                            self.cell_entries[curr_r + 1][0].focus_set()
                        else:
                            self.insert_row(len(self.data_rows))
                        return "break"

                    def _nav_shift_tab(e):
                        if curr_c > 0:
                            self.cell_entries[curr_r][curr_c - 1].focus_set()
                        elif curr_r > 0:
                            self.cell_entries[curr_r - 1][num_cols - 1].focus_set()
                        return "break"

                    self.cell_entries[curr_r][curr_c].bind("<Tab>", _nav_tab)
                    self.cell_entries[curr_r][curr_c].bind("<Shift-Tab>", _nav_shift_tab)
                    try:
                        self.cell_entries[curr_r][curr_c].bind("<ISO_Left_Tab>", _nav_shift_tab)
                    except tk.TclError:
                        pass
                _make_nav(r, c)

    def _sync(self):
        lines = ["| " + " | ".join(h.replace('|', r'\|') for h in self.headers) + " |"]
        seps = []
        for al in self.col_alignments:
            if al == 'center': seps.append(':---:')
            elif al == 'right': seps.append('---:')
            else: seps.append(':---')
        lines.append("| " + " | ".join(seps) + " |")
        for row in self.data_rows:
            lines.append("| " + " | ".join(c.replace('|', r'\|') for c in row) + " |")

        formatted, _ = format_markdown_table_lines(lines)
        old_str = "\n".join(self.raw_table_lines)
        new_str = "\n".join(formatted)
        if old_str != new_str:
            self.raw_table_lines[:] = formatted
            if self.on_change_callback:
                self.on_change_callback(old_str, new_str)

    def _on_cell_commit(self, r, c, val):
        clean_val = val.strip().replace('|', r'\|')
        if r == 0:
            if c < len(self.headers) and self.headers[c] != clean_val:
                self.headers[c] = clean_val
                self._sync()
        else:
            d_idx = r - 1
            if 0 <= d_idx < len(self.data_rows) and c < len(self.data_rows[d_idx]) and self.data_rows[d_idx][c] != clean_val:
                self.data_rows[d_idx][c] = clean_val
                self._sync()

    def insert_row(self, insert_idx):
        new_row = ['' for _ in self.headers]
        self.data_rows.insert(max(0, min(len(self.data_rows), insert_idx)), new_row)
        self._sync()
        self._build_table_ui()
        target_r = min(len(self.cell_entries) - 1, insert_idx + 1)
        if target_r < len(self.cell_entries) and self.cell_entries[target_r]:
            self.cell_entries[target_r][0].focus_set()

    def delete_row(self, row_idx):
        if 0 <= row_idx < len(self.data_rows):
            del self.data_rows[row_idx]
            self._sync()
            self._build_table_ui()

    def insert_column(self, col_idx):
        self.headers.insert(col_idx, f"Header {len(self.headers) + 1}")
        self.col_alignments.insert(col_idx, 'left')
        for r in self.data_rows:
            r.insert(col_idx, '')
        self._sync()
        self._build_table_ui()

    def delete_column(self, col_idx):
        if len(self.headers) > 1 and 0 <= col_idx < len(self.headers):
            del self.headers[col_idx]
            del self.col_alignments[col_idx]
            for r in self.data_rows:
                if col_idx < len(r):
                    del r[col_idx]
            self._sync()
            self._build_table_ui()

    def cycle_all_alignment(self):
        """Cycle column alignments across left, center, and right."""
        curr = self.col_alignments[0] if self.col_alignments else 'left'
        nxt = 'center' if curr == 'left' else ('right' if curr == 'center' else 'left')
        self.col_alignments = [nxt] * len(self.headers)
        self._sync()
        self._build_table_ui()

    def set_column_alignment(self, col_idx, align):
        if 0 <= col_idx < len(self.col_alignments):
            self.col_alignments[col_idx] = align
            self._sync()
            self._build_table_ui()

    def copy_table_tsv(self):
        rows = [self.headers] + self.data_rows
        tsv = "\n".join(["\t".join(r) for r in rows])
        self.clipboard_clear()
        self.clipboard_append(tsv)

    def _show_context_menu(self, event, r, c):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Insert Row Above", command=lambda: self.insert_row(max(0, r - 1)))
        menu.add_command(label="Insert Row Below", command=lambda: self.insert_row(r))
        menu.add_separator()
        menu.add_command(label="Insert Column Left", command=lambda: self.insert_column(c))
        menu.add_command(label="Insert Column Right", command=lambda: self.insert_column(c + 1))
        menu.add_separator()
        if r > 0:
            menu.add_command(label="Delete Row", command=lambda: self.delete_row(r - 1))
        if len(self.headers) > 1:
            menu.add_command(label="Delete Column", command=lambda: self.delete_column(c))
        menu.add_separator()
        menu.add_command(label="Align Column Left", command=lambda: self.set_column_alignment(c, 'left'))
        menu.add_command(label="Align Column Center", command=lambda: self.set_column_alignment(c, 'center'))
        menu.add_command(label="Align Column Right", command=lambda: self.set_column_alignment(c, 'right'))
        menu.add_separator()
        menu.add_command(label="Copy Table as TSV (Excel)", command=self.copy_table_tsv)
        menu.tk_popup(event.x_root, event.y_root)


class VisualFigureWidget(tk.Frame):
    """
    Interactive figure widget with image thumbnail, alignment, academic caption, and context menu.
    Updates canonical markdown syntax in-place when aligned, resized, or captioned.
    """
    def __init__(self, parent, on_change_callback, raw_match: str, img_info: dict, image_handler,
                 is_dark: bool = False, visual_font: str = None, visual_font_size: int = 11,
                 content_width: int = 600, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_change_callback = on_change_callback
        self.raw_match = raw_match
        self.img_info = dict(img_info)
        self.image_handler = image_handler
        self.is_dark = is_dark
        self.visual_font = visual_font or VISUAL_FONT or SANS_FONT
        self.visual_font_size = visual_font_size
        self.content_width = content_width

        bg_col = "#1e293b" if is_dark else "#ffffff"
        self.config(bg=bg_col)
        self._photo = None
        self._build_figure_ui()

    def _build_figure_ui(self):
        for w in self.winfo_children():
            w.destroy()

        image = self.image_handler.load_image(self.img_info)
        align = self.img_info.get('align', 'center')
        if align not in ('left', 'center', 'right'):
            align = 'center'

        align_anchor = {'left': 'w', 'center': 'center', 'right': 'e'}.get(align, 'center')
        bg_col = "#1e293b" if self.is_dark else "#ffffff"
        cap_fg = "#94a3b8" if self.is_dark else "#64748b"

        container = tk.Frame(self, bg=bg_col)
        container.pack(anchor=align_anchor, padx=12, pady=6)

        if image:
            orig_w, orig_h = image.size
            disp_w = self.img_info.get('width')
            disp_h = self.img_info.get('height')

            if disp_w and disp_h:
                target_w, target_h = disp_w, disp_h
            elif disp_w:
                target_w = disp_w
                target_h = max(1, int(orig_h * disp_w / orig_w))
            elif disp_h:
                target_h = disp_h
                target_w = max(1, int(orig_w * disp_h / orig_h))
            else:
                target_w = min(orig_w, 550)
                target_h = max(1, int(orig_h * target_w / orig_w))

            max_w = max(80, self.content_width - 30)
            if target_w > max_w:
                target_h = max(1, int(target_h * max_w / target_w))
                target_w = max_w

            display = image
            if (target_w, target_h) != image.size:
                try:
                    display = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
                except Exception:
                    display = image

            self._photo = self.image_handler.get_photo_image(display, thumbnail=False)
            if self._photo:
                img_lbl = tk.Label(container, image=self._photo, bg=bg_col, bd=0, relief=tk.FLAT)
                img_lbl.pack(anchor=align_anchor)
                img_lbl.bind("<Button-3>", self._show_context_menu)
                img_lbl.bind("<Double-Button-1>", lambda e: ImageViewerWindow(self.winfo_toplevel(), image, self.img_info.get('alt', 'Image')))
        else:
            alt = self.img_info.get('alt', 'Image')
            ph_lbl = tk.Label(container, text=f"[🖼️ {alt}]", bg=bg_col, fg=cap_fg, font=(self.visual_font, 10, 'italic'))
            ph_lbl.pack(anchor=align_anchor)
            ph_lbl.bind("<Button-3>", self._show_context_menu)

        cap = self.img_info.get('caption')
        if not cap:
            alt_text = self.img_info.get('alt', '')
            if alt_text and alt_text.lower() not in ('image', 'img', 'figure', 'pic'):
                cap = alt_text

        if cap:
            cap_lbl = tk.Label(
                container, text=cap, bg=bg_col, fg=cap_fg,
                font=(self.visual_font, max(8, int(self.visual_font_size * 0.85)), 'italic'),
                justify=tk.CENTER
            )
            cap_lbl.pack(anchor=align_anchor, pady=(3, 0))
            cap_lbl.bind("<Button-3>", self._show_context_menu)

    def _set_prop(self, prop, val):
        full_match = self.raw_match
        m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', full_match)
        if not m:
            return
        meta_str = m.group(1)
        path = m.group(2)
        parsed = self.image_handler._parse_extended_alt(meta_str)

        if prop == 'width':
            parsed['width'] = int(val) if val is not None else None
        elif prop == 'align':
            parsed['align'] = str(val) if val else 'center'
        elif prop == 'caption':
            parsed['caption'] = str(val).strip() if val else None

        base_alt = parsed.get('alt', '')
        params = []
        if parsed.get('width'):
            params.append(f"width={parsed['width']}")
        if parsed.get('align') and parsed['align'] != 'center':
            params.append(f"align={parsed['align']}")
        if parsed.get('caption'):
            params.append(f'caption="{parsed["caption"]}"')

        new_match = f"![{base_alt}|{','.join(params)}]({path})" if params else f"![{base_alt}]({path})"

        old_match = self.raw_match
        self.raw_match = new_match
        self.img_info['width'] = parsed.get('width')
        self.img_info['align'] = parsed.get('align', 'center')
        self.img_info['caption'] = parsed.get('caption')
        self._build_figure_ui()

        if self.on_change_callback:
            self.on_change_callback(old_match, new_match)

    def _prompt_caption(self):
        curr_cap = self.img_info.get('caption') or self.img_info.get('alt', '')
        from tkinter import simpledialog
        new_cap = simpledialog.askstring("Figure Caption", "Enter figure caption:", initialvalue=curr_cap, parent=self)
        if new_cap is not None:
            self._set_prop('caption', new_cap.strip())

    def _show_context_menu(self, event):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Align Left", command=lambda: self._set_prop('align', 'left'))
        menu.add_command(label="Align Center", command=lambda: self._set_prop('align', 'center'))
        menu.add_command(label="Align Right", command=lambda: self._set_prop('align', 'right'))
        menu.add_separator()
        menu.add_command(label="Size: 25%", command=lambda: self._set_prop('width', int(self.content_width * 0.25)))
        menu.add_command(label="Size: 50%", command=lambda: self._set_prop('width', int(self.content_width * 0.50)))
        menu.add_command(label="Size: 75%", command=lambda: self._set_prop('width', int(self.content_width * 0.75)))
        menu.add_command(label="Size: 100% (Full Width)", command=lambda: self._set_prop('width', int(self.content_width)))
        menu.add_command(label="Size: Original", command=lambda: self._set_prop('width', None))
        menu.add_separator()
        menu.add_command(label="Edit Caption...", command=self._prompt_caption)
        menu.tk_popup(event.x_root, event.y_root)



class MarkdownVisualWidget(tk.Frame):
    """
    A widget for rendering markdown visually with image support.
    Editable - changes sync back to source.
    Optimized for performance with large documents using a single Text widget.
    """
    
    # Chunk size for progressive rendering
    RENDER_CHUNK_SIZE = 100  # lines per chunk
    
    def __init__(self, parent, base_path=None, on_content_change=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.content = ""
        self.base_path = base_path or os.getcwd()
        self.image_handler = ImageHandler(self.base_path)
        self.embedded_images = {}  # Track embedded images by text index
        self._render_job = None  # For cancelling pending renders
        self.on_content_change = on_content_change  # Callback for content changes
        self._last_rendered_hash = ''
        self.current_theme = None
        self.zoom = 1.0
        self._rerender_job = None
        self._pending_yview = None
        self._rendered_width = None
        self._hr_color = "#cbd5e1"
        self.on_zoom_wheel = None
        self._anchor_marks = {}
        self._anchor_occurrence_marks = {}
        self._heading_marks = {}
        self._footnotes = {}
        self._footnote_def_marks = {}
        self._footnote_ref_marks = {}
        self._footnote_occurrence_marks = {}
        self._fn_tooltip = None
        self._nav_history = []
        self._link_counter = 0
        self._setup_widget()
    
    def _setup_widget(self):
        """Setup the visual rendering widget - read-only presentation on a page"""
        self.page_view = PageView(self, desk_bg="#d9dce1", page_bg="#ffffff")
        self.page_view.pack(fill=tk.BOTH, expand=True)
        self.page_view.on_resized = self._on_page_resized
        
        self.text_widget = tk.Text(
            self.page_view.sheet,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            padx=40,
            pady=24,
            bg="#ffffff",
            fg="#1a1a1a",
            cursor="xterm",
            selectbackground="#b3d9ff",
            selectforeground="#000000",
            inactiveselectbackground="#b3d9ff",
            spacing1=2,  # Space above lines
            spacing2=3,  # Proportional space between wrapped lines  
            spacing3=6,  # Space below lines
            state=tk.DISABLED  # Strictly read-only presentation
        )
        self.page_view.attach_text(self.text_widget)
        # Backwards-compatible alias
        self.scrollbar = self.page_view.scrollbar
        
        # Setup text tags for styling
        self._setup_text_tags()
        
        # Bind mousewheel, zoom, and navigation
        self.text_widget.bind("<MouseWheel>", self._on_mousewheel)
        self.text_widget.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.text_widget.bind("<Alt-Left>", lambda e: self.navigate_back())
        self.bind("<Alt-Left>", lambda e: self.navigate_back())

        # Right-click context menu for selection and copying
        self._setup_context_menu()
        
        self.render_mode = "text"
    
    def _setup_context_menu(self):
        """Create right-click context menu for copying selected text"""
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self._copy_selection, accelerator="Ctrl+C")
        self.context_menu.add_command(label="Select All", command=self._select_all, accelerator="Ctrl+A")
        self.text_widget.bind("<Button-3>", self._show_context_menu)

    def _show_context_menu(self, event):
        try:
            has_sel = bool(self.text_widget.tag_ranges(tk.SEL))
        except Exception:
            has_sel = False
        self.context_menu.entryconfigure("Copy", state=tk.NORMAL if has_sel else tk.DISABLED)
        self.context_menu.tk_popup(event.x_root, event.y_root)

    def _copy_selection(self):
        try:
            sel_text = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.clipboard_clear()
            self.clipboard_append(sel_text)
        except tk.TclError:
            pass

    def _select_all(self):
        self.text_widget.tag_add(tk.SEL, "1.0", tk.END)
        return "break"

    # --- zoom / page helpers -------------------------------------------------

    def _px(self, pt, scale=1.0):
        """Font size in pixels (negative = pixels in Tk) for a point size at the current zoom."""
        return -max(4, round(pt * scale * self.page_view.dpi() / 72.0 * self.zoom))

    def _s(self, n):
        """Scale a pixel measurement (defined at 96 dpi, 100% zoom) to the current zoom/dpi."""
        return max(0, int(n * self.page_view.dpi() / 96.0 * self.zoom))

    def content_width_px(self):
        return self.page_view.content_width

    def set_zoom(self, zoom):
        """Apply a zoom factor: text styling immediately, embedded widgets after a short debounce."""
        self.zoom = zoom
        self.page_view.set_zoom(zoom)
        self._apply_base_font()
        self._update_tags_for_theme(self.current_theme or {})
        self._schedule_rerender()

    def _apply_base_font(self):
        theme = self.current_theme or {}
        family = theme.get('visual_font') or VISUAL_FONT or 'Segoe UI'
        self.text_widget.config(font=(family, self._px(theme.get('visual_font_size', 11))))

    def set_page(self, size=None, margin=None, enabled=None):
        self.page_view.set_page(size=size, margin=margin, enabled=enabled)

    def _on_page_resized(self):
        """The text column changed width; rebuild width-dependent content (tables, images, rules)."""
        width = self.content_width_px()
        if self._rendered_width is not None and abs(width - self._rendered_width) <= 8:
            return
        self._schedule_rerender()

    def _schedule_rerender(self):
        if not self.content:
            return
        if self._rerender_job:
            self.after_cancel(self._rerender_job)
        self._rerender_job = self.after(150, self.rerender)

    def rerender(self):
        """Force a re-render of the current content, keeping the scroll position."""
        self._rerender_job = None
        if not self.content:
            return
        first = self.text_widget.index('@0,0')  # top visible logical line is stable across zoom/resize
        self._last_rendered_hash = ''
        self._pending_yview = first
        self.set_content(self.content)
        if self._render_job is None:
            self._restore_yview()

    def _restore_yview(self):
        if self._pending_yview is not None:
            self.text_widget.update_idletasks()
            self.text_widget.yview(self._pending_yview)
            self._pending_yview = None

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        delta = int(-event.delta / 40) if event.delta else 0
        if delta:
            self.text_widget.yview_scroll(delta, "units")
        return "break"
    
    def _on_ctrl_mousewheel(self, event):
        """Handle Ctrl+MouseWheel for zoom"""
        if self.on_zoom_wheel:
            return self.on_zoom_wheel(event)
        return
    
    def set_base_path(self, path):
        """Set the base path for resolving relative image paths"""
        self.base_path = path
        self.image_handler.base_path = path
    
    def set_content(self, markdown_text):
        """Set and render markdown content with images in read-only presentation mode"""
        import hashlib as _hl
        content_hash = _hl.md5(markdown_text.encode()).hexdigest()
        if content_hash == self._last_rendered_hash and self.text_widget.get('1.0', tk.END).strip():
            return
        self._last_rendered_hash = content_hash

        # Cancel any pending render
        if self._render_job:
            self.after_cancel(self._render_job)
            self._render_job = None
        
        self.content = markdown_text
        self._rendered_width = self.content_width_px()
        
        # Enable text widget temporarily to replace content
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        self.embedded_images.clear()
        self.image_handler.clear_caches()
        self._anchor_marks.clear()
        self._anchor_occurrence_marks.clear()
        self._heading_marks.clear()
        self._footnotes.clear()
        self._footnote_def_marks.clear()
        self._footnote_ref_marks.clear()
        self._footnote_occurrence_marks.clear()
        self._link_counter = 0

        # Pre-scan footnote definitions so hover tooltips work regardless of position
        for fn_match in re.finditer(r'^\[\^([^\]]+)\]:\s*(.+)$', markdown_text, re.MULTILINE):
            self._footnotes[fn_match.group(1).strip()] = fn_match.group(2).strip()
        
        # Find all images in the text
        images = self.image_handler.find_images(markdown_text)
        
        line_count = markdown_text.count('\n')
        if line_count < 500:
            self._render_content(markdown_text, images)
            self.text_widget.config(state=tk.DISABLED)
        else:
            # Progressive rendering for large documents
            self._render_progressive(markdown_text, images, 0)
    
    def _render_progressive(self, full_text, images, start_line):
        """Render content progressively to avoid UI freeze"""
        self.text_widget.config(state=tk.NORMAL)
        lines = full_text.split('\n')
        end_line = min(start_line + self.RENDER_CHUNK_SIZE, len(lines))
        
        # Render this chunk
        chunk = '\n'.join(lines[start_line:end_line])
        chunk_lines_text = '\n'.join(lines[start_line:end_line])
        chunk_images = [img for img in images 
                        if img.get('full_match') and img['full_match'] in chunk_lines_text]
        
        self._render_content(chunk, chunk_images, is_chunk=True)
        
        if end_line < len(lines):
            self._render_job = self.after(10, 
                lambda: self._render_progressive(full_text, images, end_line))
        else:
            self._render_job = None
            self.text_widget.config(state=tk.DISABLED)
            self._restore_yview()
    
    # Inline markdown pattern: captures bold, italic, highlight, code, math, footnotes, anchors, links in order
    _INLINE_RE = re.compile(
        r'(\*\*|__)(?P<bold>.+?)\1'        # **bold** or __bold__
        r'|(?<!\*)\*(?!\*)(?P<italic>.+?)(?<!\*)\*(?!\*)'  # *italic*
        r'|(?<!_)_(?!_)(?P<italic2>.+?)(?<!_)_(?!_)'       # _italic_
        r'|==(?P<highlight>[^=\n]+)=='     # ==highlight==
        r'|<mark>(?P<mark_text>.*?)</mark>' # <mark>highlight</mark>
        r'|`(?P<code>[^`]+)`'              # `code`
        r'|(?<!\$)\$(?!\$)(?P<math_inline>[^$\n]+?)(?<!\$)\$(?!\$)' # $math$
        r'|\[\^(?P<footnote_ref>[^\]\n]+)\]' # [^1]
        r'|\[@(?:anchor|def):\s*(?P<anchor_name>[^\]\n]+)\]' # [@anchor: point]
        r'|\[@(?!(?:anchor|def):)(?P<ref_name>[^\]\n]+)\]'   # [@point]
        r'|\[(?P<link_label>[^\]]+)\]\((?P<link_url>[^\)]+)\)' # [label](url)
    )

    def _insert_inline(self, text, base_tags=()):
        """Parse inline markdown in *text* and insert it into the text widget with tags.

        Markers are stripped; the plain content is inserted with the appropriate
        inline tag plus any *base_tags* passed in (e.g. a block-level tag like
        "blockquote" or "list_item").
        """
        tw = self.text_widget
        pos = 0
        for m in self._INLINE_RE.finditer(text):
            # Insert any plain text before this match
            if m.start() > pos:
                tw.insert(tk.END, text[pos:m.start()], base_tags if base_tags else '')
            # Determine matched group and inline tag
            if m.group('bold') is not None:
                tw.insert(tk.END, m.group('bold'), base_tags + ('bold',) if base_tags else 'bold')
            elif m.group('italic') is not None:
                tw.insert(tk.END, m.group('italic'), base_tags + ('italic',) if base_tags else 'italic')
            elif m.group('italic2') is not None:
                tw.insert(tk.END, m.group('italic2'), base_tags + ('italic',) if base_tags else 'italic')
            elif m.group('highlight') is not None:
                tw.insert(tk.END, m.group('highlight'), base_tags + ('highlight',) if base_tags else 'highlight')
            elif m.group('mark_text') is not None:
                tw.insert(tk.END, m.group('mark_text'), base_tags + ('highlight',) if base_tags else 'highlight')
            elif m.group('code') is not None:
                tw.insert(tk.END, m.group('code'), base_tags + ('code',) if base_tags else 'code')
            elif m.group('math_inline') is not None:
                rendered_math = _latex_to_unicode(m.group('math_inline'))
                tw.insert(tk.END, rendered_math, base_tags + ('math_inline',) if base_tags else 'math_inline')
            elif m.group('footnote_ref') is not None:
                fn_id = m.group('footnote_ref').strip()
                fn_key = fn_id.lower()
                self._link_counter += 1
                ref_mark = f"fnref_{self._link_counter}"
                tw.mark_set(ref_mark, "end-1c")
                tw.mark_gravity(ref_mark, tk.LEFT)
                if fn_id not in self._footnote_ref_marks:
                    self._footnote_ref_marks[fn_id] = ref_mark
                fn_occs = self._footnote_occurrence_marks.setdefault(fn_key, [])
                fn_occ_mark = f"fnocc_{self._link_counter}"
                tw.mark_set(fn_occ_mark, "end-1c")
                tw.mark_gravity(fn_occ_mark, tk.LEFT)
                fn_occs.append(fn_occ_mark)
                fn_tag = f"fn_ref_tag_{self._link_counter}"
                tw.insert(tk.END, f"[{fn_id}]", base_tags + ('footnote_ref', fn_tag) if base_tags else ('footnote_ref', fn_tag))
                for seq in ("<Button-1>", "<Control-Button-1>", "<Shift-Button-1>", "<Control-Shift-Button-1>"):
                    tw.tag_bind(fn_tag, seq, lambda e, fid=fn_id, om=fn_occ_mark: self._cycle_footnote_occurrence(fid, om, reverse=bool(e.state & 0x0001)))
                tw.tag_bind(fn_tag, "<Enter>", lambda e, fid=fn_id: self._show_footnote_tooltip(e, fid))
                tw.tag_bind(fn_tag, "<Leave>", lambda e: self._hide_footnote_tooltip())
            elif m.group('anchor_name') is not None:
                raw_anchor = m.group('anchor_name').strip()
                key = _normalize_anchor_key(raw_anchor)
                self._link_counter += 1
                mark_name = f"anc_{self._link_counter}"
                tw.mark_set(mark_name, "end-1c")
                tw.mark_gravity(mark_name, tk.LEFT)
                self._anchor_marks[key] = mark_name
                occ_list = self._anchor_occurrence_marks.setdefault(key, [])
                occ_mark = f"occ_{self._link_counter}"
                tw.mark_set(occ_mark, "end-1c")
                tw.mark_gravity(occ_mark, tk.LEFT)
                occ_list.append(occ_mark)
                anc_tag = f"anc_badge_{self._link_counter}"
                tw.insert(tk.END, f"⚓ {raw_anchor}", base_tags + ('anchor_badge', anc_tag) if base_tags else ('anchor_badge', anc_tag))
                for seq in ("<Button-1>", "<Control-Button-1>", "<Shift-Button-1>", "<Control-Shift-Button-1>"):
                    tw.tag_bind(anc_tag, seq, lambda e, k=key, r=raw_anchor, om=occ_mark: self._cycle_anchor_occurrence(k, r, om, reverse=bool(e.state & 0x0001)))
                tw.tag_bind(anc_tag, "<Button-3>", lambda e, r=raw_anchor: self._copy_anchor_ref(r))
                tw.tag_bind(anc_tag, "<Enter>", lambda e: tw.config(cursor="hand2"))
                tw.tag_bind(anc_tag, "<Leave>", lambda e: tw.config(cursor=""))
            elif m.group('ref_name') is not None:
                raw_ref = m.group('ref_name').strip()
                key = _normalize_anchor_key(raw_ref)
                self._link_counter += 1
                occ_list = self._anchor_occurrence_marks.setdefault(key, [])
                occ_mark = f"occ_{self._link_counter}"
                tw.mark_set(occ_mark, "end-1c")
                tw.mark_gravity(occ_mark, tk.LEFT)
                occ_list.append(occ_mark)
                tag_id = f"link_ref_{self._link_counter}"
                tw.insert(tk.END, f"[@{raw_ref}]", base_tags + ('link', tag_id) if base_tags else ('link', tag_id))
                for seq in ("<Button-1>", "<Control-Button-1>", "<Shift-Button-1>", "<Control-Shift-Button-1>"):
                    tw.tag_bind(tag_id, seq, lambda e, k=key, r=raw_ref, om=occ_mark: self._cycle_anchor_occurrence(k, r, om, reverse=bool(e.state & 0x0001)))
                tw.tag_bind(tag_id, "<Button-3>", lambda e, r=raw_ref: self._copy_anchor_ref(r))
                tw.tag_bind(tag_id, "<Enter>", lambda e: tw.config(cursor="hand2"))
                tw.tag_bind(tag_id, "<Leave>", lambda e: tw.config(cursor=""))
            elif m.group('link_label') is not None:
                label = m.group('link_label')
                url = (m.group('link_url') or '').strip()
                self._link_counter += 1
                tag_id = f"link_url_{self._link_counter}"
                if url.startswith(('#', '@')):
                    raw_target = url.lstrip('#@').strip()
                    target_key = _normalize_anchor_key(raw_target)
                    occ_list = self._anchor_occurrence_marks.setdefault(target_key, [])
                    occ_mark = f"occ_{self._link_counter}"
                    tw.mark_set(occ_mark, "end-1c")
                    tw.mark_gravity(occ_mark, tk.LEFT)
                    occ_list.append(occ_mark)
                    tw.insert(tk.END, label, base_tags + ('link', tag_id) if base_tags else ('link', tag_id))
                    for seq in ("<Button-1>", "<Control-Button-1>", "<Shift-Button-1>", "<Control-Shift-Button-1>"):
                        tw.tag_bind(tag_id, seq, lambda e, k=target_key, r=raw_target, om=occ_mark: self._cycle_anchor_occurrence(k, r, om, reverse=bool(e.state & 0x0001)))
                else:
                    tw.insert(tk.END, label, base_tags + ('link', tag_id) if base_tags else ('link', tag_id))
                    tw.tag_bind(tag_id, "<Button-1>", lambda e, u=url: self._open_external_link(u))
                tw.tag_bind(tag_id, "<Enter>", lambda e: tw.config(cursor="hand2"))
                tw.tag_bind(tag_id, "<Leave>", lambda e: tw.config(cursor=""))
            pos = m.end()
        # Insert any remaining plain text
        if pos < len(text):
            tw.insert(tk.END, text[pos:], base_tags if base_tags else '')

    def _render_content(self, markdown_text, images, is_chunk=False):
        """Render markdown content efficiently into the single text widget"""
        if not is_chunk:
            self._anchor_marks.clear()
            self._anchor_occurrence_marks.clear()
            self._heading_marks.clear()
            self._footnote_def_marks.clear()
            self._footnote_ref_marks.clear()
            self._footnote_occurrence_marks.clear()
            self._link_counter = 0
        # Build a map of image positions
        image_positions = {img['start']: img for img in images}
        
        lines = markdown_text.split('\n')
        in_code_block = False
        code_block_content = []
        in_math_block = False
        math_block_content = []
        in_table = False
        table_rows = []
        
        for i, line in enumerate(lines):
            # Check if this line contains an image
            img_match = re.search(r'!\[.*?\]\(.*?\)|<img\s+[^>]*src=.*?>', line.strip(), re.IGNORECASE)
            
            # Handle code blocks
            if line.strip().startswith('```'):
                # Flush any pending table
                if in_table and table_rows:
                    self._render_table(table_rows)
                    table_rows = []
                    in_table = False
                
                if in_code_block:
                    code_text = '\n'.join(code_block_content) + '\n'
                    self.text_widget.insert(tk.END, code_text, "code_block")
                    self.text_widget.insert(tk.END, '\n')
                    code_block_content = []
                    in_code_block = False
                else:
                    in_code_block = True
                continue
            
            if in_code_block:
                code_block_content.append(line)
                continue

            # Handle display math blocks ($$ ... $$)
            stripped_line = line.strip()
            if stripped_line.startswith('$$'):
                if in_table and table_rows:
                    self._render_table(table_rows)
                    table_rows = []
                    in_table = False
                # Single-line $$ ... $$
                if not in_math_block and stripped_line.endswith('$$') and len(stripped_line) > 4:
                    expr = stripped_line[2:-2].strip()
                    self.text_widget.insert(tk.END, _latex_to_unicode(expr) + '\n', "math_block")
                    continue
                elif in_math_block:
                    tail = stripped_line[:-2].strip() if stripped_line.endswith('$$') else stripped_line[2:].strip()
                    if tail:
                        math_block_content.append(tail)
                    expr = '\n'.join(math_block_content).strip()
                    self.text_widget.insert(tk.END, _latex_to_unicode(expr) + '\n', "math_block")
                    math_block_content = []
                    in_math_block = False
                    continue
                else:
                    in_math_block = True
                    head = stripped_line[2:].strip()
                    if head:
                        math_block_content.append(head)
                    continue

            if in_math_block:
                if stripped_line.endswith('$$'):
                    tail = stripped_line[:-2].strip()
                    if tail:
                        math_block_content.append(tail)
                    expr = '\n'.join(math_block_content).strip()
                    self.text_widget.insert(tk.END, _latex_to_unicode(expr) + '\n', "math_block")
                    math_block_content = []
                    in_math_block = False
                else:
                    math_block_content.append(line)
                continue
            
            # Handle tables - collect consecutive table rows
            if '|' in line and line.strip().startswith('|'):
                if not in_table:
                    in_table = True
                table_rows.append(line)
                continue
            else:
                # End of table - render it
                if in_table and table_rows:
                    self._render_table(table_rows)
                    table_rows = []
                    in_table = False
            
            # Handle footnote definition: [^id]: text
            fn_def_match = re.match(r'^\[\^([^\]]+)\]:\s*(.+)$', line.strip())
            if fn_def_match:
                fn_id = fn_def_match.group(1).strip()
                fn_key = fn_id.lower()
                fn_body = fn_def_match.group(2).strip()
                self._footnotes[fn_id] = fn_body
                self._link_counter += 1
                def_mark = f"fndef_{self._link_counter}"
                self.text_widget.mark_set(def_mark, "end-1c")
                self.text_widget.mark_gravity(def_mark, tk.LEFT)
                self._footnote_def_marks[fn_id] = def_mark
                fn_occs = self._footnote_occurrence_marks.setdefault(fn_key, [])
                fn_occ_mark = f"fnocc_{self._link_counter}"
                self.text_widget.mark_set(fn_occ_mark, "end-1c")
                self.text_widget.mark_gravity(fn_occ_mark, tk.LEFT)
                fn_occs.append(fn_occ_mark)
                back_tag = f"fn_back_{self._link_counter}"
                self.text_widget.insert(tk.END, f"[{fn_id}] ", ('footnote_def', 'footnote_badge', back_tag))
                self._insert_inline(fn_body, ('footnote_def',))
                self.text_widget.insert(tk.END, " ↩\n", ('footnote_def', 'link', back_tag))
                for seq in ("<Button-1>", "<Control-Button-1>", "<Shift-Button-1>", "<Control-Shift-Button-1>"):
                    self.text_widget.tag_bind(back_tag, seq, lambda e, fid=fn_id, om=fn_occ_mark: self._cycle_footnote_occurrence(fid, om, reverse=bool(e.state & 0x0001)))
                self.text_widget.tag_bind(back_tag, "<Enter>", lambda e: self.text_widget.config(cursor="hand2"))
                self.text_widget.tag_bind(back_tag, "<Leave>", lambda e: self.text_widget.config(cursor=""))
                continue

            # Handle images - insert placeholder with embedded image
            if img_match and PIL_AVAILABLE:
                matched_img = None
                for img in images:
                    if img.get('full_match') and img['full_match'] in line:
                        matched_img = img
                        break

                if matched_img:
                    full_match = matched_img['full_match']
                    before, _, after = line.partition(full_match)
                    if before.strip():
                        self._insert_inline(before, ('paragraph',))
                    self._insert_image_inline(matched_img)
                    if after.strip():
                        self._insert_inline(after, ('paragraph',))
                    self.text_widget.insert(tk.END, '\n')
                else:
                    self._insert_inline(line, ('paragraph',))
                    self.text_widget.insert(tk.END, '\n', 'paragraph')
                continue
            
            # Headers
            if line.startswith('######'):
                self._register_heading(line[6:].strip())
                self._insert_inline(line[6:].strip(), ('h6',))
                self.text_widget.insert(tk.END, '\n', 'h6')
            elif line.startswith('#####'):
                self._register_heading(line[5:].strip())
                self._insert_inline(line[5:].strip(), ('h5',))
                self.text_widget.insert(tk.END, '\n', 'h5')
            elif line.startswith('####'):
                self._register_heading(line[4:].strip())
                self._insert_inline(line[4:].strip(), ('h4',))
                self.text_widget.insert(tk.END, '\n', 'h4')
            elif line.startswith('###'):
                self._register_heading(line[3:].strip())
                self._insert_inline(line[3:].strip(), ('h3',))
                self.text_widget.insert(tk.END, '\n', 'h3')
            elif line.startswith('##'):
                self._register_heading(line[2:].strip())
                self._insert_inline(line[2:].strip(), ('h2',))
                self.text_widget.insert(tk.END, '\n', 'h2')
            elif line.startswith('#'):
                self._register_heading(line[1:].strip())
                self._insert_inline(line[1:].strip(), ('h1',))
                self.text_widget.insert(tk.END, '\n', 'h1')
            
            # Horizontal rule
            elif line.strip() in ['---', '***', '___']:
                self._insert_hr()
            
            # Blockquote
            elif line.startswith('>'):
                self.text_widget.insert(tk.END, '▌ ', ('blockquote', 'quote_bar'))
                self._insert_inline(line[1:].strip(), ('blockquote',))
                self.text_widget.insert(tk.END, '\n', 'blockquote')
            
            # List items (nesting by leading whitespace)
            elif re.match(r'^[\*\-\+]\s', line.strip()):
                depth = self._list_depth(line)
                bullet = ('•', '◦', '▪', '▪')[depth]
                tags = ('list_item', f'list_d{depth}')
                self.text_widget.insert(tk.END, bullet + ' ', tags)
                self._insert_inline(line.strip()[2:], tags)
                self.text_widget.insert(tk.END, '\n', tags)
            elif re.match(r'^\d+\.\s', line.strip()):
                m_li = re.match(r'^(\d+\.\s)', line.strip())
                tags = ('list_item', f'list_d{self._list_depth(line)}')
                self.text_widget.insert(tk.END, m_li.group(1), tags)
                self._insert_inline(line.strip()[m_li.end():], tags)
                self.text_widget.insert(tk.END, '\n', tags)
            
            # Normal text
            else:
                self._insert_inline(line, ('paragraph',))
                self.text_widget.insert(tk.END, '\n', 'paragraph')
        
        # Flush any remaining table at end of content
        if in_table and table_rows:
            self._render_table(table_rows)

    @staticmethod
    def _list_depth(line):
        """Nesting depth (0-3) of a list item from its leading whitespace."""
        expanded = line.expandtabs(4)
        indent = len(expanded) - len(expanded.lstrip(' '))
        return max(0, min(3, indent // 2))

    def _insert_hr(self):
        """Insert a thin rule spanning the text column (rebuilt on zoom/resize)."""
        rule = tk.Frame(self.text_widget, bg=self._hr_color, height=max(1, self._s(1)),
                        width=self.content_width_px())
        self.text_widget.window_create(tk.END, window=rule, pady=self._s(8))
        self.text_widget.insert(tk.END, '\n', 'hr')
    
    def _is_dark(self, hex_color):
        """Helper to determine if a background color is dark"""
        if not hex_color or not isinstance(hex_color, str):
            return False
        c = hex_color.lstrip('#')
        if len(c) == 6:
            try:
                r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                return lum < 0.5
            except ValueError:
                pass
        return False

    def _render_table(self, table_rows):
        """Render a markdown table as an interactive, styled MS Word-like grid frame inside the text widget."""
        if not table_rows:
            return
        is_dark = self._is_dark(self.text_widget.cget('bg'))
        font_family = (self.current_theme.get('visual_font') if self.current_theme else None) or VISUAL_FONT or SANS_FONT
        font_size = self.current_theme.get('visual_font_size', 11) if self.current_theme else 11

        tbl_widget = VisualTableWidget(
            self.text_widget,
            on_change_callback=lambda old, new: self.on_content_change(old, new) if self.on_content_change else None,
            table_lines=table_rows,
            is_dark=is_dark,
            visual_font=font_family,
            visual_font_size=font_size,
            content_width=self.content_width_px()
        )
        self.text_widget.insert(tk.END, '\n')
        self.text_widget.window_create(tk.END, window=tbl_widget, padx=self._s(6), pady=self._s(8))
        self.text_widget.insert(tk.END, '\n\n')

    def _insert_image_inline(self, img_info):
        """Insert an image inline in the text widget with true alignment, academic captions, and context controls"""
        raw_match = img_info.get('full_match', '')
        is_dark = self._is_dark(self.text_widget.cget('bg'))
        font_family = (self.current_theme.get('visual_font') if self.current_theme else None) or VISUAL_FONT or SANS_FONT
        font_size = self.current_theme.get('visual_font_size', 11) if self.current_theme else 11

        fig_widget = VisualFigureWidget(
            self.text_widget,
            on_change_callback=lambda old, new: self.on_content_change(old, new) if self.on_content_change else None,
            raw_match=raw_match,
            img_info=img_info,
            image_handler=self.image_handler,
            is_dark=is_dark,
            visual_font=font_family,
            visual_font_size=font_size,
            content_width=self.content_width_px()
        )
        self.text_widget.insert(tk.END, '\n')
        self.text_widget.window_create(tk.END, window=fig_widget, padx=self._s(6), pady=self._s(8))
        self.text_widget.insert(tk.END, '\n\n')

    
    def _on_image_click(self, event):
        """Handle click on embedded image"""
        index = self.text_widget.index(f"@{event.x},{event.y}")
        for img_index, (photo, full_image, alt) in self.embedded_images.items():
            if self.text_widget.compare(index, ">=", img_index + "-1c") and \
               self.text_widget.compare(index, "<=", img_index + "+1c"):
                self._show_full_image(full_image, alt)
                break
    
    def _show_full_image(self, image, alt_text=""):
        """Show full-size image in popup window"""
        title = f"Image: {alt_text}" if alt_text else "Image Viewer"
        ImageViewerWindow(self.winfo_toplevel(), image, title)
    
    def _setup_text_tags(self):
        """Setup text tags for markdown styling on the main text widget"""
        self._update_tags_for_theme(self.current_theme or {})

    def _update_tags_for_theme(self, theme):
        """Update tag colors and fonts dynamically according to active theme"""
        tw = self.text_widget
        bg_col = theme.get('visual_bg', '#ffffff')
        fg_col = theme.get('visual_fg', '#1a1a1a')
        font_family = theme.get('visual_font') or VISUAL_FONT or SANS_FONT or "TkDefaultFont"
        font_size = theme.get('visual_font_size', 11)

        is_dark = self._is_dark(bg_col)

        if is_dark:
            h1_fg = "#f8fafc"
            h2_fg = "#f1f5f9"
            h3_fg = "#e2e8f0"
            h4_fg = "#cbd5e1"
            h5_fg = "#94a3b8"
            h6_fg = "#94a3b8"
            code_bg = "#1e1e2e"
            code_fg = "#cdd6f4"
            inline_code_bg = "#282a36"
            inline_code_fg = "#f38ba8"
            quote_bg = "#181825"
            quote_fg = "#a6adc8"
            link_fg = "#89b4fa"
            hr_fg = "#45475a"
            cap_fg = "#94a3b8"
        else:
            h1_fg = "#0f172a"
            h2_fg = "#1e293b"
            h3_fg = "#334155"
            h4_fg = "#475569"
            h5_fg = "#64748b"
            h6_fg = "#64748b"
            code_bg = "#282c34"
            code_fg = "#abb2bf"
            inline_code_bg = "#f1f5f9"
            inline_code_fg = "#be185d"
            quote_bg = "#f8fafc"
            quote_fg = "#475569"
            link_fg = "#2563eb"
            hr_fg = "#e2e8f0"
            cap_fg = "#64748b"

        self._hr_color = hr_fg if is_dark else "#cbd5e1"
        bpx = abs(self._px(font_size))  # body font height in pixels at the current zoom
        mono = (theme.get('source_font') or MONO_FONT)

        # Heading styles (Word-like scale relative to the body size)
        heading_specs = (
            ("h1", 2.0,  24, 10, h1_fg, "bold"),
            ("h2", 1.6,  18, 8,  h2_fg, "bold"),
            ("h3", 1.3,  14, 6,  h3_fg, "bold"),
            ("h4", 1.15, 10, 4,  h4_fg, "bold"),
            ("h5", 1.0,  8,  3,  h5_fg, "bold"),
            ("h6", 1.0,  6,  3,  h6_fg, "bold italic"),
        )
        for name, mult, sp1, sp3, color, weight in heading_specs:
            tw.tag_configure(name, font=(font_family, self._px(font_size, mult), *weight.split()),
                             spacing1=self._s(sp1), spacing3=self._s(sp3), foreground=color)

        # Inline styles
        tw.tag_configure("bold", font=(font_family, self._px(font_size), "bold"))
        tw.tag_configure("italic", font=(font_family, self._px(font_size), "italic"))
        hl_bg = "#713f12" if is_dark else "#fef08a"
        hl_fg = "#fef9c3" if is_dark else "#1a1a1a"
        tw.tag_configure("highlight", background=hl_bg, foreground=hl_fg)
        tw.tag_configure("code", font=(mono, self._px(font_size, 0.9)), background=inline_code_bg,
                         foreground=inline_code_fg, relief=tk.FLAT)
        tw.tag_configure("code_block", font=(mono, self._px(font_size, 0.9)), background=code_bg, foreground=code_fg,
                         spacing1=self._s(3), spacing3=self._s(3), lmargin1=self._s(16), lmargin2=self._s(16),
                         rmargin=self._s(16))
        
        # Block styles (customizable single blockquote)
        bq_font = theme.get('blockquote_font') or font_family
        bq_size_val = theme.get('blockquote_size')
        try:
            bq_size = int(bq_size_val) if bq_size_val and int(bq_size_val) > 0 else font_size
        except (ValueError, TypeError):
            bq_size = font_size
        bq_style = theme.get('blockquote_style', 'italic')
        bq_style_parts = tuple(bq_style.split()) if bq_style and bq_style != 'normal' else ()
        bq_fg = theme.get('blockquote_fg') or quote_fg
        bq_bg = theme.get('blockquote_bg') or quote_bg
        bq_bar_fg = theme.get('blockquote_bar_fg') or link_fg
        try:
            bq_pad = int(theme.get('blockquote_padding', 16))
        except (ValueError, TypeError):
            bq_pad = 16
        try:
            bq_space = int(theme.get('blockquote_spacing', 4))
        except (ValueError, TypeError):
            bq_space = 4
        bq_justify = theme.get('blockquote_justify', 'left')
        if bq_justify not in ('left', 'center', 'right'):
            bq_justify = 'left'

        tw.tag_configure(
            "blockquote",
            font=(bq_font, self._px(bq_size), *bq_style_parts),
            foreground=bq_fg,
            background=bq_bg,
            lmargin1=self._s(bq_pad),
            lmargin2=self._s(bq_pad + 16),
            rmargin=self._s(bq_pad),
            spacing1=self._s(bq_space),
            spacing3=self._s(bq_space),
            justify=bq_justify,
        )
        tw.tag_configure("quote_bar", foreground=bq_bar_fg, font=(bq_font, self._px(bq_size)))
        tw.tag_configure("list_item", spacing1=self._s(1), spacing3=self._s(3))
        for depth in range(4):
            first = self._s(16 + 22 * depth)
            tw.tag_configure(f"list_d{depth}", lmargin1=first, lmargin2=first + self._s(16))
        tw.tag_configure("link", foreground=link_fg, underline=True)
        # Anchor badge styling
        anc_bg = "#2e1065" if is_dark else "#ede9fe"
        anc_fg = "#c4b5fd" if is_dark else "#6d28d9"
        tw.tag_configure("anchor_badge", font=(font_family, self._px(font_size, 0.85), "bold"),
                         foreground=anc_fg, background=anc_bg)
        # Footnote and Math styles
        fn_bg = "#1e3a8a" if is_dark else "#dbeafe"
        fn_fg = "#93c5fd" if is_dark else "#1d4ed8"
        tw.tag_configure("footnote_ref", font=(font_family, self._px(font_size, 0.78), "bold"),
                         offset=self._s(4), foreground=fn_fg, background=fn_bg)
        tw.tag_configure("footnote_def", font=(font_family, self._px(font_size, 0.9)),
                         foreground=quote_fg, lmargin1=self._s(16), lmargin2=self._s(32),
                         spacing1=self._s(2), spacing3=self._s(2))
        tw.tag_configure("footnote_badge", font=(font_family, self._px(font_size, 0.9), "bold"),
                         foreground=link_fg)
        math_font = "Cambria Math" if os.name == "nt" else font_family
        math_fg = "#5eead4" if is_dark else "#0f766e"
        tw.tag_configure("math_inline", font=(math_font, self._px(font_size, 1.05), "italic"),
                         foreground=math_fg, background=inline_code_bg)
        tw.tag_configure("math_block", font=(math_font, self._px(font_size, 1.15)),
                         foreground=math_fg, background=quote_bg, justify="center",
                         lmargin1=self._s(24), rmargin=self._s(24),
                         spacing1=self._s(8), spacing3=self._s(8))
        flash_bg = "#854d0e" if is_dark else "#fef08a"
        flash_fg = "#fef9c3" if is_dark else "#1a1a1a"
        tw.tag_configure("flash_target", background=flash_bg, foreground=flash_fg)
        tw.tag_configure("hr", font=(font_family, -2), foreground=hr_fg, spacing1=0, spacing3=self._s(4))

        # Image alignment and caption styles
        tw.tag_configure("image_caption", font=(font_family, self._px(font_size, 0.85), "italic"), foreground=cap_fg,
                         spacing1=self._s(4), spacing3=self._s(10), justify="center")
        tw.tag_configure("image_center", justify="center")
        tw.tag_configure("image_left", justify="left")
        tw.tag_configure("image_right", justify="right")
        tw.tag_configure("image_placeholder", font=(font_family, self._px(font_size)), foreground=cap_fg,
                         background=quote_bg, lmargin1=self._s(24), spacing1=self._s(6), spacing3=self._s(6))

        # Paragraph spacing (reading rhythm: ~1.2 line height, 8px between paragraphs)
        tw.tag_configure("paragraph", spacing1=0, spacing2=max(1, int(bpx * 0.2)), spacing3=self._s(8))

    def get_content(self):
        """Get the current content"""
        return self.content
    
    def apply_theme(self, theme):
        """Apply theme settings to the visual widget while preserving customization"""
        self.current_theme = theme
        bg = theme.get('visual_bg', '#ffffff')
        fg = theme.get('visual_fg', '#1a1a1a')
        font_family = theme.get('visual_font') or VISUAL_FONT or 'Segoe UI'
        font_size = theme.get('visual_font_size', 11)

        self.text_widget.config(
            bg=bg,
            fg=fg,
            font=(font_family, self._px(font_size)),
            insertbackground=fg,
        )
        self.page_view.set_colors(theme.get('desk_bg', '#d9dce1'), bg)
        self._update_tags_for_theme(theme)
        self._schedule_rerender()

    def _register_heading(self, text: str):
        """Register a heading as an internal jump mark and cyclic occurrence."""
        clean = text.strip()
        key = _normalize_anchor_key(clean)
        if key:
            mark_name = f"head_{len(self._heading_marks)}"
            self.text_widget.mark_set(mark_name, "end-1c")
            self.text_widget.mark_gravity(mark_name, tk.LEFT)
            self._heading_marks[key] = mark_name
            occ_list = self._anchor_occurrence_marks.setdefault(key, [])
            occ_mark = f"occ_h_{len(self._heading_marks)}_{len(occ_list)}"
            self.text_widget.mark_set(occ_mark, "end-1c")
            self.text_widget.mark_gravity(occ_mark, tk.LEFT)
            occ_list.append(occ_mark)

    def _cycle_anchor_occurrence(self, key: str, raw_label: str = "", from_mark: str = "", reverse: bool = False):
        """Cycle forward or backward through all visual occurrences of a reference/anchor."""
        tw = self.text_widget
        marks = self._anchor_occurrence_marks.get(key, [])
        if len(marks) > 1 and from_mark in marks:
            try:
                curr_idx = tw.index("@0,0")
                if not self._nav_history or self._nav_history[-1] != curr_idx:
                    self._nav_history.append(curr_idx)
                    if len(self._nav_history) > 50:
                        self._nav_history.pop(0)
            except Exception:
                pass
            idx = marks.index(from_mark)
            next_idx = (idx - 1) % len(marks) if reverse else (idx + 1) % len(marks)
            target_mark = marks[next_idx]
            if target_mark in tw.mark_names():
                tw.see(target_mark)
                self._flash_target(target_mark)
                notepad = self._get_notepad()
                if notepad and hasattr(notepad, '_set_status'):
                    disp = raw_label or key
                    notepad._set_status(
                        f"[@{disp}] — occurrence {next_idx + 1} of {len(marks)}  (Click: next | Shift+Click: prev | Alt+Left: return)"
                    )
                return
        elif len(marks) == 1 and from_mark in marks:
            self._flash_target(from_mark)
            notepad = self._get_notepad()
            if notepad and hasattr(notepad, '_set_status'):
                disp = raw_label or key
                notepad._set_status(f"[@{disp}] — only 1 occurrence in document")
            return
        self._jump_to_anchor(key, raw_label)

    def _cycle_footnote_occurrence(self, fn_id: str, from_mark: str = "", reverse: bool = False):
        """Cycle forward or backward through all footnote citations and definition in visual mode."""
        self._hide_footnote_tooltip()
        tw = self.text_widget
        marks = self._footnote_occurrence_marks.get(fn_id.lower(), [])
        if len(marks) > 1 and from_mark in marks:
            try:
                curr_idx = tw.index("@0,0")
                if not self._nav_history or self._nav_history[-1] != curr_idx:
                    self._nav_history.append(curr_idx)
                    if len(self._nav_history) > 50:
                        self._nav_history.pop(0)
            except Exception:
                pass
            idx = marks.index(from_mark)
            next_idx = (idx - 1) % len(marks) if reverse else (idx + 1) % len(marks)
            target_mark = marks[next_idx]
            if target_mark in tw.mark_names():
                tw.see(target_mark)
                self._flash_target(target_mark)
                notepad = self._get_notepad()
                if notepad and hasattr(notepad, '_set_status'):
                    notepad._set_status(
                        f"Footnote [^{fn_id}] — occurrence {next_idx + 1} of {len(marks)}  (Click: next | Shift+Click: prev | Alt+Left: return)"
                    )
                return
        self._jump_to_footnote_def(fn_id)

    def _jump_to_anchor(self, key: str, label: str = ""):
        """Jump smoothly to an anchor or heading target mark."""
        tw = self.text_widget
        target_mark = None
        if key in self._anchor_marks:
            target_mark = self._anchor_marks[key]
        elif key in self._heading_marks:
            target_mark = self._heading_marks[key]
        elif self._anchor_occurrence_marks.get(key):
            target_mark = self._anchor_occurrence_marks[key][0]
        else:
            # Fallback: check without non-alphanumeric chars
            slug = re.sub(r'[^a-z0-9]', '', key)
            for k, m in self._anchor_marks.items():
                if re.sub(r'[^a-z0-9]', '', k) == slug:
                    target_mark = m
                    break
            if not target_mark:
                for k, m in self._heading_marks.items():
                    if re.sub(r'[^a-z0-9]', '', k) == slug:
                        target_mark = m
                        break
        
        if target_mark and target_mark in tw.mark_names():
            try:
                curr_idx = tw.index("@0,0")
                if not self._nav_history or self._nav_history[-1] != curr_idx:
                    self._nav_history.append(curr_idx)
                    if len(self._nav_history) > 50:
                        self._nav_history.pop(0)
            except Exception:
                pass
            
            tw.see(target_mark)
            self._flash_target(target_mark)
            notepad = self._get_notepad()
            if notepad and hasattr(notepad, '_set_status'):
                notepad._set_status(f"Jumped to [@{label}]  (Alt+Left to return)")
        else:
            notepad = self._get_notepad()
            if notepad and hasattr(notepad, '_set_status'):
                notepad._set_status(f"Reference '[@{label}]' not found in document")

    def _show_footnote_tooltip(self, event, fn_id: str):
        """Display a hover tooltip previewing the footnote definition text."""
        self.text_widget.config(cursor="hand2")
        self._hide_footnote_tooltip()
        fn_body = self._footnotes.get(fn_id, f"(Footnote [^{fn_id}] not yet defined)")
        try:
            self._fn_tooltip = tk.Toplevel(self)
            self._fn_tooltip.wm_overrideredirect(True)
            self._fn_tooltip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 16}")
            is_dark = self._is_dark((self.current_theme or {}).get('visual_bg', '#ffffff'))
            bg = "#1e293b" if is_dark else "#0f172a"
            fg = "#f8fafc"
            lbl = tk.Label(
                self._fn_tooltip,
                text=f"[^{fn_id}]: {fn_body}",
                justify=tk.LEFT,
                wraplength=360,
                background=bg,
                foreground=fg,
                relief=tk.SOLID,
                borderwidth=1,
                font=(SANS_FONT, 9),
                padx=8,
                pady=5,
            )
            lbl.pack()
        except Exception:
            self._fn_tooltip = None

    def _hide_footnote_tooltip(self):
        """Hide any active footnote tooltip."""
        self.text_widget.config(cursor="")
        if self._fn_tooltip is not None:
            try:
                self._fn_tooltip.destroy()
            except Exception:
                pass
            self._fn_tooltip = None

    def _jump_to_footnote_def(self, fn_id: str):
        """Jump from an inline footnote badge [^id] to its definition."""
        self._hide_footnote_tooltip()
        tw = self.text_widget
        mark = self._footnote_def_marks.get(fn_id)
        if mark and mark in tw.mark_names():
            try:
                curr_idx = tw.index("@0,0")
                if not self._nav_history or self._nav_history[-1] != curr_idx:
                    self._nav_history.append(curr_idx)
            except Exception:
                pass
            tw.see(mark)
            self._flash_target(mark)
            notepad = self._get_notepad()
            if notepad and hasattr(notepad, '_set_status'):
                notepad._set_status(f"Jumped to footnote [^{fn_id}]  (Click ↩ or Alt+Left to return)")
        else:
            notepad = self._get_notepad()
            if notepad and hasattr(notepad, '_set_status'):
                notepad._set_status(f"Footnote definition '[^{fn_id}]:' not found")

    def _jump_to_footnote_ref(self, fn_id: str):
        """Jump from a footnote definition back to its inline citation."""
        tw = self.text_widget
        mark = self._footnote_ref_marks.get(fn_id)
        if mark and mark in tw.mark_names():
            tw.see(mark)
            self._flash_target(mark)
            notepad = self._get_notepad()
            if notepad and hasattr(notepad, '_set_status'):
                notepad._set_status(f"Returned to citation [^{fn_id}]")

    def _flash_target(self, mark_name: str):
        """Briefly highlight the target anchor/line for visual confirmation."""
        tw = self.text_widget
        try:
            line_start = tw.index(f"{mark_name} linestart")
            line_end = tw.index(f"{mark_name} lineend+1c")
            tw.tag_add("flash_target", line_start, line_end)
            tw.tag_raise("flash_target")
            self.after(1200, lambda: tw.tag_remove("flash_target", "1.0", tk.END))
        except tk.TclError:
            pass

    def _open_external_link(self, url: str):
        """Open an external web URL or mailto link."""
        if url.startswith(('http://', 'https://', 'mailto:')):
            try:
                webbrowser.open(url)
            except Exception:
                pass

    def _copy_anchor_ref(self, label: str):
        """Copy the reference shorthand [@label] to clipboard on anchor badge click."""
        try:
            ref_text = f"[@{label}]"
            self.clipboard_clear()
            self.clipboard_append(ref_text)
            notepad = self._get_notepad()
            if notepad and hasattr(notepad, '_set_status'):
                notepad._set_status(f"Copied '{ref_text}' to clipboard")
        except Exception:
            pass

    def navigate_back(self):
        """Navigate back to the previous jump location (Alt+Left)."""
        if self._nav_history:
            prev_idx = self._nav_history.pop()
            try:
                self.text_widget.see(prev_idx)
                notepad = self._get_notepad()
                if notepad and hasattr(notepad, '_set_status'):
                    notepad._set_status("Returned to previous location")
                return "break"
            except Exception:
                pass
        return None

    def _get_notepad(self):
        """Find the top-level MarkdownNotepad instance if available."""
        top = self.winfo_toplevel()
        return top if hasattr(top, '_set_status') else None


class DocumentTab(ttk.Frame):
    """
    Represents a single document tab with its own editor state.
    Encapsulates the source editor, visual viewer, and document state.
    Inherits from ttk.Frame so it can be added directly to a Notebook.
    """
    
    def __init__(self, parent, theme, on_modified_callback=None, on_content_change=None):
        super().__init__(parent)
        self.theme = theme
        self.on_modified = on_modified_callback
        self.on_content_change_callback = on_content_change
        
        # Document state
        self.tab_type = "document"
        self.file_path = None
        self.is_modified = False
        self.current_mode = "visual"
        self.word_wrap = True
        self.zoom = 1.0
        self.snapshots = []
        self._embedded_windows = []
        self._updating_from_widget = False
        self._last_active_line = None
        self._target_counts = {}
        self._nav_history = []
        
        desk_bg = theme.get('desk_bg', '#d9dce1')
        
        # Source mode editor (a Text on a Word-style page)
        self.source_frame = ttk.Frame(self)
        self.source_page = PageView(self.source_frame, desk_bg=desk_bg, page_bg=theme['source_bg'],
                                    max_margin_in=0.75)
        self.source_editor = tk.Text(
            self.source_page.sheet,
            wrap=tk.WORD,
            font=self._source_font(),
            undo=True,
            padx=15,
            pady=12,
            bg=theme.get('visual_bg', '#ffffff'),
            fg=theme.get('visual_fg', '#1a1a1a'),
            insertbackground=theme.get('visual_fg', '#1a1a1a'),
            insertwidth=2,
            selectbackground="#b3d9ff",
            selectforeground="#000000",
            inactiveselectbackground="#b3d9ff",
            spacing1=2,
            spacing2=3,
            spacing3=6,
        )
        self.source_page.attach_text(self.source_editor)
        self.source_page.pack(fill=tk.BOTH, expand=True)
        self._setup_syntax_tags()
        self.source_frame.pack(fill=tk.BOTH, expand=True)
        
        # Visual mode viewer (retained for backward compatibility and image handler)
        self.visual_frame = ttk.Frame(self)
        self.visual_viewer = MarkdownVisualWidget(
            self.visual_frame,
            base_path=os.getcwd(),
            on_content_change=self._on_visual_edit
        )
        self.visual_viewer.pack(fill=tk.BOTH, expand=True)
        self.visual_viewer.apply_theme(theme)
        self.apply_page_settings()
        
        # Setup modification tracking, Ctrl+Click navigation, autocomplete, active line elision, and table Tab navigation
        self._ac_job = None
        self._ac_popup = None
        self._ac_listbox = None
        self._ac_items = []
        self.source_editor.bind("<<Modified>>", self._on_text_modified)
        self.source_editor.bind("<Control-Button-1>", self._on_source_ctrl_click)
        self.source_editor.bind("<Control-Shift-Button-1>", self._on_source_ctrl_click)
        self.source_editor.bind("<KeyRelease>", self._on_autocomplete_key_release, add='+')
        self.source_editor.bind("<KeyRelease>", self._update_active_line_elision, add='+')
        self.source_editor.bind("<ButtonRelease-1>", self._update_active_line_elision, add='+')
        self.source_editor.bind("<Down>", self._on_ac_down)
        self.source_editor.bind("<Up>", self._on_ac_up)
        self.source_editor.bind("<Return>", self._on_ac_return)
        self.source_editor.bind("<Escape>", self._on_ac_escape)
        self.source_editor.bind("<Button-1>", lambda e: self._close_autocomplete(), add='+')
        self.source_editor.bind("<FocusOut>", lambda e: self.after(150, self._check_ac_focus), add='+')
        self.source_editor.bind("<Tab>", self._on_source_tab)
        self.source_editor.bind("<Shift-Tab>", self._on_source_shift_tab)
        try:
            self.source_editor.bind("<ISO_Left_Tab>", self._on_source_shift_tab)
        except tk.TclError:
            pass
        self._last_embedded_sig = None
        self._mini_toolbar = FloatingMiniToolbar(self)

    # --- page / zoom -----------------------------------------------------------

    def _source_font(self):
        """Source editor font as (family, -pixels) for the current zoom."""
        family = self.theme.get('source_font') or MONO_FONT
        size = self.theme.get('source_font_size', 11)
        px = max(4, round(size * self.source_page.dpi() / 72.0 * self.zoom))
        return (family, -px)

    def apply_page_settings(self):
        """Push page size / margin / layout preferences from the theme to both pages."""
        size = self.theme.get('page_size', 'Letter')
        margin = self.theme.get('page_margin', 'Normal')
        layout = bool(self.theme.get('page_layout', True))
        self.source_page.set_page(size=size, margin=margin, enabled=layout and self.word_wrap)
        self.visual_viewer.set_page(size=size, margin=margin, enabled=layout)

    def set_wrap(self, word_wrap):
        """Word wrap on/off for the source editor (off = full-width page + horizontal scrollbar)."""
        self.word_wrap = word_wrap
        self.source_editor.config(wrap=tk.WORD if word_wrap else tk.NONE)
        self.source_page.set_hscroll(not word_wrap)
        self.apply_page_settings()

    def set_zoom(self, zoom):
        """Apply a zoom factor (1.0 = 100%) to both the source page and the visual page."""
        self.zoom = zoom
        self.source_page.set_zoom(zoom)
        self.source_editor.config(font=self._source_font())
        self._setup_syntax_tags()
        self.visual_viewer.set_zoom(zoom)

    def apply_desk_color(self, desk_bg):
        self.source_page.set_colors(desk_bg, self.theme['source_bg'])
        self.visual_viewer.page_view.set_colors(desk_bg, self.theme.get('visual_bg', '#ffffff'))
    
    def _on_text_modified(self, event=None):
        """Handle text modification"""
        if getattr(self, '_updating_from_widget', False):
            self.source_editor.edit_modified(False)
            return
        if self.source_editor.edit_modified():
            self.is_modified = True
            self.source_editor.edit_modified(False)
            if self.on_modified:
                self.on_modified()
    
    def clear_embedded_widgets(self):
        """Remove any embedded visual table/figure widgets from source_editor without polluting undo stack."""
        if not getattr(self, '_embedded_windows', None):
            return
        prev_undo = self.source_editor.cget('undo')
        self.source_editor.config(undo=False)
        try:
            for win in list(self._embedded_windows):
                try:
                    idx = self.source_editor.index(str(win))
                    self.source_editor.delete(idx)
                except tk.TclError:
                    pass
                try:
                    win.destroy()
                except Exception:
                    pass
            self._embedded_windows.clear()
            self._last_embedded_sig = None
        finally:
            self.source_editor.config(undo=prev_undo)

    def _on_embedded_block_edit(self, old_text, new_text):
        """Handle cell edits, structural changes, or figure edits from embedded widgets."""
        self._updating_from_widget = True
        try:
            pos = self.source_editor.search(old_text, "1.0", tk.END)
            if pos:
                end_pos = f"{pos} + {len(old_text)}c"
                self.source_editor.delete(pos, end_pos)
                self.source_editor.insert(pos, new_text)
                new_end_pos = f"{pos} + {len(new_text)}c"
                self.source_editor.tag_add("syn_block_elide", pos, new_end_pos)
                self.is_modified = True
                if self.on_modified:
                    self.on_modified()
                if self.on_content_change_callback:
                    self.on_content_change_callback(self.get_content())
        finally:
            self._updating_from_widget = False

    def _render_embedded_blocks(self):
        """Embed interactive VisualTableWidget and VisualFigureWidget for tables and images in visual mode."""
        if self.current_mode != "visual" or getattr(self, '_updating_from_widget', False):
            return

        content = self.source_editor.get("1.0", tk.END)
        lines = content.split('\n')
        dark = _is_dark_color(self.theme.get('visual_bg', '#ffffff'))
        font_family = self.theme.get('visual_font') or VISUAL_FONT or SANS_FONT
        font_size = self.theme.get('visual_font_size', 11)
        content_w = getattr(self.source_page, 'content_width', 600)

        # Build candidate signature of blocks to render
        candidate_blocks = []
        in_tbl = False
        cur_tbl = []
        tbl_start = 0

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                if not in_tbl:
                    in_tbl = True
                    cur_tbl = [stripped]
                    tbl_start = lineno
                else:
                    cur_tbl.append(stripped)
            else:
                if in_tbl:
                    if any(re.match(r'^\|(\s*:?-+:?\s*\|)+$', tl) for tl in cur_tbl):
                        candidate_blocks.append(('table', tbl_start, lineno - 1, tuple(cur_tbl)))
                    in_tbl = False
                    cur_tbl = []

            m_img = re.match(r'^\s*(!\[([^\]]*)\]\(([^)]+)\))\s*$', line)
            if m_img and not in_tbl:
                candidate_blocks.append(('image', lineno, m_img.group(1), m_img.group(2), m_img.group(3)))

        if in_tbl and any(re.match(r'^\|(\s*:?-+:?\s*\|)+$', tl) for tl in cur_tbl):
            candidate_blocks.append(('table', tbl_start, len(lines), tuple(cur_tbl)))

        new_sig = (tuple(candidate_blocks), dark, font_family, font_size, content_w)
        if getattr(self, '_last_embedded_sig', None) == new_sig:
            return

        self._updating_from_widget = True
        was_modified = self.is_modified
        prev_undo = self.source_editor.cget('undo')
        self.source_editor.config(undo=False)
        try:
            self.clear_embedded_widgets()
            for blk in candidate_blocks:
                if blk[0] == 'table':
                    _, s_line, e_line, t_lines = blk
                    self.source_editor.tag_add("syn_block_elide", f"{s_line}.0", f"{e_line}.end")
                    tbl_widget = VisualTableWidget(
                        self.source_editor,
                        on_change_callback=self._on_embedded_block_edit,
                        table_lines=list(t_lines),
                        is_dark=dark,
                        visual_font=font_family,
                        visual_font_size=font_size,
                        content_width=content_w
                    )
                    self.source_editor.window_create(f"{s_line}.0", window=tbl_widget)
                    self._embedded_windows.append(tbl_widget)
                elif blk[0] == 'image':
                    _, lineno, raw_match, alt_meta, img_path = blk
                    img_info = self.visual_viewer.image_handler._parse_extended_alt(alt_meta)
                    img_info['path'] = img_path
                    img_info['type'] = 'file'
                    img_info['full_match'] = raw_match

                    self.source_editor.tag_add("syn_block_elide", f"{lineno}.0", f"{lineno}.end")
                    fig_widget = VisualFigureWidget(
                        self.source_editor,
                        on_change_callback=self._on_embedded_block_edit,
                        raw_match=raw_match,
                        img_info=img_info,
                        image_handler=self.visual_viewer.image_handler,
                        is_dark=dark,
                        visual_font=font_family,
                        visual_font_size=font_size,
                        content_width=content_w
                    )
                    self.source_editor.window_create(f"{lineno}.0", window=fig_widget)
                    self._embedded_windows.append(fig_widget)
            self._last_embedded_sig = new_sig
        finally:
            self.source_editor.config(undo=prev_undo)
            self.is_modified = was_modified
            self.source_editor.edit_modified(False)
            self._updating_from_widget = False

    def _update_active_line_elision(self, event=None):
        """Reveal syntax delimiters on the active cursor line and elide them on other lines in visual mode."""
        if self.current_mode != "visual":
            return
        try:
            curr_line = self.source_editor.index(tk.INSERT).split('.')[0]
            if getattr(self, '_last_active_line', None) != curr_line:
                old_line = getattr(self, '_last_active_line', None)
                self._last_active_line = curr_line
                if old_line:
                    line_start = f'{old_line}.0'
                    line_end = f'{old_line}.end'
                    line_text = self.source_editor.get(line_start, line_end)
                    h_m = re.match(r'^(#{1,6}\s+)', line_text)
                    if h_m:
                        self.source_editor.tag_add('syn_delimiter', line_start, f'{old_line}.{len(h_m.group(1))}')
                    bq_m = re.match(r'^(>\s?)', line_text)
                    if bq_m:
                        self.source_editor.tag_add('syn_delimiter', line_start, f'{old_line}.{len(bq_m.group(1))}')
                    for m in re.finditer(r'(\*\*\*|___)(.+?)\1', line_text):
                        p_len = len(m.group(1))
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.start()}', f'{old_line}.{m.start() + p_len}')
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.end() - p_len}', f'{old_line}.{m.end()}')
                    for m in re.finditer(r'(?<!\*)\*\*([^\*\n]+?)\*\*(?!\*)', line_text):
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.start()}', f'{old_line}.{m.start() + 2}')
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.end() - 2}', f'{old_line}.{m.end()}')
                    for m in re.finditer(r'(?<!\*)\*([^\*\n]+?)\*(?!\*)', line_text):
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.start()}', f'{old_line}.{m.start() + 1}')
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.end() - 1}', f'{old_line}.{m.end()}')
                    for m in re.finditer(r'==([^=\n]+)==', line_text):
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.start()}', f'{old_line}.{m.start() + 2}')
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.end() - 2}', f'{old_line}.{m.end()}')
                    for m in re.finditer(r'`([^`\n]+)`', line_text):
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.start()}', f'{old_line}.{m.start() + 1}')
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.end() - 1}', f'{old_line}.{m.end()}')
                    for m in re.finditer(r'(?<!!)(\[)([^\]\n]+)(\])\(([^\)\n]+)\)', line_text):
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.start(1)}', f'{old_line}.{m.end(1)}')
                        self.source_editor.tag_add('syn_delimiter', f'{old_line}.{m.start(3)}', f'{old_line}.{m.end(4)}')

                self.source_editor.tag_remove('syn_delimiter', f'{curr_line}.0', f'{curr_line}.end')
        except Exception:
            pass

    def switch_mode(self, mode, theme=None):
        """Switch between unified visual mode and raw source mode."""
        if mode == self.current_mode:
            return
        self.current_mode = mode
        if mode == "source":
            self.clear_embedded_widgets()
        if theme:
            self.apply_theme(theme)
        else:
            self.apply_theme(self.theme)

    def _on_visual_edit(self, old_or_new, new_text=None, rerender=False):
        """Handle visual editor changes: replaces old_text with new_text in canonical source."""
        if new_text is not None:
            old_text = old_or_new
            curr = self.source_editor.get("1.0", "end-1c")
            if old_text in curr:
                pos = self.source_editor.search(old_text, "1.0", tk.END)
                if pos:
                    end_pos = f"{pos} + {len(old_text)}c"
                    self.source_editor.delete(pos, end_pos)
                    self.source_editor.insert(pos, new_text)
                else:
                    updated = curr.replace(old_text, new_text, 1)
                    self.source_editor.delete("1.0", tk.END)
                    self.source_editor.insert("1.0", updated)
                self.is_modified = True
                if self.on_modified:
                    self.on_modified()
        else:
            self.set_content(old_or_new)
            self.is_modified = True
            if self.on_modified:
                self.on_modified()
        if self.on_content_change_callback:
            self.on_content_change_callback(self.get_content())
        if rerender:
            self._apply_syntax_highlighting()

    def get_display_name(self):
        """Get display title for tab"""
        if self.file_path:
            return os.path.basename(self.file_path)
        return "Untitled"

    def get_content(self):
        """Get canonical document content from source editor"""
        return self.source_editor.get("1.0", "end-1c").replace("\ufffc", "").rstrip()

    def set_content(self, content):
        """Set document content and refresh visual viewer if active"""
        self.source_editor.delete(1.0, tk.END)
        self.source_editor.insert(1.0, content)
        self.source_editor.edit_modified(False)
        self.is_modified = False
        if self.current_mode == "visual":
            self._apply_syntax_highlighting()
        self.visual_viewer.set_content(content)

    def apply_theme(self, theme):
        """Apply theme to this tab"""
        self.theme = theme
        is_visual = (self.current_mode == "visual")
        if is_visual:
            v_font = theme.get('visual_font') or VISUAL_FONT or SANS_FONT or 'Segoe UI'
            v_sz = theme.get('visual_font_size', 11)
            px = max(4, round(v_sz * self.source_page.dpi() / 72.0 * self.zoom))
            self.source_editor.config(
                bg=theme.get('visual_bg', '#ffffff'),
                fg=theme.get('visual_fg', '#1a1a1a'),
                insertbackground=theme.get('visual_fg', '#1a1a1a'),
                font=(v_font, -px),
                spacing1=2,
                spacing2=3,
                spacing3=6,
            )
            self.source_page.set_colors(theme.get('desk_bg', '#d9dce1'), theme.get('visual_bg', '#ffffff'))
        else:
            self.source_editor.config(
                bg=theme['source_bg'],
                fg=theme['source_fg'],
                insertbackground=theme['source_fg'],
                font=self._source_font(),
                spacing1=1,
                spacing2=2,
                spacing3=2,
            )
            self.source_page.set_colors(theme.get('desk_bg', '#d9dce1'), theme['source_bg'])

        self._setup_syntax_tags()
        self._apply_syntax_highlighting()
        self.visual_viewer.apply_theme(theme)
        self.apply_page_settings()

    def _setup_syntax_tags(self):
        """Configure colour/font tags for visual or source mode syntax highlighting (zoom aware)."""
        is_visual = (self.current_mode == "visual")
        editor = self.source_editor

        if is_visual:
            sz = self.theme.get('visual_font_size', 11)
            font_family = self.theme.get('visual_font') or VISUAL_FONT or SANS_FONT or 'Segoe UI'
            bg_col = self.theme.get('visual_bg', '#ffffff')
            fg_col = self.theme.get('visual_fg', '#1a1a1a')
        else:
            sz = self.theme.get('source_font_size', 11)
            font_family = self.theme.get('source_font') or MONO_FONT
            bg_col = self.theme.get('source_bg', '#ffffff')
            fg_col = self.theme.get('source_fg', '#1a1a1a')

        dpi_k = self.source_page.dpi() / 72.0 * self.zoom
        def px(pt):
            return -max(4, round(pt * dpi_k))

        k96 = self.source_page.dpi() / 96.0 * self.zoom
        dark = _is_dark_color(bg_col)

        if dark:
            heads = ('#f8fafc', '#e2e8f0', '#cbd5e1', '#cbd5e1')
            bold_fg, italic_fg, link_fg, quote_fg, hr_fg = '#ff7b72', '#d2a8ff', '#79c0ff', '#9ca3af', '#6b7280'
            code_bg, code_fg = '#2d2f3a', '#f38ba8'
            hl_bg, hl_fg = '#713f12', '#fef9c3'
        else:
            heads = ('#2e1065', '#3b0764', '#4c1d95', '#5b21b6')
            bold_fg, italic_fg, link_fg, quote_fg, hr_fg = '#d73a49', '#6f42c1', '#6d28d9', '#6c757d', '#aaaaaa'
            code_bg, code_fg = '#f0f0f0', '#c7254e'
            hl_bg, hl_fg = '#fef08a', '#1a1a1a'

        # Delimiters and block elision tags
        editor.tag_configure('syn_delimiter', elide=is_visual, foreground='#94a3b8' if dark else '#94a3b8')
        editor.tag_configure('syn_block_elide', elide=is_visual)

        # Headers
        h_font = font_family if is_visual else (self.theme.get('source_font') or MONO_FONT)
        editor.tag_configure('syn_h1', font=(h_font, px(sz + (10 if is_visual else 8)), 'bold'),
                             foreground=heads[0], spacing1=8 if is_visual else 1, spacing3=4 if is_visual else 1)
        editor.tag_configure('syn_h2', font=(h_font, px(sz + (7 if is_visual else 5)), 'bold'),
                             foreground=heads[1], spacing1=6 if is_visual else 1, spacing3=3 if is_visual else 1)
        editor.tag_configure('syn_h3', font=(h_font, px(sz + (4 if is_visual else 3)), 'bold'),
                             foreground=heads[2], spacing1=4 if is_visual else 1, spacing3=2 if is_visual else 1)
        editor.tag_configure('syn_h4', font=(h_font, px(sz + (2 if is_visual else 1)), 'bold'),
                             foreground=heads[3], spacing1=3 if is_visual else 1, spacing3=2 if is_visual else 1)

        # Inline styles
        b_font = font_family
        editor.tag_configure('syn_bold', font=(b_font, px(sz), 'bold'), foreground=bold_fg)
        editor.tag_configure('syn_italic', font=(b_font, px(sz), 'italic'), foreground=italic_fg)
        editor.tag_configure('syn_bold_italic', font=(b_font, px(sz), 'bold italic'), foreground=bold_fg)
        editor.tag_configure('syn_highlight', background=hl_bg, foreground=hl_fg)

        mono_font = self.theme.get('source_font') or MONO_FONT
        editor.tag_configure('syn_code_inline', font=(mono_font, px(sz * 0.95)), background=code_bg, foreground=code_fg)
        editor.tag_configure('syn_code_block', font=(mono_font, px(sz * 0.95)), background='#282c34', foreground='#abb2bf',
                             lmargin1=int(20 * k96), lmargin2=int(20 * k96), rmargin=int(20 * k96),
                             spacing1=4, spacing3=4)
        editor.tag_configure('syn_link', foreground=link_fg, underline=True)
        anc_fg = '#c084fc' if dark else '#7c3aed'
        editor.tag_configure('syn_anchor', foreground=anc_fg, font=(b_font, px(sz), 'bold'), underline=True)
        editor.tag_configure('syn_footnote', foreground='#38bdf8' if dark else '#0284c7', font=(b_font, px(sz), 'bold'))
        math_font = "Cambria Math" if os.name == "nt" else font_family
        editor.tag_configure('syn_math', foreground='#5eead4' if dark else '#0f766e', background=code_bg, font=(math_font, px(sz), 'italic'))
        editor.tag_configure('syn_flash', background=hl_bg, foreground=hl_fg)

        bq_font = self.theme.get('blockquote_font') or b_font
        bq_size_val = self.theme.get('blockquote_size')
        try:
            bq_sz = int(bq_size_val) if bq_size_val and int(bq_size_val) > 0 else sz
        except (ValueError, TypeError):
            bq_sz = sz
        bq_style = self.theme.get('blockquote_style', 'italic')
        bq_style_parts = tuple(bq_style.split()) if bq_style and bq_style != 'normal' else ()
        bq_fg = self.theme.get('blockquote_fg') or quote_fg
        bq_bg = self.theme.get('blockquote_bg') or ''
        try:
            bq_pad = int(self.theme.get('blockquote_padding', 16))
        except (ValueError, TypeError):
            bq_pad = 16
        bq_justify = self.theme.get('blockquote_justify', 'left')
        if bq_justify not in ('left', 'center', 'right'):
            bq_justify = 'left'
        bq_kwargs = {
            'font': (bq_font, px(bq_sz), *bq_style_parts),
            'foreground': bq_fg,
            'lmargin1': int(max(12, bq_pad) * k96),
            'lmargin2': int(max(20, bq_pad + 12) * k96),
            'rmargin': int(max(12, bq_pad) * k96),
            'justify': bq_justify,
            'spacing1': 4 if is_visual else 1,
            'spacing3': 4 if is_visual else 1,
        }
        if bq_bg:
            bq_kwargs['background'] = bq_bg
        editor.tag_configure('syn_blockquote', **bq_kwargs)
        editor.tag_configure('syn_list_marker', font=(b_font, px(sz), 'bold'), foreground=link_fg)
        editor.tag_configure('syn_hr', foreground=hr_fg)

    def _apply_syntax_highlighting(self):
        """Apply markdown syntax highlighting and visual elision/widgets to the source editor."""
        editor = self.source_editor
        for tag in ('syn_h1', 'syn_h2', 'syn_h3', 'syn_h4', 'syn_bold', 'syn_italic',
                    'syn_bold_italic', 'syn_highlight', 'syn_code_inline', 'syn_code_block',
                    'syn_link', 'syn_anchor', 'syn_footnote', 'syn_math', 'syn_flash',
                    'syn_blockquote', 'syn_list_marker', 'syn_hr', 'syn_delimiter', 'syn_block_elide'):
            editor.tag_remove(tag, '1.0', tk.END)

        content = editor.get('1.0', tk.END)
        lines = content.split('\n')
        in_code_block = False

        for lineno, line in enumerate(lines, start=1):
            line_start = f'{lineno}.0'
            line_end   = f'{lineno}.end'

            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                editor.tag_add('syn_code_block', line_start, line_end)
                continue

            if in_code_block:
                editor.tag_add('syn_code_block', line_start, line_end)
                continue

            h_match = re.match(r'^(#{1,6}\s+)(.*)$', line)
            if h_match:
                pfx_len = len(h_match.group(1))
                h_lvl = min(len(h_match.group(1).strip()), 4)
                editor.tag_add('syn_delimiter', line_start, f'{lineno}.{pfx_len}')
                editor.tag_add(f'syn_h{h_lvl}', f'{lineno}.{pfx_len}', line_end)
            elif line.strip() in ('---', '***', '___'):
                editor.tag_add('syn_hr', line_start, line_end)
            elif line.startswith('>'):
                bq_match = re.match(r'^(>\s?)(.*)$', line)
                if bq_match:
                    pfx_len = len(bq_match.group(1))
                    editor.tag_add('syn_delimiter', line_start, f'{lineno}.{pfx_len}')
                editor.tag_add('syn_blockquote', line_start, line_end)
            else:
                m = re.match(r'^(\s*)([-*+]|\d+\.)\s', line)
                if m:
                    editor.tag_add('syn_list_marker', line_start, f'{lineno}.{m.end()}')

            for m in re.finditer(r'(\*\*\*|___)(.+?)\1', line):
                p_len = len(m.group(1))
                editor.tag_add('syn_delimiter', f'{lineno}.{m.start()}', f'{lineno}.{m.start() + p_len}')
                editor.tag_add('syn_bold_italic', f'{lineno}.{m.start() + p_len}', f'{lineno}.{m.end() - p_len}')
                editor.tag_add('syn_delimiter', f'{lineno}.{m.end() - p_len}', f'{lineno}.{m.end()}')
            for m in re.finditer(r'(?<!\*)\*\*([^\*\n]+?)\*\*(?!\*)', line):
                editor.tag_add('syn_delimiter', f'{lineno}.{m.start()}', f'{lineno}.{m.start() + 2}')
                editor.tag_add('syn_bold', f'{lineno}.{m.start() + 2}', f'{lineno}.{m.end() - 2}')
                editor.tag_add('syn_delimiter', f'{lineno}.{m.end() - 2}', f'{lineno}.{m.end()}')
            for m in re.finditer(r'(?<!\*)\*([^\*\n]+?)\*(?!\*)', line):
                editor.tag_add('syn_delimiter', f'{lineno}.{m.start()}', f'{lineno}.{m.start() + 1}')
                editor.tag_add('syn_italic', f'{lineno}.{m.start() + 1}', f'{lineno}.{m.end() - 1}')
                editor.tag_add('syn_delimiter', f'{lineno}.{m.end() - 1}', f'{lineno}.{m.end()}')
            for m in re.finditer(r'==([^=\n]+)==', line):
                editor.tag_add('syn_delimiter', f'{lineno}.{m.start()}', f'{lineno}.{m.start() + 2}')
                editor.tag_add('syn_highlight', f'{lineno}.{m.start() + 2}', f'{lineno}.{m.end() - 2}')
                editor.tag_add('syn_delimiter', f'{lineno}.{m.end() - 2}', f'{lineno}.{m.end()}')
            for m in re.finditer(r'`([^`\n]+)`', line):
                editor.tag_add('syn_delimiter', f'{lineno}.{m.start()}', f'{lineno}.{m.start() + 1}')
                editor.tag_add('syn_code_inline', f'{lineno}.{m.start() + 1}', f'{lineno}.{m.end() - 1}')
                editor.tag_add('syn_delimiter', f'{lineno}.{m.end() - 1}', f'{lineno}.{m.end()}')
            for m in re.finditer(r'\$\$(.+?)\$\$|(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)', line):
                editor.tag_add('syn_math', f'{lineno}.{m.start()}', f'{lineno}.{m.end()}')
            for m in re.finditer(r'\[\^([^\]\n]+)\]', line):
                editor.tag_add('syn_footnote', f'{lineno}.{m.start()}', f'{lineno}.{m.end()}')
            for m in re.finditer(r'\[@(?:anchor|def):\s*([^\]\n]+)\]', line):
                editor.tag_add('syn_anchor', f'{lineno}.{m.start()}', f'{lineno}.{m.end()}')
            for m in re.finditer(r'\[@(?!(?:anchor|def):)([^\]\n]+)\]', line):
                editor.tag_add('syn_link', f'{lineno}.{m.start()}', f'{lineno}.{m.end()}')
            for m in re.finditer(r'(?<!!)(\[)([^\]\n]+)(\])\(([^\)\n]+)\)', line):
                editor.tag_add('syn_delimiter', f'{lineno}.{m.start(1)}', f'{lineno}.{m.end(1)}')
                editor.tag_add('syn_link', f'{lineno}.{m.start(2)}', f'{lineno}.{m.end(2)}')
                editor.tag_add('syn_delimiter', f'{lineno}.{m.start(3)}', f'{lineno}.{m.end(4)}')

        if self.current_mode == "visual":
            try:
                curr_line = editor.index(tk.INSERT).split('.')[0]
                editor.tag_remove('syn_delimiter', f'{curr_line}.0', f'{curr_line}.end')
            except Exception:
                pass
            self._render_embedded_blocks()

    def _find_all_ref_occurrences(self, content: str, ref_norm: str) -> list:
        """Find all unified cross-reference, anchor, internal link, and heading occurrences for *ref_norm* in document order.

        Returns a list of tuples: (lineno, col_start, col_end, display_label, kind)
        where *lineno* is 1-based and *col_start* / *col_end* are 0-based line column offsets.
        """
        if not ref_norm:
            return []
        occurrences = []
        in_code = False
        tag_pat = re.compile(r'\[@(?:(?:anchor|def):\s*)?([^\]\n]+)\]', re.IGNORECASE)
        link_pat = re.compile(r'(?<!!)\[([^\]]+)\]\(([@#][^)\s]+)\)')
        head_pat = re.compile(r'^(#{1,6})\s+(.+)$')

        for lineno, line in enumerate(content.split('\n'), start=1):
            if line.strip().startswith('```'):
                in_code = not in_code
                continue
            if in_code:
                continue

            line_hits = []
            # 1. Unified [@label] and legacy [@anchor: label] / [@def: label]
            for m in tag_pat.finditer(line):
                raw_lbl = m.group(1).strip()
                if _normalize_anchor_key(raw_lbl) == ref_norm:
                    line_hits.append((lineno, m.start(), m.end(), raw_lbl, 'tag'))

            # 2. Markdown internal links [text](#label) or [text](@label)
            for m in link_pat.finditer(line):
                raw_lbl = m.group(2).lstrip('#@').strip()
                if _normalize_anchor_key(raw_lbl) == ref_norm:
                    # Avoid double-counting if span overlaps a tag hit
                    if not any(s <= m.start() < e or m.start() <= s < m.end() for _, s, e, _, _ in line_hits):
                        line_hits.append((lineno, m.start(), m.end(), raw_lbl, 'link'))

            # 3. Markdown headings (# Heading)
            hm = head_pat.match(line.strip())
            if hm and not line_hits:
                raw_lbl = hm.group(2).strip()
                if _normalize_anchor_key(raw_lbl) == ref_norm:
                    h_pos = line.find('#')
                    line_hits.append((lineno, max(0, h_pos), len(line), raw_lbl, 'heading'))

            line_hits.sort(key=lambda item: item[1])
            occurrences.extend(line_hits)

        return occurrences

    def _find_all_footnote_occurrences(self, content: str, fn_norm: str) -> list:
        """Find all footnote definitions [^id]: and inline citations [^id] for *fn_norm* in document order."""
        if not fn_norm:
            return []
        occurrences = []
        in_code = False
        fn_pat = re.compile(r'\[\^([^\]\n]+)\](:)?')

        for lineno, line in enumerate(content.split('\n'), start=1):
            if line.strip().startswith('```'):
                in_code = not in_code
                continue
            if in_code:
                continue

            line_hits = []
            for m in fn_pat.finditer(line):
                fid = m.group(1).strip()
                if fid.lower() == fn_norm:
                    is_def = bool(m.group(2)) and line.lstrip().startswith('[^')
                    line_hits.append((lineno, m.start(), m.end(), fid, 'def' if is_def else 'ref'))
            occurrences.extend(line_hits)

        return occurrences

    def _flash_source_line(self, idx_str: str):
        """Flash the line containing *idx_str* in source_editor."""
        editor = self.source_editor
        line_start = editor.index(f"{idx_str} linestart")
        line_end = editor.index(f"{idx_str} lineend+1c")
        editor.tag_remove('syn_flash', '1.0', tk.END)
        editor.tag_add('syn_flash', line_start, line_end)
        self.after(900, lambda: editor.tag_remove('syn_flash', '1.0', tk.END))

    def _jump_to_source_occurrence(self, occurrences: list, cur_line: int, cur_col: int, reverse: bool, label: str, is_footnote: bool = False):
        """Cycle to the next (or previous if *reverse*) occurrence and update status."""
        editor = self.source_editor
        if not occurrences:
            return None

        # Locate the occurrence currently under/nearest the click
        cur_i = 0
        exact_hit = None
        for idx, (lineno, col_s, col_e, _, _) in enumerate(occurrences):
            if lineno == cur_line and col_s <= cur_col <= col_e:
                exact_hit = idx
                break
        if exact_hit is not None:
            cur_i = exact_hit
        else:
            cur_i = min(
                range(len(occurrences)),
                key=lambda i: (abs(occurrences[i][0] - cur_line), abs(occurrences[i][1] - cur_col))
            )

        prefix = f"Footnote [^{label}]" if is_footnote else f"[@{label}]"
        top = self.winfo_toplevel()

        if len(occurrences) == 1:
            only_line, only_col, _, _, _ = occurrences[0]
            only_idx = f"{only_line}.{only_col}"
            self._flash_source_line(only_idx)
            if hasattr(top, '_set_status'):
                hint = "" if is_footnote else f" Add another [@{label}] to cycle."
                top._set_status(f"{prefix} — only 1 occurrence in document (Line {only_line}).{hint}")
            return "break"

        # Record pre-jump cursor position for Alt+Left back navigation
        try:
            curr_ins = editor.index(tk.INSERT)
            if not self._nav_history or self._nav_history[-1] != curr_ins:
                self._nav_history.append(curr_ins)
                if len(self._nav_history) > 50:
                    self._nav_history.pop(0)
        except Exception:
            pass

        next_i = (cur_i - 1) % len(occurrences) if reverse else (cur_i + 1) % len(occurrences)
        target_line, target_col, _, _, _ = occurrences[next_i]
        found_idx = f"{target_line}.{target_col}"

        # Guard against rare column shift on lines containing an embedded widget
        try:
            preview_ch = editor.get(found_idx, f"{found_idx}+2c")
            if preview_ch and preview_ch[0] not in ('[', '#'):
                searched = editor.search(r'\[@|\[\^|\[|#', f"{target_line}.0", f"{target_line}.end", regexp=True)
                if searched:
                    found_idx = searched
        except tk.TclError:
            pass

        editor.mark_set(tk.INSERT, found_idx)
        editor.see(found_idx)
        editor.focus_set()
        self._flash_source_line(found_idx)

        if hasattr(top, '_set_status'):
            top._set_status(
                f"{prefix} — occurrence {next_i + 1} of {len(occurrences)} (Line {target_line})  "
                f"[Ctrl+Click: next | Ctrl+Shift+Click: prev]"
            )
        return "break"

    def _on_source_ctrl_click(self, event):
        """Handle Ctrl+Click (and Ctrl+Shift+Click) to cycle through all occurrences of a ref, anchor, heading, or footnote."""
        editor = self.source_editor
        click_idx = editor.index(f"@{event.x},{event.y}")
        try:
            line_num, col_num = map(int, click_idx.split('.'))
        except (ValueError, TypeError):
            return None
        line = editor.get(f"{line_num}.0", f"{line_num}.end")
        reverse = bool(getattr(event, 'state', 0) & 0x0001)
        content = editor.get("1.0", "end-1c")

        # Case A: Footnote [^id] or [^id]:
        for m in re.finditer(r'\[\^([^\]\n]+)\]:?', line):
            if m.start() <= col_num <= m.end():
                fn_id = m.group(1).strip()
                occurrences = self._find_all_footnote_occurrences(content, fn_id.lower())
                if occurrences:
                    return self._jump_to_source_occurrence(occurrences, line_num, col_num, reverse, fn_id, is_footnote=True)

        # Case B: Unified [@label], legacy [@anchor: label] / [@def: label], internal links, or Headings
        target_ref = None
        for m in re.finditer(r'\[@(?:(?:anchor|def):\s*)?([^\]\n]+)\]', line, re.IGNORECASE):
            if m.start() <= col_num <= m.end():
                target_ref = m.group(1).strip()
                break

        if not target_ref:
            for m in re.finditer(r'(?<!!)\[([^\]]+)\]\(([^)\s]+)\)', line):
                if m.start() <= col_num <= m.end():
                    u = m.group(2).strip()
                    if u.startswith(('#', '@')):
                        target_ref = u.lstrip('#@').strip()
                    elif u.startswith(('http://', 'https://', 'mailto:')):
                        try:
                            webbrowser.open(u)
                        except Exception:
                            pass
                        return "break"
                    break

        if not target_ref:
            hm = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if hm:
                cand = hm.group(2).strip()
                if len(self._find_all_ref_occurrences(content, _normalize_anchor_key(cand))) >= 2:
                    target_ref = cand

        if target_ref:
            ref_norm = _normalize_anchor_key(target_ref)
            occurrences = self._find_all_ref_occurrences(content, ref_norm)
            if occurrences:
                return self._jump_to_source_occurrence(occurrences, line_num, col_num, reverse, target_ref, is_footnote=False)
            top = self.winfo_toplevel()
            if hasattr(top, '_set_status'):
                top._set_status(f"Reference '[@{target_ref}]' not found in document")
        return None

    def center_cursor_vertically(self):
        """Keep the active cursor line near the vertical center (~45%) of the source editor viewport."""
        editor = self.source_editor
        try:
            bbox = editor.bbox(tk.INSERT)
            if not bbox:
                editor.see(tk.INSERT)
                return
            _, y, _, line_h = bbox
            vh = editor.winfo_height()
            if vh < 80:
                return
            target_y = int(vh * 0.45)
            diff_px = (y + line_h // 2) - target_y
            threshold = max(20, int(line_h * 1.4))
            if abs(diff_px) > threshold:
                lines_to_scroll = int(round(diff_px / max(12, line_h)))
                if lines_to_scroll != 0:
                    editor.yview_scroll(lines_to_scroll, "units")
        except Exception:
            pass

    # --- Cross-Reference Autocomplete ([@ trigger with 1s delay) ---------------

    def _on_autocomplete_key_release(self, event=None):
        """Schedule or update cross-reference autocomplete popup after 1s delay when typing [@..."""
        if event and event.keysym in (
            'Up', 'Down', 'Return', 'Escape', 'Tab', 'ISO_Left_Tab',
            'Shift_L', 'Shift_R', 'Control_L', 'Control_R', 'Alt_L', 'Alt_R',
            'Left', 'Right', 'Home', 'End', 'Prior', 'Next'
        ):
            return

        if self._ac_job:
            try:
                self.after_cancel(self._ac_job)
            except Exception:
                pass
            self._ac_job = None

        line_prefix = self.source_editor.get("insert linestart", tk.INSERT)
        m = re.search(r'\[@(?!(?:anchor|def):)([^\]\n]*)$', line_prefix)
        if m:
            if self._ac_popup and self._ac_popup.winfo_exists():
                # Already open: filter live as user types
                self._show_anchor_autocomplete()
            else:
                # Wait 1000ms (1 second) before showing popup
                self._ac_job = self.after(1000, self._show_anchor_autocomplete)
        else:
            self._close_autocomplete()

    def _collect_document_targets(self):
        """Collect all unified [@...] tags, legacy [@anchor: ...] tags, and # headings in the document with occurrence counts."""
        content = self.source_editor.get("1.0", "end-1c")
        self._target_counts = {}
        first_names = {}
        order = []

        try:
            cur_line, cur_col = map(int, self.source_editor.index(tk.INSERT).split('.'))
        except Exception:
            cur_line, cur_col = -1, -1

        tag_pat = re.compile(r'\[@(?:(?:anchor|def):\s*)?([^\]\n]+)\]', re.IGNORECASE)
        in_code = False
        lines = content.split('\n')

        # 1. Unified [@label] & legacy [@anchor: label] tags outside code blocks
        for lineno, line in enumerate(lines, start=1):
            if line.strip().startswith('```'):
                in_code = not in_code
                continue
            if in_code:
                continue
            for m in tag_pat.finditer(line):
                # Skip a token actively being edited at cursor
                if lineno == cur_line and m.start() <= cur_col <= m.end():
                    continue
                name = m.group(1).strip()
                norm = _normalize_anchor_key(name)
                if name and norm:
                    if norm not in first_names:
                        first_names[norm] = name
                        order.append(norm)
                    self._target_counts[norm] = self._target_counts.get(norm, 0) + 1

        targets = []
        for norm in order:
            name = first_names[norm]
            cnt = self._target_counts.get(norm, 1)
            occ_str = f"  ({cnt}x)"
            targets.append((f"\u2693 {name}{occ_str}", name))

        # 2. Document headings (outside code blocks)
        in_code = False
        seen_norms = set(first_names.keys())
        for line in lines:
            if line.strip().startswith('```'):
                in_code = not in_code
                continue
            if in_code:
                continue
            hm = re.match(r'^(#{1,6})\s+(.+)', line)
            if hm:
                h_text = hm.group(2).strip()
                norm = _normalize_anchor_key(h_text)
                if h_text and norm not in seen_norms:
                    seen_norms.add(norm)
                    lvl = len(hm.group(1))
                    targets.append((f"H{lvl}: {h_text}  (Heading)", h_text))

        return targets

    def _show_anchor_autocomplete(self):
        """Display or refresh the floating cross-reference autocomplete popup."""
        self._ac_job = None
        editor = self.source_editor
        line_prefix = editor.get("insert linestart", tk.INSERT)
        m = re.search(r'\[@(?!(?:anchor|def):)([^\]\n]*)$', line_prefix)
        if not m:
            self._close_autocomplete()
            return

        query = m.group(1).strip().lower()
        all_targets = self._collect_document_targets()
        filtered = [item for item in all_targets if not query or query in item[1].lower()]

        # Always offer an option to insert a new unified reference/anchor tag at cursor
        new_label = m.group(1).strip() or "point"
        filtered.append((f"\u2795 Insert new reference [@{new_label}]", f"__NEW__:{new_label}"))

        self._ac_items = filtered

        bbox = editor.bbox(tk.INSERT)
        if not bbox:
            self._close_autocomplete()
            return
        bx, by, bw, bh = bbox
        x = editor.winfo_rootx() + bx
        y = editor.winfo_rooty() + by + bh + 4

        bg = self.theme.get('source_bg', '#ffffff')
        fg = self.theme.get('source_fg', '#1a1a1a')
        is_dark = _is_dark_color(bg)
        pop_bg = "#252526" if is_dark else "#ffffff"
        pop_fg = "#e2e8f0" if is_dark else "#1e293b"
        sel_bg = "#3b82f6" if is_dark else "#2563eb"
        border_col = "#475569" if is_dark else "#cbd5e1"

        if not self._ac_popup or not self._ac_popup.winfo_exists():
            self._ac_popup = tk.Toplevel(self)
            self._ac_popup.wm_overrideredirect(True)
            self._ac_popup.attributes("-topmost", True)
            frame = tk.Frame(self._ac_popup, bg=border_col, bd=1, relief=tk.SOLID)
            frame.pack(fill=tk.BOTH, expand=True)
            hdr = tk.Label(
                frame, text=" Cross-Reference Target (Enter/Click to insert, Esc to close)",
                font=(SANS_FONT, 8, "bold"), bg= border_col, fg=pop_fg, anchor="w", padx=4, pady=2
            )
            hdr.pack(fill=tk.X)
            self._ac_listbox = tk.Listbox(
                frame, font=(SANS_FONT, 9), bg=pop_bg, fg=pop_fg,
                selectbackground=sel_bg, selectforeground="#ffffff",
                activestyle="none", highlightthickness=0, bd=0,
                width=42, height=min(8, max(2, len(filtered)))
            )
            self._ac_listbox.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
            self._ac_listbox.bind("<ButtonRelease-1>", self._on_ac_click)
            self._ac_listbox.bind("<Return>", self._on_ac_return)
            self._ac_listbox.bind("<Escape>", self._on_ac_escape)

        self._ac_listbox. configure(height=min(8, max(2, len(filtered))), bg=pop_bg, fg=pop_fg)
        self._ac_listbox.delete(0, tk.END)
        for display_text, _ in filtered:
            self._ac_listbox.insert(tk.END, f" {display_text}")
        self._ac_listbox.selection_set(0)
        self._ac_listbox.activate(0)
        self._ac_popup.wm_geometry(f"+{x}+{y}")

    def _check_ac_focus(self):
        if not self._ac_popup or not self._ac_popup.winfo_exists():
            return
        try:
            focused = self.focus_get()
            if focused is not self.source_editor and focused is not self._ac_listbox:
                self._close_autocomplete()
        except Exception:
            self._close_autocomplete()

    def _close_autocomplete(self):
        if self._ac_job:
            try:
                self.after_cancel(self._ac_job)
            except Exception:
                pass
            self._ac_job = None
        if self._ac_popup and self._ac_popup.winfo_exists():
            try:
                self._ac_popup.destroy()
            except Exception:
                pass
        self._ac_popup = None
        self._ac_listbox = None
        self._ac_items = []

    def _on_ac_down(self, event=None):
        if self._ac_popup and self._ac_popup.winfo_exists() and self._ac_listbox:
            cur = self._ac_listbox.curselection()
            idx = (cur[0] + 1) % len(self._ac_items) if cur else 0
            self._ac_listbox.selection_clear(0, tk.END)
            self._ac_listbox.selection_set(idx)
            self._ac_listbox.activate(idx)
            self._ac_listbox.see(idx)
            return "break"
        return None

    def _on_ac_up(self, event=None):
        if self._ac_popup and self._ac_popup.winfo_exists() and self._ac_listbox:
            cur = self._ac_listbox.curselection()
            idx = (cur[0] - 1) % len(self._ac_items) if cur else 0
            self._ac_listbox.selection_clear(0, tk.END)
            self._ac_listbox.selection_set(idx)
            self._ac_listbox.activate(idx)
            self._ac_listbox.see(idx)
            return "break"
        return None

    def _on_ac_return(self, event=None):
        if self._ac_popup and self._ac_popup.winfo_exists() and self._ac_listbox:
            self._accept_autocomplete()
            return "break"
        return None

    def _on_ac_escape(self, event=None):
        if self._ac_popup and self._ac_popup.winfo_exists():
            self._close_autocomplete()
            return "break"
        return None

    def _on_ac_click(self, event=None):
        if self._ac_popup and self._ac_popup.winfo_exists():
            self._accept_autocomplete()
            self.source_editor.focus_set()

    def _accept_autocomplete(self):
        if not self._ac_listbox or not self._ac_items:
            self._close_autocomplete()
            return
        cur = self._ac_listbox.curselection()
        idx = cur[0] if cur else 0
        if not (0 <= idx < len(self._ac_items)):
            self._close_autocomplete()
            return
        _, value = self._ac_items[idx]
        self._close_autocomplete()

        editor = self.source_editor
        line_prefix = editor.get("insert linestart", tk.INSERT)
        m = re.search(r'\[@(?!(?:anchor|def):)([^\]\n]*)$', line_prefix)
        if not m:
            return

        match_len = len(m.group(0))
        start_pos = editor.index(f"insert - {match_len}c")
        end_pos = editor.index(tk.INSERT)
        # If the character immediately after cursor is a closing bracket, consume it too
        if editor.get(end_pos, f"{end_pos}+1c") == "]":
            end_pos = editor.index(f"{end_pos}+1c")

        if value.startswith("__NEW__:"):
            new_name = value.split(":", 1)[1]
            replacement = f"[@{new_name}]"
        else:
            replacement = f"[@{value}]"

        editor.delete(start_pos, end_pos)
        editor.insert(start_pos, replacement)
        editor.mark_set(tk.INSERT, f"{start_pos}+{len(replacement)}c")
        self.is_modified = True
        if self.on_modified:
            self.on_modified()
        self._apply_syntax_highlighting()

    # --- Markdown Table Auto-Formatter & Cell Navigation (Tab / Shift+Tab) -----

    def _on_source_tab(self, event=None):
        """Handle Tab in source editor: accept autocomplete if open, or auto-format/advance in Markdown tables."""
        if self._ac_popup and self._ac_popup.winfo_exists():
            self._accept_autocomplete()
            return "break"
        if self._handle_table_tab(direction=1):
            return "break"
        return None

    def _on_source_shift_tab(self, event=None):
        """Handle Shift+Tab in source editor: navigate backward and auto-format in Markdown tables."""
        if self._handle_table_tab(direction=-1):
            return "break"
        return None

    @staticmethod
    def _is_table_line(line: str) -> bool:
        s = line.strip()
        return s.startswith('|') and '|' in s[1:]

    @staticmethod
    def _parse_table_row(line: str):
        s = line.strip()
        if s.startswith('|'):
            s = s[1:]
        if s.endswith('|'):
            s = s[:-1]
        return [cell.strip() for cell in s.split('|')]

    @staticmethod
    def _is_separator_row(cells) -> bool:
        return bool(cells) and all(bool(re.match(r'^:?-+:?$', c.strip())) for c in cells if c.strip()) and any(c.strip() for c in cells)

    def _handle_table_tab(self, direction: int = 1) -> bool:
        """If cursor is inside a Markdown table, auto-pad/align all columns and jump to next/previous cell."""
        editor = self.source_editor
        try:
            insert_idx = editor.index(tk.INSERT)
            curr_line_no, curr_col_char = map(int, insert_idx.split('.'))
        except Exception:
            return False

        curr_line = editor.get(f"{curr_line_no}.0", f"{curr_line_no}.end")
        if not self._is_table_line(curr_line):
            return False

        total_lines = int(editor.index("end-1c").split('.')[0])
        first_line = curr_line_no
        while first_line > 1 and self._is_table_line(editor.get(f"{first_line - 1}.0", f"{first_line - 1}.end")):
            first_line -= 1
        last_line = curr_line_no
        while last_line < total_lines and self._is_table_line(editor.get(f"{last_line + 1}.0", f"{last_line + 1}.end")):
            last_line += 1

        raw_lines = [editor.get(f"{ln}.0", f"{ln}.end") for ln in range(first_line, last_line + 1)]
        parsed_rows = [self._parse_table_row(ln) for ln in raw_lines]
        num_cols = max((len(r) for r in parsed_rows), default=1)
        if num_cols < 1:
            return False

        sep_indices = set()
        col_alignments = ['left'] * num_cols
        for r_idx, row in enumerate(parsed_rows):
            while len(row) < num_cols:
                row.append('')
            if self._is_separator_row(row):
                sep_indices.add(r_idx)
                for c_idx, cell in enumerate(row):
                    c_str = cell.strip()
                    if c_str.startswith(':') and c_str.endswith(':'):
                        col_alignments[c_idx] = 'center'
                    elif c_str.endswith(':'):
                        col_alignments[c_idx] = 'right'
                    else:
                        col_alignments[c_idx] = 'left'

        # Determine active cell (rel_row, curr_col)
        rel_row = curr_line_no - first_line
        pipes_before = curr_line[:curr_col_char].count('|')
        curr_col = max(0, min(num_cols - 1, pipes_before - 1))

        # Compute target cell
        target_row = rel_row
        target_col = curr_col + direction

        if direction > 0:
            while target_col >= num_cols or target_row in sep_indices:
                if target_col >= num_cols:
                    target_col = 0
                    target_row += 1
                if target_row in sep_indices:
                    target_row += 1
            if target_row >= len(parsed_rows):
                parsed_rows.append(['' for _ in range(num_cols)])
                target_row = len(parsed_rows) - 1
                target_col = 0
        else:
            while target_col < 0 or target_row in sep_indices:
                if target_col < 0:
                    if target_row > 0:
                        target_row -= 1
                        target_col = num_cols - 1
                    else:
                        target_row = 0
                        target_col = 0
                        break
                if target_row in sep_indices:
                    if target_row > 0:
                        target_row -= 1
                    else:
                        target_row = 0
                        break

        # Calculate column widths across all non-separator rows
        col_widths = [3] * num_cols
        for r_idx, row in enumerate(parsed_rows):
            if r_idx in sep_indices:
                continue
            for c_idx in range(num_cols):
                col_widths[c_idx] = max(col_widths[c_idx], len(row[c_idx]))

        # Build formatted table lines
        formatted_lines = []
        for r_idx, row in enumerate(parsed_rows):
            cells_out = []
            if r_idx in sep_indices:
                for c_idx in range(num_cols):
                    w = max(3, col_widths[c_idx])
                    al = col_alignments[c_idx]
                    if al == 'center':
                        cells_out.append(':' + '-' * max(1, w - 2) + ':')
                    elif al == 'right':
                        cells_out.append('-' * max(1, w - 1) + ':')
                    else:
                        cells_out.append(':' + '-' * max(1, w - 1) if row[c_idx].startswith(':') else '-' * w)
            else:
                for c_idx in range(num_cols):
                    w = col_widths[c_idx]
                    val = row[c_idx]
                    al = col_alignments[c_idx]
                    if al == 'center':
                        cells_out.append(val.center(w))
                    elif al == 'right':
                        cells_out.append(val.rjust(w))
                    else:
                        cells_out.append(val.ljust(w))
            formatted_lines.append("| " + " | ".join(cells_out) + " |")

        new_table_text = "\n".join(formatted_lines)
        editor.delete(f"{first_line}.0", f"{last_line}.end")
        editor.insert(f"{first_line}.0", new_table_text)

        # Place cursor in target cell and highlight its text if non-empty
        target_doc_line = first_line + target_row
        cell_start_col = 2 + sum(col_widths[c] + 3 for c in range(target_col))
        cell_val = parsed_rows[target_row][target_col]
        al = col_alignments[target_col]
        w = col_widths[target_col]
        if al == 'center':
            leading_pad = (w - len(cell_val)) // 2
        elif al == 'right':
            leading_pad = w - len(cell_val)
        else:
            leading_pad = 0

        val_start = f"{target_doc_line}.{cell_start_col + leading_pad}"
        val_end = f"{target_doc_line}.{cell_start_col + leading_pad + len(cell_val)}"
        editor.tag_remove(tk.SEL, "1.0", tk.END)
        if cell_val:
            editor.tag_add(tk.SEL, val_start, val_end)
        editor.mark_set(tk.INSERT, val_end)
        editor.see(val_end)

        self.is_modified = True
        if self.on_modified:
            self.on_modified()
        self._apply_syntax_highlighting()
        return True


class OutlinePanel(ttk.Frame):
    """Multi-faceted Document Navigator — filterable by All, Headings, Anchors, Media/Tables, and Footnotes."""

    LEVEL_FG = {
        1: '#1a1a2e', 2: '#16213e', 3: '#1f4068',
        4: '#555555', 5: '#777777', 6: '#999999',
    }
    DARK_LEVEL_FG = {
        1: '#f8fafc', 2: '#e2e8f0', 3: '#cbd5e1',
        4: '#94a3b8', 5: '#64748b', 6: '#64748b',
    }

    CATEGORIES = (
        ('all', 'All'),
        ('headings', '§ H'),
        ('anchors', '⚓ Anc'),
        ('media', '⊞ Media'),
        ('footnotes', 'ᵃ Fn'),
    )

    def __init__(self, parent, on_heading_click=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_heading_click = on_heading_click
        self._item_linenos = {}
        self._last_content = ""
        self._update_job = None
        self.category_var = tk.StringVar(value='all')
        self.filter_var = tk.StringVar(value='')
        self._setup_ui()

    def _setup_ui(self):
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=4, pady=(6, 2))
        ttk.Label(header, text='Navigator', font=(SANS_FONT, 10, 'bold')).pack(side=tk.LEFT, padx=4)

        # Category filter bar with uniform columns to prevent clipping
        cat_bar = ttk.Frame(self)
        cat_bar.pack(fill=tk.X, padx=4, pady=(0, 3))
        for i in range(len(self.CATEGORIES)):
            cat_bar.columnconfigure(i, weight=1, uniform='nav_cats')
        cat_tips = {
            'all': "Display all structural items in the document including headings, anchors, tables, media, and footnotes.",
            'headings': "Filter navigator to only show Markdown headings (levels H1 through H6).",
            'anchors': "Filter navigator to display explicit cross-reference anchors and reference links.",
            'media': "Filter navigator to show tables, diagrams, and embedded image figures.",
            'footnotes': "Filter navigator to list all footnote references and footnote definitions."
        }
        for i, (key, label) in enumerate(self.CATEGORIES):
            rb = ttk.Radiobutton(
                cat_bar,
                text=label,
                value=key,
                variable=self.category_var,
                style='Toolbar.Toolbutton',
                command=self._refresh_from_cache,
            )
            rb.grid(row=0, column=i, sticky='ew', padx=1)
            ToolTip(rb, cat_tips.get(key, ""))

        # Live filter search entry
        filter_row = ttk.Frame(self)
        filter_row.pack(fill=tk.X, padx=6, pady=(0, 4))
        ttk.Label(filter_row, text='🔍', font=(SANS_FONT, 9)).pack(side=tk.LEFT, padx=(0, 3))
        self.filter_entry = ttk.Entry(filter_row, textvariable=self.filter_var)
        self.filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        clear_btn = ttk.Button(
            filter_row, text='✕', width=2, style='Toolbar.TButton',
            command=lambda: (self.filter_var.set(''), self._refresh_from_cache())
        )
        clear_btn.pack(side=tk.RIGHT, padx=(2, 0))
        ToolTip(clear_btn, "Clear search filter and display all items in active category.")
        self.filter_var.trace_add('write', lambda *_: self._refresh_from_cache())

        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(tree_frame, show='tree',
                                  selectmode='browse', yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=self.tree.yview)

        self._configure_tags(is_dark=False)
        self.tree.bind('<<TreeviewSelect>>', self._on_select)

    def _configure_tags(self, is_dark=False):
        fg_map = self.DARK_LEVEL_FG if is_dark else self.LEVEL_FG
        for lvl in range(1, 7):
            self.tree.tag_configure(
                f'h{lvl}',
                foreground=fg_map.get(lvl, '#999999'),
                font=(SANS_FONT, max(11 - lvl + 1, 9),
                      'bold' if lvl <= 3 else 'normal')
            )
        self.tree.tag_configure(
            'anchor',
            foreground='#c4b5fd' if is_dark else '#6d28d9',
            font=(SANS_FONT, 9, 'bold')
        )
        self.tree.tag_configure(
            'ref',
            foreground='#93c5fd' if is_dark else '#2563eb',
            font=(SANS_FONT, 9)
        )
        self.tree.tag_configure(
            'media',
            foreground='#5eead4' if is_dark else '#0f766e',
            font=(SANS_FONT, 9)
        )
        self.tree.tag_configure(
            'footnote',
            foreground='#fcd34d' if is_dark else '#b45309',
            font=(SANS_FONT, 9)
        )
        self.tree.tag_configure(
            'empty',
            foreground='#94a3b8' if is_dark else '#aaaaaa',
            font=(SANS_FONT, 9, 'italic')
        )

    def apply_theme(self, theme: dict):
        """Synchronize navigator colors with active theme."""
        bg = theme.get('source_bg', '#ffffff')
        is_dark = _is_dark_color(bg)
        self._configure_tags(is_dark=is_dark)

    def schedule_update(self, content: str):
        """Debounced rebuild (400 ms)."""
        self._last_content = content
        if self._update_job:
            self.after_cancel(self._update_job)
        self._update_job = self.after(400, lambda: self.update(content))

    def _refresh_from_cache(self):
        self.update(self._last_content)

    def update(self, content: str):
        """Immediately rebuild the navigator tree from document content."""
        self._update_job = None
        self._last_content = content or ""
        self.tree.delete(*self.tree.get_children())
        self._item_linenos.clear()

        cat = self.category_var.get()
        query = self.filter_var.get().strip().lower()

        items = []  # (lineno, category, tag, level_or_0, label)
        in_code_block = False
        in_table = False
        table_start_line = 0
        table_rows_count = 0
        table_cols_count = 0

        lines = self._last_content.split('\n')
        # Pre-count unified & legacy cross-reference/anchor tags outside code blocks
        anc_totals: Dict[str, int] = {}
        anc_seen_counts: Dict[str, int] = {}
        _in_code_pre = False
        _tag_pat_pre = re.compile(r'\[@(?:(?:anchor|def):\s*)?([^\]\n]+)\]', re.IGNORECASE)
        for _ln in lines:
            if _ln.strip().startswith('```'):
                _in_code_pre = not _in_code_pre
                continue
            if _in_code_pre:
                continue
            for _m in _tag_pat_pre.finditer(_ln):
                _nm = _m.group(1).strip()
                if _nm:
                    _nk = _normalize_anchor_key(_nm)
                    anc_totals[_nk] = anc_totals.get(_nk, 0) + 1

        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('```'):
                if in_table:
                    items.append((table_start_line, 'media', 'media', 0, f"⊞ Table ({table_rows_count}×{table_cols_count})  [L{table_start_line}]"))
                    in_table = False
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            # Tables
            if '|' in line and stripped.startswith('|'):
                cells = [c.strip() for c in stripped.strip('|').split('|')]
                if not in_table:
                    in_table = True
                    table_start_line = lineno
                    table_rows_count = 1
                    table_cols_count = len(cells)
                else:
                    if not all(re.match(r'^:?-+:?$', c) for c in cells if c):
                        table_rows_count += 1
                    table_cols_count = max(table_cols_count, len(cells))
            else:
                if in_table:
                    items.append((table_start_line, 'media', 'media', 0, f"⊞ Table ({table_rows_count}×{table_cols_count})  [L{table_start_line}]"))
                    in_table = False

            # Headings
            m_h = re.match(r'^(#{1,6})\s+(.+)', line)
            if m_h:
                lvl = len(m_h.group(1))
                text = m_h.group(2).strip()
                items.append((lineno, 'headings', f'h{lvl}', lvl, f"{'#' * lvl} {text}"))

            # Anchors & References (unified ring)
            for m_tag in re.finditer(r'\[@(?:(anchor|def):\s*)?([^\]\n]+)\]', line, re.IGNORECASE):
                is_legacy_def = bool(m_tag.group(1))
                name = m_tag.group(2).strip()
                if not name:
                    continue
                norm = _normalize_anchor_key(name)
                anc_seen_counts[norm] = anc_seen_counts.get(norm, 0) + 1
                occ_idx = anc_seen_counts[norm]
                total_cnt = anc_totals.get(norm, 1)
                count_badge = f" ({occ_idx}/{total_cnt})" if total_cnt > 1 else ""
                tag_style = 'anchor' if (is_legacy_def or occ_idx == 1) else 'ref'
                items.append((lineno, 'anchors', tag_style, 0, f"⚓ [@{name}]{count_badge}  [L{lineno}]"))

            # Images & Display Math
            for m_img in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', line):
                alt = m_img.group(1).strip() or os.path.basename(m_img.group(2).strip()) or "image"
                items.append((lineno, 'media', 'media', 0, f"🖼 Image: {alt}  [L{lineno}]"))
            if stripped.startswith('$$'):
                items.append((lineno, 'media', 'media', 0, f"∑ Math Block  [L{lineno}]"))

            # Footnotes
            m_fndef = re.match(r'^\[\^([^\]]+)\]:\s*(.+)$', stripped)
            if m_fndef:
                fid = m_fndef.group(1).strip()
                fbody = m_fndef.group(2).strip()
                preview = (fbody[:28] + '…') if len(fbody) > 28 else fbody
                items.append((lineno, 'footnotes', 'footnote', 0, f"ᵃ [^{fid}]: {preview}"))
            else:
                for m_fnref in re.finditer(r'\[\^([^\]\n]+)\](?!:)', line):
                    fid = m_fnref.group(1).strip()
                    items.append((lineno, 'footnotes', 'ref', 0, f"ᵃ Cite [^{fid}]  [L{lineno}]"))

        if in_table:
            items.append((table_start_line, 'media', 'media', 0, f"⊞ Table ({table_rows_count}×{table_cols_count})  [L{table_start_line}]"))

        # Filter by category and search query
        filtered = [
            it for it in items
            if (cat == 'all' or it[1] == cat) and (not query or query in it[4].lower())
        ]

        if not filtered:
            self.tree.insert('', tk.END, text='(no matching items)', tags=('empty',))
            return

        # Hierarchical tree when showing Headings (or All without search filter)
        if cat == 'headings' and not query:
            stack = []
            for lineno, _, tag, level, label in filtered:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                indent = '  ' * (level - 1)
                parent_iid = stack[-1][1] if stack else ''
                iid = self.tree.insert(parent_iid, tk.END, text=f"{indent}{label}",
                                       tags=(tag,), open=True)
                stack.append((level, iid))
                self._item_linenos[iid] = lineno
        else:
            for lineno, item_cat, tag, level, label in filtered:
                indent = '  ' * max(0, level - 1) if item_cat == 'headings' else ''
                iid = self.tree.insert('', tk.END, text=f"{indent}{label}", tags=(tag,), open=True)
                self._item_linenos[iid] = lineno

    def _on_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        lineno = self._item_linenos.get(sel[0])
        if lineno and self._on_heading_click:
            self._on_heading_click(lineno)


class DockedFindReplaceBar(ttk.Frame):
    """Non-blocking docked Find & Replace bar at the top of the editor area.
    Operates seamlessly across both Source Mode and Visual Mode with live match
    highlighting, match counter, case/regex toggles, and Replace / Replace All.
    """

    def __init__(self, parent, notepad, **kwargs):
        super().__init__(parent, padding=(6, 4), **kwargs)
        self.notepad = notepad
        self.is_visible = False
        self.replace_visible = False
        self.find_var = tk.StringVar()
        self.replace_var = tk.StringVar()
        self.case_var = tk.BooleanVar(value=False)
        self.regex_var = tk.BooleanVar(value=False)
        self._matches = []  # List of (start_idx, end_idx)
        self._active_idx = -1
        self._setup_ui()

    def _setup_ui(self):
        # Row 1: Find controls (close button packed right first to avoid occlusion)
        self.find_row = ttk.Frame(self)
        self.find_row.pack(fill=tk.X, pady=1)

        close_btn = ttk.Button(self.find_row, text="✕", width=3, style="Toolbar.TButton", command=self.hide)
        close_btn.pack(side=tk.RIGHT, padx=2)
        ToolTip(close_btn, "Close Find & Replace bar (Escape).")

        ttk.Label(self.find_row, text="Find:", width=6).pack(side=tk.LEFT)
        self.find_entry = ttk.Entry(self.find_row, textvariable=self.find_var, width=22)
        self.find_entry.pack(side=tk.LEFT, padx=(0, 4))

        prev_btn = ttk.Button(self.find_row, text="▲", width=3, style="Toolbar.TButton", command=self.find_prev)
        prev_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(prev_btn, "Navigate to previous search match in document (Shift+Enter).")

        next_btn = ttk.Button(self.find_row, text="▼", width=3, style="Toolbar.TButton", command=self.find_next)
        next_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(next_btn, "Navigate to next search match in document (Enter / F3).")

        case_cb = ttk.Checkbutton(self.find_row, text="Aa", variable=self.case_var,
                                  style="Toolbar.Toolbutton", command=self._on_query_change)
        case_cb.pack(side=tk.LEFT, padx=2)
        ToolTip(case_cb, "Toggle case-sensitive search matching.")

        regex_cb = ttk.Checkbutton(self.find_row, text=".*", variable=self.regex_var,
                                   style="Toolbar.Toolbutton", command=self._on_query_change)
        regex_cb.pack(side=tk.LEFT, padx=2)
        ToolTip(regex_cb, "Enable regular expression search pattern matching.")

        self.toggle_rep_btn = ttk.Button(
            self.find_row, text="⇄ Replace", style="Toolbar.TButton",
            command=self.toggle_replace_row
        )
        self.toggle_rep_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(self.toggle_rep_btn, "Expand or collapse the text replacement input row.")

        self.count_label = ttk.Label(self.find_row, text="", foreground="#64748b", font=(SANS_FONT, 9))
        self.count_label.pack(side=tk.LEFT, padx=8)

        # Row 2: Replace controls (packed when replace_visible is True)
        self.replace_row = ttk.Frame(self)
        ttk.Label(self.replace_row, text="Replace:", width=8).pack(side=tk.LEFT)
        self.replace_entry = ttk.Entry(self.replace_row, textvariable=self.replace_var, width=22)
        self.replace_entry.pack(side=tk.LEFT, padx=(0, 4))

        rep_btn = ttk.Button(self.replace_row, text="Replace", style="Toolbar.TButton", command=self.replace_one)
        rep_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(rep_btn, "Replace the currently focused match with replacement text.")

        rep_all_btn = ttk.Button(self.replace_row, text="Replace All", style="Toolbar.TButton", command=self.replace_all)
        rep_all_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(rep_all_btn, "Replace all occurrences throughout the entire document.")

        # Bindings
        self.find_var.trace_add("write", lambda *_: self._on_query_change())
        self.find_entry.bind("<Return>", lambda e: self.find_next())
        self.find_entry.bind("<Shift-Return>", lambda e: self.find_prev())
        self.find_entry.bind("<Escape>", lambda e: self.hide())
        self.replace_entry.bind("<Return>", lambda e: self.replace_one())
        self.replace_entry.bind("<Escape>", lambda e: self.hide())

    def _active_text_widget(self):
        tab = self.notepad.current_tab
        if not tab or getattr(tab, 'tab_type', 'document') == 'spreadsheet':
            return None
        return getattr(tab, 'source_editor', None)

    def _ensure_highlight_tags(self, widget):
        if not widget:
            return
        dark = _is_dark_color(self.notepad.theme.get('source_bg', '#ffffff'))
        match_bg = "#78350f" if dark else "#fde68a"
        match_fg = "#fef3c7" if dark else "#1e293b"
        active_bg = "#ea580c" if dark else "#f97316"
        active_fg = "#ffffff"
        widget.tag_configure("find_match", background=match_bg, foreground=match_fg)
        widget.tag_configure("find_active", background=active_bg, foreground=active_fg)
        widget.tag_raise("find_match")
        widget.tag_raise("find_active")

    def clear_highlights(self):
        tab = self.notepad.current_tab
        if not tab or getattr(tab, 'tab_type', 'document') == 'spreadsheet':
            return
        for w in (getattr(tab, 'source_editor', None), getattr(getattr(tab, 'visual_viewer', None), 'text_widget', None)):
            if w:
                try:
                    w.tag_remove("find_match", "1.0", tk.END)
                    w.tag_remove("find_active", "1.0", tk.END)
                except Exception:
                    pass

    def show(self, replace_mode=False):
        if not self.is_visible:
            self.pack(before=self.notepad.notebook, fill=tk.X, padx=2, pady=(2, 2))
            self.is_visible = True
        if replace_mode and not self.replace_visible:
            self.replace_row.pack(fill=tk.X, pady=(2, 1))
            self.replace_visible = True
        elif not replace_mode and self.replace_visible:
            self.replace_row.pack_forget()
            self.replace_visible = False

        # Pre-populate search with selected text if short single-line selection exists
        w = self._active_text_widget()
        if w:
            try:
                if w.tag_ranges(tk.SEL):
                    sel = w.get(tk.SEL_FIRST, tk.SEL_LAST)
                    if sel and '\n' not in sel and len(sel) <= 120:
                        self.find_var.set(sel)
            except tk.TclError:
                pass

        self.find_entry.focus_set()
        self.find_entry.selection_range(0, tk.END)
        self._on_query_change()

    def toggle_replace_row(self):
        if self.replace_visible:
            self.replace_row.pack_forget()
            self.replace_visible = False
        else:
            self.replace_row.pack(fill=tk.X, pady=(2, 1))
            self.replace_visible = True
            self.replace_entry.focus_set()

    def hide(self):
        self.clear_highlights()
        if self.is_visible:
            self.pack_forget()
            self.is_visible = False
        self.notepad._focus_active_editor()
        return "break"

    def _on_query_change(self):
        self.clear_highlights()
        self._matches.clear()
        self._active_idx = -1

        query = self.find_var.get()
        tab = self.notepad.current_tab
        if not query or not tab:
            self.count_label.config(text="")
            return

        flags = 0 if self.case_var.get() else re.IGNORECASE
        try:
            pattern = query if self.regex_var.get() else re.escape(query)
            compiled = re.compile(pattern, flags)
        except re.error:
            self.count_label.config(text="Invalid regex")
            return

        # Spreadsheet Tab Search
        if getattr(tab, 'tab_type', 'document') == 'spreadsheet':
            for (r, c) in sorted(tab.engine.raw_cells.keys()):
                raw_val = tab.engine.get_raw(r, c)
                comp_val = str(tab.engine.get_value(r, c))
                if (raw_val and compiled.search(raw_val)) or (comp_val and compiled.search(comp_val)):
                    self._matches.append((r, c))

            if not self._matches:
                self.count_label.config(text="No matches")
                return

            self._active_idx = 0
            self._highlight_active_match()
            return

        # Document Tab Search
        w = self._active_text_widget()
        if not w:
            self.count_label.config(text="")
            return

        self._ensure_highlight_tags(w)
        content = w.get("1.0", "end-1c")

        for m in compiled.finditer(content):
            if m.start() == m.end():
                continue
            s_idx = f"1.0 + {m.start()} chars"
            e_idx = f"1.0 + {m.end()} chars"
            self._matches.append((s_idx, e_idx))
            w.tag_add("find_match", s_idx, e_idx)

        if not self._matches:
            self.count_label.config(text="No matches")
            return

        self._active_idx = 0
        self._highlight_active_match()

    def _highlight_active_match(self):
        tab = self.notepad.current_tab
        if not tab or not self._matches or self._active_idx < 0:
            return

        if getattr(tab, 'tab_type', 'document') == 'spreadsheet':
            r, c = self._matches[self._active_idx]
            tab.grid_widget.active_cell = (r, c)
            tab.grid_widget.sel_start = (r, c)
            tab.grid_widget.sel_end = (r, c)
            tab.grid_widget._ensure_cell_visible(r, c)
            tab.grid_widget._notify_select()
            tab.grid_widget.redraw()
            ref = cell_to_ref(r, c) if SPREADSHEET_AVAILABLE else f"R{r+1}C{c+1}"
            self.count_label.config(text=f"{self._active_idx + 1} of {len(self._matches)} (Cell {ref})")
            return

        w = self._active_text_widget()
        if not w:
            return
        w.tag_remove("find_active", "1.0", tk.END)
        s_idx, e_idx = self._matches[self._active_idx]
        w.tag_add("find_active", s_idx, e_idx)
        w.see(s_idx)
        if w is getattr(self.notepad.current_tab, 'source_editor', None):
            w.mark_set(tk.INSERT, e_idx)
        self.count_label.config(text=f"{self._active_idx + 1} of {len(self._matches)}")

    def find_next(self):
        if not self._matches:
            self._on_query_change()
            return "break"
        self._active_idx = (self._active_idx + 1) % len(self._matches)
        self._highlight_active_match()
        return "break"

    def find_prev(self):
        if not self._matches:
            self._on_query_change()
            return "break"
        self._active_idx = (self._active_idx - 1) % len(self._matches)
        self._highlight_active_match()
        return "break"

    def replace_one(self):
        tab = self.notepad.current_tab
        if not tab:
            return "break"
        if not self._matches or self._active_idx < 0:
            self._on_query_change()
            return "break"

        query = self.find_var.get()
        rep_text = self.replace_var.get()
        flags = 0 if self.case_var.get() else re.IGNORECASE
        try:
            pattern = query if self.regex_var.get() else re.escape(query)
            compiled = re.compile(pattern, flags)
        except re.error:
            return "break"

        if getattr(tab, 'tab_type', 'document') == 'spreadsheet':
            r, c = self._matches[self._active_idx]
            raw = tab.engine.get_raw(r, c)
            new_raw = compiled.sub(rep_text, raw, count=1)
            if new_raw != raw:
                tab.grid_widget._save_undo_state({"type": "set_cell", "r": r, "c": c, "old": raw, "new": new_raw})
                tab.engine.set_cell(r, c, new_raw)
                tab.engine.recalculate_all()
                tab.grid_widget.redraw()
                tab.is_modified = True
                self.notepad._update_tab_title(tab)
            self._on_query_change()
            return "break"

        editor = getattr(tab, 'source_editor', None)
        if not editor:
            return "break"
        s_idx, e_idx = self._matches[self._active_idx]
        editor.delete(s_idx, e_idx)
        editor.insert(s_idx, rep_text)
        tab.is_modified = True
        self.notepad._update_tab_title(tab)
        self.notepad._schedule_syntax_highlight()
        self._on_query_change()
        return "break"

    def replace_all(self):
        tab = self.notepad.current_tab
        query = self.find_var.get()
        if not tab or not query:
            return "break"
        rep_text = self.replace_var.get()
        flags = 0 if self.case_var.get() else re.IGNORECASE
        try:
            pattern = query if self.regex_var.get() else re.escape(query)
            compiled = re.compile(pattern, flags)
        except re.error:
            return "break"

        if getattr(tab, 'tab_type', 'document') == 'spreadsheet':
            old_cells = {}
            new_cells = {}
            count = 0
            for (r, c) in list(tab.engine.raw_cells.keys()):
                raw = tab.engine.get_raw(r, c)
                new_raw, n = compiled.subn(rep_text, raw)
                if n > 0:
                    old_cells[(r, c)] = raw
                    new_cells[(r, c)] = new_raw
                    tab.engine.set_cell(r, c, new_raw)
                    count += n
            if count > 0:
                tab.grid_widget._save_undo_state({"type": "batch_cells", "old_cells": old_cells, "new_cells": new_cells})
                tab.engine.recalculate_all()
                tab.grid_widget.redraw()
                tab.is_modified = True
                self.notepad._update_tab_title(tab)
                self.notepad._set_status(f"Replaced {count} occurrence(s) in spreadsheet")
            self._on_query_change()
            return "break"

        content = tab.get_content()
        new_content, count = compiled.subn(rep_text, content)
        if count > 0:
            tab.set_content(new_content)
            tab.is_modified = True
            self.notepad._update_tab_title(tab)
            self.notepad._schedule_syntax_highlight()
            self.notepad._set_status(f"Replaced {count} occurrence(s)")
        self._on_query_change()
        return "break"


class MarkdownNotepad(tk.Tk):
    """Main application window"""
    
    # Default theme settings. Font entries are resolved in __init__ once a Tk root exists.
    DEFAULT_THEME = {
        'source_bg': '#ffffff',
        'source_fg': '#1a1a1a',
        'source_font': None,
        'source_font_size': 11,
        'visual_bg': '#ffffff',
        'visual_fg': '#1a1a1a',
        'visual_font': None,
        'visual_font_size': 11,
        # Page layout / zoom / Royal Purple Word 2010 theme
        'desk_bg': '#dcd6e8',
        'ribbon_bg': '#f3effa',
        'status_bg': '#3b1d6e',
        'status_fg': '#f5f3ff',
        'accent_purple': '#5b21b6',
        'page_layout': True,
        'page_size': 'Letter',
        'page_margin': 'Normal',
        'zoom_pct': 100,
        'typewriter_scrolling': False,
        # Blockquote customization (empty strings/0 mean inherit from active theme)
        'blockquote_font': '',
        'blockquote_size': 0,
        'blockquote_style': 'italic',
        'blockquote_fg': '',
        'blockquote_bg': '',
        'blockquote_bar_fg': '',
        'blockquote_padding': 16,
        'blockquote_spacing': 4,
        'blockquote_justify': 'left',
    }

    ZOOM_STEPS = [25, 50, 75, 90, 100, 110, 125, 150, 175, 200, 250, 300, 400]
    
    def __init__(self):
        super().__init__()
        
        # Initialize fonts now that Tkinter root exists
        global SANS_FONT, MONO_FONT, VISUAL_FONT
        if SANS_FONT is None:
            SANS_FONT = select_best_font(SANS_SERIF_FONTS, "TkDefaultFont")
        if MONO_FONT is None:
            MONO_FONT = select_best_font(MONOSPACE_FONTS, "TkFixedFont")
        if VISUAL_FONT is None:
            VISUAL_FONT = select_best_font(VISUAL_FONTS, SANS_FONT)
        
        self.title("MarkItDown Notepad")
        ui_scale = max(1.0, self.winfo_fpixels('1i') / 96.0)
        self.geometry(f"{int(1280 * ui_scale)}x{int(820 * ui_scale)}")
        self.minsize(int(760 * ui_scale), int(520 * ui_scale))
        self.configure(bg=self.DEFAULT_THEME.get('desk_bg', '#d9dce1'))
        
        # Application state - now tab-based
        self.tabs = []  # List of DocumentTab instances
        self.current_tab = None  # Currently active tab
        self.word_wrap = True  # Word wrap toggle (global setting)
        self.extracted_images_dir = None  # Directory for extracted images
        
        # Session manager for document caching
        self.session_manager = get_session_manager() if SESSION_MANAGER_AVAILABLE else None
        self._auto_save_job = None
        
        # Focus / full-screen mode state
        self.focus_mode_active = False
        self.focus_escape_binding = None
        self._pre_fs = None
        self._fs_hint = None
        
        # Recent files
        self.recent_files = []
        self.max_recent_files = 10
        self._load_recent_files()
        
        # Theme settings
        self.theme = self.DEFAULT_THEME.copy()
        self.theme['source_font'] = MONO_FONT
        self.theme['visual_font'] = VISUAL_FONT
        self._load_theme()
        
        # Zoom state (global, percent)
        self.zoom_pct = self._clamp_zoom(self.theme.get('zoom_pct', 100))
        self._zoom_updating = False
        self._zoom_save_job = None
        
        # Initialize MarkItDown
        if MARKITDOWN_AVAILABLE:
            self.md_converter = MarkItDown(enable_plugins=False)
        else:
            self.md_converter = None
        
        self.live_render_var = tk.BooleanVar(value=self.theme.get('live_render_enabled', True))
        self.page_layout_var = tk.BooleanVar(value=bool(self.theme.get('page_layout', True)))
        self.page_size_var = tk.StringVar(value=self.theme.get('page_size', 'Letter'))
        self.page_margin_var = tk.StringVar(value=self.theme.get('page_margin', 'Normal'))
        self.typewriter_var = tk.BooleanVar(value=bool(self.theme.get('typewriter_scrolling', False)))
        self.source_view_var = tk.BooleanVar(value=False)
        self.mode_var = tk.StringVar(value="visual")
        self._setup_ui()
        self._setup_bindings()
        self._restore_session()
        self._update_title()
        
        # Start auto-save timer for session
        self._start_auto_save()
    
    def _load_theme(self):
        """Load theme from settings file"""
        settings_path = Path.home() / '.markitdown_notepad_settings.json'
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    saved_theme = json.load(f)
                if isinstance(saved_theme, dict):
                    # Ignore empty font names so defaults survive
                    for key in ('source_font', 'visual_font'):
                        if not saved_theme.get(key):
                            saved_theme.pop(key, None)
                    self.theme.update(saved_theme)
            except Exception:
                pass
    
    def _save_theme(self):
        """Save theme to settings file"""
        settings_path = Path.home() / '.markitdown_notepad_settings.json'
        try:
            with open(settings_path, 'w') as f:
                json.dump(self.theme, f, indent=2)
        except:
            pass
    
    def _load_recent_files(self):
        """Load recent files list from settings"""
        recent_path = Path.home() / '.markitdown_recent_files.json'
        if recent_path.exists():
            try:
                with open(recent_path, 'r') as f:
                    self.recent_files = json.load(f)
            except:
                self.recent_files = []
    
    def _save_recent_files(self):
        """Save recent files list to settings"""
        recent_path = Path.home() / '.markitdown_recent_files.json'
        try:
            with open(recent_path, 'w') as f:
                json.dump(self.recent_files, f, indent=2)
        except:
            pass
    
    def _add_to_recent_files(self, filepath):
        """Add a file to the recent files list"""
        # Normalize path
        filepath = os.path.normpath(filepath)
        
        # Remove if already exists (will re-add at top)
        if filepath in self.recent_files:
            self.recent_files.remove(filepath)
        
        # Add to beginning
        self.recent_files.insert(0, filepath)
        
        # Trim to max
        self.recent_files = self.recent_files[:self.max_recent_files]
        
        # Save and update menu
        self._save_recent_files()
        self._update_recent_files_menu()
    
    def _update_recent_files_menu(self):
        """Update the recent files submenu"""
        # Clear existing items
        self.recent_menu.delete(0, tk.END)
        
        if self.recent_files:
            for filepath in self.recent_files:
                # Show just filename with full path in a truncated form
                display_name = os.path.basename(filepath)
                # Use lambda with default arg to capture filepath correctly
                self.recent_menu.add_command(
                    label=display_name,
                    command=lambda f=filepath: self._open_recent_file(f)
                )
            
            self.recent_menu.add_separator()
            self.recent_menu.add_command(label="Clear Recent Files", command=self._clear_recent_files)
        else:
            self.recent_menu.add_command(label="(No recent files)", state=tk.DISABLED)
    
    def _open_recent_file(self, filepath):
        """Open a file from the recent files list"""
        if not os.path.exists(filepath):
            if messagebox.askyesno("File Not Found", 
                                   f"File no longer exists:\n{filepath}\n\nRemove from recent files?"):
                if filepath in self.recent_files:
                    self.recent_files.remove(filepath)
                    self._save_recent_files()
                    self._update_recent_files_menu()
            return
        
        self.open_file(filepath)
    
    def _clear_recent_files(self):
        """Clear the recent files list"""
        self.recent_files = []
        self._save_recent_files()
        self._update_recent_files_menu()
        self._set_status("Recent files cleared")
    
    def _setup_ui(self):
        """Setup the user interface"""
        self._setup_styles()
        
        # Menu bar
        self._create_menu()
        
        # Toolbar
        self._create_toolbar()
        
        # Main content area with PanedWindow for editor and AI sidebar
        # Outer paned: [outline?] | [main_paned]
        self.outer_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.outer_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.outline_panel   = OutlinePanel(self.outer_paned,
                                             on_heading_click=self._jump_to_line,
                                             width=220)
        self.outline_panel.apply_theme(self.theme)
        self.outline_visible = False
        # outline_panel is NOT added to outer_paned until first toggle

        # Inner paned: editor | AI sidebar (unchanged)
        self.main_paned = ttk.PanedWindow(self.outer_paned, orient=tk.HORIZONTAL)
        self.outer_paned.add(self.main_paned, weight=1)
        
        # Left pane: Editor content with tabs
        self.content_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.content_frame, weight=1)

        # Docked non-blocking Find & Replace bar (packed above notebook when active)
        self.find_bar = DockedFindReplaceBar(self.content_frame, notepad=self)
        
        # Notebook widget for tabs
        self.notebook = ttk.Notebook(self.content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Configure tab events
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.notebook.bind("<Button-3>", self._show_tab_context_menu)
        self.notebook.bind("<Button-2>", self._on_tab_middle_click)
        
        # Right pane: AI Sidebar (initially hidden)
        self.ai_sidebar = None
        self.ai_sidebar_visible = False
        self._setup_ai_sidebar()
        
        # Status bar (packed before the paned area so it is never squeezed out)
        self._create_status_bar()
        self._update_zoom_widgets()
    
    def _setup_styles(self):
        """Configure ttk look & feel with MS Word 2010 Royal Purple styling."""
        style = ttk.Style()
        if 'clam' in style.theme_names():
            try:
                style.theme_use('clam')
            except Exception:
                pass

        for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont'):
            try:
                tkfont.nametofont(name).configure(family=SANS_FONT, size=10)
            except Exception:
                pass

        ribbon_bg = self.theme.get('ribbon_bg', '#f3effa')
        desk_bg = self.theme.get('desk_bg', '#dcd6e8')
        status_bg = self.theme.get('status_bg', '#3b1d6e')
        status_fg = self.theme.get('status_fg', '#f5f3ff')
        accent_purple = self.theme.get('accent_purple', '#5b21b6')

        style.configure("Ribbon.TFrame", background=ribbon_bg)
        style.configure("RibbonDeck.TFrame", background=ribbon_bg)
        style.configure("Toolbar.TButton", background=ribbon_bg, relief='flat', padding=(6, 3), width=0)
        style.map("Toolbar.TButton",
            background=[('pressed', '#ddd6fe'), ('active', '#ede9fe')],
            relief=[('pressed', 'groove'), ('!pressed', 'flat')]
        )
        style.configure("Toolbar.Toolbutton", background=ribbon_bg, padding=(8, 3))
        style.map("Toolbar.Toolbutton",
            background=[('selected', '#ddd6fe'), ('active', '#ede9fe')]
        )
        style.configure("TNotebook", background=desk_bg)
        style.configure("TNotebook.Tab", background=ribbon_bg, padding=(14, 5))
        style.map("TNotebook.Tab",
            background=[('selected', '#ffffff'), ('active', '#ede9fe')]
        )
        style.configure("StatusBar.TFrame", background=status_bg)
        style.configure("Status.TLabel", background=status_bg, foreground=status_fg, font=(SANS_FONT, 9))
        style.configure("StatusPill.TButton", background='#4c1d95', foreground='#ffffff', relief='flat', padding=(6, 1))
        style.map("StatusPill.TButton",
            background=[('pressed', '#6d28d9'), ('active', '#5b21b6')]
        )
        style.configure("Zoom.TButton", background=status_bg, foreground=status_fg, padding=(4, 0), width=2)
        style.map("Zoom.TButton",
            background=[('pressed', '#6d28d9'), ('active', '#4c1d95')]
        )
        style.configure('Vertical.TScrollbar', arrowsize=12)

    def _create_menu(self):
        """Create the menu bar"""
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)
        menubar = self.menubar  # Keep local reference for compatibility
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Document Tab", command=self.new_document_tab, accelerator="Ctrl+N")
        file_menu.add_command(label="New Spreadsheet Tab", command=self.new_spreadsheet_tab, accelerator="Ctrl+Shift+N")
        file_menu.add_command(label="Open...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Open as Spreadsheet...", command=self.open_as_spreadsheet, accelerator="Ctrl+Alt+O")
        file_menu.add_command(label="Open with MarkItDown...", command=self.open_with_markitdown, accelerator="Ctrl+Shift+O")
        
        # Recent files submenu
        self.recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Open Recent", menu=self.recent_menu)
        self._update_recent_files_menu()
        
        file_menu.add_separator()
        file_menu.add_command(label="Close Tab", command=self.close_tab, accelerator="Ctrl+W")
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_file_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Save Quick Snapshot (Checkpoint)", command=self.create_quick_snapshot, accelerator="Ctrl+Alt+S")
        file_menu.add_command(label="Document Snapshots (Checkpoints)...", command=self.open_snapshots_dialog, accelerator="Ctrl+Shift+P")
        file_menu.add_separator()
        
        # Export submenu
        export_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Export As", menu=export_menu)
        if EXPORT_AVAILABLE and WEASYPRINT_AVAILABLE:
            export_menu.add_command(label="PDF...", command=self._export_pdf, accelerator="Ctrl+Shift+E")
        else:
            export_menu.add_command(label="PDF... (install weasyprint)", state=tk.DISABLED)
        if EXPORT_AVAILABLE and EXPORT_MARKDOWN_AVAILABLE:
            export_menu.add_command(label="HTML...", command=self._export_html)
        else:
            export_menu.add_command(label="HTML... (install markdown)", state=tk.DISABLED)
        if EXPORT_AVAILABLE and DOCX_AVAILABLE:
            export_menu.add_command(label="DOCX (Word)...", command=self._export_docx)
        else:
            export_menu.add_command(label="DOCX... (install python-docx)", state=tk.DISABLED)
        
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_app, accelerator="Alt+F4")
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Cut", command=self.cut, accelerator="Ctrl+X")
        edit_menu.add_command(label="Copy", command=self.copy, accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste", command=self.paste, accelerator="Ctrl+V")
        edit_menu.add_separator()
        edit_menu.add_command(label="Select All", command=self.select_all, accelerator="Ctrl+A")
        edit_menu.add_separator()
        edit_menu.add_command(label="Find...", command=self.find_text, accelerator="Ctrl+F")
        edit_menu.add_command(label="Find & Replace...", command=self.find_replace, accelerator="Ctrl+H")
        edit_menu.add_command(label="Go to Line...", command=self.goto_line, accelerator="Ctrl+G")
        edit_menu.add_separator()
        edit_menu.add_command(label="Clean & Reflow Document", command=self.clean_current_document, accelerator="Ctrl+Shift+R")
        edit_menu.add_command(label="Clean & Reflow Selection", command=self.clean_selected_text)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Source View", variable=self.source_view_var,
                                  command=self._on_source_view_toggle, accelerator="Ctrl+E")
        view_menu.add_separator()
        
        # Word wrap toggle
        self.wrap_var = tk.BooleanVar(value=True)
        view_menu.add_checkbutton(label="Word Wrap", variable=self.wrap_var, 
                                   command=self._toggle_word_wrap, accelerator="Ctrl+Shift+W")
        
        # Typewriter scrolling toggle
        view_menu.add_checkbutton(label="Typewriter Scrolling (Keep Cursor Centered)",
                                   variable=self.typewriter_var,
                                   command=self._toggle_typewriter_scrolling,
                                   accelerator="Ctrl+Shift+T")
        
        # Large file mode
        self.large_file_var = tk.BooleanVar(value=False)
        view_menu.add_checkbutton(label="Large File Mode (faster)", variable=self.large_file_var,
                                   command=self._toggle_large_file_mode)
        
        view_menu.add_separator()
        view_menu.add_command(label="Zoom In", command=self.zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self.zoom_out, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset Zoom", command=self.zoom_reset, accelerator="Ctrl+0")
        view_menu.add_command(label="Zoom to Fit Width", command=self.zoom_fit_width, accelerator="Ctrl+Shift+0")
        
        view_menu.add_separator()
        
        # Page layout (Word-style sheet on a desk)
        view_menu.add_checkbutton(label="Page Layout", variable=self.page_layout_var,
                                  command=self._apply_page_settings)
        size_menu = tk.Menu(view_menu, tearoff=0)
        for name in PageView.PAGE_INCHES:
            size_menu.add_radiobutton(label=name, value=name, variable=self.page_size_var,
                                      command=self._apply_page_settings)
        view_menu.add_cascade(label="Page Size", menu=size_menu)
        margin_menu = tk.Menu(view_menu, tearoff=0)
        for name in PageView.MARGIN_INCHES:
            margin_menu.add_radiobutton(label=name, value=name, variable=self.page_margin_var,
                                        command=self._apply_page_settings)
        view_menu.add_cascade(label="Page Margins", menu=margin_menu)
        
        view_menu.add_separator()
        
        # Document caching toggle
        self.cache_var = tk.BooleanVar(value=True)
        if self.session_manager:
            self.cache_var.set(self.session_manager.is_caching_enabled())
        view_menu.add_checkbutton(
            label="Cache Documents (restore on restart)", 
            variable=self.cache_var,
            command=self._toggle_caching
        )
        
        view_menu.add_separator()
        
        # Focus Mode
        view_menu.add_command(label="Full Screen / Focus Mode", command=self._toggle_focus_mode, accelerator="F11")
        
        view_menu.add_separator()
        
        # AI Sidebar
        view_menu.add_command(label="AI Sidebar", command=self._toggle_ai_sidebar, accelerator="Ctrl+Shift+A")
        view_menu.add_command(label="AI Settings...", command=self._open_ai_settings)
        view_menu.add_separator()
        view_menu.add_command(label='Document Navigator & Outline',
                              command=self._toggle_outline_panel,
                              accelerator='Ctrl+Shift+L')
        view_menu.add_separator()
        view_menu.add_checkbutton(
            label='Live Re-render in Visual Mode',
            variable=self.live_render_var,
            command=self._toggle_live_render
        )
        
        # Format menu (for source mode)
        format_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Format", menu=format_menu)
        format_menu.add_command(label="Bold", command=lambda: self._insert_format("**", "**"), accelerator="Ctrl+B")
        format_menu.add_command(label="Italic", command=lambda: self._insert_format("*", "*"), accelerator="Ctrl+I")
        format_menu.add_command(label="Highlight", command=lambda: self._insert_format("==", "=="), accelerator="Ctrl+Shift+H")
        format_menu.add_command(label="Inline Code", command=lambda: self._insert_format("`", "`"), accelerator="Ctrl+`")
        format_menu.add_separator()
        format_menu.add_command(label="Heading 1", command=lambda: self._insert_line_prefix("# "))
        format_menu.add_command(label="Heading 2", command=lambda: self._insert_line_prefix("## "))
        format_menu.add_command(label="Heading 3", command=lambda: self._insert_line_prefix("### "))
        format_menu.add_command(label="Heading 4", command=lambda: self._insert_line_prefix("#### "))
        format_menu.add_separator()
        format_menu.add_command(label="Bullet List", command=lambda: self._insert_line_prefix("- "))
        format_menu.add_command(label="Numbered List", command=lambda: self._insert_line_prefix("1. "))
        format_menu.add_command(label="Blockquote", command=lambda: self._insert_line_prefix("> "), accelerator="Ctrl+Shift+Q")
        format_menu.add_command(label="Customize Blockquote Style...", command=self.open_blockquote_style_dialog)
        format_menu.add_separator()
        format_menu.add_command(label="Insert Table...", command=self.insert_table, accelerator="Alt+T")
        format_menu.add_command(label="Code Block", command=lambda: self._insert_text("\n```\n\n```\n"), accelerator="Ctrl+Shift+K")
        format_menu.add_command(label="Horizontal Rule", command=lambda: self._insert_text("\n---\n"))
        format_menu.add_separator()
        format_menu.add_command(label="Link", command=lambda: self._insert_format("[", "](url)"), accelerator="Ctrl+K")
        format_menu.add_command(label="Insert Reference / Anchor [@...]", command=lambda: self._insert_format("[@", "]"), accelerator="Ctrl+Shift+M")
        format_menu.add_command(label="Insert Footnote", command=self.insert_footnote, accelerator="Ctrl+Shift+F")
        format_menu.add_command(label="Math & Symbols...", command=self.open_math_symbol_picker, accelerator="Ctrl+Shift+Y")
        format_menu.add_separator()
        format_menu.add_command(label="Insert Image...", command=self.insert_image, accelerator="Ctrl+Shift+I")
        format_menu.add_command(label="Manage Images...", command=self.manage_images)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="MarkItDown Info", command=self.show_markitdown_info)
    
    def _create_toolbar(self):
        """Create the MS Word 2010 style two-deck ribbon toolbar with grouped tools and 1-2 sentence tooltips."""
        self.toolbar_frame = ttk.Frame(self, style="Ribbon.TFrame")
        self.toolbar_frame.pack(fill=tk.X, padx=5, pady=(4, 2))
        
        # Two distinct decks: Top Deck (Workflow & Workspace) and Bottom Deck (Typography & Formatting)
        self.ribbon_top_deck = ttk.Frame(self.toolbar_frame, style="RibbonDeck.TFrame")
        self.ribbon_top_deck.pack(fill=tk.X, padx=2, pady=(0, 2))
        
        self.ribbon_bottom_deck = ttk.Frame(self.toolbar_frame, style="RibbonDeck.TFrame")
        self.ribbon_bottom_deck.pack(fill=tk.X, padx=2, pady=(0, 2))
        self._ribbon_collapsed = False

        # -------------------------------------------------------------
        # DECK 1 (Top Deck): File, Workflow, View Mode, Panels
        # -------------------------------------------------------------
        top = self.ribbon_top_deck

        new_doc_btn = ttk.Button(top, text="\U0001f5cb New Doc", command=self.new_document_tab, style="Toolbar.TButton")
        new_doc_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(new_doc_btn, "Create a new blank Markdown document tab (Ctrl+N).")

        new_sheet_btn = ttk.Button(top, text="\u229e New Sheet", command=self.new_spreadsheet_tab, style="Toolbar.TButton")
        new_sheet_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(new_sheet_btn, "Create a new Excel-style spreadsheet tab (Ctrl+Shift+N) with formula evaluation and AI integration.")

        open_btn = ttk.Button(top, text="\U0001f4c2 Open", command=self.open_file, style="Toolbar.TButton")
        open_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(open_btn, "Open an existing Markdown (.md, .txt) file from your local disk (Ctrl+O).")

        open_sheet_btn = ttk.Button(top, text="\U0001f4ca Open Sheet", command=self.open_as_spreadsheet, style="Toolbar.TButton")
        open_sheet_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(open_sheet_btn, "Open a CSV, TSV, XLSX, or Markdown table file directly as an interactive spreadsheet (Ctrl+Alt+O).")

        convert_btn = ttk.Button(top, text="\u26a1 Convert", command=self.open_with_markitdown, style="Toolbar.TButton")
        convert_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(convert_btn, "Convert document formats (PDF, DOCX, PPTX, EPUB, XLSX) to clean Markdown using MarkItDown (Ctrl+Shift+O).")

        save_btn = ttk.Button(top, text="\U0001f4be Save", command=self.save_file, style="Toolbar.TButton")
        save_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(save_btn, "Save the current document to disk (Ctrl+S). If untitled, prompts for a destination file path.")

        snap_btn = ttk.Button(top, text="\U0001f552 Snapshots", command=self.open_snapshots_dialog, style="Toolbar.TButton")
        snap_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(snap_btn, "Open Document Snapshots / Checkpoints dialog (Ctrl+Alt+S / Ctrl+Shift+P) to compare, review diffs, or restore prior versions.")

        reflow_btn = ttk.Button(top, text="\u21bb Reflow", command=self.clean_current_document, style="Toolbar.TButton")
        reflow_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(reflow_btn, "Clean and reflow lines, fixing artificial line breaks, hyphenated breaks, and soft wrapped prose (Ctrl+Shift+R).")

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)

        self.source_view_cb = ttk.Checkbutton(
            top, text="Source View", variable=self.source_view_var, style="Toolbar.Toolbutton",
            command=self._on_source_view_toggle
        )
        self.source_view_cb.pack(side=tk.LEFT, padx=2)
        ToolTip(self.source_view_cb, "Toggle raw Markdown syntax view (Ctrl+E). When unticked, displays a seamless Word-like visual canvas with embedded tables and figures.")

        # Right-aligned controls in Deck 1
        self.ai_sidebar_btn = ttk.Button(top, text="\U0001f4ac AI Chat", command=self._toggle_ai_sidebar, style="Toolbar.TButton")
        self.ai_sidebar_btn.pack(side=tk.RIGHT, padx=1)
        ToolTip(self.ai_sidebar_btn, "Toggle the AI assistant sidebar (Ctrl+Shift+A) for drafting, rewriting, grammar correction, and document Q&A.")

        fullscreen_btn = ttk.Button(top, text="\u26f6 Full Screen", command=self._toggle_focus_mode, style="Toolbar.TButton")
        fullscreen_btn.pack(side=tk.RIGHT, padx=1)
        ToolTip(fullscreen_btn, "Toggle full-screen focus distraction-free writing mode (F11). Press Escape to exit.")

        outline_btn = ttk.Button(top, text="\U0001f4d1 Navigator", command=self._toggle_outline_panel, style="Toolbar.TButton")
        outline_btn.pack(side=tk.RIGHT, padx=1)
        ToolTip(outline_btn, "Toggle the Document Navigator sidebar (Ctrl+Shift+L) to browse headings, anchors, tables, and citations.")

        theme_btn = ttk.Button(top, text="\U0001f3a8 Theme", command=self.open_theme_dialog, style="Toolbar.TButton")
        theme_btn.pack(side=tk.RIGHT, padx=1)
        ToolTip(theme_btn, "Customize editor theme, fonts, margins, desk background color, and blockquote styling.")

        find_btn = ttk.Button(top, text="\U0001f50d Find", command=self.find_text, style="Toolbar.TButton")
        find_btn.pack(side=tk.RIGHT, padx=1)
        ToolTip(find_btn, "Open the docked search and replace bar (Ctrl+F / Ctrl+H) for fast in-document navigation.")

        self.ribbon_toggle_btn = ttk.Button(top, text="▲", width=3, command=self._toggle_ribbon_collapse, style="Toolbar.TButton")
        self.ribbon_toggle_btn.pack(side=tk.RIGHT, padx=2)
        self._ribbon_toggle_tip = ToolTip(self.ribbon_toggle_btn, "Collapse formatting ribbon deck (Ctrl+F1) to maximize writing area.")

        # -------------------------------------------------------------
        # DECK 2 (Bottom Deck): Typography, Structure, Insert Elements
        # -------------------------------------------------------------
        btm = self.ribbon_bottom_deck
        self._format_buttons = []

        def add_format_button(text, command, tip, width=None):
            kwargs = {'text': text, 'command': command, 'style': "Toolbar.TButton"}
            if width is not None and width > 0:
                kwargs['width'] = width
            btn = ttk.Button(btm, **kwargs)
            btn.pack(side=tk.LEFT, padx=1)
            ToolTip(btn, tip)
            self._format_buttons.append(btn)
            return btn

        # Typography
        add_format_button("\U0001d401", lambda: self._insert_format("**", "**"), "Format selected text as bold (**text**) (Ctrl+B).", width=3)
        add_format_button("\U0001d43c", lambda: self._insert_format("*", "*"), "Format selected text as italic (*text*) (Ctrl+I).", width=3)
        add_format_button("\U0001f58d HL", lambda: self._insert_format("==", "=="), "Format selected text with a highlight marker (==text==) (Ctrl+Shift+H).")
        add_format_button("</>", lambda: self._insert_format("`", "`"), "Format selected text as inline code (`code`) (Ctrl+`).")

        ttk.Separator(btm, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)

        # Headings & Blocks
        heading_mb = ttk.Menubutton(btm, text="H\u25be", style="Toolbar.TButton")
        heading_menu = tk.Menu(heading_mb, tearoff=0)
        heading_menu.add_command(label="H1 — Heading 1 (#)", command=lambda: self._insert_line_prefix("# "))
        heading_menu.add_command(label="H2 — Heading 2 (##)", command=lambda: self._insert_line_prefix("## "))
        heading_menu.add_command(label="H3 — Heading 3 (###)", command=lambda: self._insert_line_prefix("### "))
        heading_menu.add_command(label="H4 — Heading 4 (####)", command=lambda: self._insert_line_prefix("#### "))
        heading_mb.config(menu=heading_menu)
        heading_mb.pack(side=tk.LEFT, padx=1)
        ToolTip(heading_mb, "Insert a markdown section heading (levels H1 through H4) at the active cursor line.")
        self._format_buttons.append(heading_mb)

        quote_btn = add_format_button("\u275d Quote", lambda: self._insert_line_prefix("> "), "Insert a blockquote prefix (> text) on the active line (Ctrl+Shift+Q). Right-click to configure custom colors and styling.")
        quote_btn.bind("<Button-3>", lambda e: self.open_blockquote_style_dialog())

        add_format_button("\u2022 List", lambda: self._insert_line_prefix("- "), "Insert a bulleted list item (- item) at the current line.")
        add_format_button("1. List", lambda: self._insert_line_prefix("1. "), "Insert a numbered list item (1. item) at the current line.")
        add_format_button("\u229e Table", self.insert_table, "Insert a structured Markdown table (Alt+T). You can specify columns, rows, and header text.")
        add_format_button("{ }", lambda: self._insert_text("\n```\n\n```\n"), "Insert a fenced code block with optional syntax highlighting language tag (Ctrl+Shift+K).")
        add_format_button("\u2015 HR", lambda: self._insert_text("\n---\n"), "Insert a thematic horizontal divider rule (---) separating document sections.")

        ttk.Separator(btm, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)

        # Cross-References & Media
        add_format_button("\u2693 Anchor", lambda: self._insert_format("[@", "]"), "Insert a cross-reference / anchor tag [@name] (Ctrl+Shift+M). Ctrl+Click any [@name] tag to cycle through all occurrences in the document.")
        add_format_button("\U0001f517 Link", lambda: self._insert_format("[", "](url)"), "Insert a markdown hyperlink [text](url) (Ctrl+K) around the selected text or at cursor.")
        add_format_button("[^] Note", self.insert_footnote, "Insert a footnote reference marker [^1] and generate a footnote definition stub (Ctrl+Shift+F).")
        add_format_button("\u2211 Math", self.open_math_symbol_picker, "Open the mathematical symbol and Greek alphabet picker (Ctrl+Shift+Y) to insert Unicode math notation.")
        add_format_button("\U0001f5bc Image", self.insert_image, "Insert an image tag with optional extended caption and alignment metadata (Ctrl+Shift+I).")

    def _toggle_ribbon_collapse(self):
        """Toggle bottom ribbon deck collapse/expand state (Ctrl+F1)."""
        if getattr(self, '_ribbon_collapsed', False):
            self.ribbon_bottom_deck.pack(fill=tk.X, padx=2, pady=(0, 2))
            self._ribbon_collapsed = False
            self.ribbon_toggle_btn.config(text="▲")
            self._ribbon_toggle_tip.set_text("Collapse formatting ribbon deck (Ctrl+F1) to maximize writing area.")
        else:
            self.ribbon_bottom_deck.pack_forget()
            self._ribbon_collapsed = True
            self.ribbon_toggle_btn.config(text="▼")
            self._ribbon_toggle_tip.set_text("Expand formatting ribbon deck (Ctrl+F1) to show typography and insert tools.")

    def _set_format_buttons_enabled(self, enabled=True):
        """Formatting buttons remain active across both Source View and Visual View."""
        for btn in getattr(self, '_format_buttons', []):
            try:
                btn.state(['!disabled'])
            except Exception:
                pass
    
    def _create_status_bar(self):
        """Create the MS Word 2010 Royal Purple status bar with interactive pills and tooltips."""
        self.status_bar = ttk.Frame(self, style="StatusBar.TFrame")
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, before=self.outer_paned)
        
        self.status_label = ttk.Label(self.status_bar, text="Ready", style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Zoom controls (right-aligned): [-] slider [+] 100%
        self.zoom_label = ttk.Menubutton(self.status_bar, text="100%", width=6)
        self.zoom_label.pack(side=tk.RIGHT, padx=(2, 8), pady=1)
        zoom_menu = tk.Menu(self.zoom_label, tearoff=0)
        for pct in (50, 75, 100, 125, 150, 200):
            zoom_menu.add_command(label=f"{pct}%", command=lambda p=pct: self._set_zoom(p))
        zoom_menu.add_separator()
        zoom_menu.add_command(label="Fit Width", command=self.zoom_fit_width)
        self.zoom_label.config(menu=zoom_menu)
        ToolTip(self.zoom_label, "Current canvas zoom level. Click to select a preset zoom percentage or fit width (Ctrl+0 to reset).")
        
        zoom_in_btn = ttk.Button(self.status_bar, text="+", style="Zoom.TButton", command=self.zoom_in)
        zoom_in_btn.pack(side=tk.RIGHT, pady=1)
        ToolTip(zoom_in_btn, "Zoom in the document canvas by 10% (Ctrl++).")

        self.zoom_scale = ttk.Scale(self.status_bar, from_=25, to=300, orient=tk.HORIZONTAL, length=110,
                                    command=self._on_zoom_scale)
        self.zoom_scale.pack(side=tk.RIGHT, padx=4)
        ToolTip(self.zoom_scale, "Drag to adjust zoom between 25% and 400% of standard letter width. You can also hold Ctrl and scroll your mouse wheel.")

        zoom_out_btn = ttk.Button(self.status_bar, text="−", style="Zoom.TButton", command=self.zoom_out)
        zoom_out_btn.pack(side=tk.RIGHT, pady=1)
        ToolTip(zoom_out_btn, "Zoom out the document canvas by 10% (Ctrl+-).")
        
        ttk.Separator(self.status_bar, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=3)
        
        self.mode_label = ttk.Label(self.status_bar, text="Visual View", style="Status.TLabel", cursor="hand2")
        self.mode_label.pack(side=tk.RIGHT, padx=8)
        self.mode_label.bind("<Button-1>", lambda e: self._on_status_mode_click())
        ToolTip(self.mode_label, "Current document editing mode. Click to toggle between Visual View and raw Markdown Source View (Ctrl+E).")
        
        self.wrap_label = ttk.Label(self.status_bar, text="Wrap: On", style="Status.TLabel", cursor="hand2")
        self.wrap_label.pack(side=tk.RIGHT, padx=8)
        self.wrap_label.bind("<Button-1>", lambda e: self.toggle_word_wrap())
        ToolTip(self.wrap_label, "Soft word wrap status. Click to toggle between visual line wrapping and unwrapped continuous lines.")
        
        # Word and character count
        self.word_count_label = ttk.Label(self.status_bar, text="Words: 0 | Chars: 0", style="Status.TLabel", cursor="hand2")
        self.word_count_label.pack(side=tk.RIGHT, padx=8)
        self.word_count_label.bind("<Button-1>", lambda e: self.open_statistics_dialog())
        ToolTip(self.word_count_label, "Document word and character counts. Click to open comprehensive Document Statistics with reading time and structure analysis.")
        
        self.position_label = ttk.Label(self.status_bar, text="Line 1, Col 1", style="Status.TLabel", cursor="hand2")
        self.position_label.pack(side=tk.RIGHT, padx=8)
        self.position_label.bind("<Button-1>", lambda e: self.goto_line())
        ToolTip(self.position_label, "Current cursor line and column position. Click to jump to a specific line number (Ctrl+G).")
        
        # MarkItDown status
        md_status = "MarkItDown: ✓" if MARKITDOWN_AVAILABLE else "MarkItDown: ✗"
        self.md_status_label = ttk.Label(self.status_bar, text=md_status, style="Status.TLabel", cursor="hand2")
        self.md_status_label.pack(side=tk.RIGHT, padx=8)
        self.md_status_label.bind("<Button-1>", lambda e: self.show_markitdown_info())
        ToolTip(self.md_status_label, "MarkItDown document conversion engine availability. Click for supported format information and installation instructions.")

    def open_statistics_dialog(self):
        """Open comprehensive document statistics dialog."""
        if self.current_tab:
            DocumentStatisticsDialog(self, self.current_tab)

    def _on_status_mode_click(self):
        """Toggle source view from status bar click."""
        self.source_view_var.set(not self.source_view_var.get())
        self._on_source_view_toggle()

    def _setup_bindings(self):
        """Setup keyboard bindings"""
        self.bind("<Control-F1>", lambda e: self._toggle_ribbon_collapse())
        # Neutralize conflicting default Text class bindings (e.g. Ctrl+I inserting tab)
        try:
            self.unbind_class("Text", "<Control-Key-i>")
            self.unbind_class("Text", "<Control-i>")
        except Exception:
            pass

        self.bind("<Control-n>", lambda e: self.new_document_tab())
        self.bind("<Control-N>", lambda e: self.new_spreadsheet_tab())
        self.bind("<Control-Shift-N>", lambda e: self.new_spreadsheet_tab())
        self.bind("<Control-Shift-Key-N>", lambda e: self.new_spreadsheet_tab())
        self.bind("<Control-o>", lambda e: self.open_file())
        self.bind("<Control-Alt-o>", lambda e: self.open_as_spreadsheet())
        self.bind("<Control-Alt-O>", lambda e: self.open_as_spreadsheet())
        self.bind("<Control-O>", lambda e: self.open_with_markitdown())  # Ctrl+Shift+O
        self.bind("<Control-s>", lambda e: self.save_file())
        self.bind("<Control-S>", lambda e: self.save_file_as())  # Ctrl+Shift+S
        self.bind("<Control-Alt-s>", self._doc_shortcut(self.create_quick_snapshot))
        self.bind("<Control-Alt-S>", self._doc_shortcut(self.create_quick_snapshot))
        self.bind("<Control-P>", self._doc_shortcut(self.open_snapshots_dialog))
        self.bind("<Control-Shift-P>", self._doc_shortcut(self.open_snapshots_dialog))
        self.bind("<Control-I>", lambda e: self.insert_image())  # Ctrl+Shift+I
        self.bind("<Control-w>", lambda e: self.close_tab())  # Close tab
        self.bind("<Control-W>", lambda e: self._toggle_word_wrap())  # Ctrl+Shift+W for word wrap
        self.bind("<Control-T>", lambda e: self._toggle_typewriter_scrolling())  # Ctrl+Shift+T
        self.bind("<Control-Shift-T>", lambda e: self._toggle_typewriter_scrolling())
        self.bind("<Control-z>", self._doc_shortcut(self.undo))
        self.bind("<Control-y>", self._doc_shortcut(self.redo))
        self.bind("<Control-a>", self._doc_shortcut(self.select_all))
        self.bind("<Control-v>", self._doc_shortcut(self.paste))
        self.bind("<Control-V>", self._doc_shortcut(self.paste))
        self.bind("<Control-f>", lambda e: self.find_text())
        self.bind("<Control-h>", lambda e: self.find_replace())
        self.bind("<Control-g>", lambda e: self.goto_line())
        self.bind("<Control-b>", self._doc_shortcut(lambda: self._insert_format("**", "**")))
        self.bind("<Control-B>", self._doc_shortcut(lambda: self._insert_format("**", "**")))
        self.bind("<Control-i>", self._doc_shortcut(lambda: self._insert_format("*", "*")))
        self.bind("<Control-Key-i>", self._doc_shortcut(lambda: self._insert_format("*", "*")))
        self.bind("<Control-grave>", self._doc_shortcut(lambda: self._insert_format("`", "`")))
        self.bind("<Control-H>", self._doc_shortcut(lambda: self._insert_format("==", "==")))
        self.bind("<Control-Shift-H>", self._doc_shortcut(lambda: self._insert_format("==", "==")))
        self.bind("<Control-Shift-h>", self._doc_shortcut(lambda: self._insert_format("==", "==")))
        self.bind("<Control-Shift-Key-H>", self._doc_shortcut(lambda: self._insert_format("==", "==")))
        self.bind("<Control-k>", self._doc_shortcut(lambda: self._insert_format("[", "](url)")))
        self.bind("<Control-K>", self._doc_shortcut(lambda: self._insert_text("\n```\n\n```\n")))
        self.bind("<Control-Q>", self._doc_shortcut(lambda: self._insert_line_prefix("> ")))
        self.bind("<Control-M>", self._doc_shortcut(lambda: self._insert_format("[@", "]")))
        self.bind("<Control-F>", self._doc_shortcut(self.insert_footnote))
        self.bind("<Control-Y>", self._doc_shortcut(self.open_math_symbol_picker))
        self.bind("<Alt-t>", self._doc_shortcut(self.insert_table))
        self.bind("<Alt-T>", self._doc_shortcut(self.insert_table))
        self.bind("<Control-Key-1>", lambda e: self._show_source_mode())
        self.bind("<Control-Key-2>", lambda e: self._show_visual_mode())
        self.bind("<Control-e>", lambda e: self._toggle_mode())
        self.bind("<F3>", lambda e: self._find_next_f3())
        
        # Tab navigation
        self.bind("<Control-Tab>", lambda e: self._next_tab())
        self.bind("<Control-Shift-Tab>", lambda e: self._prev_tab())
        
        # Zoom bindings
        self.bind("<Control-plus>", lambda e: self.zoom_in())
        self.bind("<Control-equal>", lambda e: self.zoom_in())  # For keyboards without numpad
        self.bind("<Control-minus>", lambda e: self.zoom_out())
        self.bind("<Control-0>", lambda e: self.zoom_reset())
        self.bind("<Control-parenright>", lambda e: self.zoom_fit_width())  # Ctrl+Shift+0
        self.bind("<Control-Shift-Key-0>", lambda e: self.zoom_fit_width())
        
        # AI sidebar toggle
        self.bind("<Control-A>", lambda e: self._toggle_ai_sidebar())  # Ctrl+Shift+A
        
        # Document Outline toggle
        self.bind("<Control-L>", lambda e: self._toggle_outline_panel())
        
        # Export shortcut
        self.bind("<Control-E>", lambda e: self._export_pdf())  # Ctrl+Shift+E

        # Clean & Reflow shortcut
        self.bind("<Control-R>", lambda e: self.clean_current_document())  # Ctrl+Shift+R
        
        # Focus mode toggle
        self.bind("<F11>", lambda e: self._toggle_focus_mode())
        
        # Navigation back (Alt+Left)
        self.bind("<Alt-Left>", lambda e: self._navigate_back())
        self.bind("<Alt-Key-Left>", lambda e: self._navigate_back())
        
        # Position update state
        self._position_update_pending = False
        
        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.quit_app)

    def _schedule_syntax_highlight(self, event=None):
        if not getattr(self, '_syntax_hl_pending', False):
            self._syntax_hl_pending = True
            self.after(300, self._do_syntax_highlight)

    def _do_syntax_highlight(self):
        self._syntax_hl_pending = False
        if self.current_tab:
            self.current_tab._apply_syntax_highlighting()

    def _navigate_back(self):
        """Navigate back to previous jump location (Alt+Left) in source/visual mode."""
        tab = self.current_tab
        if tab:
            nav_hist = getattr(tab, '_nav_history', None)
            if nav_hist:
                prev_idx = nav_hist.pop()
                try:
                    editor = tab.source_editor
                    editor.mark_set(tk.INSERT, prev_idx)
                    editor.see(prev_idx)
                    editor.focus_set()
                    tab._flash_source_line(prev_idx)
                    self._set_status("Returned to previous location")
                    return "break"
                except Exception:
                    pass
            return tab.visual_viewer.navigate_back()
        return None

    def _jump_to_line(self, lineno: int):
        if not self.current_tab:
            return
        editor = self.current_tab.source_editor
        try:
            editor.see(f'{lineno}.0')
            editor.mark_set(tk.INSERT, f'{lineno}.0')
            editor.focus_set()
        except Exception:
            pass

    def _toggle_outline_panel(self):
        if self.outline_visible:
            self.outer_paned.forget(self.outline_panel)
            self.outline_visible = False
        else:
            self.outer_paned.insert(0, self.outline_panel, weight=0)
            try:
                self.outer_paned.paneconfig(self.outline_panel, minsize=240, width=255)
            except Exception:
                pass
            self.outline_visible = True
            self._schedule_outline_update()

    def _schedule_outline_update(self):
        if self.outline_visible and self.current_tab:
            self.outline_panel.schedule_update(self.current_tab.get_content())
    
    # === Session Management Methods ===
    
    def _restore_session(self):
        """Restore session from previous run, or create empty tab"""
        restored = False
        
        if self.session_manager and self.session_manager.has_saved_session():
            state = self.session_manager.load_session()
            
            if state.caching_enabled and state.tabs:
                # Restore window geometry
                if state.window_geometry:
                    try:
                        self.geometry(state.window_geometry)
                    except:
                        pass
                
                # Restore all tabs
                for tab_state in state.tabs:
                    t_type = getattr(tab_state, 'tab_type', 'document')
                    if t_type == 'spreadsheet' and SPREADSHEET_AVAILABLE:
                        tab = SpreadsheetTab(
                            self.notebook,
                            theme=self.theme,
                            on_modified_callback=lambda: self._on_tab_modified()
                        )
                        tab.file_path = tab_state.file_path
                        tab.set_content(tab_state.content)
                        tab.is_modified = tab_state.is_modified
                        tab.snapshots = list(getattr(tab_state, 'snapshots', []) or [])
                        tab_name = tab.get_display_name()
                        if tab.is_modified:
                            tab_name = f"*{tab_name}"
                        self.notebook.add(tab, text=tab_name)
                        self.tabs.append(tab)
                        continue

                    tab = DocumentTab(
                        self.notebook,
                        theme=self.theme,
                        on_modified_callback=lambda: self._on_tab_modified(),
                        on_content_change=lambda c: self._on_visual_edit(c)
                    )
                    
                    # Set tab properties
                    tab.file_path = tab_state.file_path
                    tab.set_content(tab_state.content)
                    tab.is_modified = tab_state.is_modified
                    tab.current_mode = tab_state.current_mode
                    tab.snapshots = list(getattr(tab_state, 'snapshots', []) or [])
                    
                    # Add to notebook
                    tab_name = tab.get_display_name()
                    if tab.is_modified:
                        tab_name = f"*{tab_name}"
                    self.notebook.add(tab, text=tab_name)
                    self.tabs.append(tab)
                    
                    # Setup bindings
                    self._setup_tab_bindings(tab)
                    tab.set_zoom(self.zoom_pct / 100.0)
                    
                    # Restore cursor position
                    try:
                        tab.source_editor.mark_set(tk.INSERT, tab_state.cursor_position)
                        tab.source_editor.see(tab_state.cursor_position)
                    except:
                        pass
                    
                    # Restore scroll position
                    try:
                        tab.source_editor.yview_moveto(tab_state.scroll_position)
                    except:
                        pass
                    
                    # Update visual viewer base path
                    if tab.file_path:
                        tab.visual_viewer.set_base_path(os.path.dirname(tab.file_path))
                    
                    tab.visual_viewer.live_render_enabled = self.theme.get('live_render_enabled', True)
                    
                    # Show correct mode
                    if tab_state.current_mode == "visual":
                        tab.switch_mode("visual", self.theme)
                    else:
                        tab.switch_mode("source", self.theme)
                
                # Select the active tab
                if state.active_tab_index < len(self.tabs):
                    self.notebook.select(self.tabs[state.active_tab_index])
                    self.current_tab = self.tabs[state.active_tab_index]
                elif self.tabs:
                    self.current_tab = self.tabs[0]
                
                # Update cache checkbox
                self.cache_var.set(state.caching_enabled)
                
                restored = True
                self._set_status(f"Restored {len(self.tabs)} tab(s) from previous session")
        
        # If no session restored, create empty tab
        if not restored:
            self.new_tab()
    
    def _save_session(self):
        """Save current session state"""
        if not self.session_manager:
            return
        
        # Clear existing tabs in session
        self.session_manager.state.tabs = []
        
        # Save each tab's state
        for i, tab in enumerate(self.tabs):
            t_type = getattr(tab, 'tab_type', 'document')
            if t_type == 'spreadsheet':
                tab_state = TabState(
                    file_path=tab.file_path,
                    content=tab.get_content(),
                    cursor_position="A1",
                    scroll_position=0.0,
                    current_mode="spreadsheet",
                    is_modified=tab.is_modified,
                    snapshots=getattr(tab, 'snapshots', []),
                    tab_type="spreadsheet"
                )
            else:
                tab_state = TabState(
                    file_path=tab.file_path,
                    content=tab.get_content(),
                    cursor_position=tab.source_editor.index(tk.INSERT),
                    scroll_position=tab.source_editor.yview()[0],
                    current_mode=tab.current_mode,
                    is_modified=tab.is_modified,
                    snapshots=getattr(tab, 'snapshots', []),
                    tab_type="document"
                )
            self.session_manager.state.tabs.append(tab_state)
        
        # Save active tab index
        if self.current_tab and self.current_tab in self.tabs:
            self.session_manager.set_active_tab(self.tabs.index(self.current_tab))
        
        # Save window geometry
        self.session_manager.set_window_geometry(self.geometry())
        
        # Save to file
        self.session_manager.save_session()
    
    def _start_auto_save(self):
        """Start periodic auto-save of session"""
        def auto_save():
            if self.session_manager and self.session_manager.is_caching_enabled():
                self._save_session()
            # Schedule next auto-save (every 30 seconds)
            self._auto_save_job = self.after(30000, auto_save)
        
        # Start the auto-save cycle
        self._auto_save_job = self.after(30000, auto_save)
    
    def _toggle_caching(self):
        """Toggle document caching on/off"""
        if self.session_manager:
            enabled = self.cache_var.get()
            self.session_manager.set_caching_enabled(enabled)
            
            if enabled:
                self._save_session()
                self._set_status("Document caching enabled - session will be restored on restart")
            else:
                self._set_status("Document caching disabled - documents will not be cached")
    
    # === AI Sidebar Methods ===
    
    def _setup_ai_sidebar(self):
        """Setup the AI chat sidebar"""
        if not AI_CHAT_AVAILABLE:
            return
        
        # Create sidebar frame (will be added to paned window when shown)
        self.ai_sidebar_frame = ttk.Frame(self.main_paned)
        
        # Create the chat sidebar widget
        self.ai_sidebar = ChatSidebar(
            self.ai_sidebar_frame,
            get_document_content_callback=self._get_current_document_content,
            get_document_images_callback=self._get_current_document_images,
            apply_edit_callback=self._on_apply_edit_from_chat
        )
        self.ai_sidebar.pack(fill=tk.BOTH, expand=True)
        self.ai_sidebar.apply_theme(self.theme)

    def _focus_in_chat(self):
        """True when keyboard focus is inside the AI chat sidebar."""
        sb = getattr(self, 'ai_sidebar', None)
        if sb is None:
            return False
        try:
            w = self.focus_get()
        except (KeyError, tk.TclError):
            return False
        while w is not None:
            if w is sb:
                return True
            w = getattr(w, 'master', None)
        return False

    def _doc_shortcut(self, func):
        """Wrap a document-level shortcut so it is ignored while the chat has focus."""
        def handler(event=None):
            if self._focus_in_chat():
                return None
            return func()
        return handler
    
    def _toggle_ai_sidebar(self):
        """Toggle AI sidebar visibility"""
        if not AI_CHAT_AVAILABLE:
            messagebox.showwarning("AI Chat Unavailable", 
                "AI chat module not available.\nEnsure ai_chat.py is present and anthropic is installed.")
            return
        
        if self.ai_sidebar_visible:
            # Hide sidebar (save any in-progress Edit Chat session first)
            if self.ai_sidebar:
                self.ai_sidebar.commit_edit_mode()
            self.main_paned.forget(self.ai_sidebar_frame)
            self.ai_sidebar_visible = False
            self.ai_sidebar_btn.config(text="AI Chat")
            self._set_status("AI sidebar hidden")
        else:
            # Show sidebar
            self.main_paned.add(self.ai_sidebar_frame, weight=0)
            # Set initial width for sidebar (about 350 pixels)
            self.update_idletasks()
            total_width = self.main_paned.winfo_width()
            if total_width > 400:
                sash_pos = total_width - 380
                try:
                    self.main_paned.sashpos(0, sash_pos)
                except:
                    pass
            self.ai_sidebar_visible = True
            self.ai_sidebar_btn.config(text="AI Chat [ON]")
            self._set_status("AI sidebar shown - Configure API key in Settings")
            
            # Update sidebar with current document context
            self._update_ai_sidebar_context()
    
    def _update_ai_sidebar_context(self):
        """Update AI sidebar with current document context"""
        if not self.ai_sidebar or not self.ai_sidebar_visible:
            return
        
        # Get current document ID for chat history
        doc_id = None
        if self.current_tab and self.current_tab.file_path:
            doc_id = self.current_tab.file_path
        elif self.current_tab:
            # Use a hash of content for unsaved documents
            content = self.current_tab.get_content()
            if content:
                doc_id = f"unsaved_{hash(content[:100])}"
        
        self.ai_sidebar.set_document_id(doc_id)
    
    def _get_current_document_content(self):
        """Get current document content for AI context"""
        if not self.current_tab:
            return ""
        if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
            return self.current_tab.get_ai_context()
        return self.current_tab.get_content()
    
    def _get_current_document_images(self):
        """Get images from current document for AI context"""
        images = []
        if not self.current_tab or not PIL_AVAILABLE:
            return images
        
        content = self.current_tab.get_content()
        
        # Extract base64 images
        base64_pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        for match in re.finditer(base64_pattern, content):
            alt_text = match.group(1)
            img_type = match.group(2)
            img_data = match.group(3)
            try:
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                images.append({'image': img, 'alt': alt_text, 'type': img_type})
            except:
                pass
        
        # Extract file path images
        file_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        base_path = ""
        if self.current_tab.file_path:
            base_path = os.path.dirname(self.current_tab.file_path)
        
        for match in re.finditer(file_pattern, content):
            alt_text = match.group(1)
            path = match.group(2)
            
            # Skip base64 images (already handled)
            if path.startswith('data:'):
                continue
            
            # Resolve relative paths
            if base_path and not os.path.isabs(path):
                path = os.path.join(base_path, path)
            
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    ext = os.path.splitext(path)[1].lower()
                    img_type = ext.lstrip('.') if ext else 'png'
                    if img_type == 'jpg':
                        img_type = 'jpeg'
                    images.append({'image': img, 'alt': alt_text, 'type': img_type})
                except:
                    pass
        
        return images
    
    def _open_ai_settings(self):
        """Open AI settings dialog"""
        if not AI_CHAT_AVAILABLE:
            messagebox.showwarning("AI Chat Unavailable", 
                "AI chat module not available.\nEnsure ai_chat.py is present.")
            return

        def _on_settings_saved():
            if self.ai_sidebar:
                self.ai_sidebar.reload_settings()

        AISettingsDialog(self, on_save=_on_settings_saved)

    # === Right-Click Context Menu ===

    def _show_editor_context_menu(self, event):
        """Show context menu on right-click in the editor"""
        menu = tk.Menu(self, tearoff=0)
        
        # Determine which widget was clicked
        widget = event.widget
        
        # Standard editing actions
        menu.add_command(label="Cut", command=self.cut, accelerator="Ctrl+X")
        menu.add_command(label="Copy", command=self.copy, accelerator="Ctrl+C")
        menu.add_command(label="Paste", command=self.paste, accelerator="Ctrl+V")
        menu.add_separator()
        menu.add_command(label="Select All", command=self.select_all, accelerator="Ctrl+A")
        
        # Check if there's a text selection
        has_selection = False
        selected_text = ""
        try:
            selected_text = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            has_selection = bool(selected_text.strip())
        except tk.TclError:
            pass
        
        # Formatting actions
        if has_selection and self.current_tab:
            menu.add_separator()
            menu.add_command(label="Bold", command=lambda: self._insert_format("**", "**"), accelerator="Ctrl+B")
            menu.add_command(label="Italic", command=lambda: self._insert_format("*", "*"), accelerator="Ctrl+I")
            menu.add_command(label="Highlight", command=lambda: self._insert_format("==", "=="), accelerator="Ctrl+Shift+H")

        # AI-powered actions (only when text is selected and AI is available)
        if has_selection and AI_CHAT_AVAILABLE:
            menu.add_separator()
            
            # Get enabled context actions from settings
            ai_settings = get_ai_settings()
            actions = ai_settings.get_enabled_context_actions()
            
            for action in actions:
                name = action.get('name', '')
                prompt = action.get('prompt', '')
                
                if name == 'Transfer to Chat':
                    menu.add_command(
                        label=f"\U0001f4ac {name}",
                        command=lambda w=widget: self._transfer_to_chat(w)
                    )
                elif prompt:
                    menu.add_command(
                        label=f"\U0001f916 {name}",
                        command=lambda w=widget, n=name, p=prompt: self._run_context_action(w, n, p)
                    )
        
        # Show the menu
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _get_selection_info(self, widget):
        """Get the selected text and its indices from a text widget"""
        try:
            start = widget.index(tk.SEL_FIRST)
            end = widget.index(tk.SEL_LAST)
            text = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            return text, start, end
        except tk.TclError:
            return None, None, None

    def _transfer_to_chat(self, widget):
        """Transfer selected text to the AI chat sidebar"""
        text, start, end = self._get_selection_info(widget)
        if not text:
            return
        
        # Open sidebar if not visible
        if not self.ai_sidebar_visible:
            self._toggle_ai_sidebar()
        
        # Set pending selection in the sidebar
        if self.ai_sidebar:
            self.ai_sidebar.set_pending_selection(text, start, end)
            self._set_status(f"Selection transferred to chat ({len(text)} chars)")

    def _run_context_action(self, widget, action_name: str, prompt_template: str):
        """Run an AI context menu action on selected text"""
        text, start, end = self._get_selection_info(widget)
        if not text:
            return
        
        if not AI_CHAT_AVAILABLE:
            messagebox.showwarning("AI Unavailable", "AI chat module is not available.")
            return
        
        ai_settings = get_ai_settings()
        if not ai_settings.is_configured():
            messagebox.showwarning("AI Not Configured",
                "Please configure your AI provider and API key in AI Settings first.")
            return
        
        # Build the prompt
        prompt = prompt_template.replace('{selection}', text)
        
        # Get or create LLM client
        from ai_chat import get_llm_client
        llm_client = get_llm_client(ai_settings)
        if not llm_client:
            messagebox.showerror("Error", "Could not initialize AI client.")
            return
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Open the diff panel
        if DIFF_PANEL_AVAILABLE:
            DiffPanel(
                self,
                original_text=text,
                start_index=start,
                end_index=end,
                action_name=action_name,
                on_accept=self._apply_diff,
                llm_client=llm_client,
                settings=ai_settings,
                messages=messages,
                theme=self.theme,
            )
        else:
            messagebox.showwarning("Unavailable", "Diff panel module not found.")

    def _apply_diff(self, replacement_text, start_index, end_index, original_text, apply_mode="replace"):
        """Apply a diff replacement or insertion to the current document."""
        if not self.current_tab:
            return
        
        editor = self.current_tab.source_editor
        
        if apply_mode == "cursor":
            insert_pos = editor.index(tk.INSERT)
            editor.insert(insert_pos, replacement_text)
            action_desc = f"Inserted at cursor ({len(replacement_text)} chars)"
        elif apply_mode == "append":
            existing = editor.get('1.0', 'end-1c')
            prefix = "\n\n" if existing.strip() else ""
            editor.insert(tk.END, prefix + replacement_text)
            action_desc = f"Appended to document ({len(replacement_text)} chars)"
        elif apply_mode == "full":
            editor.delete('1.0', tk.END)
            editor.insert('1.0', replacement_text)
            action_desc = f"Replaced entire document ({len(replacement_text)} chars)"
        else:
            # Default: replace target range
            start_idx = start_index or '1.0'
            end_idx = end_index or tk.END
            try:
                current_text = editor.get(start_idx, end_idx)
                if end_idx == tk.END:
                    current_text = current_text.rstrip('\n')
            except tk.TclError:
                messagebox.showwarning("Warning", "Could not locate the original text position.")
                return
            
            if original_text and current_text.rstrip('\n') != original_text.rstrip('\n'):
                if not messagebox.askyesno("Text Changed",
                    "The original text has been modified since the suggestion was generated.\n\n"
                    "Apply the replacement anyway?"):
                    return
            
            editor.delete(start_idx, end_idx)
            editor.insert(start_idx, replacement_text)
            action_desc = f"Applied edit ({len(replacement_text)} chars)"
        
        # Mark as modified and refresh UI state
        self.current_tab.is_modified = True
        self._update_tab_title(self.current_tab)
        self._schedule_syntax_highlight()
        self._schedule_outline_update()
        
        # Update visual viewer if in visual mode
        if self.current_tab.current_mode == "visual":
            content = self.current_tab.source_editor.get('1.0', tk.END).rstrip()
            self.current_tab.visual_viewer.set_content(content)
        
        self._set_status(action_desc)

    def _on_apply_edit_from_chat(self, original_text, replacement_text, start_index, end_index):
        """Callback from ChatSidebar 'Apply to Document' or 'Review & Apply' — opens diff panel or spreadsheet dialog."""
        if not self.current_tab:
            return

        if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
            patches = SpreadsheetPatchParser.parse_patches(replacement_text)
            if patches:
                def _do_apply_patches(selected):
                    old_cells = {}
                    new_cells = {}
                    for coord_str, val in selected.items():
                        pos = parse_cell_ref(coord_str)
                        if pos:
                            r, c = pos
                            old_cells[(r, c)] = self.current_tab.engine.get_raw(r, c)
                            new_cells[(r, c)] = val
                            self.current_tab.engine.set_cell(r, c, val)
                    self.current_tab.grid_widget._save_undo_state({"type": "batch_cells", "old_cells": old_cells, "new_cells": new_cells})
                    self.current_tab.engine.recalculate_all()
                    self.current_tab.grid_widget.recompute_bounds()
                    self.current_tab.grid_widget.redraw()
                    self.current_tab.is_modified = True
                    self._update_tab_title(self.current_tab)
                    self._set_status(f"Applied {len(selected)} spreadsheet cell changes from AI")

                SpreadsheetDiffDialog(self, self.current_tab, patches, on_apply=_do_apply_patches)
                return
            else:
                messagebox.showinfo(
                    "Spreadsheet Edits",
                    "No structured cell edits (e.g. C3: value or spreadsheet-edits block) were found in this AI message.",
                    parent=self
                )
                self._set_status("No structured spreadsheet patches found in AI message")
                return

        default_mode = "replace"
        if original_text is None or start_index is None:
            # Check if the user currently has text selected in the active editor
            editor = self.current_tab.source_editor
            sel_text, sel_start, sel_end = self._get_selection_info(editor)
            if sel_text:
                original_text = sel_text
                start_index = sel_start
                end_index = sel_end
                default_mode = "replace"
            else:
                original_text = self.current_tab.get_content()
                start_index = "1.0"
                end_index = tk.END
                default_mode = "replace" if original_text.strip() else "cursor"
        
        if not DIFF_PANEL_AVAILABLE:
            self._apply_diff(replacement_text, start_index, end_index, original_text, default_mode)
            return
        
        panel = DiffPanel(
            self,
            original_text=original_text,
            start_index=start_index,
            end_index=end_index,
            action_name="Review & Apply from Chat",
            on_accept=self._apply_diff,
            theme=self.theme,
            default_apply_mode=default_mode,
        )
        panel.set_suggestion(replacement_text)

    # === Document Snapshots (Checkpoints) Methods ===

    def create_quick_snapshot(self, name: Optional[str] = None):
        """Create a named snapshot checkpoint of the current document."""
        if not self.current_tab:
            return
        from datetime import datetime
        content = self.current_tab.get_content()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        short_time = datetime.now().strftime("%H:%M:%S")
        if not hasattr(self.current_tab, 'snapshots') or self.current_tab.snapshots is None:
            self.current_tab.snapshots = []
        idx = len(self.current_tab.snapshots) + 1
        snap_name = name.strip() if (name and name.strip()) else f"Checkpoint #{idx} ({short_time})"
        word_count = len(content.split()) if content.strip() else 0
        snapshot = {
            'name': snap_name,
            'timestamp': now_str,
            'content': content,
            'word_count': word_count,
        }
        self.current_tab.snapshots.append(snapshot)
        if self.session_manager and self.session_manager.is_caching_enabled():
            self._save_session()
        self._set_status(f"Saved snapshot: '{snap_name}' ({word_count} words)")
        return snapshot

    def open_snapshots_dialog(self):
        """Open the Document Snapshots / Checkpoints manager dialog."""
        if not self.current_tab:
            return
        SnapshotDialog(self, self.current_tab)

    # === Export Methods ===

    def _export_pdf(self):
        """Export current document to PDF"""
        if not self.current_tab:
            return
        if not EXPORT_AVAILABLE or not WEASYPRINT_AVAILABLE:
            messagebox.showwarning("Unavailable",
                "PDF export requires 'weasyprint' and 'markdown'.\n"
                "Install with: pip install weasyprint markdown")
            return
        
        default_name = "document.pdf"
        if self.current_tab.file_path:
            default_name = os.path.splitext(os.path.basename(self.current_tab.file_path))[0] + ".pdf"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=default_name
        )
        if not filepath:
            return
        
        self._set_status("Exporting to PDF...")
        self.update_idletasks()
        
        try:
            content = self.current_tab.to_markdown_document() if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet' else self.current_tab.get_content()
            base_path = os.path.dirname(self.current_tab.file_path) if self.current_tab.file_path else os.getcwd()
            export_to_pdf(content, filepath, theme=self.theme, base_path=base_path)
            self._set_status(f"Exported to: {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export PDF:\n{e}")
            self._set_status("Export failed")

    def _export_html(self):
        """Export current document to HTML"""
        if not self.current_tab:
            return
        if not EXPORT_AVAILABLE or not EXPORT_MARKDOWN_AVAILABLE:
            messagebox.showwarning("Unavailable",
                "HTML export requires 'markdown'.\nInstall with: pip install markdown")
            return
        
        default_name = "document.html"
        if self.current_tab.file_path:
            default_name = os.path.splitext(os.path.basename(self.current_tab.file_path))[0] + ".html"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html *.htm")],
            initialfile=default_name
        )
        if not filepath:
            return
        
        try:
            content = self.current_tab.to_markdown_document() if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet' else self.current_tab.get_content()
            base_path = os.path.dirname(self.current_tab.file_path) if self.current_tab.file_path else os.getcwd()
            export_to_html(content, filepath, theme=self.theme, base_path=base_path)
            self._set_status(f"Exported to: {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export HTML:\n{e}")
            self._set_status("Export failed")

    def _export_docx(self):
        """Export current document to DOCX"""
        if not self.current_tab:
            return
        if not EXPORT_AVAILABLE or not DOCX_AVAILABLE:
            messagebox.showwarning("Unavailable",
                "DOCX export requires 'python-docx'.\nInstall with: pip install python-docx")
            return
        
        default_name = "document.docx"
        if self.current_tab.file_path:
            default_name = os.path.splitext(os.path.basename(self.current_tab.file_path))[0] + ".docx"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".docx",
            filetypes=[("Word documents", "*.docx")],
            initialfile=default_name
        )
        if not filepath:
            return
        
        try:
            content = self.current_tab.to_markdown_document() if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet' else self.current_tab.get_content()
            base_path = os.path.dirname(self.current_tab.file_path) if self.current_tab.file_path else os.getcwd()
            export_to_docx(content, filepath, theme=self.theme, base_path=base_path)
            self._set_status(f"Exported to: {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export DOCX:\n{e}")
            self._set_status("Export failed")

    # === Tab Management Methods ===

    def new_document_tab(self, file_path=None, content=""):
        """Create a new Markdown document tab (Ctrl+N)."""
        return self.new_tab(file_path=file_path, content=content)

    def new_spreadsheet_tab(self, file_path=None, content=""):
        """Create a new Excel-style spreadsheet tab (Ctrl+Shift+N)."""
        if not SPREADSHEET_AVAILABLE:
            messagebox.showwarning("Unavailable", "Spreadsheet module could not be loaded.", parent=self)
            return None

        tab = SpreadsheetTab(
            self.notebook,
            theme=self.theme,
            on_modified_callback=lambda: self._on_tab_modified(),
            on_selection_change=self._schedule_position_update
        )
        if file_path:
            tab.file_path = file_path
        if content:
            tab.set_content(content)

        tab_name = tab.get_display_name() if not file_path else os.path.basename(file_path)
        self.notebook.add(tab, text=tab_name)
        self.tabs.append(tab)
        self.notebook.select(tab)
        self.current_tab = tab
        self._update_title()
        self._set_status("Created new spreadsheet tab")
        return tab

    def open_as_spreadsheet(self, filepath=None):
        """Open a CSV, TSV, XLSX, or Markdown table file in an interactive Spreadsheet tab."""
        if not SPREADSHEET_AVAILABLE:
            messagebox.showwarning("Unavailable", "Spreadsheet module could not be loaded.", parent=self)
            return

        if not filepath:
            filetypes = [
                ("Spreadsheet files", "*.csv *.tsv *.xlsx *.xls *.mdsheet"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("Tab-separated files", "*.tsv"),
                ("MarkItDown Spreadsheet", "*.mdsheet"),
                ("Markdown files (extract table)", "*.md *.markdown *.txt"),
                ("All files", "*.*")
            ]
            filepath = filedialog.askopenfilename(title="Open as Spreadsheet", filetypes=filetypes)
            if not filepath:
                return

        filepath = os.path.normpath(filepath)

        # Check if already open
        for tab in self.tabs:
            if tab.file_path and os.path.normpath(tab.file_path) == filepath:
                self.notebook.select(tab)
                self._set_status(f"Spreadsheet already open: {os.path.basename(filepath)}")
                return

        tab = self.new_spreadsheet_tab(file_path=filepath)
        if not tab:
            return

        ok = open_spreadsheet_file(filepath, tab)
        if ok:
            self._update_tab_title(tab)
            self._set_status(f"Opened spreadsheet: {filepath}")
            self._add_to_recent_files(filepath)
        else:
            # Check if file has Markdown pipe table
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                if "|" in content:
                    tab.set_content(content)
                    tab.file_path = filepath
                    self._update_tab_title(tab)
                    self._set_status(f"Opened Markdown table as spreadsheet: {filepath}")
                    self._add_to_recent_files(filepath)
                else:
                    self.close_tab(tab)
                    messagebox.showerror("Error", f"Could not parse '{os.path.basename(filepath)}' as a spreadsheet.", parent=self)
            except Exception as e:
                self.close_tab(tab)
                messagebox.showerror("Error", f"Could not open spreadsheet:\n{e}", parent=self)

    def new_tab(self, file_path=None, content=""):
        """Create a new tab"""
        tab = DocumentTab(
            self.notebook,
            theme=self.theme,
            on_modified_callback=lambda: self._on_tab_modified(),
            on_content_change=lambda c: self._on_visual_edit(c)
        )
        
        # Add to notebook
        tab_name = tab.get_display_name() if not file_path else os.path.basename(file_path)
        self.notebook.add(tab, text=tab_name)
        self.tabs.append(tab)
        
        # Set file path if provided
        if file_path:
            tab.file_path = file_path
        
        # Set content if provided
        if content:
            tab.set_content(content)
        
        # Setup bindings for this tab
        self._setup_tab_bindings(tab)
        tab.set_zoom(self.zoom_pct / 100.0)
        if not self.word_wrap:
            tab.set_wrap(False)
        
        tab.visual_viewer.live_render_enabled = self.theme.get('live_render_enabled', True)
        
        # Select the new tab
        self.notebook.select(tab)
        self.current_tab = tab
        
        self._update_title()
        self._do_syntax_highlight()
        self._schedule_outline_update()
        return tab
    
    def _setup_tab_bindings(self, tab):
        """Setup bindings for a specific tab"""
        # Mouse wheel zoom (text widgets and the desk around the page)
        tab.source_editor.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        tab.visual_viewer.text_widget.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        tab.source_page.on_zoom_wheel = self._on_ctrl_mousewheel
        tab.visual_viewer.page_view.on_zoom_wheel = self._on_ctrl_mousewheel
        tab.visual_viewer.on_zoom_wheel = self._on_ctrl_mousewheel
        
        # Position & selection tracking (both source and visual modes)
        tab.source_editor.bind("<KeyRelease>", self._schedule_position_update)
        tab.source_editor.bind("<KeyRelease>", self._schedule_syntax_highlight, add='+')
        tab.source_editor.bind("<ButtonRelease>", self._schedule_position_update)
        tab.source_editor.bind("<<Selection>>", self._schedule_position_update)
        tab.visual_viewer.text_widget.bind("<ButtonRelease>", self._schedule_position_update)
        tab.visual_viewer.text_widget.bind("<KeyRelease>", self._schedule_position_update)
        tab.visual_viewer.text_widget.bind("<<Selection>>", self._schedule_position_update)
        
        # Right-click context menu on editor
        tab.source_editor.bind("<Button-3>", self._show_editor_context_menu)
        tab.visual_viewer.text_widget.bind("<Button-3>", self._show_editor_context_menu)

        # Formatting keyboard shortcuts (direct on source editor with propagation break)
        for seq, pfx, sfx in (
            ("<Control-b>", "**", "**"),
            ("<Control-B>", "**", "**"),
            ("<Control-i>", "*", "*"),
            ("<Control-Key-i>", "*", "*"),
            ("<Control-grave>", "`", "`"),
            ("<Control-H>", "==", "=="),
            ("<Control-Shift-H>", "==", "=="),
            ("<Control-Shift-h>", "==", "=="),
            ("<Control-Shift-Key-H>", "==", "=="),
            ("<Control-k>", "[", "](url)"),
            ("<Control-M>", "[@", "]"),
        ):
            tab.source_editor.bind(seq, lambda e, p=pfx, s=sfx: self._insert_format(p, s))
    
    def close_tab(self, tab=None):
        """Close a tab"""
        if tab is None:
            tab = self.current_tab
        
        if tab is None:
            return
        
        # Check if modified
        if tab.is_modified:
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                f"'{tab.get_display_name()}' has unsaved changes.\n\nDo you want to save before closing?",
                parent=self
            )
            if response is None:  # Cancel
                return
            if response:  # Yes - save
                self.current_tab = tab
                if not self.save_file():
                    return  # Save was cancelled
        
        # Get tab index
        tab_index = self.tabs.index(tab)
        
        # Remove from notebook
        self.notebook.forget(tab)
        self.tabs.remove(tab)
        
        # If no tabs left, create new empty tab
        if not self.tabs:
            self.new_tab()
        else:
            # Select adjacent tab
            new_index = min(tab_index, len(self.tabs) - 1)
            self.notebook.select(self.tabs[new_index])
        
        self._update_title()
    
    def _on_tab_changed(self, event=None):
        """Handle tab selection change"""
        try:
            current = self.notebook.select()
            if current:
                # Find the tab widget
                for tab in self.tabs:
                    if str(tab) == current:
                        self.current_tab = tab
                        is_sheet = getattr(tab, 'tab_type', 'document') == 'spreadsheet'
                        if is_sheet:
                            if not getattr(self, '_ribbon_collapsed', False):
                                self.ribbon_bottom_deck.pack_forget()
                        else:
                            if not getattr(self, '_ribbon_collapsed', False):
                                self.ribbon_bottom_deck.pack(fill=tk.X, padx=2, pady=(0, 2))
                        self._update_title()
                        self._update_mode_display()
                        self._update_position()
                        self._update_word_count()
                        # Update AI sidebar context for new tab
                        self._update_ai_sidebar_context()
                        if not is_sheet:
                            self._do_syntax_highlight()
                            self._schedule_outline_update()
                        if getattr(self, 'find_bar', None) and self.find_bar.is_visible:
                            self.find_bar._on_query_change()
                        break
        except:
            pass
    
    def _on_tab_modified(self):
        """Called when any tab is modified"""
        if self.current_tab:
            # Update tab title to show modified indicator
            self._update_tab_title(self.current_tab)
            self._update_title()
    
    def _update_tab_title(self, tab):
        """Update the tab title in the notebook"""
        try:
            tab_id = self.tabs.index(tab)
            name = tab.get_display_name()
            if tab.is_modified:
                name = f"● {name}"
            self.notebook.tab(tab, text=name)
        except:
            pass
    
    def _next_tab(self):
        """Switch to next tab"""
        if len(self.tabs) > 1:
            current_idx = self.tabs.index(self.current_tab)
            next_idx = (current_idx + 1) % len(self.tabs)
            self.notebook.select(self.tabs[next_idx])
    
    def _prev_tab(self):
        """Switch to previous tab"""
        if len(self.tabs) > 1:
            current_idx = self.tabs.index(self.current_tab)
            prev_idx = (current_idx - 1) % len(self.tabs)
            self.notebook.select(self.tabs[prev_idx])
    
    def _show_tab_context_menu(self, event):
        """Show context menu for tab"""
        # Find which tab was clicked
        try:
            clicked_tab_idx = self.notebook.index(f"@{event.x},{event.y}")
            if clicked_tab_idx >= 0 and clicked_tab_idx < len(self.tabs):
                clicked_tab = self.tabs[clicked_tab_idx]
                
                menu = tk.Menu(self, tearoff=0)
                menu.add_command(label="Close Tab", command=lambda: self.close_tab(clicked_tab))
                menu.add_command(label="Close Other Tabs", command=lambda: self._close_other_tabs(clicked_tab))
                menu.add_separator()
                menu.add_command(label="🗋 New Document Tab", command=self.new_document_tab)
                menu.add_command(label="⊞ New Spreadsheet Tab", command=self.new_spreadsheet_tab)
                menu.tk_popup(event.x_root, event.y_root)
        except:
            pass
    
    def _on_tab_middle_click(self, event):
        """Middle-click on a tab closes it"""
        try:
            clicked_tab_idx = self.notebook.index(f"@{event.x},{event.y}")
            if 0 <= clicked_tab_idx < len(self.tabs):
                self.close_tab(self.tabs[clicked_tab_idx])
        except Exception:
            pass
    
    def _close_other_tabs(self, keep_tab):
        """Close all tabs except the specified one"""
        tabs_to_close = [t for t in self.tabs if t != keep_tab]
        for tab in tabs_to_close:
            self.close_tab(tab)
    
    def _schedule_position_update(self, event=None):
        """Schedule position update with debouncing for performance"""
        if not self._position_update_pending:
            self._position_update_pending = True
            self.after(50, self._do_position_update)
        self._schedule_outline_update()
    
    def _do_position_update(self):
        """Actually update position after debounce"""
        self._position_update_pending = False
        self._update_position()
        self._update_word_count()
        if self.typewriter_var.get() and self.current_tab:
            self.current_tab.center_cursor_vertically()
    
    def _on_modified(self, event=None):
        """Handle text modification - called by tab's source editor"""
        if self.current_tab and hasattr(self.current_tab, 'source_editor'):
            if self.current_tab.source_editor.edit_modified():
                self.current_tab.is_modified = True
                self._update_tab_title(self.current_tab)
                self._update_title()
                self.current_tab.source_editor.edit_modified(False)
    
    def _update_position(self, event=None):
        """Update cursor position in status bar"""
        try:
            if self.current_tab:
                if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
                    self.position_label.config(text=f"Cell {self.current_tab.coord_var.get()}")
                else:
                    pos = self.current_tab.source_editor.index(tk.INSERT)
                    line, col = pos.split('.')
                    self.position_label.config(text=f"Line {line}, Col {int(col)+1}")
        except:
            pass
    
    def _update_word_count(self):
        """Update word and character count in status bar, including highlighted selection counts"""
        try:
            if self.current_tab:
                if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
                    cell_count = len(self.current_tab.engine.raw_cells)
                    (min_r, min_c), (max_r, max_c) = self.current_tab.grid_widget.get_selected_range()
                    sel_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
                    if sel_cells > 1:
                        s_nums = []
                        for r in range(min_r, max_r + 1):
                            for c in range(min_c, max_c + 1):
                                v = self.current_tab.engine.get_value(r, c)
                                if isinstance(v, (int, float)) and not isinstance(v, bool):
                                    s_nums.append(float(v))
                        if s_nums:
                            s_sum = sum(s_nums)
                            s_avg = s_sum / len(s_nums)
                            self.word_count_label.config(text=f"Sel: {sel_cells} | Sum: {s_sum:,.2f} | Avg: {s_avg:,.2f}")
                        else:
                            self.word_count_label.config(text=f"Selected: {sel_cells} cells | Populated: {cell_count}")
                    else:
                        self.word_count_label.config(text=f"Cells: {cell_count} populated")
                else:
                    content = self.current_tab.get_content()
                    char_count = len(content)
                    words = content.split()
                    word_count = len(words)
                    
                    # Check if there is a highlighted selection in the active view
                    active_widget = self.current_tab.source_editor
                    sel_text = ""
                    try:
                        if active_widget.tag_ranges(tk.SEL):
                            sel_text = active_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
                    except tk.TclError:
                        sel_text = ""
                    
                    if sel_text:
                        sel_words = len(sel_text.split())
                        sel_chars = len(sel_text)
                        self.word_count_label.config(
                            text=f"Words: {sel_words} / {word_count} | Chars: {sel_chars} / {char_count}"
                        )
                    else:
                        self.word_count_label.config(text=f"Words: {word_count} | Chars: {char_count}")
            else:
                self.word_count_label.config(text="Words: 0 | Chars: 0")
        except Exception:
            pass
    
    def _update_title(self):
        """Update window title"""
        title = "MarkItDown Notepad"
        if self.current_tab:
            if self.current_tab.file_path:
                title = f"{os.path.basename(self.current_tab.file_path)} - {title}"
            else:
                title = f"{self.current_tab.get_display_name()} - {title}"
            if self.current_tab.is_modified:
                title = f"*{title}"
        self.title(title)
    
    def _update_mode_display(self):
        """Update mode display in status bar for current tab"""
        if self.current_tab:
            if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
                self.mode_var.set("spreadsheet")
                self.source_view_var.set(False)
                self.mode_label.config(text="Spreadsheet View")
                if hasattr(self, 'source_view_cb'):
                    self.source_view_cb.config(state="disabled")
            else:
                if hasattr(self, 'source_view_cb'):
                    self.source_view_cb.config(state="normal")
                mode = self.current_tab.current_mode
                self.mode_var.set(mode)
                self.source_view_var.set(mode == "source")
                self.mode_label.config(text="Source View" if mode == "source" else "Visual View")
                self._set_format_buttons_enabled(True)
    
    def _set_status(self, message):
        """Set status bar message"""
        self.status_label.config(text=message)
    
    # === View Mode Methods ===
    
    def _toggle_word_wrap(self):
        """Toggle word wrap in source editor"""
        self.word_wrap = not self.word_wrap
        self.wrap_var.set(self.word_wrap)
        
        for tab in self.tabs:
            tab.set_wrap(self.word_wrap)
        
        # Update status bar
        self.wrap_label.config(text=f"Wrap: {'On' if self.word_wrap else 'Off'}")
        status = "Word wrap enabled" if self.word_wrap else "Word wrap disabled (horizontal scroll)"
        self._set_status(status)

    def _toggle_typewriter_scrolling(self):
        """Toggle typewriter scrolling (keep cursor line vertically centered)"""
        enabled = bool(self.typewriter_var.get())
        self.theme['typewriter_scrolling'] = enabled
        self._save_theme()
        if enabled and self.current_tab:
            self.current_tab.center_cursor_vertically()
        self._set_status(
            "Typewriter scrolling enabled (cursor centered)"
            if enabled else "Typewriter scrolling disabled"
        )
    
    def _toggle_large_file_mode(self):
        """Toggle large file mode for better performance with big files"""
        is_large = self.large_file_var.get()
        
        if self.current_tab and hasattr(self.current_tab, 'source_editor'):
            if is_large:
                # Disable expensive operations
                self.current_tab.source_editor.config(undo=False)
                self._set_status("Large file mode: undo disabled for performance")
            else:
                # Re-enable
                self.current_tab.source_editor.config(undo=True)
                self._set_status("Normal mode: full features enabled")
    
    def _toggle_live_render(self):
        enabled = self.live_render_var.get()
        for tab in self.tabs:
            if hasattr(tab, 'visual_viewer'):
                tab.visual_viewer.live_render_enabled = enabled
        self.theme['live_render_enabled'] = enabled
        self._save_theme()
    
    # === Zoom & Page Layout ===
    
    @staticmethod
    def _clamp_zoom(pct):
        try:
            return max(25, min(400, int(pct)))
        except (TypeError, ValueError):
            return 100
    
    def _on_ctrl_mousewheel(self, event):
        """Handle Ctrl+MouseWheel for zoom"""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        return "break"  # Prevent default scrolling
    
    def _set_zoom(self, pct, save=True):
        """Set the global zoom (percent) and apply it to every tab."""
        pct = self._clamp_zoom(pct)
        self.zoom_pct = pct
        self.theme['zoom_pct'] = pct
        for tab in self.tabs:
            tab.set_zoom(pct / 100.0)
        self._update_zoom_widgets()
        if save:
            if self._zoom_save_job:
                self.after_cancel(self._zoom_save_job)
            self._zoom_save_job = self.after(500, self._save_theme)
    
    def zoom_in(self):
        """Move to the next zoom step"""
        for step in self.ZOOM_STEPS:
            if step > self.zoom_pct:
                self._set_zoom(step)
                return
    
    def zoom_out(self):
        """Move to the previous zoom step"""
        for step in reversed(self.ZOOM_STEPS):
            if step < self.zoom_pct:
                self._set_zoom(step)
                return
    
    def zoom_reset(self):
        """Reset zoom to 100%"""
        self._set_zoom(100)
    
    def zoom_fit_width(self):
        """Zoom so the page exactly fills the available width"""
        if not self.current_tab:
            return
        tab = self.current_tab
        if getattr(tab, 'tab_type', 'document') == 'spreadsheet':
            self._set_zoom(100)
            return
        page = tab.source_page
        self._set_zoom(int(page.fit_zoom() * 100))
    
    def _update_zoom_widgets(self):
        """Sync the status-bar zoom controls with the current zoom"""
        if not hasattr(self, 'zoom_label'):
            return
        self._zoom_updating = True
        try:
            self.zoom_label.config(text=f"{self.zoom_pct}%")
            self.zoom_scale.set(self.zoom_pct)
        finally:
            self._zoom_updating = False
    
    def _on_zoom_scale(self, value):
        if self._zoom_updating:
            return
        pct = int(round(float(value) / 5.0) * 5)
        if pct != self.zoom_pct:
            self._set_zoom(pct)
    
    def _apply_page_settings(self):
        """Persist page layout choices from the View menu and push them to every tab"""
        self.theme['page_layout'] = bool(self.page_layout_var.get())
        self.theme['page_size'] = self.page_size_var.get()
        self.theme['page_margin'] = self.page_margin_var.get()
        for tab in self.tabs:
            tab.apply_page_settings()
        self._save_theme()
    
    def _show_source_mode(self):
        """Switch current tab to raw Markdown source editing view"""
        if not self.current_tab:
            return

        self.current_tab.switch_mode("source", self.theme)
        self.source_view_var.set(True)
        self.mode_var.set("source")
        if hasattr(self, 'mode_label'):
            self.mode_label.config(text="Source View")
        self._set_format_buttons_enabled(True)
        self.current_tab.source_editor.focus_set()
        if getattr(self, 'find_bar', None) and self.find_bar.is_visible:
            self.find_bar._on_query_change()
        self._set_status("Source View — raw Markdown syntax")

    def _show_visual_mode(self):
        """Switch current tab to formatted visual presentation view"""
        if not self.current_tab:
            return

        self.current_tab.switch_mode("visual", self.theme)
        self.source_view_var.set(False)
        self.mode_var.set("visual")
        if hasattr(self, 'mode_label'):
            self.mode_label.config(text="Visual View")
        self._set_format_buttons_enabled(True)
        self.current_tab.source_editor.focus_set()
        if getattr(self, 'find_bar', None) and self.find_bar.is_visible:
            self.find_bar._on_query_change()
        self._set_status("Visual View — formatted document canvas")

    def _on_source_view_toggle(self):
        """Toggle callback for the Source View checkbox"""
        if not self.current_tab or getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
            return
        if self.source_view_var.get():
            self._show_source_mode()
        else:
            self._show_visual_mode()

    def toggle_source_view(self):
        """Shortcut action for Ctrl+E"""
        self.source_view_var.set(not self.source_view_var.get())
        self._on_source_view_toggle()
        return "break"

    def _toggle_mode(self):
        """Toggle between source view and visual view"""
        return self.toggle_source_view()

    def _refresh_active_visual_view(self):
        """Refresh visual viewer content if currently in visual mode"""
        if self.current_tab and self.current_tab.current_mode == "visual":
            try:
                ypos = self.current_tab.visual_viewer.text_widget.yview()[0]
            except Exception:
                ypos = 0.0
            content = self.current_tab.source_editor.get("1.0", tk.END).rstrip()
            self.current_tab.visual_viewer.set_content(content)
            try:
                self.current_tab.visual_viewer.text_widget.yview_moveto(ypos)
            except Exception:
                pass

    def _on_visual_edit(self, content):
        """Called when visual editor content changes"""
        pass
    
    # === Focus Mode ===
    
    def _toggle_focus_mode(self):
        """Toggle focus mode on/off"""
        if self.focus_mode_active:
            self._exit_focus_mode()
        else:
            self._enter_focus_mode()
    
    def _fullscreen_desk_color(self):
        """Darker desk colour used while in full-screen focus mode."""
        return _shade_color(self.theme.get('desk_bg', '#d9dce1'), -0.55)

    def _focus_active_editor(self):
        if not self.current_tab:
            return
        if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
            self.current_tab.grid_widget.focus_set()
        elif hasattr(self.current_tab, 'source_editor'):
            self.current_tab.source_editor.focus_set()

    def _enter_focus_mode(self):
        """Enter distraction-free full screen: hide all chrome and show the same page widgets on a dark desk."""
        if self.focus_mode_active or not self.current_tab:
            return
        self.focus_mode_active = True
        self._pre_fs = {
            'geometry': self.geometry(),
            'state': self.state(),
            'outline': self.outline_visible,
            'ai': self.ai_sidebar_visible,
        }

        # Hide side panels, then the chrome
        if self.outline_visible:
            self._toggle_outline_panel()
        if self.ai_sidebar_visible:
            self._toggle_ai_sidebar()
        self.config(menu="")
        self.toolbar_frame.pack_forget()
        self.status_bar.pack_forget()
        self.outer_paned.pack_configure(padx=0, pady=0)

        # Hide the tab strip but keep the notebook (and every tab widget) intact
        style = ttk.Style()
        style.layout('Hidden.TNotebook.Tab', [])
        style.configure('Hidden.TNotebook', borderwidth=0, tabmargins=0)
        self.notebook.configure(style='Hidden.TNotebook')

        desk = self._fullscreen_desk_color()
        self.configure(bg=desk)
        for tab in self.tabs:
            tab.apply_desk_color(desk)

        self.attributes('-fullscreen', True)
        self.focus_escape_binding = self.bind("<Escape>", self._on_fullscreen_escape)
        self._show_fullscreen_hint()
        self.after(100, self._focus_active_editor)

    def _on_fullscreen_escape(self, event=None):
        """Escape leaves full screen, unless a dialog window has the focus."""
        widget = self.focus_get()
        if widget is not None and widget.winfo_toplevel() is not self:
            return
        self._exit_focus_mode()

    def _show_fullscreen_hint(self):
        self._hide_fullscreen_hint()
        self._fs_hint = tk.Label(self, text="Press Esc or F11 to exit full screen", bg="#222222", fg="#f0f0f0",
                                 font=(SANS_FONT, 10), padx=14, pady=6)
        self._fs_hint.place(relx=0.5, rely=0.96, anchor='s')
        self._fs_hint_job = self.after(2500, self._hide_fullscreen_hint)

    def _hide_fullscreen_hint(self):
        job = getattr(self, '_fs_hint_job', None)
        if job:
            try:
                self.after_cancel(job)
            except Exception:
                pass
            self._fs_hint_job = None
        if self._fs_hint is not None:
            self._fs_hint.destroy()
            self._fs_hint = None

    def _exit_focus_mode(self):
        """Leave full screen and restore the normal window chrome."""
        if not self.focus_mode_active:
            return
        self.focus_mode_active = False
        self._hide_fullscreen_hint()

        if self.focus_escape_binding:
            self.unbind("<Escape>", self.focus_escape_binding)
            self.focus_escape_binding = None

        self.attributes('-fullscreen', False)

        # Restore chrome in pack order: toolbar (top), status bar (bottom), then the paned area
        self.configure(bg=self.theme.get('desk_bg', '#dcd6e8'))
        self.config(menu=self.menubar)
        self.toolbar_frame.pack(fill=tk.X, padx=5, pady=2, before=self.outer_paned)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, before=self.outer_paned)
        self.outer_paned.pack_configure(padx=5, pady=5)
        self.notebook.configure(style='TNotebook')
        for tab in self.tabs:
            tab.apply_desk_color(self.theme.get('desk_bg', '#d9dce1'))

        pre = self._pre_fs or {}
        self.update_idletasks()
        if pre.get('state') == 'zoomed':
            self.state('zoomed')
        elif pre.get('geometry'):
            self.geometry(pre['geometry'])

        if pre.get('outline') and not self.outline_visible:
            self._toggle_outline_panel()
        if pre.get('ai') and not self.ai_sidebar_visible:
            self._toggle_ai_sidebar()
        self._pre_fs = None

        self._focus_active_editor()
        self._set_status("Ready")
    
    # === File Operations ===

    def new_file(self):
        """Create a new file - now creates a new tab"""
        self.new_tab()
        self._set_status("New file created")
    
    def open_file(self, filepath=None):
        """Open a markdown, text, or converted document in a new tab"""
        if not filepath:
            filetypes = [
                ("All Supported Files", "*.md *.markdown *.txt *.pdf *.epub *.docx *.pptx *.xlsx *.xls *.html *.htm *.csv *.json *.xml *.msg *.py *.js *.wav *.mp3"),
                ("Markdown & Text files", "*.md *.markdown *.txt *.csv *.json *.xml *.py *.js"),
                ("PDF documents", "*.pdf"),
                ("EPUB eBooks", "*.epub"),
                ("Microsoft Word", "*.docx"),
                ("Microsoft PowerPoint", "*.pptx"),
                ("Microsoft Excel", "*.xlsx *.xls"),
                ("HTML files", "*.html *.htm"),
                ("All files", "*.*")
            ]
            filepath = filedialog.askopenfilename(filetypes=filetypes)
            if not filepath:
                return

        filepath = os.path.normpath(filepath)

        # Check if already open in a tab
        for tab in self.tabs:
            if tab.file_path and os.path.normpath(tab.file_path) == filepath:
                self.notebook.select(tab)
                self._set_status(f"File already open: {os.path.basename(filepath)}")
                return

        ext = Path(filepath).suffix.lower()

        # If it's a spreadsheet format, route directly to open_as_spreadsheet
        if ext in ('.csv', '.tsv', '.xlsx', '.xls', '.mdsheet') and SPREADSHEET_AVAILABLE:
            self.open_as_spreadsheet(filepath)
            return

        # If it's a binary/document format supported by MarkItDown, route directly to MarkItDown
        if MARKITDOWN_AVAILABLE and ext in MARKITDOWN_DOCUMENT_EXTENSIONS:
            self.open_with_markitdown(filepath)
            return

        # Attempt reading as plain text / markdown
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            tab = self.new_tab(file_path=filepath, content=content)
            tab.is_modified = False
            self._update_tab_title(tab)
            self._set_status(f"Opened: {filepath}")
            tab.visual_viewer.set_base_path(os.path.dirname(filepath))
            self._add_to_recent_files(filepath)
        except UnicodeDecodeError:
            # Fallback to MarkItDown if available
            if MARKITDOWN_AVAILABLE:
                self.open_with_markitdown(filepath)
            else:
                messagebox.showerror("Error", f"Could not decode file '{os.path.basename(filepath)}' as text/UTF-8.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file:\n{e}")

    def open_with_markitdown(self, filepath=None):
        """Open any file and convert with MarkItDown in a background thread."""
        if not MARKITDOWN_AVAILABLE or self.md_converter is None:
            messagebox.showwarning(
                "MarkItDown Not Available",
                "MarkItDown is not installed.\n\nInstall with:\npip install 'markitdown[all]'"
            )
            return
        
        if not filepath:
            filetypes = [
                ("All Supported Files", "*.pdf *.epub *.docx *.pptx *.xlsx *.xls *.html *.htm *.csv *.json *.xml *.msg *.jpg *.jpeg *.png *.gif *.bmp *.wav *.mp3 *.md *.txt"),
                ("PDF documents", "*.pdf"),
                ("EPUB eBooks", "*.epub"),
                ("Word documents", "*.docx"),
                ("PowerPoint", "*.pptx"),
                ("Excel files", "*.xlsx *.xls"),
                ("HTML files", "*.html *.htm"),
                ("Images", "*.jpg *.jpeg *.png *.gif *.bmp"),
                ("Audio", "*.wav *.mp3"),
                ("All files", "*.*")
            ]
            filepath = filedialog.askopenfilename(filetypes=filetypes)
            if not filepath:
                return

        filepath = os.path.normpath(filepath)

        # Check if already open in a tab
        for tab in self.tabs:
            if tab.file_path and os.path.normpath(tab.file_path) == filepath:
                self.notebook.select(tab)
                self._set_status(f"File already open: {os.path.basename(filepath)}")
                return

        # Check image extraction before starting background thread to avoid UI dialog races
        extract_images = False
        default_assets_dir = DocumentImageExtractor.get_default_assets_dir(filepath) if DOC_IMAGE_EXTRACTOR_AVAILABLE else os.path.join(os.path.dirname(filepath), "assets")
        output_dir = default_assets_dir

        if ImageExtractor.can_extract(filepath):
            extract_images = messagebox.askyesno(
                "Extract & Display Images?",
                "This document contains embedded images.\n\n"
                f"Would you like to extract them inline to:\n{os.path.basename(default_assets_dir)}/\n\n"
                "Images will be displayed inline throughout the markdown text.",
                default=messagebox.YES
            )
            if not extract_images:
                output_dir = None

        # Create non-blocking progress dialog
        progress_dialog = tk.Toplevel(self)
        progress_dialog.title("Converting Document")
        progress_dialog.geometry("400x140")
        progress_dialog.transient(self)
        progress_dialog.grab_set()
        progress_dialog.resizable(False, False)
        
        frame = ttk.Frame(progress_dialog, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"Converting with MarkItDown:\n{os.path.basename(filepath)}", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 10))
        pbar = ttk.Progressbar(frame, mode="indeterminate")
        pbar.pack(fill=tk.X, pady=(0, 8))
        pbar.start(10)
        
        status_lbl = ttk.Label(frame, text="Processing document in background...", foreground="gray")
        status_lbl.pack(anchor=tk.W)

        self._set_status(f"Converting: {os.path.basename(filepath)}...")

        conv_queue = queue.Queue()

        def worker():
            try:
                result = self.md_converter.convert(filepath)
                raw_content = str(getattr(result, 'text_content', None) or getattr(result, 'markdown', '') or "")
                extracted = {}

                if extract_images and output_dir and DOC_IMAGE_EXTRACTOR_AVAILABLE:
                    content, extracted, assets_path = DocumentImageExtractor.extract_and_integrate(
                        filepath, output_dir=output_dir, raw_markdown=raw_content, clean_text=True
                    )
                else:
                    if TEXT_CLEANER_AVAILABLE:
                        content = clean_document(raw_content)
                    else:
                        content = raw_content

                conv_queue.put(('success', content, extracted, output_dir if extract_images else None, filepath))
            except Exception as e:
                conv_queue.put(('error', str(e), filepath))

        def poll_queue():
            try:
                msg_type, *args = conv_queue.get_nowait()
                try:
                    pbar.stop()
                    progress_dialog.destroy()
                except Exception:
                    pass

                if msg_type == 'success':
                    content, extracted, out_dir, fpath = args
                    tab = self.new_tab(file_path=fpath, content=content)
                    tab.is_modified = False
                    self._update_tab_title(tab)
                    
                    if out_dir:
                        tab.visual_viewer.set_base_path(out_dir)
                    else:
                        tab.visual_viewer.set_base_path(os.path.dirname(fpath))
                    
                    self._add_to_recent_files(fpath)
                    
                    img_count = len(extracted)
                    if img_count > 0:
                        self._set_status(f"Converted: {os.path.basename(fpath)} ({img_count} images extracted)")
                        messagebox.showinfo(
                            "Conversion Complete",
                            f"Successfully converted '{os.path.basename(fpath)}' to Markdown.\n\n"
                            f"Extracted {img_count} image(s) to: {out_dir}"
                        )
                    else:
                        self._set_status(f"Converted: {os.path.basename(fpath)}")
                elif msg_type == 'error':
                    err_msg, fpath = args
                    messagebox.showerror("Conversion Error", f"Could not convert '{os.path.basename(fpath)}':\n{err_msg}")
                    self._set_status("Conversion failed")
            except queue.Empty:
                if progress_dialog.winfo_exists():
                    self.after(50, poll_queue)

        threading.Thread(target=worker, daemon=True).start()
        self.after(50, poll_queue)
    
    def save_file(self):
        """Save the current tab's file"""
        if not self.current_tab:
            return False

        if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
            if self.current_tab.file_path:
                ok = save_spreadsheet_file(self.current_tab, self.current_tab.file_path)
                if ok:
                    self.current_tab.is_modified = False
                    self._update_tab_title(self.current_tab)
                    self._update_title()
                    self._set_status(f"Saved spreadsheet: {self.current_tab.file_path}")
                return ok
            else:
                return self.save_file_as()
            
        if self.current_tab.file_path:
            try:
                content = self.current_tab.get_content()
                with open(self.current_tab.file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.current_tab.is_modified = False
                self._update_tab_title(self.current_tab)
                self._update_title()
                self._set_status(f"Saved: {self.current_tab.file_path}")
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file:\n{e}")
                return False
        else:
            return self.save_file_as()
    
    def save_file_as(self):
        """Save the current tab's file with a new name"""
        if not self.current_tab:
            return False

        if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
            filetypes = [
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("Tab-separated files", "*.tsv"),
                ("MarkItDown Spreadsheet", "*.mdsheet"),
                ("All files", "*.*")
            ]
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=filetypes
            )
            if filepath:
                ok = save_spreadsheet_file(self.current_tab, filepath)
                if ok:
                    self.current_tab.file_path = filepath
                    self.current_tab.is_modified = False
                    self._update_tab_title(self.current_tab)
                    self._update_title()
                    self._set_status(f"Saved spreadsheet: {filepath}")
                    self._add_to_recent_files(filepath)
                return ok
            return False
            
        filetypes = [
            ("Markdown files", "*.md"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                content = self.current_tab.get_content()
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.current_tab.file_path = filepath
                self.current_tab.is_modified = False
                self._update_tab_title(self.current_tab)
                self._update_title()
                self._set_status(f"Saved: {filepath}")
                
                # Update visual viewer base path
                self.current_tab.visual_viewer.set_base_path(os.path.dirname(filepath))
                
                # Add to recent files
                self._add_to_recent_files(filepath)
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file:\n{e}")
                return False
        return False
    
    def _confirm_discard(self):
        """Ask user to confirm discarding changes - returns True if OK to proceed"""
        if not self.current_tab or not self.current_tab.is_modified:
            return True
            
        result = messagebox.askyesnocancel(
            "Unsaved Changes",
            "You have unsaved changes. Do you want to save them?"
        )
        if result is None:  # Cancel
            return False
        if result:  # Yes
            return self.save_file()
        return True  # No
    
    # === Edit Operations ===
    
    def undo(self):
        """Undo last action"""
        if self.current_tab:
            if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
                self.current_tab.grid_widget.undo()
            elif hasattr(self.current_tab, 'source_editor'):
                try:
                    self.current_tab.source_editor.edit_undo()
                except tk.TclError:
                    pass
    
    def redo(self):
        """Redo last undone action"""
        if self.current_tab:
            if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
                self.current_tab.grid_widget.redo()
            elif hasattr(self.current_tab, 'source_editor'):
                try:
                    self.current_tab.source_editor.edit_redo()
                except tk.TclError:
                    pass
    
    def cut(self):
        """Cut selected text"""
        if self.current_tab:
            if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
                self.current_tab.cut()
            elif hasattr(self.current_tab, 'source_editor'):
                self.current_tab.source_editor.event_generate("<<Cut>>")
    
    def copy(self):
        """Copy selected text"""
        if self.current_tab:
            if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
                self.current_tab.grid_widget.copy_selection()
            elif hasattr(self.current_tab, 'source_editor'):
                self.current_tab.source_editor.event_generate("<<Copy>>")
    
    def paste(self):
        """Paste from clipboard with smart detection of tables and images"""
        if self.current_tab:
            if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
                self.current_tab.grid_widget.paste_clipboard()
                return "break"
            return self._handle_smart_paste()
        return "break"

    def _insert_smart_pasted_text(self, editor, text_to_insert):
        try:
            if editor.tag_ranges(tk.SEL):
                editor.delete(tk.SEL_FIRST, tk.SEL_LAST)
        except Exception:
            pass
        editor.insert(tk.INSERT, text_to_insert)
        self.current_tab.is_modified = True
        self._update_tab_title(self.current_tab)
        self._schedule_syntax_highlight()
        self._refresh_active_visual_view()
        editor.see(tk.INSERT)

    def _get_document_assets_dir(self):
        if self.current_tab and self.current_tab.file_path:
            doc_dir = os.path.dirname(self.current_tab.file_path)
            stem = os.path.splitext(os.path.basename(self.current_tab.file_path))[0]
            return os.path.join(doc_dir, f"{stem}_assets")
        home = os.path.expanduser("~")
        return os.path.join(home, ".markitdown_assets")

    def _compute_relative_asset_path(self, fpath):
        if self.current_tab and self.current_tab.file_path:
            doc_dir = os.path.dirname(self.current_tab.file_path)
            try:
                return os.path.relpath(fpath, doc_dir).replace('\\', '/')
            except ValueError:
                pass
        return fpath.replace('\\', '/')

    def _handle_smart_paste(self, event=None):
        """Smart paste: converts Excel/Sheets/Calc TSV tables, HTML tables, pipe tables, or clipboard images."""
        if not self.current_tab:
            return "break"
        editor = self.current_tab.source_editor

        # 1. Try clipboard text first
        clip_text = None
        try:
            clip_text = self.clipboard_get()
        except tk.TclError:
            pass

        if clip_text:
            # HTML table detection
            html_table = parse_html_table_to_markdown(clip_text)
            if html_table:
                self._insert_smart_pasted_text(editor, html_table)
                self._set_status("Pasted formatted table from HTML")
                return "break"

            # Spreadsheet TSV detection (Excel / Google Sheets / Calc)
            tsv_table = parse_spreadsheet_tsv_to_markdown(clip_text)
            if tsv_table:
                self._insert_smart_pasted_text(editor, tsv_table)
                self._set_status("Pasted spreadsheet cells as Markdown table")
                return "break"

            # Existing messy pipe table detection
            pipe_table = format_existing_pipe_table_on_paste(clip_text)
            if pipe_table:
                self._insert_smart_pasted_text(editor, pipe_table)
                self._set_status("Pasted and auto-aligned Markdown table")
                return "break"

            # Standard text paste
            self._insert_smart_pasted_text(editor, clip_text)
            return "break"

        # 2. Check for clipboard image (screenshots)
        if PIL_AVAILABLE:
            try:
                from PIL import ImageGrab
                img = ImageGrab.grabclipboard()
                if img is not None:
                    if hasattr(img, 'save'):
                        assets_dir = self._get_document_assets_dir()
                        os.makedirs(assets_dir, exist_ok=True)
                        import time
                        ts = int(time.time() * 1000)
                        fname = f"figure_{ts}.png"
                        fpath = os.path.join(assets_dir, fname)
                        img.save(fpath, format="PNG")
                        rel_path = self._compute_relative_asset_path(fpath)
                        img_md = f"\n![Figure|width=550,align=center]({rel_path})\n\n"
                        self._insert_smart_pasted_text(editor, img_md)
                        self._set_status("Pasted image from clipboard into document assets")
                        return "break"
                    elif isinstance(img, list) and img:
                        inserted_any = False
                        for p in img:
                            if isinstance(p, str) and os.path.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'):
                                rel_p = self._compute_relative_asset_path(p)
                                stem = os.path.splitext(os.path.basename(p))[0]
                                img_md = f"\n![{stem}|width=550,align=center]({rel_p})\n\n"
                                self._insert_smart_pasted_text(editor, img_md)
                                inserted_any = True
                        if inserted_any:
                            self._set_status("Pasted image file(s) as Markdown figure(s)")
                            return "break"
            except Exception as e:
                print(f"Error grabbing clipboard image: {e}")

        editor.event_generate("<<Paste>>")
        return "break"
    
    def select_all(self):
        """Select all text or all populated spreadsheet cells"""
        if self.current_tab:
            if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
                self.current_tab.select_all()
            elif hasattr(self.current_tab, 'source_editor'):
                self.current_tab.source_editor.tag_add(tk.SEL, "1.0", tk.END)
        return "break"

    def clean_current_document(self):
        """Clean and reflow the current tab's markdown text using algorithmic normalization."""
        if not self.current_tab:
            return
        if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
            old_cells = {}
            new_cells = {}
            for (r, c), val in list(self.current_tab.engine.raw_cells.items()):
                if isinstance(val, str) and not val.startswith("="):
                    trimmed = val.strip()
                    if trimmed != val:
                        old_cells[(r, c)] = val
                        new_cells[(r, c)] = trimmed
                        self.current_tab.engine.set_cell(r, c, trimmed)
            if old_cells:
                self.current_tab.grid_widget._save_undo_state({"type": "batch_cells", "old_cells": old_cells, "new_cells": new_cells})
                self.current_tab.engine.recalculate_all()
                self.current_tab.grid_widget.redraw()
                self.current_tab.is_modified = True
                self._update_tab_title(self.current_tab)
                self._set_status(f"Trimmed whitespace in {len(old_cells)} spreadsheet cell(s)")
            else:
                self._set_status("Spreadsheet cells are already clean")
            return

        if not TEXT_CLEANER_AVAILABLE:
            messagebox.showwarning("Feature Unavailable", "Text cleaner module is not available.")
            return

        current_content = self.current_tab.get_content()
        if not current_content.strip():
            return

        cleaned = clean_document(current_content)
        if cleaned != current_content:
            self.current_tab.set_content(cleaned)
            self.current_tab.is_modified = True
            self._update_tab_title(self.current_tab)
            self._set_status("Document text cleaned and reflowed")
        else:
            self._set_status("Document text is already clean")

    def clean_selected_text(self):
        """Clean and reflow only the currently selected text in the active tab."""
        if not self.current_tab:
            return
        if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
            (min_r, min_c), (max_r, max_c) = self.current_tab.grid_widget.get_selected_range()
            old_cells = {}
            new_cells = {}
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    val = self.current_tab.engine.get_raw(r, c)
                    if isinstance(val, str) and not val.startswith("="):
                        trimmed = val.strip()
                        if trimmed != val:
                            old_cells[(r, c)] = val
                            new_cells[(r, c)] = trimmed
                            self.current_tab.engine.set_cell(r, c, trimmed)
            if old_cells:
                self.current_tab.grid_widget._save_undo_state({"type": "batch_cells", "old_cells": old_cells, "new_cells": new_cells})
                self.current_tab.engine.recalculate_all()
                self.current_tab.grid_widget.redraw()
                self.current_tab.is_modified = True
                self._update_tab_title(self.current_tab)
                self._set_status(f"Trimmed whitespace in {len(old_cells)} selected cell(s)")
            else:
                self._set_status("Selected cells are already clean")
            return

        if not TEXT_CLEANER_AVAILABLE:
            messagebox.showwarning("Feature Unavailable", "Text cleaner module is not available.")
            return

        # Canonical editor widget
        text_widget = getattr(self.current_tab, 'source_editor', None)
        if not text_widget:
            return

        try:
            sel_start = text_widget.index(tk.SEL_FIRST)
            sel_end = text_widget.index(tk.SEL_LAST)
            selected_text = text_widget.get(sel_start, sel_end)
        except tk.TclError:
            messagebox.showinfo("No Selection", "Please select text to clean and reflow.")
            return

        if not selected_text.strip():
            return

        cleaned = clean_document(selected_text)
        text_widget.delete(sel_start, sel_end)
        text_widget.insert(sel_start, cleaned)
        self.current_tab.is_modified = True
        self._update_tab_title(self.current_tab)
        self._schedule_syntax_highlight()
        self._set_status("Selected text cleaned and reflowed")
    
    def find_text(self):
        """Show the non-blocking docked Find bar (works in Source and Visual mode)"""
        if self.current_tab and getattr(self, 'find_bar', None):
            self.find_bar.show(replace_mode=False)
        return "break"
    
    # === Format Operations ===
    
    def _insert_format(self, prefix, suffix):
        """Insert formatting around selected text, or toggle formatting if already applied."""
        if not self.current_tab:
            return "break"

        if getattr(self.current_tab, 'tab_type', 'document') == 'spreadsheet':
            if prefix == "**":
                self.current_tab.toggle_bold_selection()
            return "break"
        
        tab = self.current_tab
        editor = tab.source_editor
        p_len = len(prefix)
        s_len = len(suffix)
        
        try:
            first = editor.index(tk.SEL_FIRST)
            last = editor.index(tk.SEL_LAST)
            selected = editor.get(first, last)
            
            # Case A: Selection itself starts with prefix and ends with suffix (toggle off)
            is_wrapped_inside = False
            if prefix == "*" and (selected.startswith("**") or selected.endswith("**")):
                is_wrapped_inside = False
            elif len(selected) >= p_len + s_len and selected.startswith(prefix) and selected.endswith(suffix):
                is_wrapped_inside = True

            if is_wrapped_inside:
                inner = selected[p_len:-s_len]
                editor.delete(first, last)
                editor.insert(first, inner)
                new_last = f"{first}+{len(inner)}c"
                editor.tag_add(tk.SEL, first, new_last)
                editor.mark_set(tk.INSERT, new_last)
                self.current_tab.is_modified = True
                self._update_tab_title(self.current_tab)
                self._schedule_syntax_highlight()
                self._refresh_active_visual_view()
                return "break"

            # Case B: Prefix and suffix are immediately outside the selection (toggle off)
            can_unwrap_outside = False
            before = editor.get(f"{first}-{p_len}c", first)
            after = editor.get(last, f"{last}+{s_len}c")
            if before == prefix and after == suffix:
                can_unwrap_outside = True
                # Ensure single asterisk italic does not strip from bold (**)
                if prefix == "*":
                    char_before = editor.get(f"{first}-{p_len+1}c", f"{first}-{p_len}c")
                    char_after = editor.get(f"{last}+{s_len}c", f"{last}+{s_len+1}c")
                    if char_before == "*" or char_after == "*":
                        can_unwrap_outside = False

            if can_unwrap_outside:
                editor.delete(last, f"{last}+{s_len}c")
                editor.delete(f"{first}-{p_len}c", first)
                new_first = f"{first}-{p_len}c"
                new_last = f"{new_first}+{len(selected)}c"
                editor.tag_add(tk.SEL, new_first, new_last)
                editor.mark_set(tk.INSERT, new_last)
                self.current_tab.is_modified = True
                self._update_tab_title(self.current_tab)
                self._schedule_syntax_highlight()
                self._refresh_active_visual_view()
                return "break"

            # Case C: Apply formatting around selected text
            editor.delete(first, last)
            editor.insert(first, f"{prefix}{selected}{suffix}")
            new_sel_first = f"{first}+{p_len}c"
            new_sel_last = f"{first}+{p_len + len(selected)}c"
            editor.tag_add(tk.SEL, new_sel_first, new_sel_last)
            editor.mark_set(tk.INSERT, new_sel_last)
            editor.see(new_sel_last)
            self.current_tab.is_modified = True
            self._update_tab_title(self.current_tab)
            self._schedule_syntax_highlight()
            self._refresh_active_visual_view()
            return "break"

        except tk.TclError:
            # No selection, insert the formatting markers and place cursor in between
            insert_pos = editor.index(tk.INSERT)
            editor.insert(insert_pos, f"{prefix}{suffix}")
            new_pos = f"{insert_pos}+{p_len}c"
            editor.mark_set(tk.INSERT, new_pos)
            editor.see(new_pos)
            self.current_tab.is_modified = True
            self._update_tab_title(self.current_tab)
            self._schedule_syntax_highlight()
            self._refresh_active_visual_view()
            return "break"
    
    def _insert_line_prefix(self, prefix):
        """Insert prefix at the beginning of the current line (with toggle/replace support)"""
        if not self.current_tab:
            return "break"
        
        editor = self.current_tab.source_editor
        pos = editor.index(tk.INSERT)
        line = pos.split('.')[0]
        line_text = editor.get(f"{line}.0", f"{line}.end")
        
        heading_match = re.match(r'^(#{1,6}\s+)(.*)$', line_text)
        if prefix.startswith('#'):
            if heading_match:
                old_prefix, rest = heading_match.groups()
                if old_prefix == prefix:
                    editor.delete(f"{line}.0", f"{line}.{len(old_prefix)}")
                else:
                    editor.delete(f"{line}.0", f"{line}.{len(old_prefix)}")
                    editor.insert(f"{line}.0", prefix)
            else:
                editor.insert(f"{line}.0", prefix)
        elif prefix in ("> ", "- ", "1. "):
            if line_text.startswith(prefix):
                editor.delete(f"{line}.0", f"{line}.{len(prefix)}")
            else:
                editor.insert(f"{line}.0", prefix)
        else:
            editor.insert(f"{line}.0", prefix)

        self.current_tab.is_modified = True
        self._update_tab_title(self.current_tab)
        self._schedule_syntax_highlight()
        self._refresh_active_visual_view()
        return "break"
    
    def _insert_text(self, text):
        """Insert text at cursor position"""
        if not self.current_tab:
            return "break"
        self.current_tab.source_editor.insert(tk.INSERT, text)
        self.current_tab.is_modified = True
        self._update_tab_title(self.current_tab)
        self._schedule_syntax_highlight()
        self._refresh_active_visual_view()
        return "break"

    def insert_table(self):
        """Open dialog to insert a formatted Markdown table with configurable rows, columns, and header."""
        if not self.current_tab or not hasattr(self.current_tab, 'source_editor'):
            return "break"
        InsertTableDialog(self, self.current_tab.source_editor, self._on_table_inserted)
        return "break"

    def _on_table_inserted(self):
        if self.current_tab:
            self.current_tab.is_modified = True
            self._update_tab_title(self.current_tab)
            self._schedule_syntax_highlight()
            self._schedule_outline_update()
            self._refresh_active_visual_view()
            self._set_status("Markdown table inserted (use Tab / Shift+Tab to navigate & align cells)")

    def insert_footnote(self):
        """Insert a numbered footnote citation [^n] at cursor and append [^n]: definition at bottom."""
        if not self.current_tab or not hasattr(self.current_tab, 'source_editor'):
            return "break"
        editor = self.current_tab.source_editor
        content = editor.get("1.0", tk.END)
        existing = [int(m) for m in re.findall(r'\[\^(\d+)\]', content)]
        next_num = (max(existing) + 1) if existing else 1
        ref_token = f"[^{next_num}]"

        # If selection exists, place citation immediately after selection
        try:
            insert_pos = editor.index(tk.SEL_LAST)
        except tk.TclError:
            insert_pos = editor.index(tk.INSERT)

        editor.insert(insert_pos, ref_token)
        # Append footnote definition at the end of the document
        end_text = editor.get("1.0", "end-1c")
        prefix = "\n\n" if not end_text.endswith("\n\n") else ""
        if end_text.endswith("\n") and not end_text.endswith("\n\n"):
            prefix = "\n"
        def_token = f"{prefix}[^{next_num}]: "
        editor.insert("end-1c", def_token)
        editor.mark_set(tk.INSERT, "end-1c")
        editor.see(tk.INSERT)
        self.current_tab.is_modified = True
        self._update_tab_title(self.current_tab)
        self._schedule_syntax_highlight()
        self._schedule_outline_update()
        self._refresh_active_visual_view()
        self._set_status(f"Inserted footnote {ref_token} — enter footnote text below")
        return "break"

    def open_math_symbol_picker(self):
        """Open the non-modal Math & Unicode Symbol palette."""
        if not self.current_tab:
            return "break"
        if hasattr(self, '_math_picker_win') and self._math_picker_win and self._math_picker_win.winfo_exists():
            self._math_picker_win.lift()
            self._math_picker_win.focus_set()
            return "break"
        self._math_picker_win = MathSymbolDialog(self)
        return "break"
    
    # === Image Management ===
    
    def _get_assets_folder(self):
        """Get or create the assets folder for the current document"""
        if not self.current_tab or not self.current_tab.file_path:
            return None
        
        doc_dir = os.path.dirname(self.current_tab.file_path)
        doc_name = os.path.splitext(os.path.basename(self.current_tab.file_path))[0]
        assets_folder = os.path.join(doc_dir, f"{doc_name}_assets")
        
        if not os.path.exists(assets_folder):
            os.makedirs(assets_folder)
        
        return assets_folder
    
    def insert_image(self):
        """Insert an image into the document with proper asset management"""
        if not self.current_tab or not hasattr(self.current_tab, 'source_editor'):
            return
        
        # Open image insert dialog
        InsertImageDialog(self, self.current_tab.source_editor, self.current_tab.file_path, self._get_assets_folder)
    
    def manage_images(self):
        """Open image manager dialog"""
        if not self.current_tab or not self.current_tab.file_path or not hasattr(self.current_tab, 'source_editor'):
            messagebox.showinfo(
                "Save Required",
                "Please save your document first to manage images.\n\n"
                "Images are stored in a folder alongside your document."
            )
            return
        
        ImageManagerDialog(self, self.current_tab.file_path, self.current_tab.source_editor)
    
    def _find_document_images(self):
        """Find all image references in the current document"""
        if not self.current_tab:
            return []
            
        content = self.current_tab.source_editor.get(1.0, tk.END)
        
        # Find markdown image syntax: ![alt](path)
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(pattern, content)
        
        images = []
        doc_dir = os.path.dirname(self.current_tab.file_path) if self.current_tab.file_path else os.getcwd()
        
        for alt, path in matches:
            # Skip base64 images
            if path.startswith('data:'):
                images.append({
                    'alt': alt,
                    'path': path[:50] + '...',
                    'type': 'embedded',
                    'exists': True
                })
            else:
                # Check if file exists
                if os.path.isabs(path):
                    full_path = path
                else:
                    full_path = os.path.join(doc_dir, path)
                
                images.append({
                    'alt': alt,
                    'path': path,
                    'full_path': full_path,
                    'type': 'file',
                    'exists': os.path.exists(full_path)
                })
        
        return images

    # === Search Methods ===
    
    def find_replace(self):
        """Show the non-blocking docked Find & Replace bar"""
        if self.current_tab and getattr(self, 'find_bar', None):
            self.find_bar.show(replace_mode=True)
        return "break"
    
    def goto_line(self):
        """Open go to line dialog"""
        if self.current_tab and hasattr(self.current_tab, 'source_editor'):
            GoToLineDialog(self, self.current_tab.source_editor)
    
    def _find_next_f3(self):
        """Find next using F3 (uses docked find bar)"""
        if getattr(self, 'find_bar', None):
            if not self.find_bar.is_visible:
                self.find_bar.show(replace_mode=False)
            else:
                self.find_bar.find_next()
            return "break"
        self.find_text()
    
    # === Application Methods ===
    
    def quit_app(self):
        """Exit the application"""
        # Save session state before checking for unsaved changes
        if getattr(self, 'ai_sidebar', None):
            self.ai_sidebar.commit_edit_mode()
        self._save_session()
        
        # Check all tabs for unsaved changes (only if caching is disabled)
        # If caching is enabled, documents will be restored on next launch
        if not (self.session_manager and self.session_manager.is_caching_enabled()):
            for tab in self.tabs:
                if tab.is_modified:
                    response = messagebox.askyesnocancel(
                        "Unsaved Changes",
                        f"'{tab.get_display_name()}' has unsaved changes.\n\nDo you want to save before exiting?",
                        parent=self
                    )
                    if response is None:  # Cancel
                        return
                    if response:  # Yes - save
                        self.current_tab = tab
                        self.notebook.select(tab)
                        if not self.save_file():
                            return  # Save was cancelled
        
        # Stop auto-save timer
        if self._auto_save_job:
            self.after_cancel(self._auto_save_job)
        
        self.destroy()
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About MarkItDown Notepad",
            "MarkItDown Notepad v1.0\n\n"
            "A lightweight markdown editor using Microsoft's MarkItDown package.\n\n"
            "Features:\n"
            "• Open any file type via MarkItDown conversion\n"
            "• Source and Visual editing modes\n"
            "• Standard notepad operations\n\n"
            "Built with Python and Tkinter"
        )
    
    def show_markitdown_info(self):
        """Show MarkItDown information"""
        if MARKITDOWN_AVAILABLE:
            info = (
                "MarkItDown is available!\n\n"
                "Supported file types:\n"
                "• PDF files\n"
                "• Word documents (.docx)\n"
                "• PowerPoint presentations (.pptx)\n"
                "• Excel spreadsheets (.xlsx, .xls)\n"
                "• HTML files\n"
                "• Images (with EXIF/OCR)\n"
                "• Audio files (with transcription)\n"
                "• CSV, JSON, XML\n"
                "• ZIP files\n"
                "• YouTube URLs\n"
                "• EPub books\n"
                "• Outlook messages (.msg)\n\n"
                "Use 'Open with MarkItDown' to convert any of these files to Markdown."
            )
        else:
            info = (
                "MarkItDown is NOT installed.\n\n"
                "Install with:\n"
                "pip install 'markitdown[all]'\n\n"
                "Or install specific features:\n"
                "pip install 'markitdown[pdf,docx,pptx]'"
            )
        
        messagebox.showinfo("MarkItDown Info", info)
    
    def open_theme_dialog(self):
        """Open theme settings dialog"""
        ThemeDialog(self, self.theme, self._apply_theme)

    def open_blockquote_style_dialog(self):
        """Open blockquote customization dialog"""
        BlockquoteStyleDialog(self, self.theme, self._apply_theme)
    
    def _apply_theme(self, new_theme):
        """Apply a new theme to the application"""
        self.theme = new_theme
        self.theme['zoom_pct'] = self.zoom_pct
        self.configure(bg=self.theme.get('desk_bg', '#d9dce1'))
        self._save_theme()
        
        # Apply to all tabs (each tab re-styles its source editor, syntax tags and visual page)
        for tab in self.tabs:
            tab.apply_theme(self.theme)
            tab.set_zoom(self.zoom_pct / 100.0)
        
        if getattr(self, 'ai_sidebar', None):
            self.ai_sidebar.apply_theme(self.theme)
        if getattr(self, 'outline_panel', None):
            try:
                self.outline_panel.apply_theme(self.theme)
            except Exception:
                pass
        self._setup_styles()
        
        self._set_status("Theme applied")


class ThemeDialog(tk.Toplevel):
    """Dialog for customizing theme settings"""
    
    # Preset themes
    PRESETS = {
        'Royal Purple (Word)': {
            'source_bg': '#ffffff', 'source_fg': '#1a1a1a',
            'visual_bg': '#ffffff', 'visual_fg': '#1a1a1a',
            'desk_bg': '#dcd6e8', 'ribbon_bg': '#f3effa',
            'status_bg': '#3b1d6e', 'status_fg': '#f5f3ff',
            'accent_purple': '#5b21b6',
        },
        'Royal Nocturne': {
            'source_bg': '#191124', 'source_fg': '#e9d5ff',
            'visual_bg': '#221533', 'visual_fg': '#f3e8ff',
            'desk_bg': '#120b1a', 'ribbon_bg': '#2a1a40',
            'status_bg': '#3b1d6e', 'status_fg': '#f5f3ff',
            'accent_purple': '#9333ea',
        },
        'Light': {
            'source_bg': '#fefefe', 'source_fg': '#1a1a1a',
            'visual_bg': '#ffffff', 'visual_fg': '#1a1a1a',
            'desk_bg': '#d9dce1', 'ribbon_bg': '#eef1f6',
            'status_bg': '#334155', 'status_fg': '#f8fafc',
            'accent_purple': '#2563eb',
        },
        'Paper (Word)': {
            'source_bg': '#ffffff', 'source_fg': '#000000',
            'visual_bg': '#ffffff', 'visual_fg': '#000000',
            'desk_bg': '#e6e6e6', 'ribbon_bg': '#f5f5f5',
            'status_bg': '#2b579a', 'status_fg': '#ffffff',
            'accent_purple': '#2b579a',
        },
        'Dark': {
            'source_bg': '#1e1e1e', 'source_fg': '#d4d4d4',
            'visual_bg': '#252526', 'visual_fg': '#cccccc',
            'desk_bg': '#141414', 'ribbon_bg': '#2d2d30',
            'status_bg': '#007acc', 'status_fg': '#ffffff',
            'accent_purple': '#3794ff',
        },
        'Sepia': {
            'source_bg': '#f4ecd8', 'source_fg': '#5b4636',
            'visual_bg': '#f9f5eb', 'visual_fg': '#5b4636',
            'desk_bg': '#cdbf9f', 'ribbon_bg': '#efe4cd',
            'status_bg': '#5b4636', 'status_fg': '#f9f5eb',
            'accent_purple': '#8b5e34',
        },
        'Solarized Light': {
            'source_bg': '#fdf6e3', 'source_fg': '#657b83',
            'visual_bg': '#fdf6e3', 'visual_fg': '#657b83',
            'desk_bg': '#eee8d5', 'ribbon_bg': '#e6dfc8',
            'status_bg': '#073642', 'status_fg': '#fdf6e3',
            'accent_purple': '#268bd2',
        },
        'Solarized Dark': {
            'source_bg': '#002b36', 'source_fg': '#839496',
            'visual_bg': '#073642', 'visual_fg': '#93a1a1',
            'desk_bg': '#001f27', 'ribbon_bg': '#003847',
            'status_bg': '#002b36', 'status_fg': '#93a1a1',
            'accent_purple': '#2aa198',
        },
    }
    
    def __init__(self, parent, current_theme, on_apply):
        super().__init__(parent)
        self.parent = parent
        self.theme = current_theme.copy()
        self.on_apply = on_apply
        
        self.title("Theme Settings")
        self.transient(parent)
        
        self._setup_ui()
        _fit_and_center_dialog(self, parent, 860, 720)
        
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda e: self.destroy())
    
    def _setup_ui(self):
        """Setup two-column Theme dialog with MS Word 2003-style live template preview."""
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        content_row = ttk.Frame(main_frame)
        content_row.pack(fill=tk.BOTH, expand=True)

        controls_col = ttk.Frame(content_row)
        controls_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        preview_col = ttk.Frame(content_row)
        preview_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        
        # Presets section
        preset_frame = ttk.LabelFrame(controls_col, text="Presets", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        preset_btn_frame = ttk.Frame(preset_frame)
        preset_btn_frame.pack(fill=tk.X)
        
        for col in range(3):
            preset_btn_frame.columnconfigure(col, weight=1, uniform="preset_btn")
        for i, (name, _) in enumerate(self.PRESETS.items()):
            btn = ttk.Button(preset_btn_frame, text=name,
                           command=lambda n=name: self._apply_preset(n))
            btn.grid(row=i//3, column=i%3, padx=3, pady=3, sticky="ew")
        
        # Source Editor section
        source_frame = ttk.LabelFrame(controls_col, text="Source Editor", padding=10)
        source_frame.pack(fill=tk.X, pady=5)
        
        # Background color
        bg_frame = ttk.Frame(source_frame)
        bg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bg_frame, text="Background:", width=12).pack(side=tk.LEFT)
        self.source_bg_var = tk.StringVar(value=self.theme['source_bg'])
        self.source_bg_preview = tk.Label(bg_frame, width=3, bg=self.theme['source_bg'], relief=tk.SUNKEN)
        self.source_bg_preview.pack(side=tk.LEFT, padx=5)
        ttk.Entry(bg_frame, textvariable=self.source_bg_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(bg_frame, text="Choose...", command=lambda: self._choose_color('source_bg')).pack(side=tk.LEFT, padx=2)
        
        # Foreground color
        fg_frame = ttk.Frame(source_frame)
        fg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(fg_frame, text="Text Color:", width=12).pack(side=tk.LEFT)
        self.source_fg_var = tk.StringVar(value=self.theme['source_fg'])
        self.source_fg_preview = tk.Label(fg_frame, width=3, bg=self.theme['source_fg'], relief=tk.SUNKEN)
        self.source_fg_preview.pack(side=tk.LEFT, padx=5)
        ttk.Entry(fg_frame, textvariable=self.source_fg_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(fg_frame, text="Choose...", command=lambda: self._choose_color('source_fg')).pack(side=tk.LEFT, padx=2)
        
        # Font
        font_frame = ttk.Frame(source_frame)
        font_frame.pack(fill=tk.X, pady=2)
        ttk.Label(font_frame, text="Font:", width=12).pack(side=tk.LEFT)
        self.source_font_var = tk.StringVar(value=self.theme.get('source_font') or MONO_FONT or 'Consolas')
        self.source_font_combo = ttk.Combobox(font_frame, textvariable=self.source_font_var, width=16,
                                  values=['Consolas', 'Courier New', 'Cascadia Code', 'Fira Code', 
                                         'Source Code Pro', 'Monaco', 'Menlo'])
        self.source_font_combo.pack(side=tk.LEFT, padx=2)
        
        # Font size
        ttk.Label(font_frame, text="Size:").pack(side=tk.LEFT, padx=(10, 2))
        self.source_size_var = tk.IntVar(value=self.theme['source_font_size'])
        size_spin = ttk.Spinbox(font_frame, from_=8, to=24, width=5, textvariable=self.source_size_var)
        size_spin.pack(side=tk.LEFT, padx=2)
        
        # Visual Mode section
        visual_frame = ttk.LabelFrame(controls_col, text="Visual Mode", padding=10)
        visual_frame.pack(fill=tk.X, pady=5)
        
        # Background color
        vbg_frame = ttk.Frame(visual_frame)
        vbg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(vbg_frame, text="Background:", width=12).pack(side=tk.LEFT)
        self.visual_bg_var = tk.StringVar(value=self.theme['visual_bg'])
        self.visual_bg_preview = tk.Label(vbg_frame, width=3, bg=self.theme['visual_bg'], relief=tk.SUNKEN)
        self.visual_bg_preview.pack(side=tk.LEFT, padx=5)
        ttk.Entry(vbg_frame, textvariable=self.visual_bg_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(vbg_frame, text="Choose...", command=lambda: self._choose_color('visual_bg')).pack(side=tk.LEFT, padx=2)
        
        # Foreground color
        vfg_frame = ttk.Frame(visual_frame)
        vfg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(vfg_frame, text="Text Color:", width=12).pack(side=tk.LEFT)
        self.visual_fg_var = tk.StringVar(value=self.theme['visual_fg'])
        self.visual_fg_preview = tk.Label(vfg_frame, width=3, bg=self.theme['visual_fg'], relief=tk.SUNKEN)
        self.visual_fg_preview.pack(side=tk.LEFT, padx=5)
        ttk.Entry(vfg_frame, textvariable=self.visual_fg_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(vfg_frame, text="Choose...", command=lambda: self._choose_color('visual_fg')).pack(side=tk.LEFT, padx=2)
        
        # Visual Font
        vfont_frame = ttk.Frame(visual_frame)
        vfont_frame.pack(fill=tk.X, pady=2)
        ttk.Label(vfont_frame, text="Font:", width=12).pack(side=tk.LEFT)
        self.visual_font_var = tk.StringVar(value=self.theme.get('visual_font') or VISUAL_FONT or 'Segoe UI')
        self.visual_font_combo = ttk.Combobox(vfont_frame, textvariable=self.visual_font_var, width=16,
                                   values=['Segoe UI', 'Arial', 'Calibri', 'Georgia', 'Times New Roman',
                                          'Verdana', 'Trebuchet MS'])
        self.visual_font_combo.pack(side=tk.LEFT, padx=2)
        
        # Visual Font size
        ttk.Label(vfont_frame, text="Size:").pack(side=tk.LEFT, padx=(10, 2))
        self.visual_size_var = tk.IntVar(value=self.theme['visual_font_size'])
        vsize_spin = ttk.Spinbox(vfont_frame, from_=8, to=24, width=5, textvariable=self.visual_size_var)
        vsize_spin.pack(side=tk.LEFT, padx=2)
        
        # Desk (background behind the page)
        desk_frame = ttk.LabelFrame(controls_col, text="Page Layout & Blockquote", padding=10)
        desk_frame.pack(fill=tk.X, pady=5)
        
        dbg_frame = ttk.Frame(desk_frame)
        dbg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dbg_frame, text="Desk colour:", width=12).pack(side=tk.LEFT)
        self.desk_bg_var = tk.StringVar(value=self.theme.get('desk_bg', '#d9dce1'))
        self.desk_bg_preview = tk.Label(dbg_frame, width=3, bg=self.desk_bg_var.get(), relief=tk.SUNKEN)
        self.desk_bg_preview.pack(side=tk.LEFT, padx=5)
        ttk.Entry(dbg_frame, textvariable=self.desk_bg_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(dbg_frame, text="Choose...", command=lambda: self._choose_color('desk_bg')).pack(side=tk.LEFT, padx=2)

        bq_btn_row = ttk.Frame(desk_frame)
        bq_btn_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            bq_btn_row,
            text="❝ Customize Blockquote Style...",
            command=lambda: BlockquoteStyleDialog(self, self.theme, self._on_blockquote_updated)
        ).pack(side=tk.LEFT)
        
        # Font Detection Section
        detect_frame = ttk.Frame(controls_col)
        detect_frame.pack(fill=tk.X, pady=(8, 0))
        detect_btn = ttk.Button(detect_frame, text="🔍 Detect System Fonts", 
                                command=self._detect_fonts)
        detect_btn.pack(anchor=tk.CENTER)
        ttk.Label(detect_frame, text="Populates font lists with all available system fonts",
                 font=(SANS_FONT, 8), foreground="#666666").pack(anchor=tk.CENTER, pady=(2, 0))

        # Right Pane: MS Word 2003-style Live Document Template Preview
        preview_frame = ttk.LabelFrame(preview_col, text="Document Template Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_canvas = tk.Canvas(
            preview_frame, width=340, height=490,
            relief=tk.SUNKEN, bd=2, highlightthickness=0, bg="#ffffff"
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<Configure>", self._update_template_preview)
        ttk.Label(
            preview_frame,
            text="Live preview of Ribbon, Desk, Visual Document Page, Blockquote, Source Code Inset, and Status Bar.",
            font=(SANS_FONT, 8), foreground="#666666", wraplength=330, justify=tk.LEFT
        ).pack(fill=tk.X, pady=(6, 0))

        # Wire live updates on variable changes
        for var in (
            self.source_bg_var, self.source_fg_var, self.source_font_var, self.source_size_var,
            self.visual_bg_var, self.visual_fg_var, self.visual_font_var, self.visual_size_var,
            self.desk_bg_var,
        ):
            var.trace_add("write", self._update_template_preview)

        self._update_template_preview()
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(12, 0))
        
        ttk.Button(btn_frame, text="Apply", command=self._apply, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset to Default", command=self._reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy, width=10).pack(side=tk.RIGHT, padx=5)

    @staticmethod
    def _safe_hex(color_val: str, fallback: str) -> str:
        val = (color_val or "").strip()
        if re.match(r'^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$', val):
            return val
        return fallback

    @staticmethod
    def _safe_int(var, fallback: int, lo: int = 8, hi: int = 24) -> int:
        try:
            v = int(var.get())
            return max(lo, min(hi, v))
        except Exception:
            return fallback

    @staticmethod
    def _is_dark_hex(hex_color: str) -> bool:
        try:
            h = hex_color.lstrip('#')
            if len(h) == 3:
                h = ''.join(ch * 2 for ch in h)
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return (0.299 * r + 0.587 * g + 0.114 * b) < 128
        except Exception:
            return False

    def _update_template_preview(self, *args):
        """Render MS Word 2003-style live mini document & source template preview."""
        if not hasattr(self, "preview_canvas"):
            return
        c = self.preview_canvas
        c.delete("all")

        default = MarkdownNotepad.DEFAULT_THEME
        s_bg = self._safe_hex(self.source_bg_var.get(), default['source_bg'])
        s_fg = self._safe_hex(self.source_fg_var.get(), default['source_fg'])
        v_bg = self._safe_hex(self.visual_bg_var.get(), default['visual_bg'])
        v_fg = self._safe_hex(self.visual_fg_var.get(), default['visual_fg'])
        d_bg = self._safe_hex(self.desk_bg_var.get(), default.get('desk_bg', '#dcd6e8'))

        # Keep swatch indicators synchronized cleanly when typing hex codes
        swatch_map = (
            (self.source_bg_preview, s_bg),
            (self.source_fg_preview, s_fg),
            (self.visual_bg_preview, v_bg),
            (self.visual_fg_preview, v_fg),
            (self.desk_bg_preview, d_bg),
        )
        for sw, hex_col in swatch_map:
            try:
                sw.config(bg=hex_col)
            except tk.TclError:
                pass

        s_font = (self.source_font_var.get() or MONO_FONT or 'Consolas').strip()
        v_font = (self.visual_font_var.get() or VISUAL_FONT or 'Segoe UI').strip()
        s_size_raw = self._safe_int(self.source_size_var, 11)
        v_size_raw = self._safe_int(self.visual_size_var, 11)

        # Scaled font sizes suited for the 340x490 mini template canvas
        v_scaled = max(7, min(13, int(round(v_size_raw * 0.78))))
        v_h1_scaled = max(10, min(17, v_scaled + 3))
        s_scaled = max(7, min(12, int(round(s_size_raw * 0.75))))

        rib_bg = self._safe_hex(self.theme.get('ribbon_bg', default['ribbon_bg']), '#f3effa')
        stat_bg = self._safe_hex(self.theme.get('status_bg', default['status_bg']), '#3b1d6e')
        stat_fg = self._safe_hex(self.theme.get('status_fg', default['status_fg']), '#f5f3ff')
        accent = self._safe_hex(self.theme.get('accent_purple', default['accent_purple']), '#5b21b6')
        rib_fg = "#f3e8ff" if self._is_dark_hex(rib_bg) else "#1e1b4b"
        rib_btn_bg = "#3b235a" if self._is_dark_hex(rib_bg) else "#ffffff"
        rib_border = "#6b21a8" if self._is_dark_hex(rib_bg) else "#d8b4fe"

        cw = max(340, c.winfo_width())
        ch = max(490, c.winfo_height())
        sb_y0 = ch - 28
        page_w = min(cw - 32, 340)
        px0 = max(16, (cw - page_w) // 2)
        px1 = px0 + page_w
        py0 = 46
        page_h = max(250, min(320, ch - 208))
        py1 = py0 + page_h

        # 1. Mini Word Ribbon Bar (y=0..32)
        c.create_rectangle(0, 0, cw, 32, fill=rib_bg, outline="", tags="preview_ribbon")
        c.create_line(0, 32, cw, 32, fill=accent, width=2, tags="preview_ribbon")
        px = 10
        for pill_lbl in ("New", "Save", "B", "I", "Theme"):
            pw = max(22, len(pill_lbl) * 6 + 10)
            c.create_rectangle(px, 6, px + pw, 25, fill=rib_btn_bg, outline=rib_border, width=1, tags="preview_ribbon")
            c.create_text(
                px + pw // 2, 16, text=pill_lbl,
                font=("Segoe UI", 7, "bold" if pill_lbl == "B" else "normal"),
                fill=rib_fg, tags="preview_ribbon"
            )
            px += pw + 6

        # 2. Desk Surface (y=33..sb_y0)
        c.create_rectangle(0, 33, cw, sb_y0, fill=d_bg, outline="", tags="preview_desk")

        # 3. Visual Document Page with Drop Shadow
        c.create_rectangle(px0 + 4, py0 + 4, px1 + 4, py1 + 4, fill="#555555", outline="", tags="preview_page")
        c.create_rectangle(px0, py0, px1, py1, fill=v_bg, outline="#888888", width=1, tags="preview_page")

        # Classic MS Word 2003 Margin Crop Marks (18px inner margins)
        ml, mr, mt, mb = px0 + 18, px1 - 18, py0 + 18, py1 - 18
        text_wrap_w = max(180, (mr - ml) - 10)
        crop_col = "#94a3b8"
        c.create_line(ml - 7, mt, ml, mt, ml, mt - 7, fill=crop_col, width=1, tags="preview_page")
        c.create_line(mr + 7, mt, mr, mt, mr, mt - 7, fill=crop_col, width=1, tags="preview_page")
        c.create_line(ml - 7, mb, ml, mb, ml, mb + 7, fill=crop_col, width=1, tags="preview_page")
        c.create_line(mr + 7, mb, mr, mb, mr, mb + 7, fill=crop_col, width=1, tags="preview_page")

        # Document H1 Title + Horizontal Rule
        h1_col = accent if not self._is_dark_hex(v_bg) else v_fg
        c.create_text(
            ml + 4, mt + 6, text="Document Template",
            font=(v_font, v_h1_scaled, "bold"), fill=h1_col, anchor="nw", tags="preview_page"
        )
        c.create_line(ml + 4, mt + 28, mr - 4, mt + 28, fill=accent, width=1, tags="preview_page")

        # Body Prose Paragraph
        body_text = (
            "The quick brown fox jumps over the lazy dog. "
            "Preview how headings, prose typography, and margins appear on the printed page."
        )
        c.create_text(
            ml + 4, mt + 36, text=body_text,
            font=(v_font, v_scaled), fill=v_fg, width=text_wrap_w, anchor="nw", tags="preview_page"
        )

        # Blockquote Sample Box inside Page
        bq_bg = self._safe_hex(self.theme.get('blockquote_bg', ''), "#2d1d42" if self._is_dark_hex(v_bg) else "#f5f3ff")
        bq_fg = self._safe_hex(self.theme.get('blockquote_fg', ''), v_fg)
        bq_bar = self._safe_hex(self.theme.get('blockquote_bar_fg', ''), accent)
        bq_font = (self.theme.get('blockquote_font') or v_font).strip()
        bq_raw_sz = int(self.theme.get('blockquote_size') or 0)
        bq_scaled = max(7, min(13, int(round((bq_raw_sz if bq_raw_sz > 0 else v_size_raw) * 0.78))))
        bq_style = self.theme.get('blockquote_style') or 'italic'
        bq_justify = self.theme.get('blockquote_justify') or 'left'
        bq_y0, bq_y1 = mt + 106, mt + 150
        c.create_rectangle(ml + 6, bq_y0, mr - 6, bq_y1, fill=bq_bg, outline="", tags="preview_page")
        c.create_line(ml + 8, bq_y0, ml + 8, bq_y1, fill=bq_bar, width=4, tags="preview_page")
        bq_tx = ml + 18 if bq_justify == 'left' else ((ml + mr) // 2 if bq_justify == 'center' else mr - 14)
        bq_anc = 'w' if bq_justify == 'left' else ('center' if bq_justify == 'center' else 'e')
        c.create_text(
            bq_tx, (bq_y0 + bq_y1) // 2,
            text="\"Blockquote style & indentation preview.\"",
            font=(bq_font, bq_scaled, bq_style), fill=bq_fg, anchor=bq_anc,
            width=max(160, text_wrap_w - 30), tags="preview_page"
        )

        # Sample Bulleted List Rows inside Page
        c.create_text(
            ml + 10, bq_y1 + 12,
            text="• Primary section outline item\n• Secondary formatted note sample",
            font=(v_font, max(7, v_scaled - 1)), fill=v_fg, anchor="nw", tags="preview_page"
        )

        # 4. Source Editor Template Card
        sy0 = py1 + 12
        sy1 = min(sb_y0 - 12, sy0 + 110)
        c.create_rectangle(px0 + 4, sy0 + 4, px1 + 4, sy1 + 4, fill="#555555", outline="", tags="preview_source")
        c.create_rectangle(px0, sy0, px1, sy1, fill=s_bg, outline="#888888", width=1, tags="preview_source")
        c.create_rectangle(px0, sy0, px0 + 155, sy0 + 18, fill=accent, outline="", tags="preview_source")
        c.create_text(
            px0 + 8, sy0 + 9, text="Source View (Markdown)",
            font=("Segoe UI", 7, "bold"), fill="#ffffff", anchor="w", tags="preview_source"
        )
        c.create_text(
            px0 + 12, sy0 + 28,
            text="# Heading 1\n**Bold text** and `inline_code`\n> Markdown source preview",
            font=(s_font, s_scaled), fill=s_fg, anchor="nw", tags="preview_source"
        )

        # 5. Mini Status Bar
        c.create_rectangle(0, sb_y0, cw, ch, fill=stat_bg, outline="", tags="preview_status")
        c.create_text(
            12, ch - 14, text="Page 1  |  142 words  |  Visual View",
            font=("Segoe UI", 8), fill=stat_fg, anchor="w", tags="preview_status"
        )

    def _on_blockquote_updated(self, updated_theme):
        """Merge blockquote updates from BlockquoteStyleDialog into ThemeDialog's working theme."""
        self.theme.update(updated_theme)
        self._update_template_preview()
        self.on_apply(self.theme)
    
    def _detect_fonts(self):
        """Detect all available system fonts and populate the comboboxes"""
        fonts = get_available_fonts()
        
        if not fonts:
            messagebox.showwarning(
                "Font Detection",
                "Could not detect system fonts. Using default font lists.",
                parent=self
            )
            return
        
        # Get current selections
        current_source = self.source_font_var.get()
        current_visual = self.visual_font_var.get()
        
        # Update combobox values with detected fonts
        self.source_font_combo['values'] = fonts
        self.visual_font_combo['values'] = fonts
        
        # Restore selections if they exist in the new list
        if current_source in fonts:
            self.source_font_var.set(current_source)
        if current_visual in fonts:
            self.visual_font_var.set(current_visual)
        
        messagebox.showinfo(
            "Font Detection",
            f"Successfully detected {len(fonts)} system fonts!\n\n"
            f"Font lists have been updated. You can now select any installed font.",
            parent=self
        )
    
    def _choose_color(self, key):
        """Open color chooser for the specified key"""
        targets = {
            'source_bg': (self.source_bg_var, self.source_bg_preview),
            'source_fg': (self.source_fg_var, self.source_fg_preview),
            'visual_bg': (self.visual_bg_var, self.visual_bg_preview),
            'visual_fg': (self.visual_fg_var, self.visual_fg_preview),
            'desk_bg': (self.desk_bg_var, self.desk_bg_preview),
        }
        var, preview = targets[key]
        color = colorchooser.askcolor(color=var.get(), title=f"Choose {key.replace('_', ' ').title()}")
        if color[1]:
            var.set(color[1])
            preview.config(bg=color[1])
            self._update_template_preview()
    
    def _apply_preset(self, preset_name):
        """Apply a preset theme including chrome keys and update preview."""
        preset = self.PRESETS[preset_name]
        default = MarkdownNotepad.DEFAULT_THEME

        for chrome_key in ('ribbon_bg', 'status_bg', 'status_fg', 'accent_purple'):
            self.theme[chrome_key] = preset.get(chrome_key, default[chrome_key])

        self.source_bg_var.set(preset['source_bg'])
        self.source_bg_preview.config(bg=preset['source_bg'])
        self.source_fg_var.set(preset['source_fg'])
        self.source_fg_preview.config(bg=preset['source_fg'])
        self.visual_bg_var.set(preset['visual_bg'])
        self.visual_bg_preview.config(bg=preset['visual_bg'])
        self.visual_fg_var.set(preset['visual_fg'])
        self.visual_fg_preview.config(bg=preset['visual_fg'])
        if 'desk_bg' in preset:
            self.desk_bg_var.set(preset['desk_bg'])
            self.desk_bg_preview.config(bg=preset['desk_bg'])
        self._update_template_preview()
    
    def _apply(self):
        """Apply the current settings (merged into the existing theme so other keys survive)"""
        self.theme = {
            **self.theme,
            'source_bg': self.source_bg_var.get(),
            'source_fg': self.source_fg_var.get(),
            'source_font': self.source_font_var.get() or MONO_FONT,
            'source_font_size': self._safe_int(self.source_size_var, 11),
            'visual_bg': self.visual_bg_var.get(),
            'visual_fg': self.visual_fg_var.get(),
            'visual_font': self.visual_font_var.get() or VISUAL_FONT,
            'visual_font_size': self._safe_int(self.visual_size_var, 11),
            'desk_bg': self.desk_bg_var.get(),
        }
        self.on_apply(self.theme)
        self.destroy()
    
    def _reset(self):
        """Reset to default theme"""
        default = MarkdownNotepad.DEFAULT_THEME

        for chrome_key in ('ribbon_bg', 'status_bg', 'status_fg', 'accent_purple'):
            self.theme[chrome_key] = default[chrome_key]

        self.source_bg_var.set(default['source_bg'])
        self.source_bg_preview.config(bg=default['source_bg'])
        self.source_fg_var.set(default['source_fg'])
        self.source_fg_preview.config(bg=default['source_fg'])
        self.source_font_var.set(default['source_font'] or MONO_FONT)
        self.source_size_var.set(default['source_font_size'])
        self.visual_bg_var.set(default['visual_bg'])
        self.visual_bg_preview.config(bg=default['visual_bg'])
        self.visual_fg_var.set(default['visual_fg'])
        self.visual_fg_preview.config(bg=default['visual_fg'])
        self.visual_font_var.set(default['visual_font'] or VISUAL_FONT)
        self.visual_size_var.set(default['visual_font_size'])
        self.desk_bg_var.set(default['desk_bg'])
        self.desk_bg_preview.config(bg=default['desk_bg'])
        self._update_template_preview()


class BlockquoteStyleDialog(tk.Toplevel):
    """Dialog for customizing the single Blockquote (>) appearance:
    font family, size, style, text color, background color, bar color,
    padding, vertical spacing, and justification.
    """

    def __init__(self, parent, current_theme, on_apply):
        super().__init__(parent)
        self.parent = parent
        self.theme = current_theme.copy()
        self.on_apply = on_apply

        self.title("Customize Blockquote Style (>)")
        self.transient(parent)
        self.grab_set()

        self._setup_ui()
        _fit_and_center_dialog(self, parent, 500, 520)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda e: self.destroy())

    def _setup_ui(self):
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        # Typography group
        typo = ttk.LabelFrame(frame, text="Typography", padding=10)
        typo.pack(fill=tk.X, pady=(0, 8))

        r1 = ttk.Frame(typo)
        r1.pack(fill=tk.X, pady=3)
        ttk.Label(r1, text="Font Family:", width=15).pack(side=tk.LEFT)
        self.font_var = tk.StringVar(value=self.theme.get('blockquote_font', ''))
        fonts = ['', 'Georgia', 'Times New Roman', 'Segoe UI', 'Calibri', 'Arial', 'Consolas', 'Verdana']
        ttk.Combobox(r1, textvariable=self.font_var, values=fonts, width=22).pack(side=tk.LEFT, padx=4)
        ttk.Label(r1, text="(blank = theme)", foreground="#64748b", font=(SANS_FONT, 8)).pack(side=tk.LEFT)

        r2 = ttk.Frame(typo)
        r2.pack(fill=tk.X, pady=3)
        ttk.Label(r2, text="Font Size (pt):", width=15).pack(side=tk.LEFT)
        self.size_var = tk.IntVar(value=int(self.theme.get('blockquote_size') or 0))
        ttk.Spinbox(r2, from_=0, to=36, textvariable=self.size_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(r2, text="(0 = inherit body size)", foreground="#64748b", font=(SANS_FONT, 8)).pack(side=tk.LEFT)

        r3 = ttk.Frame(typo)
        r3.pack(fill=tk.X, pady=3)
        ttk.Label(r3, text="Font Style:", width=15).pack(side=tk.LEFT)
        self.style_var = tk.StringVar(value=self.theme.get('blockquote_style', 'italic'))
        ttk.Combobox(
            r3, textvariable=self.style_var,
            values=['italic', 'normal', 'bold', 'bold italic'],
            state='readonly', width=16
        ).pack(side=tk.LEFT, padx=4)

        # Colors group
        col_frame = ttk.LabelFrame(frame, text="Colors (leave blank for automatic theme colors)", padding=10)
        col_frame.pack(fill=tk.X, pady=6)

        self.fg_var = tk.StringVar(value=self.theme.get('blockquote_fg', ''))
        self.bg_var = tk.StringVar(value=self.theme.get('blockquote_bg', ''))
        self.bar_var = tk.StringVar(value=self.theme.get('blockquote_bar_fg', ''))

        for label_text, var in (
            ("Text Color:", self.fg_var),
            ("Background:", self.bg_var),
            ("Accent Bar Color:", self.bar_var),
        ):
            row = ttk.Frame(col_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label_text, width=15).pack(side=tk.LEFT)
            ent = ttk.Entry(row, textvariable=var, width=12)
            ent.pack(side=tk.LEFT, padx=4)
            ttk.Button(
                row, text="Pick...", width=8,
                command=lambda v=var: self._pick_color(v)
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(
                row, text="Auto", width=6,
                command=lambda v=var: v.set('')
            ).pack(side=tk.LEFT, padx=2)

        # Layout & Justification group
        layout = ttk.LabelFrame(frame, text="Padding, Spacing & Justification", padding=10)
        layout.pack(fill=tk.X, pady=6)

        l1 = ttk.Frame(layout)
        l1.pack(fill=tk.X, pady=3)
        ttk.Label(l1, text="Side Padding (px):", width=15).pack(side=tk.LEFT)
        self.pad_var = tk.IntVar(value=int(self.theme.get('blockquote_padding', 16)))
        ttk.Spinbox(l1, from_=4, to=80, textvariable=self.pad_var, width=6).pack(side=tk.LEFT, padx=4)

        ttk.Label(l1, text="Vertical Space:", width=13).pack(side=tk.LEFT, padx=(12, 0))
        self.space_var = tk.IntVar(value=int(self.theme.get('blockquote_spacing', 4)))
        ttk.Spinbox(l1, from_=0, to=32, textvariable=self.space_var, width=6).pack(side=tk.LEFT, padx=4)

        l2 = ttk.Frame(layout)
        l2.pack(fill=tk.X, pady=3)
        ttk.Label(l2, text="Justification:", width=15).pack(side=tk.LEFT)
        self.justify_var = tk.StringVar(value=self.theme.get('blockquote_justify', 'left'))
        for val, lbl in (('left', 'Left'), ('center', 'Center'), ('right', 'Right')):
            ttk.Radiobutton(l2, text=lbl, value=val, variable=self.justify_var).pack(side=tk.LEFT, padx=6)

        # Action buttons
        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(btns, text="Apply", command=self._apply, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Reset Defaults", command=self._reset).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=self.destroy, width=10).pack(side=tk.RIGHT, padx=4)

    def _pick_color(self, var: tk.StringVar):
        initial = var.get().strip() or "#475569"
        color = colorchooser.askcolor(color=initial, parent=self)
        if color and color[1]:
            var.set(color[1])

    def _reset(self):
        self.font_var.set('')
        self.size_var.set(0)
        self.style_var.set('italic')
        self.fg_var.set('')
        self.bg_var.set('')
        self.bar_var.set('')
        self.pad_var.set(16)
        self.space_var.set(4)
        self.justify_var.set('left')

    def _apply(self):
        self.theme.update({
            'blockquote_font': self.font_var.get().strip(),
            'blockquote_size': int(self.size_var.get() or 0),
            'blockquote_style': self.style_var.get(),
            'blockquote_fg': self.fg_var.get().strip(),
            'blockquote_bg': self.bg_var.get().strip(),
            'blockquote_bar_fg': self.bar_var.get().strip(),
            'blockquote_padding': int(self.pad_var.get() or 16),
            'blockquote_spacing': int(self.space_var.get() or 4),
            'blockquote_justify': self.justify_var.get(),
        })
        if self.on_apply:
            self.on_apply(self.theme)
        self.destroy()


class InsertImageDialog(tk.Toplevel):
    """Dialog for inserting images with size options"""
    
    def __init__(self, parent, text_widget, current_file, get_assets_folder_func):
        super().__init__(parent)
        self.parent = parent
        self.text_widget = text_widget
        self.current_file = current_file
        self.get_assets_folder = get_assets_folder_func
        self.image_path = None
        self.original_size = (0, 0)
        
        self.title("Insert Image")
        self.transient(parent)
        
        self._setup_ui()
        _fit_and_center_dialog(self, parent, 540, 440)
        
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda e: self.destroy())
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Image File", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_var, width=50)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(file_frame, text="Browse...", command=self._browse_image).pack(side=tk.LEFT)
        
        # Preview area
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.preview_label = ttk.Label(preview_frame, text="No image selected", anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        self.size_info_label = ttk.Label(preview_frame, text="")
        self.size_info_label.pack()
        
        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Alt text
        alt_frame = ttk.Frame(options_frame)
        alt_frame.pack(fill=tk.X, pady=2)
        ttk.Label(alt_frame, text="Alt text:", width=12).pack(side=tk.LEFT)
        self.alt_var = tk.StringVar()
        ttk.Entry(alt_frame, textvariable=self.alt_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Size options
        size_frame = ttk.Frame(options_frame)
        size_frame.pack(fill=tk.X, pady=2)
        ttk.Label(size_frame, text="Display size:", width=12).pack(side=tk.LEFT)
        
        self.size_mode = tk.StringVar(value="original")
        ttk.Radiobutton(size_frame, text="Original", variable=self.size_mode, 
                       value="original", command=self._update_size_preview).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(size_frame, text="Custom", variable=self.size_mode,
                       value="custom", command=self._update_size_preview).pack(side=tk.LEFT, padx=5)
        
        # Custom size inputs
        custom_frame = ttk.Frame(options_frame)
        custom_frame.pack(fill=tk.X, pady=2)
        ttk.Label(custom_frame, text="", width=12).pack(side=tk.LEFT)
        
        ttk.Label(custom_frame, text="Width:").pack(side=tk.LEFT)
        self.width_var = tk.StringVar()
        self.width_entry = ttk.Entry(custom_frame, textvariable=self.width_var, width=6)
        self.width_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(custom_frame, text="Height:").pack(side=tk.LEFT, padx=(10, 0))
        self.height_var = tk.StringVar()
        self.height_entry = ttk.Entry(custom_frame, textvariable=self.height_var, width=6)
        self.height_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(custom_frame, text="(leave blank to auto-calculate)").pack(side=tk.LEFT, padx=5)
        
        # Syntax choice
        syntax_frame = ttk.Frame(options_frame)
        syntax_frame.pack(fill=tk.X, pady=2)
        ttk.Label(syntax_frame, text="Syntax:", width=12).pack(side=tk.LEFT)
        
        self.syntax_mode = tk.StringVar(value="markdown")
        ttk.Radiobutton(syntax_frame, text="Markdown ![alt|size](path)", 
                       variable=self.syntax_mode, value="markdown").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(syntax_frame, text="HTML <img>", 
                       variable=self.syntax_mode, value="html").pack(side=tk.LEFT, padx=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Insert", command=self._insert_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _browse_image(self):
        """Browse for an image file"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
            ("All files", "*.*")
        ]
        
        path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        if path:
            self.file_var.set(path)
            self.image_path = path
            self.alt_var.set(os.path.splitext(os.path.basename(path))[0])
            self._load_preview()
    
    def _load_preview(self):
        """Load and display image preview"""
        if not self.image_path or not PIL_AVAILABLE:
            return
        
        try:
            img = Image.open(self.image_path)
            self.original_size = img.size
            
            # Create thumbnail for preview
            preview_size = (200, 150)
            img.thumbnail(preview_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo  # Keep reference
            
            self.size_info_label.config(
                text=f"Original size: {self.original_size[0]} × {self.original_size[1]} pixels"
            )
            
            # Set default custom size
            self.width_var.set(str(self.original_size[0]))
            self.height_var.set(str(self.original_size[1]))
            
        except Exception as e:
            self.preview_label.config(image="", text=f"Error loading image: {e}")
            self.size_info_label.config(text="")
    
    def _update_size_preview(self):
        """Update size entry states based on mode"""
        if self.size_mode.get() == "custom":
            self.width_entry.config(state="normal")
            self.height_entry.config(state="normal")
        else:
            self.width_entry.config(state="disabled")
            self.height_entry.config(state="disabled")
    
    def _insert_image(self):
        """Insert the image into the document"""
        if not self.image_path:
            messagebox.showwarning("No Image", "Please select an image file first.")
            return
        
        alt_text = self.alt_var.get()
        
        # Determine the path to use
        if self.current_file:
            # Copy to assets folder
            assets_folder = self.get_assets_folder()
            if assets_folder:
                image_filename = os.path.basename(self.image_path)
                dest_path = os.path.join(assets_folder, image_filename)
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(image_filename)
                    dest_path = os.path.join(assets_folder, f"{name}_{counter}{ext}")
                    counter += 1
                
                shutil.copy2(self.image_path, dest_path)
                doc_dir = os.path.dirname(self.current_file)
                rel_path = os.path.relpath(dest_path, doc_dir).replace('\\', '/')
            else:
                rel_path = self.image_path.replace('\\', '/')
        else:
            # No saved document - use absolute path with warning
            if not messagebox.askyesno("Unsaved Document",
                    "Document not saved. Image will use absolute path which may break "
                    "if you move files. Continue?"):
                return
            rel_path = self.image_path.replace('\\', '/')
        
        # Build the markdown/HTML
        if self.size_mode.get() == "custom":
            width = self.width_var.get().strip()
            height = self.height_var.get().strip()
        else:
            width = height = ""
        
        if self.syntax_mode.get() == "html":
            # HTML img tag
            attrs = [f'src="{rel_path}"']
            if alt_text:
                attrs.append(f'alt="{alt_text}"')
            if width:
                attrs.append(f'width="{width}"')
            if height:
                attrs.append(f'height="{height}"')
            
            markdown = f'<img {" ".join(attrs)}>'
        else:
            # Markdown syntax with optional size in alt text
            if width or height:
                size_str = f"{width}x{height}" if width and height else (width or height)
                markdown = f"![{alt_text}|{size_str}]({rel_path})"
            else:
                markdown = f"![{alt_text}]({rel_path})"
        
        # Insert into editor
        self.text_widget.insert(tk.INSERT, markdown)
        if hasattr(self.parent, '_schedule_syntax_highlight'):
            self.parent._schedule_syntax_highlight()
        if hasattr(self.parent, '_refresh_active_visual_view'):
            self.parent._refresh_active_visual_view()
        
        # Update status
        if hasattr(self.parent, '_set_status'):
            self.parent._set_status(f"Image inserted: {rel_path}")
        
        self.destroy()


class ImageManagerDialog(tk.Toplevel):
    """Dialog for managing document images"""
    
    def __init__(self, parent, document_path, text_widget):
        super().__init__(parent)
        self.parent = parent
        self.document_path = document_path
        self.text_widget = text_widget
        self.doc_dir = os.path.dirname(document_path)
        
        self.title("Image Manager")
        self.geometry("600x450")
        self.resizable(True, True)
        self.transient(parent)
        
        self._setup_ui()
        self._load_images()
        
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda e: self.destroy())
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Info label
        info_text = f"Document: {os.path.basename(self.document_path)}"
        ttk.Label(main_frame, text=info_text, font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        
        # Image list
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Treeview for images
        columns = ("alt", "path", "status")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=12)
        self.tree.heading("alt", text="Alt Text")
        self.tree.heading("path", text="Path")
        self.tree.heading("status", text="Status")
        self.tree.column("alt", width=150)
        self.tree.column("path", width=300)
        self.tree.column("status", width=80)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="Add Image...", command=self._add_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Fix Broken Paths", command=self._fix_broken).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Open Assets Folder", command=self._open_assets_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Refresh", command=self._load_images).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=2)
        
        # Stats
        self.stats_label = ttk.Label(main_frame, text="")
        self.stats_label.pack(anchor=tk.W, pady=(10, 0))
    
    def _load_images(self):
        """Load and display images from the document"""
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Find images in document
        content = self.text_widget.get(1.0, tk.END)
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(pattern, content)
        
        total = len(matches)
        found = 0
        missing = 0
        embedded = 0
        
        for alt, path in matches:
            if path.startswith('data:'):
                self.tree.insert("", tk.END, values=(alt or "(no alt)", "Embedded base64", "✓ Embedded"))
                embedded += 1
            else:
                # Check if file exists
                if os.path.isabs(path):
                    full_path = path
                else:
                    full_path = os.path.join(self.doc_dir, path)
                
                if os.path.exists(full_path):
                    self.tree.insert("", tk.END, values=(alt or "(no alt)", path, "✓ Found"))
                    found += 1
                else:
                    self.tree.insert("", tk.END, values=(alt or "(no alt)", path, "✗ Missing"), tags=("missing",))
                    missing += 1
        
        # Style missing items
        self.tree.tag_configure("missing", foreground="red")
        
        # Update stats
        self.stats_label.config(
            text=f"Total: {total} images | Found: {found} | Missing: {missing} | Embedded: {embedded}"
        )
    
    def _add_image(self):
        """Add a new image to the document"""
        self.parent.insert_image()
        self._load_images()
    
    def _remove_image(self):
        """Remove selected image reference from document"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select an image to remove.")
            return
        
        item = selection[0]
        values = self.tree.item(item, "values")
        alt, path, status = values
        
        if messagebox.askyesno("Confirm Removal", 
                               f"Remove this image reference from the document?\n\n"
                               f"Alt: {alt}\nPath: {path}\n\n"
                               "Note: This only removes the reference, not the actual file."):
            # Find and remove from document
            content = self.text_widget.get(1.0, tk.END)
            
            # Escape special regex characters in path
            escaped_path = re.escape(path)
            escaped_alt = re.escape(alt) if alt != "(no alt)" else "[^\\]]*"
            
            pattern = rf'!\[{escaped_alt}\]\({escaped_path}\)'
            new_content = re.sub(pattern, '', content, count=1)
            
            # Update document
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(1.0, new_content.rstrip())
            
            self._load_images()
    
    def _fix_broken(self):
        """Try to fix broken image paths"""
        content = self.text_widget.get(1.0, tk.END)
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        fixed_count = 0
        
        def fix_match(match):
            nonlocal fixed_count
            alt = match.group(1)
            path = match.group(2)
            
            # Skip embedded images
            if path.startswith('data:'):
                return match.group(0)
            
            # Check if file exists
            if os.path.isabs(path):
                full_path = path
            else:
                full_path = os.path.join(self.doc_dir, path)
            
            if os.path.exists(full_path):
                return match.group(0)
            
            # Try to find the file
            filename = os.path.basename(path)
            
            # Search in document directory and assets folder
            search_locations = [
                self.doc_dir,
                os.path.join(self.doc_dir, f"{os.path.splitext(os.path.basename(self.document_path))[0]}_assets"),
            ]
            
            for location in search_locations:
                if os.path.exists(location):
                    for root, dirs, files in os.walk(location):
                        if filename in files:
                            found_path = os.path.join(root, filename)
                            rel_path = os.path.relpath(found_path, self.doc_dir).replace('\\', '/')
                            fixed_count += 1
                            return f"![{alt}]({rel_path})"
            
            return match.group(0)
        
        new_content = re.sub(pattern, fix_match, content)
        
        if fixed_count > 0:
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(1.0, new_content.rstrip())
            messagebox.showinfo("Fixed", f"Fixed {fixed_count} broken image path(s).")
            self._load_images()
        else:
            messagebox.showinfo("No Fixes", "No broken paths could be automatically fixed.")
    
    def _open_assets_folder(self):
        """Open the assets folder in file explorer"""
        doc_name = os.path.splitext(os.path.basename(self.document_path))[0]
        assets_folder = os.path.join(self.doc_dir, f"{doc_name}_assets")
        
        if not os.path.exists(assets_folder):
            if messagebox.askyesno("Create Folder?", 
                                   f"Assets folder doesn't exist:\n{assets_folder}\n\nCreate it?"):
                os.makedirs(assets_folder)
            else:
                return
        
        # Open in file explorer (Windows)
        os.startfile(assets_folder)


class FindReplaceDialog(tk.Toplevel):
    """Advanced Find & Replace dialog with regex support"""
    
    def __init__(self, parent, text_widget):
        super().__init__(parent)
        self.text_widget = text_widget
        self.parent = parent
        self.title("Find & Replace")
        self.transient(parent)
        
        # Track state
        self.search_pos = "1.0"
        self.match_count = 0
        self.all_matches = []
        
        self._setup_ui()
        self._setup_bindings()
        _fit_and_center_dialog(self, parent, 520, 260)
        
        # Clear highlights when dialog closes
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Focus search entry
        self.search_entry.focus()
    
    def _on_close(self):
        """Handle dialog close - clear highlights"""
        self.clear_highlights()
        self.destroy()
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Search field
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=2)
        ttk.Label(search_frame, text="Find:", width=10).pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=40)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Replace field
        replace_frame = ttk.Frame(main_frame)
        replace_frame.pack(fill=tk.X, pady=2)
        ttk.Label(replace_frame, text="Replace:", width=10).pack(side=tk.LEFT)
        self.replace_var = tk.StringVar()
        self.replace_entry = ttk.Entry(replace_frame, textvariable=self.replace_var, width=40)
        self.replace_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Options frame
        options_frame = ttk.Frame(main_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        self.case_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Match case", variable=self.case_var).pack(side=tk.LEFT, padx=5)
        
        self.regex_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Regex", variable=self.regex_var).pack(side=tk.LEFT, padx=5)
        
        self.whole_word_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Whole word", variable=self.whole_word_var).pack(side=tk.LEFT, padx=5)
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Find Next", command=self.find_next, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Find All", command=self.find_all, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Replace", command=self.replace, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Replace All", command=self.replace_all, width=12).pack(side=tk.LEFT, padx=2)
        
        # Second row of buttons
        btn_frame2 = ttk.Frame(main_frame)
        btn_frame2.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_frame2, text="Clear Highlights", command=self.clear_highlights, width=14).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame2, text="Close", command=self._on_close, width=12).pack(side=tk.RIGHT, padx=2)
        
        # Status label
        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="gray")
        self.status_label.pack(fill=tk.X, pady=5)
    
    def _setup_bindings(self):
        """Setup keyboard bindings"""
        self.search_entry.bind("<Return>", lambda e: self.find_next())
        self.replace_entry.bind("<Return>", lambda e: self.replace())
        self.bind("<Escape>", lambda e: self._on_close())
        self.bind("<F3>", lambda e: self.find_next())
        self.search_var.trace_add("write", lambda *args: self._on_search_change())
    
    def _on_search_change(self):
        """Reset search position when search text changes"""
        self.search_pos = "1.0"
        self.status_var.set("")
    
    def _get_search_pattern(self):
        """Build search pattern based on options"""
        pattern = self.search_var.get()
        if not pattern:
            return None
        
        if not self.regex_var.get():
            pattern = re.escape(pattern)
        
        if self.whole_word_var.get():
            pattern = r'\b' + pattern + r'\b'
        
        flags = 0 if self.case_var.get() else re.IGNORECASE
        
        try:
            return re.compile(pattern, flags)
        except re.error as e:
            self.status_var.set(f"Regex error: {e}")
            return None
    
    def find_next(self):
        """Find the next occurrence"""
        pattern = self._get_search_pattern()
        if not pattern:
            return
        
        # Clear previous highlight
        self.text_widget.tag_remove("search_current", "1.0", tk.END)
        
        # Get text from current position to end
        content = self.text_widget.get(self.search_pos, tk.END)
        match = pattern.search(content)
        
        if match:
            # Calculate absolute position
            start_idx = self.text_widget.index(f"{self.search_pos}+{match.start()}c")
            end_idx = self.text_widget.index(f"{self.search_pos}+{match.end()}c")
            
            # Highlight current match
            self.text_widget.tag_add("search_current", start_idx, end_idx)
            self.text_widget.tag_config("search_current", background="#ffff00", foreground="#000000")
            
            # Move cursor and scroll
            self.text_widget.mark_set(tk.INSERT, start_idx)
            self.text_widget.see(start_idx)
            
            # Update position for next search
            self.search_pos = end_idx
            self.status_var.set(f"Found at line {start_idx.split('.')[0]}")
        else:
            # Try from beginning
            if self.search_pos != "1.0":
                self.search_pos = "1.0"
                self.status_var.set("Wrapped to beginning...")
                self.find_next()
            else:
                self.status_var.set("No matches found")
    
    def find_all(self):
        """Find and highlight all occurrences"""
        pattern = self._get_search_pattern()
        if not pattern:
            return
        
        # Clear previous highlights
        self.text_widget.tag_remove("search", "1.0", tk.END)
        self.text_widget.tag_remove("search_current", "1.0", tk.END)
        
        # Search entire content
        content = self.text_widget.get("1.0", tk.END)
        self.all_matches = list(pattern.finditer(content))
        
        # Highlight all matches
        for match in self.all_matches:
            start_idx = f"1.0+{match.start()}c"
            end_idx = f"1.0+{match.end()}c"
            self.text_widget.tag_add("search", start_idx, end_idx)
        
        self.text_widget.tag_config("search", background="#90EE90")
        
        count = len(self.all_matches)
        self.status_var.set(f"Found {count} occurrence{'s' if count != 1 else ''}")
        self.match_count = count
    
    def replace(self):
        """Replace current occurrence"""
        # Check if there's a current selection matching search
        try:
            sel_start = self.text_widget.index(tk.SEL_FIRST)
            sel_end = self.text_widget.index(tk.SEL_LAST)
            selected = self.text_widget.get(sel_start, sel_end)
            
            pattern = self._get_search_pattern()
            if pattern and pattern.fullmatch(selected):
                # Replace the selection
                replacement = self.replace_var.get()
                if self.regex_var.get():
                    replacement = pattern.sub(replacement, selected)
                
                self.text_widget.delete(sel_start, sel_end)
                self.text_widget.insert(sel_start, replacement)
                self.status_var.set("Replaced")
        except tk.TclError:
            pass
        
        # Find next
        self.find_next()
    
    def replace_all(self):
        """Replace all occurrences"""
        pattern = self._get_search_pattern()
        if not pattern:
            return
        
        replacement = self.replace_var.get()
        content = self.text_widget.get("1.0", tk.END)
        
        # Count and replace
        new_content, count = pattern.subn(replacement, content)
        
        if count > 0:
            # Remember cursor position
            cursor_pos = self.text_widget.index(tk.INSERT)
            
            # Replace all content
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", new_content.rstrip('\n'))
            
            # Try to restore cursor
            try:
                self.text_widget.mark_set(tk.INSERT, cursor_pos)
            except:
                pass
            
            self.status_var.set(f"Replaced {count} occurrence{'s' if count != 1 else ''}")
        else:
            self.status_var.set("No matches found")
    
    def clear_highlights(self):
        """Clear all search highlights"""
        self.text_widget.tag_remove("search", "1.0", tk.END)
        self.text_widget.tag_remove("search_current", "1.0", tk.END)
        self.status_var.set("Highlights cleared")


class GoToLineDialog(tk.Toplevel):
    """Simple go-to-line dialog"""
    
    def __init__(self, parent, text_widget):
        super().__init__(parent)
        self.text_widget = text_widget
        self.title("Go to Line")
        self.transient(parent)
        
        # Get current line count
        self.max_line = int(text_widget.index('end-1c').split('.')[0])
        
        frame = ttk.Frame(self, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"Line number (1-{self.max_line}):").pack(anchor=tk.W)
        
        self.line_var = tk.StringVar()
        self.line_entry = ttk.Entry(frame, textvariable=self.line_var, width=20)
        self.line_entry.pack(fill=tk.X, pady=5)
        self.line_entry.focus()
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Go", command=self.goto, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy, width=10).pack(side=tk.LEFT, padx=2)
        
        self.line_entry.bind("<Return>", lambda e: self.goto())
        self.bind("<Escape>", lambda e: self.destroy())
        _fit_and_center_dialog(self, parent, 300, 140)
    
    def goto(self):
        """Go to the specified line"""
        try:
            line = int(self.line_var.get())
            if 1 <= line <= self.max_line:
                self.text_widget.mark_set(tk.INSERT, f"{line}.0")
                self.text_widget.see(f"{line}.0")
                self.destroy()
            else:
                messagebox.showwarning("Invalid Line", f"Line must be between 1 and {self.max_line}")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid line number")


class ProgressDialog(tk.Toplevel):
    """Progress dialog for long operations"""
    
    def __init__(self, parent, title="Loading...", message="Please wait..."):
        super().__init__(parent)
        self.title(title)
        self.geometry("350x120")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        # Prevent closing
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        
        frame = ttk.Frame(self, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.message_var = tk.StringVar(value=message)
        ttk.Label(frame, textvariable=self.message_var).pack(pady=5)
        
        self.progress = ttk.Progressbar(frame, mode='determinate', length=300)
        self.progress.pack(pady=10)
        
        self.detail_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.detail_var, foreground="gray").pack()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def update_progress(self, value, message=None, detail=None):
        """Update progress bar and messages"""
        self.progress['value'] = value
        if message:
            self.message_var.set(message)
        if detail:
            self.detail_var.set(detail)
        self.update()


class FindDialog(tk.Toplevel):
    """Find dialog for searching text"""
    
    def __init__(self, parent, text_widget):
        super().__init__(parent)
        self.text_widget = text_widget
        self.title("Find")
        
        # Search field
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Find:").grid(row=0, column=0, padx=5, pady=5)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(frame, textvariable=self.search_var, width=30)
        self.search_entry.grid(row=0, column=1, padx=5, pady=5)
        self.search_entry.focus()
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Find Next", command=self.find_next).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=self._on_close).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter and Escape keys
        self.search_entry.bind("<Return>", lambda e: self.find_next())
        self.bind("<Escape>", lambda e: self._on_close())
        _fit_and_center_dialog(self, parent, 380, 140)
        
        # Clear highlights when dialog closes
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Track search position
        self.search_pos = "1.0"
    
    def _on_close(self):
        """Handle dialog close - clear highlights"""
        self.text_widget.tag_remove("search", "1.0", tk.END)
        self.destroy()
    
    def find_next(self):
        """Find the next occurrence"""
        search_text = self.search_var.get()
        if not search_text:
            return
        
        # Remove previous highlight
        self.text_widget.tag_remove("search", "1.0", tk.END)
        
        # Search from current position
        pos = self.text_widget.search(search_text, self.search_pos, tk.END, nocase=True)
        
        if pos:
            # Highlight found text
            end_pos = f"{pos}+{len(search_text)}c"
            self.text_widget.tag_add("search", pos, end_pos)
            self.text_widget.tag_config("search", background="yellow")
            
            # Move cursor and scroll to position
            self.text_widget.mark_set(tk.INSERT, pos)
            self.text_widget.see(pos)
            
            # Update search position for next search
            self.search_pos = end_pos
        else:
            # Wrap around to beginning
            self.search_pos = "1.0"
            messagebox.showinfo("Find", "Reached end of document. Search will continue from beginning.")


class InsertTableDialog(tk.Toplevel):
    """Dialog for inserting a formatted Markdown table with rows, columns, header, and alignment."""

    def __init__(self, parent, text_widget, on_inserted=None):
        super().__init__(parent)
        self.parent = parent
        self.text_widget = text_widget
        self.on_inserted = on_inserted

        self.title("Insert Markdown Table")
        self.transient(parent)
        self.grab_set()

        self._setup_ui()
        _fit_and_center_dialog(self, parent, 400, 300)

        self.bind("<Return>", lambda e: self._insert())
        self.bind("<Escape>", lambda e: self.destroy())

    def _setup_ui(self):
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Configure Table Dimensions", font=(SANS_FONT, 10, "bold")).pack(anchor=tk.W, pady=(0, 10))

        grid = ttk.Frame(frame)
        grid.pack(fill=tk.X, pady=4)

        ttk.Label(grid, text="Columns:", width=16).grid(row=0, column=0, sticky=tk.W, pady=4)
        self.cols_var = tk.IntVar(value=3)
        cols_spin = ttk.Spinbox(grid, from_=1, to=12, textvariable=self.cols_var, width=8)
        cols_spin.grid(row=0, column=1, sticky=tk.W, pady=4)

        ttk.Label(grid, text="Data Rows:", width=16).grid(row=1, column=0, sticky=tk.W, pady=4)
        self.rows_var = tk.IntVar(value=3)
        rows_spin = ttk.Spinbox(grid, from_=1, to=50, textvariable=self.rows_var, width=8)
        rows_spin.grid(row=1, column=1, sticky=tk.W, pady=4)

        ttk.Label(grid, text="Column Alignment:", width=16).grid(row=2, column=0, sticky=tk.W, pady=4)
        self.align_var = tk.StringVar(value="Left")
        align_combo = ttk.Combobox(grid, textvariable=self.align_var, values=["Left", "Center", "Right"], state="readonly", width=10)
        align_combo.grid(row=2, column=1, sticky=tk.W, pady=4)

        self.header_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Include Header Row", variable=self.header_var).pack(anchor=tk.W, pady=(8, 4))

        ttk.Label(frame, text="Tip: Use Tab and Shift+Tab inside tables to jump cells and auto-align.",
                  font=(SANS_FONT, 8), foreground="#666666").pack(anchor=tk.W, pady=(2, 10))

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(btn_frame, text="Insert Table", command=self._insert, width=14).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="Cancel", command=self.destroy, width=10).pack(side=tk.RIGHT)

        cols_spin.focus_set()

    def _insert(self):
        try:
            cols = max(1, min(12, int(self.cols_var.get())))
        except (ValueError, tk.TclError):
            cols = 3
        try:
            rows = max(1, min(50, int(self.rows_var.get())))
        except (ValueError, tk.TclError):
            rows = 3

        include_header = bool(self.header_var.get())
        align = self.align_var.get()

        col_width = 10
        if align == "Center":
            sep_cell = ":" + "-" * (col_width - 2) + ":"
        elif align == "Right":
            sep_cell = "-" * (col_width - 1) + ":"
        else:
            sep_cell = ":" + "-" * (col_width - 1)

        lines = []
        if include_header:
            headers = [f"Header {c+1}".ljust(col_width) for c in range(cols)]
            lines.append("| " + " | ".join(headers) + " |")
        else:
            blank_top = [" " * col_width for _ in range(cols)]
            lines.append("| " + " | ".join(blank_top) + " |")

        seps = [sep_cell for _ in range(cols)]
        lines.append("| " + " | ".join(seps) + " |")

        data_row_count = rows if include_header else max(0, rows - 1)
        for _ in range(data_row_count):
            cells = [" " * col_width for _ in range(cols)]
            lines.append("| " + " | ".join(cells) + " |")

        table_md = "\n" + "\n".join(lines) + "\n\n"
        insert_idx = self.text_widget.index(tk.INSERT)
        self.text_widget.insert(insert_idx, table_md)

        # Place cursor inside the first cell
        first_cell_start = f"{insert_idx}+1l+2c"
        first_cell_end = f"{first_cell_start}+{col_width if not include_header else len('Header 1')}c"
        try:
            self.text_widget.mark_set(tk.INSERT, first_cell_start)
            if include_header:
                self.text_widget.tag_remove(tk.SEL, "1.0", tk.END)
                self.text_widget.tag_add(tk.SEL, first_cell_start, first_cell_end)
            self.text_widget.see(first_cell_start)
            self.text_widget.focus_set()
        except tk.TclError:
            pass

        if self.on_inserted:
            self.on_inserted()
        self.destroy()


class MathSymbolDialog(tk.Toplevel):
    """Non-modal palette for inserting Unicode mathematical symbols, Greek letters, and LaTeX templates."""

    CATEGORIES = {
        "Greek": [
            ("α", "alpha"), ("β", "beta"), ("γ", "gamma"), ("δ", "delta"),
            ("ε", "epsilon"), ("ζ", "zeta"), ("η", "eta"), ("θ", "theta"),
            ("ι", "iota"), ("κ", "kappa"), ("λ", "lambda"), ("μ", "mu"),
            ("ν", "nu"), ("ξ", "xi"), ("π", "pi"), ("ρ", "rho"),
            ("σ", "sigma"), ("τ", "tau"), ("υ", "upsilon"), ("φ", "phi"),
            ("χ", "chi"), ("ψ", "psi"), ("ω", "omega"), ("Γ", "Gamma"),
            ("Δ", "Delta"), ("Θ", "Theta"), ("Λ", "Lambda"), ("Ξ", "Xi"),
            ("Π", "Pi"), ("Σ", "Sigma"), ("Φ", "Phi"), ("Ψ", "Psi"), ("Ω", "Omega")
        ],
        "Operators": [
            ("±", "Plus-minus"), ("∓", "Minus-plus"), ("×", "Multiply"), ("÷", "Divide"),
            ("√", "Square root"), ("∛", "Cube root"), ("∞", "Infinity"), ("∂", "Partial derivative"),
            ("∇", "Nabla / Del"), ("∑", "Summation"), ("∏", "Product"), ("∫", "Integral"),
            ("∬", "Double integral"), ("∮", "Contour integral"), ("≈", "Approximately equal"), ("≠", "Not equal"),
            ("≡", "Identical to"), ("≤", "Less than or equal"), ("≥", "Greater than or equal"), ("≪", "Much less than"),
            ("≫", "Much greater than"), ("∝", "Proportional to"), ("∠", "Angle"), ("°", "Degree"),
            ("′", "Prime"), ("″", "Double prime"), ("‰", "Per mille"), ("⋅", "Dot operator")
        ],
        "Sets & Logic": [
            ("∈", "Element of"), ("∉", "Not element of"), ("∋", "Contains"), ("⊂", "Subset"),
            ("⊃", "Superset"), ("⊆", "Subset or equal"), ("⊇", "Superset or equal"), ("∪", "Union"),
            ("∩", "Intersection"), ("∅", "Empty set"), ("∀", "For all"), ("∃", "There exists"),
            ("∄", "Does not exist"), ("¬", "Logical NOT"), ("∧", "Logical AND"), ("∨", "Logical OR"),
            ("⊕", "XOR / Direct sum"), ("⇒", "Implies"), ("⇔", "If and only if"), ("→", "Right arrow"),
            ("←", "Left arrow"), ("↔", "Left-right arrow"), ("↦", "Maps to"), ("∴", "Therefore"),
            ("∵", "Because"), ("ℕ", "Natural numbers"), ("ℤ", "Integers"), ("ℝ", "Real numbers")
        ],
        "Sub & Super": [
            ("⁰", "^0"), ("¹", "^1"), ("²", "^2"), ("³", "^3"), ("⁴", "^4"),
            ("⁵", "^5"), ("⁶", "^6"), ("⁷", "^7"), ("⁸", "^8"), ("⁹", "^9"),
            ("⁺", "^+"), ("⁻", "^-"), ("⁼", "^="), ("⁽", "^("), ("⁾", "^)"),
            ("ⁿ", "^n"), ("ⁱ", "^i"), ("₀", "_0"), ("₁", "_1"), ("₂", "_2"),
            ("₃", "_3"), ("₄", "_4"), ("₅", "_5"), ("₆", "_6"), ("₇", "_7"),
            ("₈", "_8"), ("₉", "_9"), ("₊", "_+"), ("₋", "_-"), ("₌", "_="),
            ("ᵢ", "_i"), ("ⱼ", "_j"), ("ₖ", "_k"), ("ₙ", "_n")
        ],
    }

    LATEX_TEMPLATES = [
        ("Inline $...$", "$", "$", "Wrap inline math $...$"),
        ("Block $$...$$", "\n$$\n", "\n$$\n", "Insert display math block $$...$$"),
        ("\\frac{a}{b}", "\\frac{", "}{b}", "LaTeX fraction"),
        ("\\sqrt{x}", "\\sqrt{", "}", "LaTeX square root"),
        ("\\sum_{i=1}^{n}", "\\sum_{i=1}^{n} ", "", "LaTeX summation"),
        ("\\int_{a}^{b}", "\\int_{a}^{b} ", "", "LaTeX integral"),
    ]

    def __init__(self, notepad: MarkdownNotepad):
        super().__init__(notepad)
        self.notepad = notepad
        self.title("Math & Symbol Palette")
        self.transient(notepad)

        self._setup_ui()
        _fit_and_center_dialog(self, notepad, 480, 360)
        self.bind("<Escape>", lambda e: self.destroy())

    def _setup_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        nb = ttk.Notebook(main)
        nb.pack(fill=tk.BOTH, expand=True)

        for cat_name, symbols in self.CATEGORIES.items():
            page = ttk.Frame(nb, padding=6)
            nb.add(page, text=cat_name)
            cols = 7
            for idx, (sym, desc) in enumerate(symbols):
                r, c = divmod(idx, cols)
                btn = ttk.Button(
                    page, text=sym, width=4,
                    command=lambda s=sym: self._insert_symbol(s)
                )
                btn.grid(row=r, column=c, padx=2, pady=2, sticky="nsew")
                ToolTip(btn, f"{sym} — {desc}")

        # LaTeX templates tab
        latex_page = ttk.Frame(nb, padding=8)
        nb.add(latex_page, text="LaTeX Math")
        ttk.Label(
            latex_page,
            text="Insert LaTeX math (rendered with Unicode symbols in Visual Mode):",
            font=(SANS_FONT, 8), foreground="#555555"
        ).pack(anchor=tk.W, pady=(0, 6))
        t_grid = ttk.Frame(latex_page)
        t_grid.pack(fill=tk.BOTH, expand=True)
        for idx, (label, pfx, sfx, tip) in enumerate(self.LATEX_TEMPLATES):
            r, c = divmod(idx, 2)
            btn = ttk.Button(
                t_grid, text=label, width=18,
                command=lambda p=pfx, s=sfx: self._insert_template(p, s)
            )
            btn.grid(row=r, column=c, padx=4, pady=4, sticky="ew")
            ToolTip(btn, tip)

        footer = ttk.Frame(main)
        footer.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(footer, text="Click any symbol to insert at cursor", font=(SANS_FONT, 8), foreground="gray").pack(side=tk.LEFT)
        ttk.Button(footer, text="Close", command=self.destroy, width=8).pack(side=tk.RIGHT)

    def _insert_symbol(self, sym: str):
        if self.notepad.current_tab:
            editor = self.notepad.current_tab.source_editor
            editor.insert(tk.INSERT, sym)
            self.notepad.current_tab.is_modified = True
            self.notepad._update_tab_title(self.notepad.current_tab)
            self.notepad._schedule_syntax_highlight()
            self.notepad._refresh_active_visual_view()
            editor.focus_set()

    def _insert_template(self, prefix: str, suffix: str):
        if self.notepad.current_tab:
            if suffix:
                self.notepad._insert_format(prefix, suffix)
            else:
                self.notepad._insert_text(prefix)
            self.notepad.current_tab.source_editor.focus_set()


class SnapshotDialog(tk.Toplevel):
    """
    Dialog for creating, previewing, comparing (via DiffPanel), and restoring
    named document snapshots (Git-like version checkpoints).
    """

    def __init__(self, notepad: 'MarkdownNotepad', tab: 'DocumentTab'):
        super().__init__(notepad)
        self.notepad = notepad
        self.tab = tab
        if not hasattr(self.tab, 'snapshots') or self.tab.snapshots is None:
            self.tab.snapshots = []

        self.title(f"Document Snapshots — {self.tab.get_display_name()}")
        self.geometry("760x520")
        self.minsize(600, 380)
        self.transient(notepad)

        self._setup_ui()
        self._refresh_list()

        # Center on parent
        self.update_idletasks()
        x = notepad.winfo_rootx() + (notepad.winfo_width() - self.winfo_width()) // 2
        y = notepad.winfo_rooty() + (notepad.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")
        self.bind("<Escape>", lambda e: self.destroy())
        self.name_entry.focus_set()

    def _setup_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Create new snapshot bar
        top_bar = ttk.LabelFrame(main, text="Create New Checkpoint Snapshot", padding=(8, 6))
        top_bar.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(top_bar, text="Checkpoint Label:", font=(SANS_FONT, 9)).pack(side=tk.LEFT, padx=(0, 6))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(top_bar, textvariable=self.name_var, font=(SANS_FONT, 9))
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.name_entry.bind("<Return>", lambda e: self._on_create())

        ttk.Button(
            top_bar, text="\u2795 Save Snapshot", command=self._on_create
        ).pack(side=tk.RIGHT)

        # Paned window: snapshot list on top, content preview on bottom
        paned = ttk.PanedWindow(main, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        list_frame = ttk.LabelFrame(paned, text="Saved Snapshots (Newest at Bottom)", padding=4)
        paned.add(list_frame, weight=1)

        cols = ("idx", "name", "timestamp", "words", "delta")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", selectmode="browse", height=7)
        self.tree.heading("idx", text="#")
        self.tree.heading("name", text="Checkpoint Name")
        self.tree.heading("timestamp", text="Timestamp")
        self.tree.heading("words", text="Words")
        self.tree.heading("delta", text="Delta vs Current")

        self.tree.column("idx", width=40, anchor="center", stretch=False)
        self.tree.column("name", width=260, anchor="w")
        self.tree.column("timestamp", width=155, anchor="center", stretch=False)
        self.tree.column("words", width=75, anchor="e", stretch=False)
        self.tree.column("delta", width=120, anchor="center", stretch=False)

        vbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda e: self._on_compare())

        preview_frame = ttk.LabelFrame(paned, text="Snapshot Content Preview", padding=4)
        paned.add(preview_frame, weight=1)

        bg = self.notepad.theme.get('source_bg', '#ffffff')
        fg = self.notepad.theme.get('source_fg', '#1a1a1a')
        self.preview_text = scrolledtext.ScrolledText(
            preview_frame, wrap=tk.WORD, font=(MONO_FONT, 9),
            bg=bg, fg=fg, state=tk.DISABLED, relief=tk.FLAT, padx=6, pady=4, height=8
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True)

        # Action buttons
        btn_bar = ttk.Frame(main)
        btn_bar.pack(fill=tk.X)

        self.compare_btn = ttk.Button(
            btn_bar, text="\u21c4 Compare & Review in Diff Panel",
            command=self._on_compare, state=tk.DISABLED
        )
        self.compare_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.restore_btn = ttk.Button(
            btn_bar, text="\u23ea Restore Snapshot",
            command=self._on_restore, state=tk.DISABLED
        )
        self.restore_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.delete_btn = ttk.Button(
            btn_bar, text="\U0001f5d1 Delete",
            command=self._on_delete, state=tk.DISABLED
        )
        self.delete_btn.pack(side=tk.LEFT)

        ttk.Button(btn_bar, text="Close", command=self.destroy, width=9).pack(side=tk.RIGHT)

    def _refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        curr_content = self.tab.get_content()
        curr_words = len(curr_content.split()) if curr_content.strip() else 0

        for idx, snap in enumerate(self.tab.snapshots):
            snap_words = snap.get('word_count', len(snap.get('content', '').split()))
            diff_w = snap_words - curr_words
            if snap.get('content', '') == curr_content:
                delta_str = "Identical"
            elif diff_w > 0:
                delta_str = f"+{diff_w} words"
            elif diff_w < 0:
                delta_str = f"{diff_w} words"
            else:
                delta_str = "Modified (0w net)"

            self.tree.insert(
                "", tk.END, iid=str(idx),
                values=(idx + 1, snap.get('name', f"Checkpoint #{idx + 1}"),
                        snap.get('timestamp', ''), snap_words, delta_str)
            )

        children = self.tree.get_children()
        if children:
            last_id = children[-1]
            self.tree.selection_set(last_id)
            self.tree.see(last_id)
            self._on_select()
        else:
            self._clear_preview()

    def _clear_preview(self):
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete('1.0', tk.END)
        self.preview_text.config(state=tk.DISABLED)
        self.compare_btn.config(state=tk.DISABLED)
        self.restore_btn.config(state=tk.DISABLED)
        self.delete_btn.config(state=tk.DISABLED)

    def _get_selected_index(self) -> Optional[int]:
        sel = self.tree.selection()
        if not sel:
            return None
        try:
            idx = int(sel[0])
            if 0 <= idx < len(self.tab.snapshots):
                return idx
        except ValueError:
            pass
        return None

    def _on_select(self, event=None):
        idx = self._get_selected_index()
        if idx is None:
            self._clear_preview()
            return
        snap = self.tab.snapshots[idx]
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete('1.0', tk.END)
        content_to_show = snap.get('content', '')
        if getattr(self.tab, 'tab_type', 'document') == 'spreadsheet' and content_to_show.strip().startswith('{'):
            try:
                data = json.loads(content_to_show)
                cells = data.get("cells", {})
                preview_lines = [f"Spreadsheet Snapshot: {len(cells)} cells populated\n"]
                for k, v in list(cells.items())[:35]:
                    r, c = map(int, k.split(','))
                    ref = cell_to_ref(r, c) if SPREADSHEET_AVAILABLE else f"R{r+1}C{c+1}"
                    preview_lines.append(f"{ref}: {v}")
                if len(cells) > 35:
                    preview_lines.append(f"... ({len(cells) - 35} more cells)")
                content_to_show = "\n".join(preview_lines)
            except Exception:
                pass
        self.preview_text.insert('1.0', content_to_show)
        self.preview_text.config(state=tk.DISABLED)
        self.compare_btn.config(state=tk.NORMAL if getattr(self.tab, 'tab_type', 'document') != 'spreadsheet' else tk.DISABLED)
        self.restore_btn.config(state=tk.NORMAL)
        self.delete_btn.config(state=tk.NORMAL)

    def _on_create(self):
        label = self.name_var.get().strip()
        self.notepad.create_quick_snapshot(label if label else None)
        self.name_var.set("")
        self._refresh_list()

    def _on_compare(self):
        """Open DiffPanel comparing current document (Original) with the selected snapshot (Suggested)."""
        idx = self._get_selected_index()
        if idx is None or not DIFF_PANEL_AVAILABLE:
            return
        snap = self.tab.snapshots[idx]
        snap_name = snap.get('name', f"Checkpoint #{idx + 1}")
        current_text = self.tab.get_content()
        snap_text = snap.get('content', '')

        panel = DiffPanel(
            self.notepad,
            original_text=current_text,
            start_index="1.0",
            end_index=tk.END,
            action_name=f"Snapshot Comparison: {snap_name}",
            on_accept=self.notepad._apply_diff,
            theme=self.notepad.theme,
            default_apply_mode="full",
        )
        panel.set_suggestion(snap_text)

    def _on_restore(self):
        """Restore the selected snapshot after creating an automatic safety backup checkpoint."""
        idx = self._get_selected_index()
        if idx is None:
            return
        snap = self.tab.snapshots[idx]
        snap_name = snap.get('name', f"Checkpoint #{idx + 1}")
        snap_text = snap.get('content', '')
        current_text = self.tab.get_content()

        if snap_text == current_text:
            messagebox.showinfo("Snapshot Identical", "The document already matches this snapshot.", parent=self)
            return

        if not messagebox.askyesno(
            "Restore Snapshot",
            f"Restore document to '{snap_name}'?\n\n"
            "An automatic backup snapshot of your current text will be saved first.",
            parent=self
        ):
            return

        self.notepad.create_quick_snapshot(f"Auto-backup before restoring '{snap_name}'")
        if getattr(self.tab, 'tab_type', 'document') == 'spreadsheet':
            self.tab.set_content(snap_text)
            self.tab.is_modified = True
            self.notepad._update_tab_title(self.tab)
        else:
            self.notepad._apply_diff(snap_text, "1.0", tk.END, current_text, apply_mode="full")
        self._refresh_list()
        self.notepad._set_status(f"Restored snapshot: '{snap_name}'")

    def _on_delete(self):
        idx = self._get_selected_index()
        if idx is None:
            return
        snap = self.tab.snapshots[idx]
        snap_name = snap.get('name', f"Checkpoint #{idx + 1}")
        if not messagebox.askyesno("Delete Snapshot", f"Delete snapshot '{snap_name}'?", parent=self):
            return
        del self.tab.snapshots[idx]
        if self.notepad.session_manager and self.notepad.session_manager.is_caching_enabled():
            self.notepad._save_session()
        self._refresh_list()


def main():
    """Main entry point"""
    if os.name == 'nt':
        # Crisp text on scaled displays
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    app = MarkdownNotepad()
    import sys
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if os.path.exists(arg):
                app.open_file(arg)
    app.mainloop()


if __name__ == "__main__":
    main()
