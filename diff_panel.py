"""
Diff Panel Module for MarkItDown Notepad

Provides a side-by-side diff panel for reviewing AI-suggested edits or
document snapshots before applying them to the document. Supports streaming
AI responses, word-level inline diff highlighting, selective paragraph/hunk
acceptance, synchronized scrolling, manual editing, and flexible apply modes.
"""

import difflib
import re
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
import queue
import threading
from typing import Optional, Callable, Dict, Any, List, Tuple

# Font detection (mirrors main app)
def _select_best_font(preferred, fallback="TkDefaultFont"):
    try:
        available = [f.lower() for f in font.families()]
        for f in preferred:
            if f.lower() in available:
                idx = available.index(f.lower())
                return font.families()[idx] if idx < len(font.families()) else f
        return fallback
    except Exception:
        return fallback

SANS_FONTS = ["Segoe UI", "SF Pro Display", "Ubuntu", "DejaVu Sans", "Arial", "Helvetica"]
MONO_FONTS = ["Consolas", "SF Mono", "Monaco", "Ubuntu Mono", "DejaVu Sans Mono", "Courier New"]


class ToolTip:
    """Rich, screen-edge-clamped tooltip widget for diff panel."""
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
            font=(SANS_FONTS[0], 9), padx=8, pady=5,
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


class DiffPanel(tk.Toplevel):
    """
    Side-by-side diff panel for reviewing AI-suggested edits or snapshot comparisons.

    Shows original text on the left and suggested/snapshot text on the right with
    word-level inline diff highlighting, synchronized vertical scrolling,
    per-hunk selective checkboxes, and flexible apply modes.
    """

    APPLY_MODES = [
        ("replace", "Replace Target Range"),
        ("cursor", "Insert at Cursor"),
        ("append", "Append to Document"),
        ("full", "Replace Entire Document"),
    ]

    def __init__(self, parent, original_text: str, start_index: Optional[str], end_index: Optional[str],
                 action_name: str = "AI Edit",
                 on_accept: Optional[Callable] = None,
                 on_reject: Optional[Callable] = None,
                 llm_client=None, settings=None,
                 messages=None, images=None,
                 theme: Optional[Dict[str, Any]] = None,
                 default_apply_mode: str = "replace"):
        super().__init__(parent)

        self.parent = parent
        self.original_text = original_text or ""
        self.start_index = start_index or "1.0"
        self.end_index = end_index or tk.END
        self.action_name = action_name
        self.on_accept = on_accept
        self.on_reject = on_reject
        self.llm_client = llm_client
        self.settings = settings
        self.messages = messages or []
        self.images = images or []
        self.theme = theme or getattr(parent, 'theme', {}) or {}
        self.default_apply_mode = default_apply_mode

        self.response_text = ""
        self.full_suggestion_text = ""
        self.response_queue = queue.Queue()
        self.is_generating = False
        self.is_editable = False
        self._scrolling_lock = False
        self._hunks: List[Dict[str, Any]] = []
        self._hunk_vars: List[tk.BooleanVar] = []

        # Resolve fonts
        self._sans_font = _select_best_font(SANS_FONTS)
        self._mono_font = _select_best_font(MONO_FONTS)

        # Resolve theme palette
        self._pal = self._compute_palette()

        self.title(f"Review & Diff: {action_name}")
        self.geometry("920x600")
        self.minsize(680, 420)
        self.transient(parent)

        self._setup_ui()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

        # Start generation if we have a client and messages
        if self.llm_client and self.messages:
            self._start_generation()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_reject)

    @staticmethod
    def _is_dark_hex(hex_color: str) -> bool:
        try:
            c = hex_color.lstrip('#')
            if len(c) == 6:
                r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                return (0.299 * r + 0.587 * g + 0.114 * b) < 128
        except Exception:
            pass
        return False

    def _compute_palette(self) -> Dict[str, str]:
        bg_sample = self.theme.get('source_bg', '#ffffff')
        is_dark = self._is_dark_hex(bg_sample)
        if is_dark:
            return {
                'orig_bg': '#241b20',
                'sugg_bg': '#19241e',
                'edit_bg': '#1e293b',
                'fg': '#e2e8f0',
                'del_bg': '#7f1d1d',
                'del_fg': '#fecaca',
                'ins_bg': '#14532d',
                'ins_fg': '#bbf7d0',
                'muted': '#94a3b8',
            }
        return {
            'orig_bg': '#fff8f6',
            'sugg_bg': '#f4fdf7',
            'edit_bg': '#ffffff',
            'fg': '#1a1a1a',
            'del_bg': '#fecdd3',
            'del_fg': '#881337',
            'ins_bg': '#bbf7d0',
            'ins_fg': '#14532d',
            'muted': '#555555',
        }

    def _setup_ui(self):
        """Build the diff panel UI"""
        P = self._pal
        main = ttk.Frame(self, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(header, text=f"  {self.action_name}",
                  font=(self._sans_font, 11, 'bold')).pack(side=tk.LEFT)

        self.diff_stats_label = ttk.Label(
            header, text="",
            font=(self._sans_font, 9, 'bold'),
            foreground=P['muted']
        )
        self.diff_stats_label.pack(side=tk.LEFT, padx=(12, 4))

        self.status_label = ttk.Label(header, text="Generating...",
                                      font=(self._sans_font, 9))
        self.status_label.pack(side=tk.RIGHT, padx=5)

        self.sync_scroll_var = tk.BooleanVar(value=True)
        sync_cb = ttk.Checkbutton(
            header, text="Sync Scroll", variable=self.sync_scroll_var
        )
        sync_cb.pack(side=tk.RIGHT, padx=8)
        ToolTip(sync_cb, "Lock vertical scrolling so both the original document and suggested changes scroll simultaneously.")

        self.word_diff_var = tk.BooleanVar(value=True)
        word_cb = ttk.Checkbutton(
            header, text="Highlight Word Diffs", variable=self.word_diff_var,
            command=self._refresh_word_diff_tags
        )
        word_cb.pack(side=tk.RIGHT, padx=4)
        ToolTip(word_cb, "Highlight granular word-by-word deletions in red strikethrough and additions in green.")

        # Paned window for side-by-side
        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        # Left pane — Original
        left_frame = ttk.LabelFrame(paned, text="Original  (Deletions highlighted)", padding=4)
        paned.add(left_frame, weight=1)

        self.original_display = scrolledtext.ScrolledText(
            left_frame, wrap=tk.WORD,
            font=(self._mono_font, 10),
            bg=P['orig_bg'], fg=P['fg'],
            insertbackground=P['fg'],
            state=tk.NORMAL,
            relief=tk.FLAT, borderwidth=1,
            padx=8, pady=6
        )
        self.original_display.pack(fill=tk.BOTH, expand=True)
        self.original_display.tag_configure(
            'diff_del',
            background=P['del_bg'],
            foreground=P['del_fg'],
            overstrike=True
        )
        self.original_display.insert('1.0', self.original_text)
        self.original_display.config(state=tk.DISABLED)

        # Right pane — Suggestion
        right_frame = ttk.LabelFrame(paned, text="Suggested  (Additions highlighted)", padding=4)
        paned.add(right_frame, weight=1)

        self.suggestion_display = scrolledtext.ScrolledText(
            right_frame, wrap=tk.WORD,
            font=(self._mono_font, 10),
            bg=P['sugg_bg'], fg=P['fg'],
            insertbackground=P['fg'],
            state=tk.DISABLED,
            relief=tk.FLAT, borderwidth=1,
            padx=8, pady=6
        )
        self.suggestion_display.pack(fill=tk.BOTH, expand=True)
        self.suggestion_display.tag_configure(
            'diff_ins',
            background=P['ins_bg'],
            foreground=P['ins_fg']
        )

        # Configure synchronized vertical scrolling
        self._setup_sync_scroll()

        # Selective Hunk / Paragraph Acceptance Bar (populated once suggestion finishes)
        self.hunks_outer = ttk.LabelFrame(main, text="Selective Hunk / Paragraph Acceptance", padding=(6, 4))
        self.hunks_header = ttk.Frame(self.hunks_outer)
        self.hunks_header.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(
            self.hunks_header,
            text="Toggle individual changed blocks to include or exclude them from the final result:",
            font=(self._sans_font, 8), foreground=P['muted']
        ).pack(side=tk.LEFT)
        sel_all_btn = ttk.Button(
            self.hunks_header, text="Select All", width=9,
            command=lambda: self._set_all_hunks(True)
        )
        sel_all_btn.pack(side=tk.RIGHT, padx=(2, 0))
        ToolTip(sel_all_btn, "Accept and include all changed paragraphs and hunks in the final applied document.")

        desel_all_btn = ttk.Button(
            self.hunks_header, text="Deselect All", width=10,
            command=lambda: self._set_all_hunks(False)
        )
        desel_all_btn.pack(side=tk.RIGHT, padx=2)
        ToolTip(desel_all_btn, "Exclude all changed blocks, keeping the original document text across all hunks.")

        self.hunks_flow = ttk.Frame(self.hunks_outer)
        self.hunks_flow.pack(fill=tk.X)

        # Button bar
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)

        self.reject_btn = ttk.Button(btn_frame, text="Reject", command=self._on_reject)
        self.reject_btn.pack(side=tk.RIGHT, padx=(5, 0))
        ToolTip(self.reject_btn, "Discard all suggested changes and close this diff review panel without modifying your document.")

        self.edit_btn = ttk.Button(btn_frame, text="Edit", command=self._toggle_edit)
        self.edit_btn.pack(side=tk.RIGHT, padx=(5, 0))
        ToolTip(self.edit_btn, "Toggle manual editing mode on the suggested text pane to fine-tune revisions before applying.")

        self.accept_btn = ttk.Button(btn_frame, text="✔ Accept & Apply", command=self._on_accept,
                                     state=tk.DISABLED)
        self.accept_btn.pack(side=tk.RIGHT)
        ToolTip(self.accept_btn, "Apply the accepted changes directly to your document using the selected apply mode.")

        # Apply mode selector
        ttk.Label(btn_frame, text="Apply mode:", font=(self._sans_font, 9)).pack(side=tk.LEFT, padx=(0, 4))
        mode_labels = [label for _, label in self.APPLY_MODES]
        default_label = dict(self.APPLY_MODES).get(self.default_apply_mode, mode_labels[0])
        self.apply_mode_var = tk.StringVar(value=default_label)
        self.apply_mode_combo = ttk.Combobox(
            btn_frame, textvariable=self.apply_mode_var,
            values=mode_labels, state="readonly", width=21
        )
        self.apply_mode_combo.pack(side=tk.LEFT, padx=(0, 12))
        ToolTip(self.apply_mode_combo, "Select how applied text is inserted: replace target range, insert at cursor, append to end, or replace whole document.")

        # Info label
        ttk.Label(btn_frame, text="Target: Lines " + self._format_range(),
                  font=(self._sans_font, 9), foreground=P['muted']).pack(side=tk.LEFT)

    def _setup_sync_scroll(self):
        """Synchronize vertical scrolling between original and suggestion text widgets."""
        orig_vbar = self.original_display.vbar
        sugg_vbar = self.suggestion_display.vbar

        def on_orig_yscroll(*args):
            orig_vbar.set(*args)
            if self.sync_scroll_var.get() and not self._scrolling_lock:
                self._scrolling_lock = True
                try:
                    self.suggestion_display.yview_moveto(args[0])
                finally:
                    self._scrolling_lock = False

        def on_sugg_yscroll(*args):
            sugg_vbar.set(*args)
            if self.sync_scroll_var.get() and not self._scrolling_lock:
                self._scrolling_lock = True
                try:
                    self.original_display.yview_moveto(args[0])
                finally:
                    self._scrolling_lock = False

        self.original_display.config(yscrollcommand=on_orig_yscroll)
        self.suggestion_display.config(yscrollcommand=on_sugg_yscroll)

    def _format_range(self) -> str:
        """Format the line range for display"""
        try:
            start_line = str(self.start_index).split('.')[0]
            end_line = str(self.end_index).split('.')[0]
            if end_line == tk.END:
                return f"{start_line}–End"
            if start_line == end_line:
                return f"{start_line}"
            return f"{start_line}–{end_line}"
        except Exception:
            return "?"

    def _start_generation(self):
        """Start AI response generation in background thread"""
        self.is_generating = True
        self.status_label.config(text="Generating...")
        self.accept_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._generate, daemon=True)
        thread.start()
        self._poll_queue()

    def _generate(self):
        """Background thread: stream LLM response"""
        try:
            response_text = ""
            is_agy = self.settings and getattr(self.settings, 'provider', '') in ('agy', 'antigravity')
            target_model = self.settings.agy_model if (is_agy and getattr(self.settings, 'agy_model', '')) else (self.settings.model if self.settings else None)
            target_effort = self.settings.agy_effort if (is_agy and getattr(self.settings, 'agy_effort', '')) else None

            for chunk in self.llm_client.send_message_stream(
                self.messages,
                system_prompt=self.settings.system_prompt if self.settings else "",
                max_tokens=self.settings.max_tokens if self.settings else 1000000,
                temperature=self.settings.temperature if self.settings else 0.7,
                top_p=self.settings.top_p if self.settings else 1.0,
                top_k=self.settings.top_k if self.settings else 0,
                images=self.images if self.images else None,
                model=target_model,
                effort=target_effort
            ):
                response_text += chunk
                self.response_queue.put(('chunk', chunk))

            self.response_text = response_text
            self.response_queue.put(('done', None))
        except Exception as e:
            self.response_queue.put(('error', str(e)))

    def _poll_queue(self):
        """Poll response queue on main thread"""
        if not self.winfo_exists():
            return
        try:
            while True:
                msg_type, content = self.response_queue.get_nowait()
                if msg_type == 'chunk':
                    self.suggestion_display.config(state=tk.NORMAL)
                    self.suggestion_display.insert(tk.END, content)
                    self.suggestion_display.config(state=tk.DISABLED)
                    self.suggestion_display.see(tk.END)
                elif msg_type == 'done':
                    self.is_generating = False
                    self.full_suggestion_text = self.response_text
                    self._build_hunks_and_highlight()
                    self.status_label.config(text="Ready for review")
                    self.accept_btn.config(state=tk.NORMAL)
                    return
                elif msg_type == 'error':
                    self.is_generating = False
                    self.status_label.config(text=f"Error: {content}")
                    self.accept_btn.config(state=tk.NORMAL)
                    return
        except queue.Empty:
            pass

        self.after(50, self._poll_queue)

    def set_suggestion(self, text: str):
        """Set suggestion text directly (non-streaming mode)"""
        self.response_text = text or ""
        self.full_suggestion_text = self.response_text
        self.suggestion_display.config(state=tk.NORMAL)
        self.suggestion_display.delete('1.0', tk.END)
        self.suggestion_display.insert('1.0', self.response_text)
        self.suggestion_display.config(state=tk.DISABLED)
        self._build_hunks_and_highlight()
        self.status_label.config(text="Ready for review")
        self.accept_btn.config(state=tk.NORMAL)

    # === Word-Level Diff & Selective Hunk Methods ===

    def _build_hunks_and_highlight(self):
        """Compute line/paragraph hunks and highlight word-level differences."""
        orig_lines = self.original_text.splitlines(keepends=True)
        sugg_lines = self.full_suggestion_text.splitlines(keepends=True)

        matcher = difflib.SequenceMatcher(None, orig_lines, sugg_lines)
        opcodes = matcher.get_opcodes()

        self._hunks = []
        self._hunk_vars = []
        for child in self.hunks_flow.winfo_children():
            child.destroy()

        changed_count = 0
        for tag, i1, i2, j1, j2 in opcodes:
            orig_chunk = "".join(orig_lines[i1:i2])
            sugg_chunk = "".join(sugg_lines[j1:j2])
            if tag == 'equal':
                self._hunks.append({
                    'equal': True,
                    'orig': orig_chunk,
                    'sugg': sugg_chunk,
                    'var': None
                })
            else:
                changed_count += 1
                var = tk.BooleanVar(value=True)
                self._hunk_vars.append(var)
                self._hunks.append({
                    'equal': False,
                    ' index': changed_count,
                    'orig': orig_chunk,
                    'sugg': sugg_chunk,
                    'i1': i1 + 1,
                    'i2': max(i1 + 1, i2),
                    'var': var
                })

        # Show hunk selector bar if there are multiple changed hunks (or at least 1 when original is non-empty)
        if changed_count >= 2:
            self.hunks_outer.pack(fill=tk.X, pady=(0, 6), before=self.accept_btn.master)
            for idx, h in enumerate([hk for hk in self._hunks if not hk['equal']], 1):
                orig_w = len(h['orig'].split())
                sugg_w = len(h['sugg'].split())
                preview_src = (h['sugg'].strip() or h['orig'].strip()).replace('\n', ' ')
                if len(preview_src) > 28:
                    preview_src = preview_src[:28] + "…"
                lbl = f"Hunk {idx} (L{h['i1']}–{h['i2']}, -{orig_w}w/+{sugg_w}w): {preview_src}"
                cb = ttk.Checkbutton(
                    self.hunks_flow,
                    text=lbl,
                    variable=h['var'],
                    command=self._rebuild_from_hunks
                )
                r, c = divmod(idx - 1, 2)
                cb.grid(row=r, column=c, sticky="w", padx=6, pady=1)
        else:
            self.hunks_outer.pack_forget()

        self._highlight_word_diffs()

    def _set_all_hunks(self, state: bool):
        for var in self._hunk_vars:
            var.set(state)
        self._rebuild_from_hunks()

    def _rebuild_from_hunks(self):
        """Rebuild the suggestion display based on which hunks are checked."""
        if not self._hunks:
            return
        parts = []
        for h in self._hunks:
            if h['equal'] or (h['var'] is not None and h['var'].get()):
                parts.append(h['sugg'])
            else:
                parts.append(h['orig'])
        rebuilt = "".join(parts)
        self.response_text = rebuilt

        was_disabled = str(self.suggestion_display.cget('state')) == 'disabled'
        if was_disabled:
            self.suggestion_display.config(state=tk.NORMAL)
        self.suggestion_display.delete('1.0', tk.END)
        self.suggestion_display.insert('1.0', rebuilt)
        if was_disabled:
            self.suggestion_display.config(state=tk.DISABLED)

        self._highlight_word_diffs()

    def _refresh_word_diff_tags(self):
        if self.word_diff_var.get():
            self._highlight_word_diffs()
        else:
            self.original_display.tag_remove('diff_del', '1.0', tk.END)
            self.suggestion_display.tag_remove('diff_ins', '1.0', tk.END)

    def _highlight_word_diffs(self):
        """Compute word-level inline diffs using SequenceMatcher and highlight spans."""
        orig_text = self.original_display.get('1.0', 'end-1c')
        sugg_text = self.suggestion_display.get('1.0', 'end-1c')

        self.original_display.tag_remove('diff_del', '1.0', tk.END)
        self.suggestion_display.tag_remove('diff_ins', '1.0', tk.END)

        orig_tokens = re.findall(r'\S+|\s+', orig_text)
        sugg_tokens = re.findall(r'\S+|\s+', sugg_text)

        # Precompute character offsets for each token
        orig_offsets: List[Tuple[int, int]] = []
        pos = 0
        for tok in orig_tokens:
            nxt = pos + len(tok)
            orig_offsets.append((pos, nxt))
            pos = nxt

        sugg_offsets: List[Tuple[int, int]] = []
        pos = 0
        for tok in sugg_tokens:
            nxt = pos + len(tok)
            sugg_offsets.append((pos, nxt))
            pos = nxt

        matcher = difflib.SequenceMatcher(None, orig_tokens, sugg_tokens, autojunk=False)
        del_words = 0
        ins_words = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ('delete', 'replace') and i1 < i2 and orig_offsets:
                c_start = orig_offsets[i1][0]
                c_end = orig_offsets[i2 - 1][1]
                del_words += sum(1 for t in orig_tokens[i1:i2] if not t.isspace())
                if self.word_diff_var.get():
                    self.original_display.tag_add(
                        'diff_del', f"1.0+{c_start}c", f"1.0+{c_end}c"
                    )
            if tag in ('insert', 'replace') and j1 < j2 and sugg_offsets:
                c_start = sugg_offsets[j1][0]
                c_end = sugg_offsets[j2 - 1][1]
                ins_words += sum(1 for t in sugg_tokens[j1:j2] if not t.isspace())
                if self.word_diff_var.get():
                    self.suggestion_display.tag_add(
                        'diff_ins', f"1.0+{c_start}c", f"1.0+{c_end}c"
                    )

        self.diff_stats_label.config(text=f"(+{ins_words} words / −{del_words} words)")

    def _toggle_edit(self):
        """Toggle editability of the suggestion pane"""
        if self.is_generating:
            return

        P = self._pal
        self.is_editable = not self.is_editable
        if self.is_editable:
            self.suggestion_display.config(state=tk.NORMAL, bg=P['edit_bg'])
            self.edit_btn.config(text="Lock & Re-Diff")
            self.status_label.config(text="Editing suggestion...")
        else:
            self.suggestion_display.config(state=tk.DISABLED, bg=P['sugg_bg'])
            self.edit_btn.config(text="Edit")
            self._highlight_word_diffs()
            self.status_label.config(text="Ready for review")

    def _get_selected_apply_mode(self) -> str:
        label = self.apply_mode_var.get()
        for key, lbl in self.APPLY_MODES:
            if lbl == label:
                return key
        return "replace"

    def _on_accept(self):
        """Accept the suggestion and apply to document"""
        if self.is_generating:
            return

        replacement = self.suggestion_display.get('1.0', tk.END).rstrip('\n')
        apply_mode = self._get_selected_apply_mode()

        if not replacement.strip() and apply_mode in ("replace", "full"):
            if not messagebox.askyesno("Empty Suggestion",
                                       "The suggestion is empty. This will delete the target text. Continue?",
                                       parent=self):
                return

        if self.on_accept:
            try:
                self.on_accept(replacement, self.start_index, self.end_index, self.original_text, apply_mode)
            except TypeError:
                self.on_accept(replacement, self.start_index, self.end_index, self.original_text)

        self.destroy()

    def _on_reject(self):
        """Reject and close"""
        if self.is_generating:
            if not messagebox.askyesno("Cancel Generation",
                                       "AI is still generating. Close anyway?",
                                       parent=self):
                return

        if self.on_reject:
            self.on_reject()

        self.destroy()

