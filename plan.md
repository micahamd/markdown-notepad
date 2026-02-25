# UX Improvement Plan — MarkItDown Notepad

## Overview

Six UX improvements implemented in strict order. Features 1–5 are editor/chat polish. Feature 6 (CLI provider) is more architecturally distinct and is implemented last once all others are stable.

**Files touched**: `markdown_notepad.py`, `ai_chat.py`  
**New dependencies**: none — all features use stdlib and existing Tkinter machinery  
**Constraint**: all changes are purely additive; existing behaviour is preserved exactly when features are inactive/disabled.

---

## Feature 1 — Source Mode Syntax Highlighting

**File**: `markdown_notepad.py`  
**Classes modified**: `DocumentTab`, `MarkdownNotepad`

### Goal
Apply live, colour-coded text tags to markdown syntax in the raw source editor, turning it from a plain text box into a lightweight markdown-aware editor.

### 1.1 — Define syntax tags (`DocumentTab._setup_syntax_tags`)

Call `self._setup_syntax_tags()` inside `DocumentTab.__init__` immediately after `self.source_editor` is created (around line 1195). Define the following tags on `self.source_editor`:

| Tag name | Visual style | What it matches |
|---|---|---|
| `syn_h1` | bold, `foreground="#1a1a2e"`, `font_size = source_font_size + 8` | Line starting with exactly one `#` |
| `syn_h2` | bold, `foreground="#16213e"`, `+5` | `## ` |
| `syn_h3` | bold, `foreground="#1f4068"`, `+3` | `### ` |
| `syn_h4` | bold, `foreground="#1b1b2f"`, `+1` | `####` through `######` |
| `syn_bold` | bold | `**text**` or `__text__` |
| `syn_italic` | italic | `*text*` or `_text_` (single delimiters) |
| `syn_code_inline` | monospace, `background="#f0f0f0"`, `foreground="#c7254e"` | `` `text` `` |
| `syn_code_block` | monospace, `background="#282c34"`, `foreground="#abb2bf"`, `lmargin1=20` | Full lines inside ` ``` ` fences |
| `syn_link` | `foreground="#0066cc"`, `underline=True` | `[text](url)` |
| `syn_blockquote` | italic, `foreground="#6c757d"`, `lmargin1=30` | Lines starting `> ` |
| `syn_list_marker` | bold, `foreground="#0066cc"` | `- `, `* `, `+ `, or `N. ` at line start |
| `syn_hr` | `foreground="#aaaaaa"` | Lines that are exactly `---`, `***`, or `___` |

Font sizes for `syn_h1`–`syn_h4` are computed as `self.theme['source_font_size'] + delta` so they automatically reflect the current zoom level.

### 1.2 — `_apply_syntax_highlighting()` method on `DocumentTab`

```python
def _apply_syntax_highlighting(self):
    editor = self.source_editor
    # Clear all existing syntax tags in one pass
    for tag in ('syn_h1','syn_h2','syn_h3','syn_h4','syn_bold','syn_italic',
                'syn_code_inline','syn_code_block','syn_link',
                'syn_blockquote','syn_list_marker','syn_hr'):
        editor.tag_remove(tag, '1.0', tk.END)

    content = editor.get('1.0', tk.END)
    lines = content.split('\n')
    in_code_block = False

    for lineno, line in enumerate(lines, start=1):
        line_start = f'{lineno}.0'
        line_end   = f'{lineno}.end'

        # Code fence — toggle block state, style the fence line itself
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            editor.tag_add('syn_code_block', line_start, line_end)
            continue

        if in_code_block:
            editor.tag_add('syn_code_block', line_start, line_end)
            continue

        # Headers (checked most-specific first to avoid partial matches)
        stripped = line.lstrip()
        if   re.match(r'^# [^#]',    line): editor.tag_add('syn_h1', line_start, line_end)
        elif re.match(r'^## [^#]',   line): editor.tag_add('syn_h2', line_start, line_end)
        elif re.match(r'^### [^#]',  line): editor.tag_add('syn_h3', line_start, line_end)
        elif re.match(r'^#{4,6} ',   line): editor.tag_add('syn_h4', line_start, line_end)

        # Horizontal rule (whole line)
        elif line.strip() in ('---', '***', '___'):
            editor.tag_add('syn_hr', line_start, line_end)

        # Blockquote
        elif line.startswith('> '):
            editor.tag_add('syn_blockquote', line_start, line_end)

        else:
            # List marker — colour only the leading `- ` / `* ` / `1. ` characters
            m = re.match(r'^(\s*)([-*+]|\d+\.)\s', line)
            if m:
                editor.tag_add('syn_list_marker', line_start, f'{lineno}.{m.end()}')

        # Inline patterns (applied on all non-code-block lines regardless of block type)
        for m in re.finditer(r'(\*\*|__)(.+?)\1', line):
            editor.tag_add('syn_bold',    f'{lineno}.{m.start()}', f'{lineno}.{m.end()}')
        for m in re.finditer(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', line):
            editor.tag_add('syn_italic',  f'{lineno}.{m.start()}', f'{lineno}.{m.end()}')
        for m in re.finditer(r'`([^`]+)`', line):
            editor.tag_add('syn_code_inline', f'{lineno}.{m.start()}', f'{lineno}.{m.end()}')
        for m in re.finditer(r'\[([^\]]+)\]\([^\)]+\)', line):
            editor.tag_add('syn_link',    f'{lineno}.{m.start()}', f'{lineno}.{m.end()}')
```

### 1.3 — Debounced trigger in `MarkdownNotepad`

Add to `MarkdownNotepad`:

```python
def _schedule_syntax_highlight(self, event=None):
    if not getattr(self, '_syntax_hl_pending', False):
        self._syntax_hl_pending = True
        self.after(300, self._do_syntax_highlight)

def _do_syntax_highlight(self):
    self._syntax_hl_pending = False
    if self.current_tab:
        self.current_tab._apply_syntax_highlighting()
```

In `_setup_tab_bindings(tab)`, bind:
```python
tab.source_editor.bind('<KeyRelease>', self._schedule_syntax_highlight)
```
(In addition to the existing position-update binding — use `add='+'` so both handlers fire.)

Also call `self.current_tab._apply_syntax_highlighting()` at the end of:
- `new_tab()` (after content is loaded)
- `_on_tab_changed()` (after switching tabs)
- `_do_syntax_highlight()` is also called from `zoom_in()`, `zoom_out()`, `zoom_reset()` so header sizes stay correct after font changes.

---

## Feature 2 — Document Outline Panel

**File**: `markdown_notepad.py`  
**New class**: `OutlinePanel`  
**Modified class**: `MarkdownNotepad`

### Goal
A collapsible left panel that shows the document's heading hierarchy. Clicking a heading scrolls the editor to that line.

### 2.1 — `OutlinePanel` class (add just before `MarkdownNotepad`)

```python
class OutlinePanel(ttk.Frame):
    """Collapsible document outline — heading tree with click-to-navigate."""

    LEVEL_FG = {
        1: '#1a1a2e', 2: '#16213e', 3: '#1f4068',
        4: '#555555', 5: '#777777', 6: '#999999',
    }

    def __init__(self, parent, on_heading_click=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_heading_click = on_heading_click  # callable(lineno: int)
        self._linenos = []        # parallel list of line numbers for each tree row
        self._update_job = None
        self._setup_ui()

    def _setup_ui(self):
        # Header bar
        header = ttk.Frame(self)
        header.pack(fill=tk.X)
        ttk.Label(header, text='Outline', font=(SANS_FONT, 10, 'bold')).pack(
            side=tk.LEFT, padx=8, pady=(6, 4))

        # Treeview + scrollbar
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(tree_frame, show='tree',
                                  selectmode='browse', yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=self.tree.yview)

        # Per-level visual style
        for lvl in range(1, 7):
            self.tree.tag_configure(
                f'h{lvl}',
                foreground=self.LEVEL_FG[lvl],
                font=(SANS_FONT, max(11 - lvl + 1, 9),
                      'bold' if lvl <= 3 else 'normal')
            )
        self.tree.tag_configure('empty', foreground='#aaaaaa',
                                 font=(SANS_FONT, 9, 'italic'))

        self.tree.bind('<<TreeviewSelect>>', self._on_select)

    # ------------------------------------------------------------------
    def schedule_update(self, content: str):
        """Debounced rebuild (500 ms) — safe to call on every keystroke."""
        if self._update_job:
            self.after_cancel(self._update_job)
        self._update_job = self.after(500, lambda: self.update(content))

    def update(self, content: str):
        """Immediately rebuild the outline tree from document content."""
        self._update_job = None
        self.tree.delete(*self.tree.get_children())
        self._linenos = []

        headings = [
            (lineno, len(m.group(1)), m.group(2).strip())
            for lineno, line in enumerate(content.split('\n'), 1)
            if (m := re.match(r'^(#{1,6})\s+(.+)', line))
        ]

        if not headings:
            self.tree.insert('', tk.END, text='(no headings)', tags=('empty',))
            self._linenos.append(None)
            return

        stack = []   # (level, iid)
        for lineno, level, text in headings:
            while stack and stack[-1][0] >= level:
                stack.pop()
            indent = '  ' * (level - 1)
            label  = f'{indent}{"#" * level} {text}'
            parent_iid = stack[-1][1] if stack else ''
            iid = self.tree.insert(parent_iid, tk.END, text=label,
                                   tags=(f'h{level}',), open=True)
            stack.append((level, iid))
            self._linenos.append(lineno)

    # ------------------------------------------------------------------
    def _on_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        all_items = self._dfs_items()
        try:
            idx = all_items.index(sel[0])
            lineno = self._linenos[idx]
            if lineno and self._on_heading_click:
                self._on_heading_click(lineno)
        except (ValueError, IndexError):
            pass

    def _dfs_items(self):
        result = []
        def recurse(parent=''):
            for child in self.tree.get_children(parent):
                result.append(child)
                recurse(child)
        recurse()
        return result
```

### 2.2 — Layout integration in `MarkdownNotepad._setup_ui()`

Replace the existing `main_paned` creation block with an outer + inner paned structure:

```python
# Outer paned: [outline?] | [main_paned]
self.outer_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
self.outer_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

self.outline_panel   = OutlinePanel(self.outer_paned,
                                     on_heading_click=self._jump_to_line,
                                     width=200)
self.outline_visible = False
# outline_panel is NOT added to outer_paned until first toggle

# Inner paned: editor | AI sidebar (unchanged)
self.main_paned = ttk.PanedWindow(self.outer_paned, orient=tk.HORIZONTAL)
self.outer_paned.add(self.main_paned, weight=1)

self.content_frame = ttk.Frame(self.main_paned)
self.main_paned.add(self.content_frame, weight=1)
```

### 2.3 — Navigation helper

```python
def _jump_to_line(self, lineno: int):
    if not self.current_tab:
        return
    if self.current_tab.current_mode == 'source':
        editor = self.current_tab.source_editor
        editor.see(f'{lineno}.0')
        editor.mark_set(tk.INSERT, f'{lineno}.0')
        editor.focus_set()
    else:
        content = self.current_tab.get_content()
        total   = max(content.count('\n') + 1, 1)
        self.current_tab.visual_viewer.text_widget.yview_moveto((lineno - 1) / total)
```

### 2.4 — Toggle method

```python
def _toggle_outline_panel(self):
    if self.outline_visible:
        self.outer_paned.forget(self.outline_panel)
        self.outline_visible = False
    else:
        self.outer_paned.insert(0, self.outline_panel, weight=0, minsize=150)
        self.outline_visible = True
        self._schedule_outline_update()
```

### 2.5 — Outline update triggers

```python
def _schedule_outline_update(self):
    if self.outline_visible and self.current_tab:
        self.outline_panel.schedule_update(self.current_tab.get_content())
```

Call `_schedule_outline_update()` from:
- `_schedule_position_update()` (piggybacks on existing keystroke debounce)
- `_on_tab_changed()`
- After content is loaded in `new_tab()`

### 2.6 — Menu and shortcut

In `_create_menu()` View menu:
```python
view_menu.add_command(label='Document Outline',
                      command=self._toggle_outline_panel,
                      accelerator='Ctrl+Shift+L')
```

In `_setup_bindings()`:
```python
self.bind('<Control-L>', lambda e: self._toggle_outline_panel())  # Ctrl+Shift+L
```

---

## Feature 3 — AI Chat Markdown Rendering in Responses

**File**: `ai_chat.py`  
**Class modified**: `ChatSidebar`

### Goal
Render AI assistant responses with inline formatting (bold, italic, code, code blocks, headers, lists) instead of plain text.

### 3.1 — Additional text tags in `_setup_ui()`

After the existing `tag_configure` calls, add:

```python
self.chat_display.tag_configure('md_bold',
    font=(SANS_FONT, 10, 'bold'), foreground='#1a1a1a')
self.chat_display.tag_configure('md_italic',
    font=(SANS_FONT, 10, 'italic'), foreground='#1a1a1a')
self.chat_display.tag_configure('md_code_inline',
    font=(MONO_FONT, 9), background='#f0f0f0', foreground='#c7254e')
self.chat_display.tag_configure('md_code_block',
    font=(MONO_FONT, 9), background='#282c34', foreground='#abb2bf',
    lmargin1=10, lmargin2=10, spacing1=4, spacing3=4)
self.chat_display.tag_configure('md_h1',
    font=(SANS_FONT, 14, 'bold'), foreground='#1a1a2e', spacing1=8)
self.chat_display.tag_configure('md_h2',
    font=(SANS_FONT, 12, 'bold'), foreground='#16213e', spacing1=6)
self.chat_display.tag_configure('md_h3',
    font=(SANS_FONT, 11, 'bold'), foreground='#1f4068', spacing1=4)
self.chat_display.tag_configure('md_list_item', lmargin1=15, lmargin2=25)
self.chat_display.tag_configure('md_blockquote',
    foreground='#6c757d', font=(SANS_FONT, 10, 'italic'),
    lmargin1=20, background='#f8f9fa')
```

### 3.2 — Module-level `_insert_inline()` helper

Place this as a module-level function (not a method) since it is stateless:

```python
def _insert_inline(widget, text: str):
    """Insert one line of text into widget with bold/italic/code tags applied."""
    pattern = re.compile(
        r'(`[^`]+`)'           # inline code — highest priority
        r'|(\*\*[^*]+\*\*)'   # bold **
        r'|(__[^_]+__)'        # bold __
        r'|(\*[^*]+\*)'        # italic *
        r'|(_[^_]+_)'          # italic _
    )
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            widget.insert(tk.END, text[pos:m.start()])
        raw = m.group(0)
        if raw.startswith('`'):
            widget.insert(tk.END, raw[1:-1], 'md_code_inline')
        elif raw.startswith('**') or raw.startswith('__'):
            widget.insert(tk.END, raw[2:-2], 'md_bold')
        else:
            widget.insert(tk.END, raw[1:-1], 'md_italic')
        pos = m.end()
    if pos < len(text):
        widget.insert(tk.END, text[pos:])
```

### 3.3 — `_insert_markdown_text()` method on `ChatSidebar`

```python
def _insert_markdown_text(self, text: str):
    """Insert markdown-formatted assistant response into the (enabled) chat_display."""
    widget  = self.chat_display
    lines   = text.split('\n')
    in_code = False
    code_buf = []

    def flush_code():
        if code_buf:
            widget.insert(tk.END, '\n'.join(code_buf) + '\n', 'md_code_block')
            code_buf.clear()

    for line in lines:
        if line.strip().startswith('```'):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_buf.append(line)
            continue

        hm = re.match(r'^(#{1,3})\s+(.*)', line)
        if hm:
            widget.insert(tk.END, hm.group(2) + '\n', f'md_h{len(hm.group(1))}')
            continue

        if line.startswith('> '):
            widget.insert(tk.END, line[2:] + '\n', 'md_blockquote')
            continue

        lm = re.match(r'^(\s*)([-*+]|\d+\.)\s(.*)', line)
        if lm:
            prefix = '  • ' if not lm.group(2)[0].isdigit() else f'  {lm.group(2)} '
            widget.insert(tk.END, prefix, 'md_list_item')
            _insert_inline(widget, lm.group(3))
            widget.insert(tk.END, '\n')
            continue

        _insert_inline(widget, line)
        widget.insert(tk.END, '\n')

    flush_code()   # handle unclosed fence at end of response
```

### 3.4 — Streaming: plain text during generation, formatted on completion

Add to `ChatSidebar.__init__`:
```python
self._streaming_buffer: str = ''
self._streaming_start_index: str = ''
```

In `_add_chat_message()`, when `role == 'assistant'` and `streaming=True`, record the position just before the body would be inserted:
```python
self._streaming_start_index = self.chat_display.index(tk.END)
self._streaming_buffer = ''
```

`_append_to_last_message()` stays unchanged — it keeps appending raw text chunks to the widget for the "live typing" feel.

In `_process_queue()`, when `msg_type == 'done'`, after all other completion logic, add:

```python
# Re-render last assistant message with markdown formatting
if self._streaming_buffer:
    self.chat_display.config(state=tk.NORMAL)
    self.chat_display.delete(self._streaming_start_index, tk.END)
    self._insert_markdown_text(self._streaming_buffer)
    self.chat_display.config(state=tk.DISABLED)
    self.chat_display.see(tk.END)
```

And in `_generate_response()`, accumulate into `self._streaming_buffer` as well:
```python
response_text += chunk
self._streaming_buffer += chunk        # ← add this line
self.response_queue.put(('chunk', chunk))
```

This gives: live streaming feel during generation → clean formatted output on completion.

---

## Feature 4 — AI Chat Copy Button per Message

**File**: `ai_chat.py`  
**Class modified**: `ChatSidebar`

### Goal
Embed a small `⧉` copy button after every completed message so any response can be grabbed with one click.

### 4.1 — Track message texts

Add to `__init__`:
```python
self._message_texts: list = []  # plain-text content of each message in display order
```

### 4.2 — Embed button in `_add_chat_message()`

After inserting the message body, and only when `not streaming`:

```python
if not streaming and role in ('user', 'assistant'):
    self._message_texts.append(content)

    def _make_copy(text):
        def _do():
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_label.config(text='Copied!')
            self.after(1500, lambda: self.status_label.config(text='Ready'))
        return _do

    btn = tk.Button(
        self.chat_display,
        text='⧉', font=(SANS_FONT, 8),
        relief=tk.FLAT, bd=0,
        bg=self.chat_display.cget('bg'), fg='#888888',
        activeforeground='#0066cc',
        cursor='hand2', padx=2, pady=0,
        command=_make_copy(content)
    )
    self.chat_display.window_create(tk.END, window=btn, padx=2)
```

### 4.3 — Copy button for streamed messages

In `_process_queue()`, in the `'done'` block, after the markdown re-render, insert the copy button for the completed streaming message using the same pattern with `text = self._streaming_buffer` (before clearing it).

### 4.4 — Clear `_message_texts` in `_clear_display()`

```python
self._message_texts.clear()
```

---

## Feature 5 — Visual Mode Live Re-render (with toggle)

**File**: `markdown_notepad.py`  
**Classes modified**: `MarkdownVisualWidget`, `MarkdownNotepad`

### Goal
Typing in Visual mode re-renders the markdown formatting after a short pause. A View menu toggle lets the user disable this if they prefer static rendering.

### 5.1 — `MarkdownVisualWidget` additions

In `__init__`, add:
```python
self.live_render_enabled = True
self._live_render_job    = None
self._last_rendered_hash = ''
```

### 5.2 — Modified `_on_edit()`

```python
def _on_edit(self, event=None):
    content = self.text_widget.get('1.0', tk.END).rstrip('\n')
    self.content = content
    if self.on_content_change:
        self.on_content_change(content)

    if self.live_render_enabled:
        if self._live_render_job:
            self.after_cancel(self._live_render_job)
        import hashlib as _hl
        h = _hl.md5(content.encode()).hexdigest()
        if h != self._last_rendered_hash:
            self._live_render_job = self.after(450, lambda: self._do_live_render(content, h))

def _do_live_render(self, content: str, expected_hash: str):
    self._live_render_job = None
    # Abort if content has changed again since this was scheduled
    current = self.text_widget.get('1.0', tk.END).rstrip('\n')
    import hashlib as _hl
    if _hl.md5(current.encode()).hexdigest() != expected_hash:
        return

    # Save cursor position as character offset from document start
    char_offset = None
    try:
        cursor_idx  = self.text_widget.index(tk.INSERT)
        pre_text    = self.text_widget.get('1.0', cursor_idx)
        char_offset = len(pre_text)
    except Exception:
        pass

    # Re-render
    self._last_rendered_hash = expected_hash
    self.set_content(content)

    # Restore cursor (best effort)
    if char_offset is not None:
        try:
            full     = self.text_widget.get('1.0', tk.END)
            pre      = full[:char_offset]
            row      = pre.count('\n') + 1
            col      = len(pre) - (pre.rfind('\n') + 1)
            self.text_widget.mark_set(tk.INSERT, f'{row}.{col}')
            self.text_widget.see(tk.INSERT)
        except Exception:
            pass
```

### 5.3 — View menu toggle

In `MarkdownNotepad._create_menu()`:
```python
self.live_render_var = tk.BooleanVar(value=True)
view_menu.add_checkbutton(
    label='Live Re-render in Visual Mode',
    variable=self.live_render_var,
    command=self._toggle_live_render
)
```

```python
def _toggle_live_render(self):
    enabled = self.live_render_var.get()
    for tab in self.tabs:
        tab.visual_viewer.live_render_enabled = enabled
    self.theme['live_render_enabled'] = enabled
    self._save_theme()
```

### 5.4 — Restore preference on startup

In `_load_theme()`, after `self.theme.update(saved_theme)`, the key is already in `self.theme`.

In `new_tab()` and `_restore_session()`, after creating each `DocumentTab`:
```python
tab.visual_viewer.live_render_enabled = self.theme.get('live_render_enabled', True)
```

Also set `self.live_render_var` in `__init__` after loading theme:
```python
self.live_render_var = tk.BooleanVar(value=self.theme.get('live_render_enabled', True))
```
(Move the `BooleanVar` creation to before `_setup_ui()` since `_create_menu()` references it.)

---

## Feature 6 — CLI AI Provider

**File**: `ai_chat.py`  
**New class**: `CLIClient`  
**Modified**: `AISettings`, `AISettingsDialog`, `ChatSidebar`, `get_llm_client()`

### Design principles

Many CLI AI tools (e.g. `llm`, `claude`, `gh copilot`, custom wrappers) maintain their own conversation context between invocations — via session files, server state, or in-process memory. The app should not try to re-inject history from its own store; it should simply forward the user's message to the CLI tool and stream the output back.

Key decisions:

1. **One process per message** — each send spawns a fresh subprocess. Context is the CLI tool's responsibility (e.g. `llm -c` continues the last session; `gh copilot` manages its own context).
2. **No prompt template** — the user's raw message text (and any document context the user opts to include via the existing "Include Document" checkbox) is passed directly to the CLI.
3. **No history injection from app side** — the existing "Include Chat History" checkbox is hidden/disabled for the CLI provider because the CLI tool owns history. The "Include Document" checkbox remains useful.
4. **Two input methods** — `stdin` (message piped to the process) and `args` (message appended as a final isolated argument). Selected by the user once during configuration.
5. **`shell=False` always** — command is parsed with `shlex.split()`, message never interpolated into a shell string.
6. **`/clear` passthrough** — if the user types `/clear` in the chat input, it is forwarded to the CLI as a regular message. The CLI tool handles it (e.g., starting a new session). The chat display is also cleared on the app side.
7. **Validate button** — analogous to "Test" on other providers; runs the command with a trivial prompt and reports success if any stdout is received.

### 6.1 — New settings fields in `AISettings`

Add to `DEFAULT_SETTINGS`:

```python
'cli_command':      '',        # e.g. "llm -c" or "gh copilot explain"
'cli_input_method': 'stdin',   # 'stdin' | 'args'
```

Add corresponding properties:

```python
@property
def cli_command(self) -> str:
    return self.settings.get('cli_command', '')
@cli_command.setter
def cli_command(self, v: str):
    self.settings['cli_command'] = v

@property
def cli_input_method(self) -> str:
    return self.settings.get('cli_input_method', 'stdin')
@cli_input_method.setter
def cli_input_method(self, v: str):
    self.settings['cli_input_method'] = v
```

Update `is_configured()`:
```python
elif self.provider == 'cli':
    return bool(self.cli_command.strip())
```

### 6.2 — `CLIClient` class (place after `DeepSeekClient`, before `get_llm_client`)

```python
import shlex, subprocess

class CLIClient(LLMClient):
    """
    AI provider that forwards messages to any local CLI tool via subprocess.

    Context management is entirely delegated to the CLI tool. The app sends
    only the current user message; the tool is responsible for maintaining
    conversation history between calls (e.g. via session files or flags).

    Security: shell=False always. The command string is split with shlex;
    the message is passed via stdin or as an isolated final argument — it is
    never interpolated into a shell string.
    """

    # Reasonable default timeout; not user-configurable (keep UI simple)
    TIMEOUT = 120

    def __init__(self, command: str, input_method: str = 'stdin'):
        self.command      = command.strip()
        self.input_method = input_method   # 'stdin' | 'args'
        self.api_key      = ''             # unused; kept for interface compat

    # ------------------------------------------------------------------
    def send_message_stream(
        self,
        messages: List[Dict],
        system_prompt: str = '',
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        images: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        """Run the CLI command, pass the last user message, stream stdout."""
        if not self.command:
            raise RuntimeError(
                'No CLI command configured. '
                'Please set one in AI Settings → CLI Configuration.')

        # Extract the last user message
        user_message = ''
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
                break

        try:
            cmd_parts = shlex.split(self.command)
        except ValueError as e:
            raise RuntimeError(f'Invalid CLI command syntax: {e}')

        if not cmd_parts:
            raise RuntimeError('CLI command is empty.')

        popen_kw: dict = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'shell':  False,
        }

        if self.input_method == 'stdin':
            popen_kw['stdin'] = subprocess.PIPE
        elif self.input_method == 'args':
            # Append message as a separate, unquoted argument — never via shell
            cmd_parts = cmd_parts + [user_message]
        else:
            raise RuntimeError(f"Unknown input_method: '{self.input_method}'")

        # Launch
        try:
            proc = subprocess.Popen(cmd_parts, **popen_kw)
        except FileNotFoundError:
            raise RuntimeError(
                f"Command not found: '{cmd_parts[0]}'. "
                f"Is it installed and on your PATH?")
        except PermissionError:
            raise RuntimeError(f"Permission denied: '{cmd_parts[0]}'.")
        except Exception as e:
            raise RuntimeError(f'Failed to start process: {e}')

        # Write message to stdin
        if self.input_method == 'stdin':
            try:
                proc.stdin.write(user_message.encode('utf-8', errors='replace'))
                proc.stdin.close()
            except BrokenPipeError:
                pass

        # Stream stdout line by line
        try:
            for raw in proc.stdout:
                yield raw.decode('utf-8', errors='replace')
        except Exception as e:
            proc.kill()
            raise RuntimeError(f'Error reading CLI output: {e}')

        # Wait for exit
        try:
            proc.wait(timeout=self.TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(
                f'CLI command did not finish within {self.TIMEOUT} seconds.')

        if proc.returncode not in (0, None):
            stderr = proc.stderr.read().decode('utf-8', errors='replace').strip()
            if stderr:
                raise RuntimeError(
                    f'CLI exited with code {proc.returncode}: {stderr}')

    # ------------------------------------------------------------------
    def get_available_models(self) -> List[str]:
        """CLI has no model selection — the command itself specifies the tool."""
        return []

    def test_connection(self) -> tuple:
        """
        Validate the CLI by running it with a simple probe message.
        Returns (success: bool, message: str).
        """
        if not self.command.strip():
            return False, 'No command configured.'

        try:
            cmd_parts = shlex.split(self.command)
        except ValueError as e:
            return False, f'Invalid command syntax: {e}'

        probe = 'ping'  # minimal single-word probe
        popen_kw: dict = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'shell':  False,
        }

        if self.input_method == 'stdin':
            popen_kw['stdin'] = subprocess.PIPE
            run_cmd = cmd_parts
        else:
            popen_kw['stdin'] = subprocess.DEVNULL
            run_cmd = cmd_parts + [probe]

        try:
            proc = subprocess.Popen(run_cmd, **popen_kw)
            if self.input_method == 'stdin':
                stdout, stderr = proc.communicate(
                    input=probe.encode(), timeout=30)
            else:
                stdout, stderr = proc.communicate(timeout=30)

            output = stdout.decode('utf-8', errors='replace').strip()
            if output:
                preview = output[:120].replace('\n', ' ')
                return True, f'✓ Command is working. Response: "{preview}"'
            elif proc.returncode == 0:
                return True, '✓ Command ran successfully (no output received).'
            else:
                err = stderr.decode('utf-8', errors='replace').strip()
                return False, (f'Command exited {proc.returncode}. '
                               f'{err or "(no stderr output)"}')

        except FileNotFoundError:
            return False, (f"Command not found: '{cmd_parts[0]}'. "
                           f"Is it installed and available on your PATH?")
        except subprocess.TimeoutExpired:
            proc.kill()
            return False, 'Command timed out (30 s). Is the tool responsive?'
        except Exception as e:
            return False, f'Error: {e}'

    def encode_image_for_api(self, image, max_size: int = 512):
        """CLI provider does not support image attachments."""
        return None
```

### 6.3 — Update `get_llm_client()` factory

```python
elif provider == 'cli':
    return CLIClient(
        command=settings.cli_command,
        input_method=settings.cli_input_method,
    )
```

### 6.4 — `AISettingsDialog` changes

#### Add `'cli'` to the provider combobox (line ~1714):
```python
values=["anthropic", "gemini", "deepseek", "ollama", "cli"]
```

#### Add CLI configuration section in `_setup_ui()`

Insert a new `ttk.LabelFrame` **after** the Ollama row in the API Keys/Connection frame, but keep it initially hidden. It is shown/hidden by `_on_provider_change()`:

```python
# CLI Configuration section — shown only when provider == 'cli'
self.cli_config_frame = ttk.LabelFrame(main_frame, text="CLI Configuration", padding=10)
# Not packed here — managed by _on_provider_change()

# Row 0: "Command Line" label + entry
ttk.Label(self.cli_config_frame, text="Command Line:").grid(
    row=0, column=0, sticky=tk.W, pady=5)
self.cli_command_var = tk.StringVar()
cli_entry = ttk.Entry(self.cli_config_frame,
                       textvariable=self.cli_command_var, width=38)
cli_entry.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=(8, 0))

# Tooltip explaining what to put here
cli_tip = ttk.Label(self.cli_config_frame, text="ℹ️", font=('Segoe UI', 10))
cli_tip.grid(row=0, column=3, padx=(4, 0))
self._create_tooltip(cli_tip,
    "Enter the CLI command to run, for example:\n"
    "  llm -c              (Simon Willison's LLM, continue session)\n"
    "  gh copilot explain  (GitHub Copilot CLI)\n"
    "  claude              (Anthropic Claude CLI)\n"
    "  /path/to/my-script\n\n"
    "Your message is passed to this command on each send. "
    "Conversation history is managed by the CLI tool itself — "
    "the app does not inject it. Use whatever session/history flags "
    "your tool requires in the command here.\n\n"
    "Do NOT include shell operators (|, >, etc.). "
    "The command runs directly, not through a shell.")

# Row 1: Input method
ttk.Label(self.cli_config_frame, text="Input Method:").grid(
    row=1, column=0, sticky=tk.W, pady=5)
self.cli_input_method_var = tk.StringVar(value='stdin')
im_frame = ttk.Frame(self.cli_config_frame)
im_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=(8, 0))
ttk.Radiobutton(im_frame, text="stdin  (pipe message to process)",
                variable=self.cli_input_method_var, value='stdin').pack(
                    side=tk.LEFT, padx=(0, 12))
ttk.Radiobutton(im_frame, text="args  (append as argument)",
                variable=self.cli_input_method_var, value='args').pack(side=tk.LEFT)
im_tip = ttk.Label(self.cli_config_frame, text="ℹ️", font=('Segoe UI', 10))
im_tip.grid(row=1, column=3, padx=(4, 0))
self._create_tooltip(im_tip,
    "stdin: writes your message to the process's standard input. "
    "Recommended for most CLI tools (llm, claude, etc.).\n\n"
    "args: appends your message as the last command-line argument. "
    "Use for tools that accept the prompt positionally, "
    "e.g. 'gh copilot explain <text>'.")

# Row 2: Validate button
ttk.Button(self.cli_config_frame, text="Validate",
           command=lambda: self._test_connection('cli')).grid(
               row=2, column=1, sticky=tk.W, pady=(10, 0), padx=(8, 0))

# Row 3: TTY warning
ttk.Label(self.cli_config_frame,
          text="⚠  Tools that require an interactive terminal (TTY) are not supported.",
          font=(SANS_FONT, 8), foreground='#cc6600').grid(
              row=3, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))
```

Store `self.model_frame` reference (rename the local variable to `self.model_frame`) so `_on_provider_change()` can toggle its visibility.

#### Update `_on_provider_change()`

Add at the start of the method:

```python
provider = self.provider_var.get()

# Show/hide CLI config frame and model frame
if provider == 'cli':
    self.cli_config_frame.pack(fill=tk.X, pady=(0, 10))
    self.model_frame.pack_forget()
    return   # no model list for CLI
else:
    self.cli_config_frame.pack_forget()
    if not self.model_frame.winfo_ismapped():
        self.model_frame.pack(fill=tk.X, pady=(0, 10))
```

#### Update `_test_connection()` to handle `'cli'`

```python
elif provider == 'cli':
    cmd = self.cli_command_var.get().strip()
    if not cmd:
        messagebox.showwarning("Warning",
            "Please enter a CLI command first.", parent=self)
        return
    client = CLIClient(
        command=cmd,
        input_method=self.cli_input_method_var.get(),
    )
    success, message = client.test_connection()
    if success:
        messagebox.showinfo("CLI Validated", message, parent=self)
    else:
        messagebox.showerror("CLI Validation Failed", message, parent=self)
    return
```

#### Update `_load_current_settings()`

```python
self.cli_command_var.set(self.settings.cli_command)
self.cli_input_method_var.set(self.settings.cli_input_method)
```

#### Update `_save_settings()`

```python
self.settings.cli_command      = self.cli_command_var.get().strip()
self.settings.cli_input_method = self.cli_input_method_var.get()
```

### 6.5 — `ChatSidebar._send_message()` — CLI guard and `/clear` intercept

Replace the provider-specific config check block with:

```python
if self.settings.provider == 'ollama':
    if not self.settings.ollama_url:
        self._add_chat_message('system',
            'Ollama URL not configured. Please configure in Settings.')
        return
elif self.settings.provider == 'gemini':
    if not self.settings.gemini_api_key:
        self._add_chat_message('system',
            'Gemini API key not configured. Please add in Settings.')
        return
elif self.settings.provider == 'cli':
    if not self.settings.cli_command.strip():
        self._add_chat_message('system',
            'CLI command not configured. '
            'Please set one in Settings → CLI Configuration.')
        return
    # Intercept /clear: clear sidebar display and pass the command
    # to the CLI tool so it can also reset its own session state.
    if message.strip() == '/clear':
        self._clear_display()
        self.status_label.config(text='Chat display cleared')
        # Still send /clear to the CLI (it may reset the session)
        # Fall through to normal send logic below.
else:
    if not self.settings.api_key:
        self._add_chat_message('system',
            'API key not configured. Please add your API key in Settings.')
        return
```

### 6.6 — Disable "Include Chat History" for CLI provider

In `_send_message()`, after the config guard, add:

```python
# CLI tools manage their own context — history injection is irrelevant
is_cli = (self.settings.provider == 'cli')
if is_cli:
    # Override: do not pull app-side history into the message
    effective_history = False
else:
    effective_history = self.include_history_var.get()
```

Then replace `self.include_history_var.get()` in the history block with `effective_history`.

Optionally, in `_setup_ui()` (or when `_on_settings_saved()` fires), grey out the "Include Chat History" checkbox when provider is `'cli'`:

```python
def _update_controls_for_provider(self):
    is_cli = (self.settings.provider == 'cli')
    state  = tk.DISABLED if is_cli else tk.NORMAL
    self.include_history_cb.config(state=state)
```

Call `_update_controls_for_provider()` from `_on_settings_saved()` and at sidebar init.

---

## Implementation Order

Implement features strictly in this sequence. Smoke-test each before starting the next.

| Step | Feature | File(s) |
|------|---------|---------|
| 1 | Source Mode Syntax Highlighting | `markdown_notepad.py` |
| 2 | Document Outline Panel | `markdown_notepad.py` |
| 3 | AI Chat Markdown Rendering | `ai_chat.py` |
| 4 | AI Chat Copy Button | `ai_chat.py` |
| 5 | Visual Mode Live Re-render | `markdown_notepad.py` |
| 6 | CLI AI Provider | `ai_chat.py` |

---

## Smoke Tests

1. **Syntax highlighting**: Type `# Hello` in source mode — heading colours immediately. Type `**bold**` — markers embolden. Switch tabs — highlighting present on new tab.
2. **Outline**: Toggle `Ctrl+Shift+L` — panel appears. Open a file with headings — tree populates. Click a heading — editor scrolls there. Toggle again — panel hides.
3. **Chat markdown**: Send a message and elicit a response containing `**bold**`, a code block, and a list. Verify formatted rendering on completion.
4. **Copy button**: Send any message. Click `⧉`. Paste somewhere — text matches message content.
5. **Live re-render**: Switch to Visual mode. Type `**test**`. After ~450 ms observe it renders as bold.
6. **CLI**: Configure `echo` as command, `stdin` mode. Send `hello`. Response should be `hello`. Validate button should report success.

---

## Notes

- Syntax highlighting binds `<KeyRelease>` with `add='+'` so the existing position-update handler is not displaced.
- Outline update is gated on `self.outline_visible` to avoid unnecessary parsing when the panel is hidden.
- Live re-render uses MD5 of content to skip re-renders when nothing has changed.
- CLI uses `shell=False` everywhere; `shlex.split()` parses the command; message never interpolated into a shell string.
- `CLIClient.TIMEOUT` is 120 s, not user-configurable (keeps UI simple; can be made configurable later if needed).
- `_insert_inline()` is a module-level function, not a method, because it is stateless and takes the widget as a parameter.
- `/clear` in the chat input is forwarded to the CLI tool (so it can reset its own session) and also clears the sidebar display.
