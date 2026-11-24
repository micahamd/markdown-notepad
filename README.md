# MarkItDown Notepad

A dual-mode Markdown document editor and integrated Excel-compatible spreadsheet application built with Python and Tkinter, powered by Microsoft's [MarkItDown](https://github.com/microsoft/markitdown) package. Designed with a Royal Purple Word-style two-deck ribbon workspace, it pairs rich visual document authoring, live embedded tables, local background TTS dictation (`Reverb-TTS`), and multi-provider AI assistance with side-by-side diff review and publication-grade PDF, HTML, and DOCX export.

Repository: [github.com/micahamd/markdown-notepad](https://github.com/micahamd/markdown-notepad)

## Highlights

- **Dual-Tab Workspace**: Work seamlessly across rich **Document Tabs** and Excel-style **Spreadsheet Tabs** in one multi-tab window with persistent session state.
- **Always-Clean Visual Editing & Source View (`Ctrl+E`)**: Visual mode renders headings, bold, italic, underline, strikethrough, highlights, font colors/families/sizes, links, alignment tags, embedded interactive tables, and figures cleanly with formatting delimiters hidden—even on the active cursor line—while Source mode exposes the full raw Markdown and inline HTML.
- **Interactive Tables & Figures**: Insert Markdown tables with `+ Row`, `+ Col`, `⇄ Align`, and `📋 TSV`, navigate cells via keyboard (`Tab`/`Shift+Tab`, `Enter`/`Shift+Enter`, `Ctrl+Arrows`), scroll smoothly over embedded tables/figures, and customize table styles under **Theme**.
- **Integrated Excel-Compatible Spreadsheet Processor**: Virtualized 2D grid supporting cell fill/text styling, expandable formula bar with live preview, fill-handle autofill, frozen header row, multi-column sort, word wrap with auto row height, whole-column/whole-row formulas (`$E:$E`, `1:1`), `INDEX`/`MATCH`, `IFERROR`, `VLOOKUP`, `SPARKLINE`, and `.csv`/`.tsv`/`.xlsx`/`.mdsheet` I/O.
- **Local Background TTS Dictation (`Reverb-TTS`)**: Read selections, documents, or the latest AI response aloud offline using quantized `int8` `Reverb-TTS`, with multi-speaker/language selection and persistent custom voice cloning via **Dictation Properties**.
- **Multi-Provider AI Sidebar & Reusable Harnesses**: Supports **Antigravity CLI (`agy`)**, **Ollama** (local), **Google Gemini**, **Anthropic Claude**, and **DeepSeek**, with selection-aware context actions, **Transfer to Chat**, reusable instruction harnesses, side-by-side **Diff Panel**, and **Spreadsheet Diff Dialog**.
- **Publication-Grade Export**: Export documents to standalone HTML, WeasyPrint PDF, or Microsoft Word (`.docx`) preserving theme typography, custom table styles, text alignment, line spacing, inline colors/fonts/underlines, footnotes, cross-reference bookmarks/hyperlinks, and embedded images.

---

## Features

### Document Editor & Rich Formatting
- **Royal Purple Two-Deck Collapsible Ribbon (`Ctrl+F1`)**: Top deck for file, tab, view, export, navigator, theme, and dictation controls; bottom deck for typography, font properties, spacing, alignment, headings, blocks, cross-references, math, and media.
- **Selection-Based Font Properties & Floating Mini-Toolbar**:
  - Highlight any text span to pop up the non-intrusive **Floating Mini-Toolbar** (`Bold`, `Italic`, `Underline`, `Highlight`, `Code`, `Link`, `A-`/`A+` font size stepping, `Font Color`, and `Ask AI`).
  - Apply **Font Color**, **Font Family**, **Font Size** (`6–72 pt`), **Underline (`Ctrl+U`)**, **Strikethrough (`Ctrl+Shift+X`)**, **Highlight (`Ctrl+Shift+H`)**, or **Clear Font Style** across the Ribbon, Format menu, and right-click context menu.
  - Reliable cursor-only toggle handling avoids straggling delimiters (`**`, `*`, `==`, `` ` ``, `~~`), with boundary-safe `Backspace`/`Delete` around elided markers and single-step atomic Undo/Redo.
- **Paragraph Spacing & Alignment**: Set document line spacing (`1.0`, `1.5`, `2.0`) and paragraph alignment (`Left`, `Center`, `Right`, or `None (Default)`) on selected text or the active paragraph.
- **Word-Style Page Layout & Navigation**:
  - **PageView**: Letter, A4, Narrow, or Wide sheet centered on a configurable desk with selectable margins, `25%–400%` zoom (`Ctrl+Plus/Minus`, `Ctrl+MouseWheel`, Fit Width, status-bar slider), Word Wrap (`Ctrl+Shift+W`), Typewriter Scrolling (`Ctrl+Shift+T`), and distraction-free Full Screen (`F11`).
  - **Document Navigator & Outline (`Ctrl+Shift+L`)**: Browse headings, anchors/references with occurrence badges, and tables (`⊞ Table`), plus **Document Snapshots (`Ctrl+Alt+S` / `Ctrl+Shift+P`)** and **Clean & Reflow (`Ctrl+Shift+R`)**.
- **Cyclic Cross-References, Footnotes & Media**:
  - Autocomplete and unify `[@label]` / `[@anchor: label]` references and `[^id]` footnotes; `Ctrl+Click` / `Ctrl+Shift+Click` cycles occurrences forward/backward and `Alt+Left` returns to prior jump locations.
  - **Automatic Document Image Extraction (`document_image_extractor.py`)**: Extracts and links images when importing PDF, DOCX, PPTX, and EPUB files with MarkItDown, alongside `InsertImageDialog` and `ImageManagerDialog`.
- **Theme, Blockquote & Table Style Customization**:
  - **Theme Settings (`ThemeDialog`)**: 8 built-in presets (*Royal Purple (Word)*, *Royal Nocturne*, *Light*, *Paper (Word)*, *Dark*, *Sepia*, *Solarized Light*, *Solarized Dark*) with system font detection and a live Document Template Preview.
  - **Table Style Customization (`TableStyleDialog`)**: Accessible from **Theme** or **Format → Customize Table Style...** with 8 table presets, independent colors (Header Bg/Text, Row/Alt Row Bg, Cell Text, Border Color), border modes (`grid`, `horizontal`, `minimal`), striped row toggle, bold header toggle, vertical cell padding, and live preview.
  - **Blockquote Style Customization (`BlockquoteStyleDialog`)**: Customize blockquote font, size, style, colors, padding, vertical spacing, and justification.

### Interactive Embedded Markdown Tables
- Live visual grid (`VisualTableWidget`) embedded directly inside Visual mode, synchronized atomically back to canonical GFM Markdown.
- `+ Row` appends clean data rows while preserving column alignments (`:---`, `:---:`, `---:`), alongside `+ Col`, `⇄ Align`, `📋 TSV`, and right-click row/column context actions.
- **Full Keyboard Navigation**:
  - `Tab` / `Shift+Tab`: Move next/previous cell (automatically appends a new row at the last cell on `Tab`).
  - `Enter` / `Shift+Enter`: Commit edit and move down/up in the same column.
  - `Ctrl+Left` / `Ctrl+Right` / `Ctrl+Up` / `Ctrl+Down`: Commit edit and move one cell in the chosen direction.
- **Scroll Passthrough**: Mouse-wheel scroll (`<MouseWheel>`) and `Ctrl+MouseWheel` zoom pass smoothly through embedded table cells and figures to the parent document.

### Integrated Spreadsheet Processor
- **Excel-Style Virtualized Grid**: High-FPS `tk.Canvas` rendering with `O(1)` prefix-sum caching, Row 1 header freeze (`❄ Freeze`), resizable/auto-fit columns and rows, multi-column table sorting (`Sort A→Z` / `Sort Z→A`), `Ctrl+Z` / `Ctrl+Y` undo/redo, fill-handle autofill (with double-click autofill down adjacent columns), and expandable formula bar with live evaluation preview (`fx`).
- **Cell Visual Formatting**: Format cell fill colors, font colors, bold, italic, number/percent display formats, and cell/sheet text wrap (`Ctrl+Shift+W`) with dynamic row heights and horizontal overflow across empty neighbor cells.
- **Formula Engine**:
  - Supports sheet-wide **whole-column (`$E:$E`)** and **whole-row (`1:1`)** references alongside standard bounded cell ranges (`A1:D10`).
  - Math & Rounding: `SUM`, `AVERAGE`, `ROUND`, `ROUNDUP`, `ROUNDDOWN`, `MIN`, `MAX`, `ABS`, `SQRT`, `POWER`, `MOD`, `INT`, `PRODUCT`.
  - Statistics & Visualization: `MEDIAN`, `STDEV`, `VAR`, `SPARKLINE`.
  - Logic & Conditional: `IF`, `IFERROR`, `COUNT`, `COUNTA`, `SUMIF`, `COUNTIF`, `AVERAGEIF`, `AND`, `OR`, `NOT`.
  - Lookup & Text: `VLOOKUP`, `INDEX`, `MATCH`, `CONCAT`, `LEN`, `UPPER`, `LOWER`, `TRIM`, `LEFT`, `RIGHT`.
  - Built-in error checking (`#CIRC!`, `#DIV/0!`, `#NAME?`, `#REF!`, `#VALUE!`, `#N/A`, `#NUM!`) and formula AST/range safety limits.
- **Clipboard & File Formats**: Direct load/save for `.csv`, `.tsv`, `.xlsx` (`openpyxl`), and `.mdsheet`; context-menu copy as TSV, Markdown table, LaTeX `tabular`, or CSV.

### Local Background TTS Dictation (`tts_manager.py`)
- Powered by the `int8` variant of **`Reverb-TTS`** with streaming producer/consumer playback and GPU/CPU device support.
- Access from the toolbar **Dictation** menu or **Read** button (`Ctrl+Alt+D`):
  - **Read Selection / Document**, **Read Last AI Reply**, and **Stop Dictation**.
  - **Dictation Properties...**: Select preset speakers and languages, preview audio, or record/import reference audio to clone and persist custom voices across sessions (`~/.markitdown_tts/voices`).

### AI Assistant, Harnesses & Diff Review
- **Five Backends (`ai_chat.py`)**:
  - **Antigravity CLI (`agy`)**: Local authenticated agent CLI runner with interactive sign-in launcher and cancellable execution.
  - **Ollama**: Local offline models queried from your configured endpoint (default `http://localhost:11434`).
  - **Google Gemini**, **Anthropic Claude**, and **DeepSeek**: Cloud APIs with configurable Temperature, Top-p, Top-k, Max Tokens, and image attachment support.
- **Contextual Editing & Diff Workflows**:
  - **Right-Click Context Menu**: *Transfer to Chat*, *Proofread*, *Rewrite*, *Spellcheck*, *Summarize*, *Expand*, or custom `{selection}` prompts.
  - **Reusable Instruction Harnesses**: Save, switch, and manage prompt harnesses inside AI Settings.
  - **Interactive Diff Review (`diff_panel.py`)**: Side-by-side original vs. AI suggestion comparison with **Accept**, editable Lock/Unlock refinement, **Reject**, and stale-edit detection; plus coordinate-level checkboxes in **Spreadsheet Diff Dialog** for proposed `spreadsheet` blocks.
  - **Chat Message Controls**: Inline edit, delete, and redo regeneration on conversation turns with per-document persistent history.

### Supported Input & Export Formats
- **Convert with MarkItDown (`Ctrl+Shift+O`)**: PDF, Word (`.docx`), PowerPoint (`.pptx`), Excel (`.xlsx`/`.xls`), HTML, Images (EXIF/OCR), Audio, CSV/JSON/XML, ZIP, EPUB, Outlook (`.msg`), and YouTube URLs.
- **Export As (`File → Export As`)**:
  - **PDF (`Ctrl+Shift+E`)** via `weasyprint`
  - **HTML** via `markdown` (`md_in_html` + standalone CSS)
  - **DOCX** via `python-docx` (headings, lists, blockquotes, fenced code, inline styles, custom tables, bookmarks/hyperlinks, embedded images)

---

## Installation

### Prerequisites
- **Python 3.10+** with **Tkinter** (included with standard Windows/macOS Python installers; on Debian/Ubuntu install `python3-tk`).

### Quick Start (Full Installation)
```bash
git clone https://github.com/micahamd/markdown-notepad.git
cd markdown-notepad
pip install -r requirements.txt
python markdown_notepad.py
```

### Minimal or Selective Installation
The application degrades gracefully if optional packages are absent:

```bash
# Core Markdown editor + MarkItDown document conversion
pip install "markitdown[all]" Pillow

# Optional: Visual HTML rendering & PDF / DOCX export
pip install markdown tkhtmlview weasyprint python-docx

# Optional: Excel (.xlsx) spreadsheet import/export
pip install openpyxl

# Optional: Cloud / local AI SDKs
pip install anthropic google-genai ollama openai

# Optional: Offline background TTS dictation & voice cloning
pip install reverb-tts soundfile sounddevice
```

---

## Usage

```bash
python markdown_notepad.py
```

1. **Documents**: Start typing directly in **Visual Mode**, press `Ctrl+E` to inspect/edit **Source Mode**, or use `Ctrl+Shift+O` to import PDF/Word/PowerPoint/HTML/EPUB documents via MarkItDown.
2. **Spreadsheets**: Press `Ctrl+Shift+N` for a blank spreadsheet (`Ctrl+Alt+O` to open `.csv`/`.tsv`/`.xlsx`/`.mdsheet` directly as a spreadsheet).
3. **AI & Dictation**: Press `Ctrl+Shift+A` to open the AI Sidebar (configure via **Config**), right-click selected text for AI transforms, or press `Ctrl+Alt+D` to start/stop background dictation.

### Keyboard Shortcuts

| Category | Action | Shortcut |
|----------|--------|----------|
| **Files & Tabs** | New Document Tab | `Ctrl+N` |
| | New Spreadsheet Tab | `Ctrl+Shift+N` |
| | Open File | `Ctrl+O` |
| | Open as Spreadsheet | `Ctrl+Alt+O` |
| | Open with MarkItDown | `Ctrl+Shift+O` |
| | Save / Save As | `Ctrl+S` / `Ctrl+Shift+S` |
| | Quick Snapshot / Snapshots Dialog | `Ctrl+Alt+S` / `Ctrl+Shift+P` |
| | Export to PDF | `Ctrl+Shift+E` |
| | Close Tab | `Ctrl+W` |
| | Next / Previous Tab | `Ctrl+Tab` / `Ctrl+Shift+Tab` |
| **Editing & View** | Undo / Redo | `Ctrl+Z` / `Ctrl+Y` |
| | Find / Find Next / Replace | `Ctrl+F` / `F3` / `Ctrl+H` |
| | Go to Line | `Ctrl+G` |
| | Clean & Reflow Document | `Ctrl+Shift+R` |
| | Source / Visual / Toggle Mode | `Ctrl+1` / `Ctrl+2` / `Ctrl+E` |
| | Collapse / Expand Ribbon Deck | `Ctrl+F1` |
| | Document Navigator & Outline | `Ctrl+Shift+L` |
| | Word Wrap / Typewriter Scrolling | `Ctrl+Shift+W` / `Ctrl+Shift+T` |
| | Full Screen / Focus Mode | `F11` (`Esc` to exit) |
| | Zoom In / Out / Reset / Fit Width | `Ctrl++` / `Ctrl+-` / `Ctrl+0` / `Ctrl+Shift+0` |
| **Formatting** | Bold / Italic / Underline / Strike | `Ctrl+B` / `Ctrl+I` / `Ctrl+U` / `Ctrl+Shift+X` |
| | Highlight / Inline Code | `Ctrl+Shift+H` / ``Ctrl+` `` |
| | Blockquote / Code Block / Link | `Ctrl+Shift+Q` / `Ctrl+Shift+K` / `Ctrl+K` |
| | Insert Table / Image / Footnote | `Alt+T` / `Ctrl+Shift+I` / `Ctrl+Shift+F` |
| | Anchor `[@...]` / Math & Symbols | `Ctrl+Shift+M` / `Ctrl+Shift+Y` |
| | Cycle Anchor / Return to Origin | `Ctrl+Click` (`Shift` reverse) / `Alt+Left` |
| **Embedded Tables** | Next / Previous Cell (`Tab` adds row at end) | `Tab` / `Shift+Tab` |
| | Next Row Down / Up (same column) | `Enter` / `Shift+Enter` |
| | Directional Cell Jump | `Ctrl+Arrow Keys` |
| **AI & Dictation** | Toggle AI Sidebar | `Ctrl+Shift+A` |
| | Read Selection or Document / Stop | `Ctrl+Alt+D` |

---

## Architecture

```
markdown-notepad/
├── markdown_notepad.py          # Main GUI application (~13,700 lines):
│                                #   MarkdownNotepad, DocumentTab, PageView,
│                                #   MarkdownVisualWidget, VisualTableWidget,
│                                #   VisualFigureWidget, FloatingMiniToolbar,
│                                #   OutlinePanel, DockedFindReplaceBar,
│                                #   ThemeDialog, TableStyleDialog, BlockquoteStyleDialog
├── spreadsheet_tab.py           # Integrated Excel-style spreadsheet module (~6,900 lines):
│                                #   FormulaEngine, SpreadsheetGrid, SpreadsheetTab,
│                                #   SpreadsheetPatchParser, SpreadsheetDiffDialog
├── ai_chat.py                   # Multi-provider AI sidebar & harnesses (~6,420 lines):
│                                #   AISettings, ChatHistoryManager, LLMClient,
│                                #   AnthropicClient, GeminiClient, OllamaClient,
│                                #   DeepSeekClient, AntigravityClient, ChatSidebar
├── tts_manager.py               # Local Reverb-TTS manager & Dictation Properties UI (~2,270 lines)
├── export_manager.py            # Theme-aware HTML, WeasyPrint PDF, & DOCX export (~1,840 lines)
├── diff_panel.py                # Side-by-side AI suggestion diff review panel (~740 lines)
├── text_cleaner.py              # Markdown text cleanup & paragraph reflow engine (~510 lines)
├── document_image_extractor.py  # PDF / DOCX / PPTX / EPUB asset extraction (~410 lines)
├── session_manager.py           # Atomic JSON session & snapshot persistence (~350 lines)
├── requirements.txt             # Python dependencies
└── .gitignore                   # Excludes user configs, keys, caches, and logs
```

---

## Configuration Files & Security

User preferences, session caches, custom voices, and credentials are stored outside the repository in your home directory and excluded via `.gitignore`:

| Path | Purpose |
|------|---------|
| `~/.markitdown_notepad_settings.json` | Theme presets, fonts, colors, table/blockquote styles, line spacing, page layout, zoom |
| `~/.markitdown_ai_config.json` | AI provider settings, custom harnesses, context actions, and protected API keys |
| `~/.markitdown_session.json` | Open tabs, active tab index, and window geometry (atomic writes) |
| `~/.markitdown_chat_history/` | Per-document AI chat histories |
| `~/.markitdown_recent_files.json` | Recent files list (max 10) |
| `~/.markitdown_tts/` | Saved TTS speaker/language settings and persistent cloned voices |
| `~/.markitdown_assets/` | Extracted document image cache directories |

### Security Controls Built In
- **API Key Protection at Rest**: Cloud API keys stored in `~/.markitdown_ai_config.json` are encrypted via Windows DPAPI (`CryptProtectData`) on Windows, with no secrets committed to the repository.
- **Safe CLI Execution**: External CLI invocations (such as `agy`) enforce argument/executable validation and `shell=False` execution.
- **Export & Import Sanitization**: Strict URL scheme whitelists (`http`, `https`, `mailto`), WeasyPrint fetcher SSRF guards, inline HTML/CSS whitelist sanitization, archive extraction size caps, and formula AST/range limits against resource exhaustion.

---

## Troubleshooting

- **AI features disabled**: Install the provider SDKs (`pip install anthropic google-genai ollama openai`) or configure **Antigravity CLI (`agy`)** in AI Settings.
- **Export menu disabled**: Install export dependencies (`pip install weasyprint python-docx markdown`). On Windows/macOS/Linux, check WeasyPrint's GTK/Pango runtime requirements if PDF rendering fails.
- **Dictation unavailable**: Install `reverb-tts`, `soundfile`, and `sounddevice`; the `int8` `Reverb-TTS` weights download on first load.
- **Ollama shows no models**: Ensure `ollama serve` is running and at least one model is pulled (e.g., `ollama pull llama3.2`), then click **Test** in AI Settings.

## License

MIT License. This project uses Microsoft's MIT-licensed [MarkItDown](https://github.com/microsoft/markitdown) package.
