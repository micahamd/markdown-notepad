# MarkItDown Notepad

A feature-rich markdown notepad application built with Python and Tkinter, powered by Microsoft's [MarkItDown](https://github.com/microsoft/markitdown) package. Features an integrated AI assistant with context-aware editing, a smart right-click menu for AI-powered text transformations, side-by-side diff review, and professional document export to PDF, HTML, and DOCX formats.

## Features

### Core Features
- **Dual Tab Architecture**: Work seamlessly across **Document Tabs** and **Spreadsheet Tabs** in a unified tabbed workspace
- **Source Mode**: Edit raw markdown with syntax-aware formatting
- **Visual Mode**: Rendered markdown preview with styled text display (editable)
- **Integrated Spreadsheet Processor**: Excel-style 2D virtualized grid with formula engine, cell coordinates, and clipboard compatibility
- **MarkItDown Integration**: Open virtually any file type and convert to markdown
- **Tabbed Interface**: Work with multiple documents and sheets simultaneously
- **Session Persistence**: Restore open document and spreadsheet tabs, scroll positions, and window geometries on restart
- **Professional Export**: Export documents to PDF, HTML, and DOCX with theme-aware styling, or export spreadsheets to CSV, TSV, XLSX, and MDSHEET

### AI Assistant with Context Menu Integration
- **Collapsible AI Sidebar**: Integrated chat with AI models
- **Multi-Provider Support**: Anthropic Claude, Google Gemini, Ollama (local), and DeepSeek
- **Smart Context Menu**: Right-click selected text for instant AI-powered transformations
  - **Transfer to Chat**: Send selection to sidebar with position tracking for seamless replacement
  - **Rewrite**: Improve clarity and flow while preserving meaning
  - **Proofread**: Grammar and style corrections
  - **Spellcheck**: Fix spelling errors
  - **Summarize**: Create concise summaries
  - **Expand**: Add detail while maintaining tone
  - **Custom Actions**: Add your own prompts with `{selection}` placeholders
- **Side-by-Side Diff Panel**: Review AI suggestions before applying
  - Original text on left, AI suggestion on right
  - Accept, Edit (make tweaks), or Reject
  - Stale text detection warns if document changed since suggestion was generated
- **Document Context**: Optionally include current document content in AI queries
- **Chat History**: Per-document conversation history that persists across sessions
- **Advanced Parameters**: Temperature, Top-p, Top-k configuration with tooltips
- **Image Support**: Include document images in AI context

### Supported File Types (via MarkItDown & Spreadsheet Engine)
- 📄 **PDF** files
- 📝 **Word** documents (.docx)
- 📊 **PowerPoint** presentations (.pptx)
- 📈 **Excel** spreadsheets (.xlsx, .xls) — open as Markdown notes or interactive Spreadsheets
- 📊 **Spreadsheet formats** (.csv, .tsv, .xlsx, .mdsheet)
- 🌐 **HTML** files
- 🖼️ **Images** (EXIF metadata and OCR)
- 🎵 **Audio** files (metadata and transcription)
- 📋 **Text formats** (CSV, JSON, XML)
- 📦 **ZIP** files (iterates contents)
- 🎬 **YouTube** URLs
- 📚 **EPub** books
- 📚 **Outlook** messages (.msg)

### Integrated Spreadsheet Processor
- **Excel-Style 2D Grid**: Smooth, virtualized canvas grid rendering columns (A, B, C...) and rows (1, 2, 3...) with Royal Purple selection styling, interactive column and row divider resizing, and double-click auto-fit
- **Toggleable Word Wrap & Dynamic Auto-Row-Height**:
  - `↵ Wrap` toolbar toggle, context menu action, and `Ctrl+Shift+W` shortcut for cell or sheet-level text wrapping
  - Automatically calculates and expands row heights to accommodate multi-line wrapped text with zero vertical overlap
  - Expandable multi-line cell editor supporting `Alt+Enter` newline insertion
- **Smart Single-Line Text Layout**:
  - Seamless Excel-style horizontal overflow across contiguous empty right-neighbor cells
  - Crisp single-line ellipsis truncation (`…`) upon encountering populated neighbor cells, preventing row collisions
- **High-Speed Keyboard Navigation & Section Jumps**:
  - Full `Shift+Arrow` multi-cell range extension across rows and columns
  - `Ctrl+Arrow` rapid jumps across populated data blocks and empty section gaps
  - `Ctrl+Shift+Arrow` block-boundary jump selections
  - `Shift+Space` to select entire rows; `Ctrl+Space` to select entire columns
  - `Home` (Column A), `Ctrl+Home` (Cell A1), `Ctrl+End` (bottom-right populated cell), `PageUp`, `PageDown`, `<Shift-Tab>`, and `<Shift-Return>`
  - `<Up>` / `<Down>` commit-and-step transitions directly from the inline editor
- **Sub-Millisecond Rendering & Coordinate Caching**:
  - `O(1)` prefix-sum coordinate caching and `O(log N)` binary-search coordinate mapping
  - Dedicated selection-overlay fast path (`< 0.2ms`) delivering instant 60+ FPS navigation without canvas teardown
- **Freeze Top Row (`❄ Freeze`)**:
  - One-click toolbar toggle to pin Row 1 headers permanently beneath column letters while scrolling through large datasets
- **One-Click Table Sorting (`Sort A→Z` / `Sort Z→A`)**:
  - Multi-column table sorting that preserves table headers, moves entire row styles and values intact, and supports single-step `Ctrl+Z` undo
- **Live Formula Result Preview & Cell Hover Inspector**:
  - Real-time result badge (`= 1,450.00`) next to formula bar evaluating formulas as you type before committing
  - Floating cell hover inspector displaying full text for truncated cells or detailed formula breakdowns (`fx: =SUM(...) => $1,450.00`)
- **Expanded Formula Engine**:
  - Math: `=SUM(range)`, `=AVERAGE(range)`, `=ROUND(val, digits)`, `=ROUNDUP()`, `=ROUNDDOWN()`, `=MIN()`, `=MAX()`, `=ABS()`, `=SQRT()`, `=POWER()`, `=MOD()`, `=INT()`, `=PRODUCT()`
  - Statistics: `=MEDIAN()`, `=STDEV()`, `=VAR()`, `=SPARKLINE(range)` (inline Unicode trendlines ` ▂▄▆█▃`)
  - Logic & Counting: `=IF(cond, true_val, false_val)`, `=IFERROR(val, fallback)`, `=COUNT(range)`, `=COUNTA(range)`, `=SUMIF(range, criteria, sum_range)`, `=COUNTIF(range, criteria)`, `=AVERAGEIF()`, `=AND()`, `=OR()`, `=NOT()`
  - Lookup & Reference: `=VLOOKUP(val, range, col_idx, [exact])`, `=INDEX(range, row, [col])`, `=MATCH(val, range)`
  - Text: `=CONCAT()`, `=LEN()`, `=UPPER()`, `=LOWER()`, `=TRIM()`, `=LEFT()`, `=RIGHT()`
  - Error detection: Circular reference protection (`#CIRC!`), division by zero (`#DIV/0!`), invalid names (`#NAME?`), and reference errors (`#REF!`)
- **Rich Clipboard Interoperability**:
  - Smart Paste (`Ctrl+V`) automatically detects Markdown pipe tables, CSV, or TSV and populates the grid
  - Rich Copy options via context menu: Copy as TSV (standard Excel), Copy as Markdown Table, Copy as LaTeX `tabular`, and Copy as CSV
- **Bidirectional AI Chat Integration**:
  - Right-click selected cells to **Transfer Selection to AI Chat** or **Ask AI About Selection**
  - Toggle **Include Document** to furnish the AI with full structural grid geometry and populated data
  - AI can propose direct cell updates and formulas using fenced `spreadsheet` blocks
  - Interactive **Spreadsheet Diff Dialog** previews proposed changes with coordinate-level checkboxes before application
- **Native File Formats**: Direct loading and saving of `.csv`, `.tsv`, `.xlsx` (via `openpyxl`), and state-preserving `.mdsheet` files

### Editor Features
- **Multi-Tab Editing**: Open multiple documents with Ctrl+Tab navigation
- **Find & Replace**: Full search and replace functionality (Ctrl+H)
- **Go to Line**: Jump to specific line numbers (Ctrl+G)
- **Page Layout**: Word-style fixed-width page (Letter, A4, Narrow, Wide) centred on a coloured desk in both Source and Visual modes; configurable margins, or switch it off for full-width editing
- **Zoom**: 25-400% (Ctrl+Plus/Minus, Ctrl+MouseWheel, status-bar slider, Fit Width); scales the page, text, tables and images together
- **Full Screen**: F11 hides all chrome and shows the page on a dark desk (Esc or F11 to exit)
- **Word Wrap Toggle**: Configurable text wrapping
- **Theme Customization**: Customize colors and fonts for source/visual modes
- **Image Management**: Insert, resize, and manage images in documents
- **Table Rendering**: Visual mode renders markdown tables with proper formatting
- **Large File Mode**: Optimized rendering for large documents

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/note-processor.git
   cd note-processor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install core packages individually:
   ```bash
   pip install 'markitdown[all]' Pillow
   ```

3. **Optional - For AI chat functionality:**
   ```bash
   pip install anthropic google-genai
   ```

## Usage

```bash
python markdown_notepad.py
```

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New Document Tab | Ctrl+N |
| New Spreadsheet Tab | Ctrl+Shift+N |
| Open File | Ctrl+O |
| Open as Spreadsheet | Ctrl+Alt+O |
| Open with MarkItDown | Ctrl+Shift+O |
| Save | Ctrl+S |
| Save As | Ctrl+Shift+S |
| Export to PDF | Ctrl+Shift+E |
| Close Tab | Ctrl+W |
| Next Tab | Ctrl+Tab |
| Previous Tab | Ctrl+Shift+Tab |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y |
| Find | Ctrl+F |
| Find Next | F3 |
| Find & Replace | Ctrl+H |
| Go to Line | Ctrl+G |
| Select All | Ctrl+A |
| Bold | Ctrl+B |
| Italic | Ctrl+I |
| Insert Image | Ctrl+Shift+I |
| Source Mode | Ctrl+1 |
| Visual Mode | Ctrl+2 |
| Toggle Mode | Ctrl+E |
| Toggle Word Wrap | Ctrl+Shift+W |
| Toggle AI Sidebar | Ctrl+Shift+A |
| Full Screen / Focus Mode | F11 (Esc to exit) |
| Zoom In | Ctrl++ |
| Zoom Out | Ctrl+- |
| Reset Zoom | Ctrl+0 |
| Zoom to Fit Width | Ctrl+Shift+0 |

## How AI Context Menu Works

The right-click context menu transforms how you interact with AI. Instead of copying text to the chat, getting a response, and manually replacing it, the context menu creates a seamless edit-review-apply workflow.

### Basic Workflow

1. **Select text** in your document that you want to improve, proofread, or transform
2. **Right-click** the selection to open the context menu
3. **Choose an AI action** (e.g., "Proofread", "Rewrite", "Summarize")
4. **Review in diff panel**: 
   - Original text appears on the left
   - AI suggestion streams into the right pane
   - Both panes show the text in a monospaced font for easy comparison
5. **Accept, Edit, or Reject**:
   - **Accept**: Replaces the selected text in your document with the AI suggestion
   - **Edit**: Makes the suggestion editable so you can tweak it before accepting
   - **Reject**: Closes the panel and keeps your original text

### Transfer to Chat

The "Transfer to Chat" action is special—it bridges inline editing with conversational AI:

1. **Select text** and choose "Transfer to Chat" from the context menu
2. The **AI sidebar opens** (if not already visible)
3. Your selection is **pinned** with a blue indicator showing:
   - Line range (e.g., "lines 5–12")
   - Text preview (first 60 characters)
   - Your selection's position is stored internally
4. The chat input is **pre-filled** with your selection as a blockquote
5. **Type your instruction** below the quoted text (e.g., "make this more formal", "add technical details")
6. **AI responds** in the chat
7. An **"Apply to Document"** button appears below the AI's response
8. Clicking it opens the **diff panel** for review
9. **Accept** applies the change to the exact position where you originally selected text

This workflow is perfect for iterative refinement—you can have a conversation with the AI about how to improve a specific section, then apply the final result precisely where it belongs.

### Customizing Context Menu Actions

1. Toggle the AI sidebar (Ctrl+Shift+A)
2. Click **Config** in the sidebar header
3. Scroll to **"Context Menu AI Actions"**
4. Add/remove/edit actions:
   - **Name**: The label shown in the right-click menu
   - **Prompt**: The instruction sent to the AI (use `{selection}` as a placeholder for the selected text)
   - **Enabled**: Checkbox to show/hide the action
5. Click **Save Changes**

**Example custom action:**
- Name: `Technical Simplify`
- Prompt: `Rewrite the following technical text for a general audience. Use simple language and analogies where helpful. Return ONLY the rewritten text:\n\n{selection}`

All actions support any AI provider you've configured (Claude, Gemini, Ollama, DeepSeek).

## AI Assistant Setup & Provider Generalizability

The application provides a universal multi-provider AI backend supporting both remote cloud models and zero-cost local agents. Settings are persisted in `~/.markitdown_ai_config.json`, making configurations portable across different machines.

1. Click the **AI Chat** button in the toolbar or press **Ctrl+Shift+A**
2. Click the **Config** button in the sidebar header
3. Configure your preferred provider:

### Google Antigravity CLI (`agy`)
Google Antigravity CLI enables authenticated agent workflows directly on your workstation without requiring third-party API keys.

- **How it works**: The notepad launches the `agy` command-line interface in headless streaming mode using your local authenticated Google credentials.
- **Locating your `agy` executable path**:
  - **Windows (npm global)**:
    ```
    %APPDATA%\npm\agy.cmd
    ```
    *(e.g., `C:\Users\<username>\AppData\Roaming\npm\agy.cmd`)*
  - **Windows (standard binary install)**:
    ```
    %LOCALAPPDATA%\Programs\antigravity\bin\agy.exe
    ```
    or `%LOCALAPPDATA%\agy\bin\agy.exe`
  - **macOS / Linux**:
    ```bash
    /usr/local/bin/agy
    # or
    ~/.local/bin/agy
    # or
    ~/.npm-global/bin/agy
    ```
- **Configuration Steps**:
  1. In the AI Settings Dialog, choose **Antigravity CLI (agy)** from the Provider dropdown.
  2. Paste your executable path into the **CLI Path** field (or click **Browse** to select the file).
  3. Click **Test** to verify connection.
  4. If your Google credentials have expired or need initial authorization, click **Interactive Sign-in Console**; a console window will launch allowing you to complete authentication directly.

### Ollama (Local & Offline)
Run high-performance open-weight models locally with zero cloud dependencies or costs.

- **Installation**: Download and install Ollama from [ollama.ai](https://ollama.ai).
- **Start the Service**:
  ```bash
  ollama serve
  ```
- **Pull Recommended Models**:
  ```bash
  ollama pull llama3.2
  ollama pull qwen2.5:7b
  ollama pull deepseek-r1:8b
  ```
- **Configuration Steps**:
  1. Set Provider to **Ollama**.
  2. Enter the server endpoint URL (default is `http://localhost:11434`).
  3. Click **Test** to automatically query Ollama and populate all installed local models into the model selection dropdown.

### Cloud API Providers
- **Google Gemini API**:
  - Obtain an API key from [aistudio.google.com](https://aistudio.google.com).
  - Enter the key into the Gemini field, select provider **gemini**, and pick from `gemini-2.5-flash`, `gemini-2.5-pro`, or `gemini-1.5-pro`.
- **Anthropic Claude**:
  - Obtain an API key from [console.anthropic.com](https://console.anthropic.com).
  - Enter the key into the Anthropic field, select provider **anthropic**, and pick from Claude 3.5 Sonnet or Claude 3.5 Haiku.
- **DeepSeek**:
  - Obtain an API key from [deepseek.com](https://www.deepseek.com).
  - Enter the key into the DeepSeek field and select provider **deepseek**.

4. Select your preferred model from the dropdown
5. Optionally customize:
   - **System Prompt**: The AI's role/instructions
   - **Temperature**: Creativity vs. consistency (0.0–2.0)
   - **Max Tokens**: Maximum response length
   - **Context Menu Actions**: Customize right-click AI actions
6. Click **Save Changes**

### AI Parameters Explained
- **Temperature** (0.0-2.0): Controls randomness. Lower (0.3-0.5) = focused and factual, Higher (0.7-1.0) = creative and varied
- **Top-p** (0.0-1.0): Nucleus sampling - limits token choices to cumulative probability threshold. Most users should leave at 1.0
- **Top-k** (0-100): Only consider top-k most likely tokens. 0 = disabled (recommended for most use cases)
- **Max Tokens**: Maximum length of AI response (1024 = short, 4096 = long answers)

## Document Export

Export your markdown documents to professional formats with theme-aware styling.

### Export to PDF

**File → Export As → PDF** (Ctrl+Shift+E)

- Converts markdown to HTML, then renders as PDF using WeasyPrint
- Preserves your theme settings (fonts, colors, sizes)
- Resolves relative image paths and embeds images
- Supports tables, code blocks, headers, lists, blockquotes
- Configurable page size and margins

### Export to HTML

**File → Export As → HTML**

- Standalone HTML file with embedded CSS
- Theme-aware styling matches your visual mode appearance
- Images embedded or linked (depending on source)
- Ready to host on a website or share

### Export to DOCX

**File → Export As → DOCX**

- Microsoft Word format (.docx) via python-docx
- Heading styles (H1-H6) properly formatted
- Code blocks with monospace font and shading
- Tables with borders and alternating row colors
- Inline formatting: **bold**, *italic*, `code`
- Embedded images (both file paths and base64 data URIs)
- Blockquotes with indentation and italic styling
- Lists (ordered and unordered)

**Note**: Export features require additional packages. Install with:
```bash
pip install weasyprint python-docx markdown
```

## How It Works

1. **Direct Markdown Editing**: Open `.md` files directly for editing
2. **File Conversion**: Use "Open with MarkItDown" to convert any supported file to markdown
3. **Mode Switching**: Toggle between source (raw markdown) and visual (rendered) views
4. **AI Assistance**: Ask questions about your document or get writing help

## Architecture

```
note-processor/
├── markdown_notepad.py      # Main application (~9500 lines)
│   ├── ImageExtractor       # Extract images from Office documents
│   ├── ImageHandler         # Manage image embedding and display
│   ├── PageView             # Word-style page (sheet) on a desk; owns zoom/page width/margins
│   ├── MarkdownVisualWidget # Visual markdown rendering
│   ├── DocumentTab          # Individual markdown document tab
│   ├── MarkdownNotepad      # Main application window with Deck 1 & Deck 2 ribbons
│   ├── ThemeDialog          # Theme customization (auto-fitting & resizable)
│   ├── InsertImageDialog    # Image insertion wizard
│   ├── ImageManagerDialog   # Image management interface
│   ├── FindReplaceDialog    # Search and replace with regex
│   └── GoToLineDialog       # Jump to line number
│
├── spreadsheet_tab.py       # Integrated Spreadsheet Processor (~2100 lines)
│   ├── FormulaEngine        # AST formula parser, recursive evaluator, circular ref detection
│   ├── SpreadsheetGrid      # Virtualized Tkinter Canvas grid with floating headers and Royal Purple styling
│   ├── SpreadsheetTab       # Formula bar (fx), coordinate tracker, markdown table exporter, undo/redo
│   ├── SpreadsheetPatchParser # Regex extractor for AI-generated cell updates
│   ├── SpreadsheetDiffDialog # Coordinate-by-coordinate diff inspector with checkboxes
│   └── open/save_spreadsheet_file # CSV, TSV, XLSX (openpyxl), and MDSHEET I/O
│
├── ai_chat.py               # AI chat module (~3600 lines)
│   ├── AISettings           # Configuration persistence
│   ├── ChatHistoryManager   # Per-document chat history
│   ├── ChatMessage          # Message data structure
│   ├── ImageProcessor       # Image preparation for AI
│   ├── LLMClient (ABC)      # Abstract base for AI providers
│   ├── AnthropicClient      # Claude API integration
│   ├── GeminiClient         # Google Gemini integration
│   ├── OllamaClient         # Ollama local LLM integration
│   ├── DeepSeekClient       # DeepSeek API integration
│   ├── AntigravityClient    # Google Antigravity CLI (agy) integration
│   ├── AISettingsDialog     # Settings UI with dynamic button widths and CLI path config
│   └── ChatSidebar          # Chat interface widget
│
├── diff_panel.py            # Diff review panel (~300 lines)
│   └── DiffPanel            # Side-by-side diff with Accept/Edit/Reject
│
├── export_manager.py        # Document export (~700 lines)
│   ├── export_to_pdf()      # Markdown → HTML → PDF with WeasyPrint
│   ├── export_to_html()     # Markdown → Standalone HTML
│   ├── export_to_docx()     # Markdown → Word document
│   └── Helper functions     # CSS generation, image handling, formatting
│
├── session_manager.py       # Session persistence (~300 lines)
│   ├── TabState             # Document and spreadsheet tab state (tab_type aware)
│   ├── SessionState         # Full session state
│   └── SessionManager       # Save/restore functionality
│
├── requirements.txt         # Python dependencies
└── .gitignore              # Git exclusions (includes API keys)
```

### Key Design Patterns

- **Callback Architecture**: Main app and sidebar communicate via callbacks, maintaining loose coupling
- **Provider Abstraction**: `LLMClient` ABC allows easy addition of new AI providers
- **Position Tracking**: Context menu actions store text indices for precise replacement
- **Streaming UI**: AI responses stream character-by-character for responsive feel
- **Queue-Based Threading**: Background LLM calls push chunks to a queue, polled by main thread
- **Modular Export**: Each export format isolated in separate functions with shared utilities

## Configuration Files

The application stores configuration in your home directory:

| File | Purpose |
|------|---------|
| `~/.markitdown_notepad_settings.json` | Theme preferences (fonts, colors, sizes), desk colour, page size/margins/layout, zoom |
| `~/.markitdown_ai_config.json` | AI settings, API keys, and **context menu actions** |
| `~/.markitdown_session.json` | Session state (open tabs, window geometry) |
| `~/.markitdown_chat_history/` | Per-document chat histories (keyed by file path hash) |
| `~/.markitdown_recent_files.json` | Recently opened files list (max 10) |

**Security Note**: API keys are stored locally in `~/.markitdown_ai_config.json` and excluded from git via `.gitignore`. Context menu action configurations are also stored in this file, making them portable across machines (just copy the config file).

## Dependencies

### Required
- **Python 3.10+**
- **tkinter** (usually included with Python)
- **markitdown[all]** - Microsoft's document conversion library
- **Pillow** - Image handling

### Optional - AI Features
- **anthropic** - Anthropic Claude AI (GPT-4 class models)
- **google-genai** - Google Gemini AI (multimodal)
- **ollama** - Local LLM support (free, runs on your machine)
- **openai** - OpenAI-compatible APIs (used for DeepSeek)

### Optional - Spreadsheet Features
- **openpyxl** - Microsoft Excel (.xlsx) direct spreadsheet import and export

### Optional - Export Features
- **weasyprint** - PDF export with professional typography
- **python-docx** - Microsoft Word (.docx) export
- **markdown** - Enhanced markdown parsing (used by export and visual modes)

### Optional - UI Enhancements
- **tkhtmlview** - Alternative HTML-based visual rendering

## Installation

### Quick Start (Core Features Only)

```bash
git clone https://github.com/yourusername/note-processor.git
cd note-processor
pip install -r requirements.txt
python markdown_notepad.py
```

### Full Installation (All Features)

To enable all features including AI and export:

```bash
# Clone repository
git clone https://github.com/yourusername/note-processor.git
cd note-processor

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python markdown_notepad.py
```

### Minimal Installation (No AI)

If you only want the markdown editor without AI features:

```bash
pip install 'markitdown[all]' Pillow
python markdown_notepad.py
```

The application will run with AI features gracefully disabled.

### Adding Features Later

You can install optional features at any time:

```bash
# Add AI support
pip install anthropic google-genai ollama openai

# Add export support
pip install weasyprint python-docx markdown

# Add everything
pip install -r requirements.txt
```

## Quick Start Guide

### Basic Document Flow
1. **Launch**: `python markdown_notepad.py`
2. **Open** a markdown file (Ctrl+O) or create a **New Tab** (Ctrl+N)
3. **Edit** in Source mode (raw markdown) or Visual mode (rendered preview)
4. **Save** your work (Ctrl+S)
5. **Export** to PDF/DOCX/HTML when ready to share

### AI-Enhanced Writing Flow
1. **Write** a draft in your document
2. **Select** a paragraph or section you want to improve
3. **Right-click** and choose an AI action:
   - **Proofread** for quick grammar/spelling fixes
   - **Rewrite** to improve clarity and flow
   - **Summarize** to condense information
   - **Transfer to Chat** for conversational refinement
4. **Review** the suggestion in the diff panel (original on left, suggestion on right)
5. **Accept** to apply, **Edit** to tweak, or **Reject** to keep original
6. **Export** your polished document to PDF or DOCX

### Converting Other Documents
1. **File → Open with MarkItDown** (Ctrl+Shift+O)
2. Select any supported file (PDF, DOCX, PPTX, XLSX, etc.)
3. Document is converted to markdown and opened in a new tab
4. Edit, enhance with AI, and export as needed

## Tips & Best Practices

### Context Menu Actions
- Keep prompts concise and specific
- Always include `{selection}` in your custom prompts
- Add "Return ONLY the..." to prompts to avoid AI commentary
- Disable actions you don't use to keep the menu clean
- Export your `~/.markitdown_ai_config.json` to share action sets with others

### Working with AI
- Use "Transfer to Chat" for iterative refinement (back-and-forth conversation)
- Use direct context menu actions (Rewrite, Proofread) for quick one-shot improvements
- Enable "Include Document" checkbox in sidebar to give AI full document context
- Lower temperature (0.3-0.5) for factual/technical content
- Higher temperature (0.7-1.0) for creative/marketing content

### Export Quality
- **PDF**: Best for sharing, printing, or archiving—preserves exact formatting
- **DOCX**: Best when recipient needs to edit—fully editable in Word
- **HTML**: Best for web publishing—can be self-hosted or embedded

### Performance
- Enable "Large File Mode" (View menu) for documents over 500 lines
- Disable session caching if you don't need tab restoration
- Use Ollama with local models for unlimited, free AI access

## Troubleshooting

**AI features not appearing**: Install AI packages: `pip install anthropic google-genai`

**Export menu grayed out**: Install export packages: `pip install weasyprint python-docx markdown`

**WeasyPrint installation fails**: Follow platform-specific instructions at [weasyprint.org](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation)

**Ollama not detecting models**: Ensure Ollama is running and pull a model: `ollama pull llama3.2`

**Images not showing in exports**: Use absolute paths or ensure images are relative to the document location

**API key not working**: Click "Test" in AI Settings to verify connectivity. Check for typos and ensure key has correct permissions.

## License

MIT License - See LICENSE file for details.

This project uses the MIT-licensed MarkItDown package from Microsoft.
