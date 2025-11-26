# MarkItDown Notepad

A feature-rich markdown notepad application built with Python and Tkinter, powered by Microsoft's [MarkItDown](https://github.com/microsoft/markitdown) package. Includes an integrated AI assistant supporting Anthropic Claude and Google Gemini.

## Features

### Core Features
- **Source Mode**: Edit raw markdown with syntax-aware formatting
- **Visual Mode**: Rendered markdown preview with styled text display (editable)
- **MarkItDown Integration**: Open virtually any file type and convert to markdown
- **Tabbed Interface**: Work with multiple documents simultaneously
- **Session Persistence**: Optionally restore your tabs and documents on restart

### AI Assistant
- **Collapsible AI Sidebar**: Integrated chat with AI models
- **Multi-Provider Support**: Anthropic Claude and Google Gemini
- **Document Context**: Optionally include current document content in AI queries
- **Chat History**: Per-document conversation history that persists across sessions
- **Advanced Parameters**: Temperature, Top-p, Top-k configuration with tooltips
- **Image Support**: Include document images in AI context

### Supported File Types (via MarkItDown)
- 📄 **PDF** files
- 📝 **Word** documents (.docx)
- 📊 **PowerPoint** presentations (.pptx)
- 📈 **Excel** spreadsheets (.xlsx, .xls)
- 🌐 **HTML** files
- 🖼️ **Images** (EXIF metadata and OCR)
- 🎵 **Audio** files (metadata and transcription)
- 📋 **Text formats** (CSV, JSON, XML)
- 📦 **ZIP** files (iterates contents)
- 🎬 **YouTube** URLs
- 📚 **EPub** books
- 📧 **Outlook** messages (.msg)

### Editor Features
- **Multi-Tab Editing**: Open multiple documents with Ctrl+Tab navigation
- **Find & Replace**: Full search and replace functionality (Ctrl+H)
- **Go to Line**: Jump to specific line numbers (Ctrl+G)
- **Zoom**: Ctrl+Plus/Minus or Ctrl+MouseWheel
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
| New Tab | Ctrl+N |
| Open File | Ctrl+O |
| Open with MarkItDown | Ctrl+Shift+O |
| Save | Ctrl+S |
| Save As | Ctrl+Shift+S |
| Close Tab | Ctrl+W |
| Next Tab | Ctrl+Tab |
| Previous Tab | Ctrl+Shift+Tab |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y |
| Find | Ctrl+F |
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
| Zoom In | Ctrl++ |
| Zoom Out | Ctrl+- |
| Reset Zoom | Ctrl+0 |

## AI Assistant Setup

1. Click the **💬 AI** button in the toolbar or press **Ctrl+Shift+A**
2. Click the **⚙️** settings button in the sidebar
3. Enter your API key(s):
   - **Anthropic**: Get from [console.anthropic.com](https://console.anthropic.com)
   - **Gemini**: Get from [aistudio.google.com](https://aistudio.google.com)
4. Select your preferred provider and model
5. Configure generation parameters (temperature, top-p, top-k)
6. Click "Save Changes"

### AI Parameters Explained
- **Temperature** (0.0-2.0): Controls randomness. Lower = more focused, higher = more creative
- **Top-p** (0.0-1.0): Nucleus sampling - only consider tokens above this cumulative probability
- **Top-k** (0-100): Only consider the top-k most likely tokens. 0 = disabled

## How It Works

1. **Direct Markdown Editing**: Open `.md` files directly for editing
2. **File Conversion**: Use "Open with MarkItDown" to convert any supported file to markdown
3. **Mode Switching**: Toggle between source (raw markdown) and visual (rendered) views
4. **AI Assistance**: Ask questions about your document or get writing help

## Architecture

```
note-processor/
├── markdown_notepad.py    # Main application (~3500 lines)
│   ├── ImageExtractor     # Extract images from Office documents
│   ├── ImageHandler       # Manage image embedding and display
│   ├── MarkdownVisualWidget  # Visual markdown rendering
│   ├── DocumentTab        # Individual tab management
│   ├── MarkdownNotepad    # Main application window
│   ├── ThemeDialog        # Theme customization
│   └── FindReplaceDialog  # Search and replace
├── ai_chat.py             # AI chat module (~1700 lines)
│   ├── AISettings         # Configuration persistence
│   ├── ChatHistoryManager # Per-document chat history
│   ├── AnthropicClient    # Claude API integration
│   ├── GeminiClient       # Google Gemini integration
│   ├── AISettingsDialog   # Settings UI
│   └── ChatSidebar        # Chat interface widget
├── session_manager.py     # Session persistence (~230 lines)
│   ├── TabState           # Individual tab state
│   ├── SessionState       # Full session state
│   └── SessionManager     # Save/restore functionality
├── requirements.txt       # Python dependencies
└── .gitignore            # Git exclusions (includes API keys)
```

## Configuration Files

The application stores configuration in your home directory:

| File | Purpose |
|------|---------|
| `~/.markitdown_notepad_settings.json` | Theme preferences |
| `~/.markitdown_ai_config.json` | AI settings and API keys |
| `~/.markitdown_session.json` | Session state (open tabs) |
| `~/.markitdown_chat_history/` | Per-document chat histories |

**Note**: API keys are stored locally and excluded from git via `.gitignore`.

## Dependencies

### Required
- **Python 3.10+**
- **tkinter** (usually included with Python)
- **markitdown[all]** - Microsoft's document conversion library
- **Pillow** - Image handling

### Optional
- **tkhtmlview** - Enhanced visual markdown rendering
- **markdown** - HTML conversion for visual mode
- **anthropic** - Anthropic Claude AI integration
- **google-genai** - Google Gemini AI integration

## License

MIT License - See LICENSE file for details.

This project uses the MIT-licensed MarkItDown package from Microsoft.
