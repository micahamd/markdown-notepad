# MarkItDown Notepad

A lightweight markdown notepad application built with Python and Tkinter, powered by Microsoft's [MarkItDown](https://github.com/microsoft/markitdown) package.

## Features

### Core Features
- **Source Mode**: Edit raw markdown with syntax-aware formatting
- **Visual Mode**: Rendered markdown preview with styled text display
- **MarkItDown Integration**: Open virtually any file type and convert to markdown

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
- Standard notepad operations (New, Open, Save, Save As)
- Undo/Redo support
- Find functionality
- Cut/Copy/Paste
- Markdown formatting toolbar (Bold, Italic, Code, Links)
- Line and column position tracking
- Modified file indicator

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install markitdown with all features:
   ```bash
   pip install 'markitdown[all]'
   ```

2. **Optional - For enhanced visual rendering:**
   ```bash
   pip install tkhtmlview markdown
   ```

## Usage

```bash
python markdown_notepad.py
```

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New file | Ctrl+N |
| Open file | Ctrl+O |
| Open with MarkItDown | Ctrl+Shift+O |
| Save | Ctrl+S |
| Save As | Ctrl+Shift+S |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y |
| Find | Ctrl+F |
| Select All | Ctrl+A |
| Bold | Ctrl+B |
| Italic | Ctrl+I |
| Source Mode | Ctrl+1 |
| Visual Mode | Ctrl+2 |
| Toggle Mode | Ctrl+E |

## How It Works

1. **Direct Markdown Editing**: Open `.md` files directly for editing
2. **File Conversion**: Use "Open with MarkItDown" to convert any supported file to markdown
3. **Mode Switching**: Toggle between source (raw markdown) and visual (rendered) views

## Image Handling Options

MarkItDown handles images through several mechanisms:

### Current Capabilities
1. **EXIF Metadata Extraction**: Extracts and displays image metadata as markdown
2. **OCR Support**: Can extract text from images using OCR
3. **LLM Integration**: Can use OpenAI GPT-4o to generate image descriptions

### Future Enhancement Options
For full image display in the visual mode, consider these approaches:

1. **Pillow/PIL Integration**
   - Embed images directly in the tkinter canvas
   - Requires storing image references and paths

2. **HTML Rendering with tkhtmlview**
   - Render markdown as HTML with embedded base64 images
   - Best for rich document viewing

3. **Custom Image Viewer**
   - Launch external image viewer for image links
   - Keep markdown editor lightweight

4. **Hybrid Approach**
   - Display image thumbnails inline
   - Click to open full-size in separate window

## Architecture

```
markdown_notepad.py
├── MarkdownVisualWidget    # Visual markdown rendering
│   ├── HTML mode          # Uses tkhtmlview + markdown (if available)
│   └── Text mode          # Styled text fallback
├── MarkdownNotepad        # Main application window
│   ├── Source editor      # Raw markdown editing
│   ├── Visual viewer      # Rendered preview
│   ├── Menu bar          # File, Edit, View, Format, Help
│   ├── Toolbar           # Quick access buttons
│   └── Status bar        # Position, mode, status
└── FindDialog            # Search functionality
```

## Dependencies

- **Python 3.10+** (required)
- **tkinter** (usually included with Python)
- **markitdown[all]** (core functionality)
- **tkhtmlview** (optional - enhanced visual rendering)
- **markdown** (optional - HTML conversion for visual mode)

## License

This project uses the MIT-licensed MarkItDown package from Microsoft.
