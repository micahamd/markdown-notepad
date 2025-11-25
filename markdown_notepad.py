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
from tkinter import ttk, filedialog, messagebox, scrolledtext, font
import re
import os
import io
import base64
import hashlib
from pathlib import Path

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
    FILE_IMAGE_PATTERN = re.compile(
        r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpe?g|gif|webp|bmp))\)',
        re.IGNORECASE
    )
    
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
        
        Returns a list of tuples: (match_start, match_end, alt_text, image_type, image_data_or_path)
        image_type is either 'base64' or 'file'
        """
        images = []
        
        # Find base64 images
        for match in self.BASE64_IMAGE_PATTERN.finditer(markdown_text):
            alt_text = match.group(1)
            mime_type = match.group(2)
            base64_data = match.group(3)
            
            # Validate the base64 data
            if self._validate_base64(base64_data):
                images.append({
                    'start': match.start(),
                    'end': match.end(),
                    'full_match': match.group(0),
                    'alt': alt_text,
                    'type': 'base64',
                    'mime': mime_type,
                    'data': base64_data
                })
        
        # Find file path images
        for match in self.FILE_IMAGE_PATTERN.finditer(markdown_text):
            # Skip if this overlaps with a base64 match
            if any(img['start'] <= match.start() < img['end'] for img in images):
                continue
                
            alt_text = match.group(1)
            file_path = match.group(2)
            
            # Skip data URIs that might have been partially matched
            if file_path.startswith('data:'):
                continue
            
            images.append({
                'start': match.start(),
                'end': match.end(),
                'full_match': match.group(0),
                'alt': alt_text,
                'type': 'file',
                'path': file_path
            })
        
        return sorted(images, key=lambda x: x['start'])
    
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
        """Load image from file path"""
        # Resolve relative paths
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.base_path, file_path)
        
        # Normalize path
        file_path = os.path.normpath(file_path)
        
        # Check cache
        if file_path in self.image_cache:
            return self.image_cache[file_path]
        
        # Security: ensure path doesn't escape base directory too much
        # (allow some traversal but be cautious)
        if not os.path.exists(file_path):
            return None
        
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


class MarkdownVisualWidget(tk.Frame):
    """
    A widget for rendering markdown visually with image support.
    Falls back to styled text if tkhtmlview is not available.
    """
    
    def __init__(self, parent, base_path=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.content = ""
        self.base_path = base_path or os.getcwd()
        self.image_handler = ImageHandler(self.base_path)
        self.image_widgets = []  # Track embedded image widgets
        self._setup_widget()
    
    def _setup_widget(self):
        """Setup the visual rendering widget"""
        # Always use text widget for better image control
        # (tkhtmlview has limitations with images)
        
        # Create scrollable frame
        self.canvas = tk.Canvas(self, bg="#ffffff", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind canvas resize to adjust frame width
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Bind mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.render_mode = "canvas"
    
    def _on_canvas_configure(self, event):
        """Adjust scrollable frame width when canvas resizes"""
        self.canvas.itemconfig(self.canvas_frame, width=event.width - 10)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def set_base_path(self, path):
        """Set the base path for resolving relative image paths"""
        self.base_path = path
        self.image_handler.base_path = path
    
    def set_content(self, markdown_text):
        """Set and render markdown content with images"""
        self.content = markdown_text
        
        # Clear previous content
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.image_widgets.clear()
        self.image_handler.clear_caches()
        
        # Find all images in the text
        images = self.image_handler.find_images(markdown_text)
        
        # Render content with images
        self._render_with_images(markdown_text, images)
    
    def _render_with_images(self, markdown_text, images):
        """Render markdown text with inline images"""
        current_pos = 0
        row = 0
        
        for img_info in images:
            # Render text before this image
            if img_info['start'] > current_pos:
                text_before = markdown_text[current_pos:img_info['start']]
                if text_before.strip():
                    row = self._render_text_block(text_before, row)
            
            # Render the image
            row = self._render_image(img_info, row)
            current_pos = img_info['end']
        
        # Render remaining text after last image
        if current_pos < len(markdown_text):
            text_after = markdown_text[current_pos:]
            if text_after.strip():
                self._render_text_block(text_after, row)
    
    def _render_text_block(self, text, row):
        """Render a block of markdown text"""
        text_widget = tk.Text(
            self.scrollable_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            padx=20,
            pady=10,
            bg="#ffffff",
            relief=tk.FLAT,
            borderwidth=0,
            height=1  # Start small, will resize
        )
        text_widget.grid(row=row, column=0, sticky="ew", padx=5, pady=2)
        
        # Configure text tags
        self._setup_text_tags(text_widget)
        
        # Render the markdown
        self._render_markdown_to_widget(text, text_widget)
        
        # Auto-resize to content
        text_widget.update_idletasks()
        line_count = int(text_widget.index('end-1c').split('.')[0])
        text_widget.config(height=line_count)
        text_widget.config(state=tk.DISABLED)
        
        return row + 1
    
    def _render_image(self, img_info, row):
        """Render an image with thumbnail and click-to-expand"""
        # Create frame for image
        img_frame = ttk.Frame(self.scrollable_frame)
        img_frame.grid(row=row, column=0, sticky="w", padx=25, pady=10)
        
        # Try to load the image
        image = self.image_handler.load_image(img_info)
        
        if image and PIL_AVAILABLE:
            # Create thumbnail
            photo = self.image_handler.get_photo_image(image, thumbnail=True, max_size=(200, 200))
            
            if photo:
                # Create clickable image label
                img_label = ttk.Label(img_frame, image=photo, cursor="hand2")
                img_label.image = photo  # Keep reference
                img_label.pack(side=tk.LEFT)
                
                # Bind click to open full image
                img_label.bind("<Button-1>", lambda e, img=image, alt=img_info.get('alt', ''): 
                              self._show_full_image(img, alt))
                
                # Add hover effect
                img_label.bind("<Enter>", lambda e, l=img_label: l.configure(relief="solid"))
                img_label.bind("<Leave>", lambda e, l=img_label: l.configure(relief="flat"))
                
                self.image_widgets.append(img_label)
                
                # Add info label
                alt_text = img_info.get('alt', '')
                if image:
                    size_info = f"{image.size[0]}x{image.size[1]}"
                    info_text = f"{alt_text}\n{size_info} (click to expand)" if alt_text else f"{size_info} (click to expand)"
                else:
                    info_text = alt_text or "(click to expand)"
                
                info_label = ttk.Label(
                    img_frame, 
                    text=info_text,
                    font=("Segoe UI", 9, "italic"),
                    foreground="#666666"
                )
                info_label.pack(side=tk.LEFT, padx=10)
            else:
                self._render_image_placeholder(img_frame, img_info, "Could not render image")
        else:
            self._render_image_placeholder(img_frame, img_info, "Image not available")
        
        return row + 1
    
    def _render_image_placeholder(self, parent, img_info, message):
        """Render a placeholder when image can't be displayed"""
        placeholder = ttk.Frame(parent, relief="solid", borderwidth=1)
        placeholder.pack(side=tk.LEFT)
        
        inner = ttk.Frame(placeholder, padding=10)
        inner.pack()
        
        ttk.Label(inner, text="🖼️", font=("Segoe UI", 24)).pack()
        ttk.Label(inner, text=message, font=("Segoe UI", 9)).pack()
        
        alt_text = img_info.get('alt', '')
        if alt_text:
            ttk.Label(inner, text=alt_text, font=("Segoe UI", 9, "italic")).pack()
    
    def _show_full_image(self, image, alt_text=""):
        """Show full-size image in popup window"""
        title = f"Image: {alt_text}" if alt_text else "Image Viewer"
        ImageViewerWindow(self.winfo_toplevel(), image, title)
    
    def _setup_text_tags(self, text_widget):
        """Setup text tags for markdown styling"""
        # Heading styles
        text_widget.tag_configure("h1", font=("Segoe UI", 24, "bold"), spacing3=10, foreground="#1a1a2e")
        text_widget.tag_configure("h2", font=("Segoe UI", 20, "bold"), spacing3=8, foreground="#16213e")
        text_widget.tag_configure("h3", font=("Segoe UI", 16, "bold"), spacing3=6, foreground="#1f4068")
        text_widget.tag_configure("h4", font=("Segoe UI", 14, "bold"), spacing3=4, foreground="#1b1b2f")
        text_widget.tag_configure("h5", font=("Segoe UI", 12, "bold"), spacing3=3, foreground="#1b1b2f")
        text_widget.tag_configure("h6", font=("Segoe UI", 11, "bold"), spacing3=2, foreground="#1b1b2f")
        
        # Inline styles
        text_widget.tag_configure("bold", font=("Segoe UI", 11, "bold"))
        text_widget.tag_configure("italic", font=("Segoe UI", 11, "italic"))
        text_widget.tag_configure("code", font=("Consolas", 10), background="#f0f0f0", foreground="#c7254e")
        text_widget.tag_configure("code_block", font=("Consolas", 10), background="#2d2d2d", foreground="#f8f8f2", spacing1=5, spacing3=5)
        
        # Block styles
        text_widget.tag_configure("blockquote", font=("Segoe UI", 11, "italic"), foreground="#6c757d", lmargin1=30, lmargin2=30)
        text_widget.tag_configure("list_item", lmargin1=20, lmargin2=40)
        text_widget.tag_configure("link", foreground="#0066cc", underline=True)
        text_widget.tag_configure("hr", font=("Segoe UI", 6), foreground="#cccccc")
        
        # Table styles
        text_widget.tag_configure("table", font=("Consolas", 10), background="#f9f9f9")
    
    def _render_markdown_to_widget(self, markdown_text, text_widget):
        """Render markdown as styled text to a text widget"""
        lines = markdown_text.split('\n')
        in_code_block = False
        code_block_content = []
        
        for i, line in enumerate(lines):
            # Skip image lines (already rendered separately)
            if re.match(r'!\[.*?\]\(.*?\)', line.strip()):
                continue
            
            # Handle code blocks
            if line.strip().startswith('```'):
                if in_code_block:
                    code_text = '\n'.join(code_block_content) + '\n'
                    text_widget.insert(tk.END, code_text, "code_block")
                    text_widget.insert(tk.END, '\n')
                    code_block_content = []
                    in_code_block = False
                else:
                    in_code_block = True
                continue
            
            if in_code_block:
                code_block_content.append(line)
                continue
            
            # Headers
            if line.startswith('######'):
                text_widget.insert(tk.END, line[6:].strip() + '\n', "h6")
            elif line.startswith('#####'):
                text_widget.insert(tk.END, line[5:].strip() + '\n', "h5")
            elif line.startswith('####'):
                text_widget.insert(tk.END, line[4:].strip() + '\n', "h4")
            elif line.startswith('###'):
                text_widget.insert(tk.END, line[3:].strip() + '\n', "h3")
            elif line.startswith('##'):
                text_widget.insert(tk.END, line[2:].strip() + '\n', "h2")
            elif line.startswith('#'):
                text_widget.insert(tk.END, line[1:].strip() + '\n', "h1")
            
            # Horizontal rule
            elif line.strip() in ['---', '***', '___']:
                text_widget.insert(tk.END, '─' * 60 + '\n', "hr")
            
            # Blockquote
            elif line.startswith('>'):
                text_widget.insert(tk.END, line[1:].strip() + '\n', "blockquote")
            
            # List items
            elif re.match(r'^[\*\-\+]\s', line.strip()):
                text_widget.insert(tk.END, '  • ' + line.strip()[2:] + '\n', "list_item")
            elif re.match(r'^\d+\.\s', line.strip()):
                text_widget.insert(tk.END, '  ' + line.strip() + '\n', "list_item")
            
            # Table rows
            elif '|' in line:
                text_widget.insert(tk.END, line + '\n', "table")
            
            # Normal text
            else:
                text_widget.insert(tk.END, line + '\n')
    
    def get_content(self):
        """Get the current content"""
        return self.content


class MarkdownNotepad(tk.Tk):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.title("MarkItDown Notepad")
        self.geometry("1000x700")
        self.configure(bg="#f5f5f5")
        
        # Application state
        self.current_file = None
        self.is_modified = False
        self.current_mode = "source"  # "source" or "visual"
        
        # Initialize MarkItDown
        if MARKITDOWN_AVAILABLE:
            self.md_converter = MarkItDown(enable_plugins=False)
        else:
            self.md_converter = None
        
        self._setup_ui()
        self._setup_bindings()
        self._update_title()
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Menu bar
        self._create_menu()
        
        # Toolbar
        self._create_toolbar()
        
        # Main content area
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Source mode editor
        self.source_frame = ttk.Frame(self.content_frame)
        self.source_editor = scrolledtext.ScrolledText(
            self.source_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            undo=True,
            padx=10,
            pady=10,
            bg="#fefefe",
            relief=tk.FLAT,
            borderwidth=1
        )
        self.source_editor.pack(fill=tk.BOTH, expand=True)
        self.source_frame.pack(fill=tk.BOTH, expand=True)
        
        # Visual mode viewer
        self.visual_frame = ttk.Frame(self.content_frame)
        self.visual_viewer = MarkdownVisualWidget(self.visual_frame, base_path=os.getcwd())
        self.visual_viewer.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self._create_status_bar()
        
        # Start in source mode
        self._show_source_mode()
    
    def _create_menu(self):
        """Create the menu bar"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Open with MarkItDown...", command=self.open_with_markitdown, accelerator="Ctrl+Shift+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_file_as, accelerator="Ctrl+Shift+S")
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
        edit_menu.add_command(label="Find...", command=self.find_text, accelerator="Ctrl+F")
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Source Mode", command=self._show_source_mode, accelerator="Ctrl+1")
        view_menu.add_command(label="Visual Mode", command=self._show_visual_mode, accelerator="Ctrl+2")
        view_menu.add_separator()
        view_menu.add_command(label="Toggle Mode", command=self._toggle_mode, accelerator="Ctrl+E")
        
        # Format menu (for source mode)
        format_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Format", menu=format_menu)
        format_menu.add_command(label="Bold", command=lambda: self._insert_format("**", "**"), accelerator="Ctrl+B")
        format_menu.add_command(label="Italic", command=lambda: self._insert_format("*", "*"), accelerator="Ctrl+I")
        format_menu.add_command(label="Code", command=lambda: self._insert_format("`", "`"), accelerator="Ctrl+`")
        format_menu.add_separator()
        format_menu.add_command(label="Heading 1", command=lambda: self._insert_line_prefix("# "))
        format_menu.add_command(label="Heading 2", command=lambda: self._insert_line_prefix("## "))
        format_menu.add_command(label="Heading 3", command=lambda: self._insert_line_prefix("### "))
        format_menu.add_separator()
        format_menu.add_command(label="Bullet List", command=lambda: self._insert_line_prefix("- "))
        format_menu.add_command(label="Numbered List", command=lambda: self._insert_line_prefix("1. "))
        format_menu.add_command(label="Blockquote", command=lambda: self._insert_line_prefix("> "))
        format_menu.add_separator()
        format_menu.add_command(label="Horizontal Rule", command=lambda: self._insert_text("\n---\n"))
        format_menu.add_command(label="Code Block", command=lambda: self._insert_text("\n```\n\n```\n"))
        format_menu.add_command(label="Link", command=lambda: self._insert_format("[", "](url)"))
        format_menu.add_command(label="Image", command=lambda: self._insert_format("![alt](", ")"))
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="MarkItDown Info", command=self.show_markitdown_info)
    
    def _create_toolbar(self):
        """Create the toolbar"""
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=5, pady=2)
        
        # Style for buttons
        style = ttk.Style()
        style.configure("Toolbar.TButton", padding=5)
        
        # File operations
        ttk.Button(toolbar, text="📄 New", command=self.new_file, style="Toolbar.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📂 Open", command=self.open_file, style="Toolbar.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔄 MarkItDown", command=self.open_with_markitdown, style="Toolbar.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Save", command=self.save_file, style="Toolbar.TButton").pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        # Mode toggle
        self.mode_var = tk.StringVar(value="source")
        ttk.Label(toolbar, text="Mode:").pack(side=tk.LEFT, padx=(5, 2))
        
        mode_frame = ttk.Frame(toolbar)
        mode_frame.pack(side=tk.LEFT, padx=2)
        
        self.source_btn = ttk.Radiobutton(
            mode_frame, text="Source", variable=self.mode_var, 
            value="source", command=self._show_source_mode
        )
        self.source_btn.pack(side=tk.LEFT)
        
        self.visual_btn = ttk.Radiobutton(
            mode_frame, text="Visual", variable=self.mode_var,
            value="visual", command=self._show_visual_mode
        )
        self.visual_btn.pack(side=tk.LEFT)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        # Format buttons (for source mode)
        ttk.Button(toolbar, text="B", command=lambda: self._insert_format("**", "**"), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="I", command=lambda: self._insert_format("*", "*"), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="</>" , command=lambda: self._insert_format("`", "`"), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="🔗", command=lambda: self._insert_format("[", "](url)"), width=3).pack(side=tk.LEFT, padx=1)
    
    def _create_status_bar(self):
        """Create the status bar"""
        self.status_bar = ttk.Frame(self)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.position_label = ttk.Label(self.status_bar, text="Line 1, Col 1")
        self.position_label.pack(side=tk.RIGHT, padx=10)
        
        self.mode_label = ttk.Label(self.status_bar, text="Source Mode")
        self.mode_label.pack(side=tk.RIGHT, padx=10)
        
        # MarkItDown status
        md_status = "MarkItDown: Available" if MARKITDOWN_AVAILABLE else "MarkItDown: Not installed"
        self.md_status_label = ttk.Label(self.status_bar, text=md_status)
        self.md_status_label.pack(side=tk.RIGHT, padx=10)
    
    def _setup_bindings(self):
        """Setup keyboard bindings"""
        self.bind("<Control-n>", lambda e: self.new_file())
        self.bind("<Control-o>", lambda e: self.open_file())
        self.bind("<Control-O>", lambda e: self.open_with_markitdown())  # Ctrl+Shift+O
        self.bind("<Control-s>", lambda e: self.save_file())
        self.bind("<Control-S>", lambda e: self.save_file_as())  # Ctrl+Shift+S
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-y>", lambda e: self.redo())
        self.bind("<Control-a>", lambda e: self.select_all())
        self.bind("<Control-f>", lambda e: self.find_text())
        self.bind("<Control-b>", lambda e: self._insert_format("**", "**"))
        self.bind("<Control-i>", lambda e: self._insert_format("*", "*"))
        self.bind("<Control-Key-1>", lambda e: self._show_source_mode())
        self.bind("<Control-Key-2>", lambda e: self._show_visual_mode())
        self.bind("<Control-e>", lambda e: self._toggle_mode())
        
        # Track modifications
        self.source_editor.bind("<<Modified>>", self._on_modified)
        self.source_editor.bind("<KeyRelease>", self._update_position)
        self.source_editor.bind("<ButtonRelease>", self._update_position)
        
        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
    
    def _on_modified(self, event=None):
        """Handle text modification"""
        if self.source_editor.edit_modified():
            self.is_modified = True
            self._update_title()
            self.source_editor.edit_modified(False)
    
    def _update_position(self, event=None):
        """Update cursor position in status bar"""
        try:
            pos = self.source_editor.index(tk.INSERT)
            line, col = pos.split('.')
            self.position_label.config(text=f"Line {line}, Col {int(col)+1}")
        except:
            pass
    
    def _update_title(self):
        """Update window title"""
        title = "MarkItDown Notepad"
        if self.current_file:
            title = f"{os.path.basename(self.current_file)} - {title}"
        if self.is_modified:
            title = f"*{title}"
        self.title(title)
    
    def _set_status(self, message):
        """Set status bar message"""
        self.status_label.config(text=message)
    
    # === View Mode Methods ===
    
    def _show_source_mode(self):
        """Switch to source editing mode"""
        self.visual_frame.pack_forget()
        self.source_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "source"
        self.mode_var.set("source")
        self.mode_label.config(text="Source Mode")
        self._set_status("Source mode - edit raw markdown")
    
    def _show_visual_mode(self):
        """Switch to visual rendering mode"""
        # Update visual view with current content
        content = self.source_editor.get(1.0, tk.END).rstrip()
        self.visual_viewer.set_content(content)
        
        self.source_frame.pack_forget()
        self.visual_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "visual"
        self.mode_var.set("visual")
        self.mode_label.config(text="Visual Mode")
        self._set_status("Visual mode - rendered markdown preview")
    
    def _toggle_mode(self):
        """Toggle between source and visual mode"""
        if self.current_mode == "source":
            self._show_visual_mode()
        else:
            self._show_source_mode()
    
    # === File Operations ===
    
    def new_file(self):
        """Create a new file"""
        if self.is_modified:
            if not self._confirm_discard():
                return
        
        self.source_editor.delete(1.0, tk.END)
        self.current_file = None
        self.is_modified = False
        self._update_title()
        self._set_status("New file created")
    
    def open_file(self):
        """Open a markdown file directly"""
        if self.is_modified:
            if not self._confirm_discard():
                return
        
        filetypes = [
            ("Markdown files", "*.md *.markdown"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.source_editor.delete(1.0, tk.END)
                self.source_editor.insert(1.0, content)
                self.current_file = filepath
                self.is_modified = False
                self._update_title()
                self._set_status(f"Opened: {filepath}")
                
                # Update visual viewer base path for relative image resolution
                self.visual_viewer.set_base_path(os.path.dirname(filepath))
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file:\n{e}")
    
    def open_with_markitdown(self):
        """Open any file and convert with MarkItDown"""
        if not MARKITDOWN_AVAILABLE:
            messagebox.showwarning(
                "MarkItDown Not Available",
                "MarkItDown is not installed.\n\nInstall with:\npip install 'markitdown[all]'"
            )
            return
        
        if self.is_modified:
            if not self._confirm_discard():
                return
        
        filetypes = [
            ("All supported files", "*.pdf *.docx *.pptx *.xlsx *.xls *.html *.htm *.csv *.json *.xml *.epub *.jpg *.jpeg *.png *.gif *.bmp *.wav *.mp3 *.msg"),
            ("PDF files", "*.pdf"),
            ("Word documents", "*.docx"),
            ("PowerPoint", "*.pptx"),
            ("Excel files", "*.xlsx *.xls"),
            ("HTML files", "*.html *.htm"),
            ("Images", "*.jpg *.jpeg *.png *.gif *.bmp"),
            ("Audio", "*.wav *.mp3"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            try:
                self._set_status(f"Converting with MarkItDown: {filepath}...")
                self.update()
                
                result = self.md_converter.convert(filepath)
                content = result.text_content
                
                self.source_editor.delete(1.0, tk.END)
                self.source_editor.insert(1.0, content)
                
                # Keep original file reference but mark as new (not the original)
                self.current_file = None
                self.is_modified = True
                self._update_title()
                self._set_status(f"Converted: {os.path.basename(filepath)}")
                
                messagebox.showinfo(
                    "Conversion Complete",
                    f"Successfully converted '{os.path.basename(filepath)}' to Markdown.\n\n"
                    "Use 'Save As' to save the markdown output."
                )
            except Exception as e:
                messagebox.showerror("Conversion Error", f"Could not convert file:\n{e}")
                self._set_status("Conversion failed")
    
    def save_file(self):
        """Save the current file"""
        if self.current_file:
            try:
                content = self.source_editor.get(1.0, tk.END).rstrip()
                with open(self.current_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.is_modified = False
                self._update_title()
                self._set_status(f"Saved: {self.current_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file:\n{e}")
        else:
            self.save_file_as()
    
    def save_file_as(self):
        """Save the file with a new name"""
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
                content = self.source_editor.get(1.0, tk.END).rstrip()
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.current_file = filepath
                self.is_modified = False
                self._update_title()
                self._set_status(f"Saved: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file:\n{e}")
    
    def _confirm_discard(self):
        """Ask user to confirm discarding changes"""
        result = messagebox.askyesnocancel(
            "Unsaved Changes",
            "You have unsaved changes. Do you want to save them?"
        )
        if result is None:  # Cancel
            return False
        if result:  # Yes
            self.save_file()
            return not self.is_modified
        return True  # No
    
    # === Edit Operations ===
    
    def undo(self):
        """Undo last action"""
        try:
            self.source_editor.edit_undo()
        except tk.TclError:
            pass
    
    def redo(self):
        """Redo last undone action"""
        try:
            self.source_editor.edit_redo()
        except tk.TclError:
            pass
    
    def cut(self):
        """Cut selected text"""
        self.source_editor.event_generate("<<Cut>>")
    
    def copy(self):
        """Copy selected text"""
        self.source_editor.event_generate("<<Copy>>")
    
    def paste(self):
        """Paste from clipboard"""
        self.source_editor.event_generate("<<Paste>>")
    
    def select_all(self):
        """Select all text"""
        self.source_editor.tag_add(tk.SEL, "1.0", tk.END)
        return "break"
    
    def find_text(self):
        """Open find dialog"""
        FindDialog(self, self.source_editor)
    
    # === Format Operations ===
    
    def _insert_format(self, prefix, suffix):
        """Insert formatting around selected text"""
        if self.current_mode != "source":
            return
        
        try:
            selected = self.source_editor.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.source_editor.delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.source_editor.insert(tk.INSERT, f"{prefix}{selected}{suffix}")
        except tk.TclError:
            # No selection, just insert the formatting
            self.source_editor.insert(tk.INSERT, f"{prefix}{suffix}")
            # Move cursor between the markers
            pos = self.source_editor.index(tk.INSERT)
            line, col = pos.split('.')
            new_col = int(col) - len(suffix)
            self.source_editor.mark_set(tk.INSERT, f"{line}.{new_col}")
    
    def _insert_line_prefix(self, prefix):
        """Insert prefix at the beginning of the current line"""
        if self.current_mode != "source":
            return
        
        pos = self.source_editor.index(tk.INSERT)
        line = pos.split('.')[0]
        self.source_editor.insert(f"{line}.0", prefix)
    
    def _insert_text(self, text):
        """Insert text at cursor position"""
        if self.current_mode != "source":
            return
        self.source_editor.insert(tk.INSERT, text)
    
    # === Application Methods ===
    
    def quit_app(self):
        """Exit the application"""
        if self.is_modified:
            if not self._confirm_discard():
                return
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


class FindDialog(tk.Toplevel):
    """Find dialog for searching text"""
    
    def __init__(self, parent, text_widget):
        super().__init__(parent)
        self.text_widget = text_widget
        self.title("Find")
        self.geometry("350x100")
        self.resizable(False, False)
        
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
        ttk.Button(btn_frame, text="Close", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key
        self.search_entry.bind("<Return>", lambda e: self.find_next())
        
        # Track search position
        self.search_pos = "1.0"
    
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


def main():
    """Main entry point"""
    app = MarkdownNotepad()
    app.mainloop()


if __name__ == "__main__":
    main()
