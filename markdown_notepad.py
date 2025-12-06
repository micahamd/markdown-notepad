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

# Import session manager for document caching
try:
    from session_manager import get_session_manager, TabState
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False
    print("Warning: session_manager not found. Session persistence disabled.")

# Import AI chat module
try:
    from ai_chat import ChatSidebar, AISettingsDialog
    AI_CHAT_AVAILABLE = True
except ImportError:
    AI_CHAT_AVAILABLE = False
    print("Warning: ai_chat module not found. AI chat disabled.")


class ToolTip:
    """Simple tooltip widget for tkinter"""
    
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.scheduled_id = None
        
        widget.bind('<Enter>', self._on_enter)
        widget.bind('<Leave>', self._on_leave)
        widget.bind('<Button>', self._on_leave)
    
    def _on_enter(self, event=None):
        self._cancel_scheduled()
        self.scheduled_id = self.widget.after(self.delay, self._show_tooltip)
    
    def _on_leave(self, event=None):
        self._cancel_scheduled()
        self._hide_tooltip()
    
    def _cancel_scheduled(self):
        if self.scheduled_id:
            self.widget.after_cancel(self.scheduled_id)
            self.scheduled_id = None
    
    def _show_tooltip(self):
        if self.tooltip_window:
            return
        
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#ffffcc", foreground="#000000",
            relief=tk.SOLID, borderwidth=1,
            font=("Segoe UI", 9), padx=6, pady=3
        )
        label.pack()
    
    def _hide_tooltip(self):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class ImageExtractor:
    """
    Extracts images from documents alongside MarkItDown conversion.
    Supports: DOCX, PPTX, XLSX (Office Open XML formats use ZIP with embedded images)
    """
    
    SUPPORTED_EXTENSIONS = {'.docx', '.pptx', '.xlsx'}
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.emf', '.wmf'}
    
    @staticmethod
    def can_extract(filepath):
        """Check if we can extract images from this file type"""
        ext = Path(filepath).suffix.lower()
        return ext in ImageExtractor.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def extract_images(filepath, output_dir):
        """
        Extract images from Office documents.
        Returns a dict mapping original image paths to extracted file paths.
        """
        extracted = {}
        ext = Path(filepath).suffix.lower()
        
        if ext not in ImageExtractor.SUPPORTED_EXTENSIONS:
            return extracted
        
        try:
            # Office Open XML formats are ZIP files
            with zipfile.ZipFile(filepath, 'r') as zf:
                for name in zf.namelist():
                    # Images are typically in word/media/, ppt/media/, xl/media/
                    if '/media/' in name.lower():
                        img_ext = Path(name).suffix.lower()
                        if img_ext in ImageExtractor.IMAGE_EXTENSIONS:
                            # Extract to output directory
                            img_filename = Path(name).name
                            output_path = os.path.join(output_dir, img_filename)
                            
                            # Handle duplicates
                            counter = 1
                            base, ext_part = os.path.splitext(output_path)
                            while os.path.exists(output_path):
                                output_path = f"{base}_{counter}{ext_part}"
                                counter += 1
                            
                            # Extract the image
                            with zf.open(name) as src, open(output_path, 'wb') as dst:
                                dst.write(src.read())
                            
                            extracted[name] = output_path
        except Exception as e:
            print(f"Error extracting images: {e}")
        
        return extracted
    
    @staticmethod
    def insert_image_references(markdown_text, extracted_images, output_dir):
        """
        Append extracted image references to the markdown text.
        """
        if not extracted_images:
            return markdown_text
        
        # Add a section for extracted images
        image_section = "\n\n---\n\n## Extracted Images\n\n"
        for orig_path, local_path in extracted_images.items():
            filename = os.path.basename(local_path)
            # Use relative path from output directory
            rel_path = os.path.basename(local_path)
            image_section += f"![{filename}]({rel_path})\n\n"
        
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
            
            # Parse size from alt text
            alt_text, width, height = self._parse_alt_size(alt_text)
            
            # Also check title for size hints
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
                'height': height
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
                'is_html': True
            })
        
        return sorted(images, key=lambda x: x['start'])
    
    def _parse_alt_size(self, alt_text):
        """
        Parse size hints from alt text.
        Supports formats:
        - "description|200x100" -> (description, 200, 100)
        - "description|200" -> (description, 200, None)
        - "description" -> (description, None, None)
        """
        if '|' in alt_text:
            parts = alt_text.rsplit('|', 1)
            alt = parts[0].strip()
            size_str = parts[1].strip()
            width, height = self._parse_size_string(size_str)
            return alt, width, height
        return alt_text, None, None
    
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
        self._setup_widget()
    
    def _setup_widget(self):
        """Setup the visual rendering widget - single Text widget for performance"""
        # Main text widget with scrollbar
        self.text_frame = ttk.Frame(self)
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.text_frame, orient=tk.VERTICAL)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text_widget = tk.Text(
            self.text_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            padx=30,
            pady=20,
            bg="#ffffff",
            fg="#1a1a1a",
            relief=tk.FLAT,
            borderwidth=0,
            yscrollcommand=self.scrollbar.set,
            cursor="xterm",
            insertbackground="#333333",
            selectbackground="#b3d9ff",
            selectforeground="#000000",
            spacing1=2,  # Space above lines
            spacing2=1,  # Space between wrapped lines  
            spacing3=4,  # Space below lines (paragraph spacing)
        )
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.text_widget.yview)
        
        # Setup text tags for styling
        self._setup_text_tags()
        
        # Bind mousewheel and editing events
        self.text_widget.bind("<MouseWheel>", self._on_mousewheel)
        self.text_widget.bind("<KeyRelease>", self._on_edit)
        self.text_widget.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        
        self.render_mode = "text"
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.text_widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"
    
    def _on_ctrl_mousewheel(self, event):
        """Handle Ctrl+MouseWheel for zoom (propagate to parent)"""
        # Let the main app handle zoom
        return
    
    def _on_edit(self, event=None):
        """Handle text edits - notify parent of changes"""
        if self.on_content_change:
            # Get current text content (raw, without images)
            content = self.text_widget.get("1.0", tk.END).rstrip()
            self.content = content
            self.on_content_change(content)
    
    def set_base_path(self, path):
        """Set the base path for resolving relative image paths"""
        self.base_path = path
        self.image_handler.base_path = path
    
    def set_content(self, markdown_text):
        """Set and render markdown content with images - optimized for large files"""
        # Cancel any pending render
        if self._render_job:
            self.after_cancel(self._render_job)
            self._render_job = None
        
        self.content = markdown_text
        
        # Clear previous content
        self.text_widget.delete(1.0, tk.END)
        self.embedded_images.clear()
        self.image_handler.clear_caches()
        
        # Find all images in the text
        images = self.image_handler.find_images(markdown_text)
        
        # For small documents, render immediately
        # For large documents, render progressively
        line_count = markdown_text.count('\n')
        
        if line_count < 500:
            self._render_content(markdown_text, images)
        else:
            # Progressive rendering for large documents
            self._render_progressive(markdown_text, images, 0)
    
    def _render_progressive(self, full_text, images, start_line):
        """Render content progressively to avoid UI freeze"""
        lines = full_text.split('\n')
        end_line = min(start_line + self.RENDER_CHUNK_SIZE, len(lines))
        
        # Render this chunk
        chunk = '\n'.join(lines[start_line:end_line])
        chunk_lines_text = '\n'.join(lines[start_line:end_line])
        chunk_images = [img for img in images 
                        if img.get('full_match') and img['full_match'] in chunk_lines_text]
        
        self._render_content(chunk, chunk_images, is_chunk=True)
        
        # Schedule next chunk or finalize
        if end_line < len(lines):
            self._render_job = self.after(10, 
                lambda: self._render_progressive(full_text, images, end_line))
        else:
            self._render_job = None
    
    def _render_content(self, markdown_text, images, is_chunk=False):
        """Render markdown content efficiently into the single text widget"""
        # Build a map of image positions
        image_positions = {img['start']: img for img in images}
        
        lines = markdown_text.split('\n')
        in_code_block = False
        code_block_content = []
        in_table = False
        table_rows = []
        
        for i, line in enumerate(lines):
            # Check if this line contains an image
            img_match = re.match(r'!\[.*?\]\(.*?\)', line.strip())
            
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
            
            # Handle images - insert placeholder with embedded image
            if img_match and PIL_AVAILABLE:
                # Find the matching image info
                for img in images:
                    if img.get('full_match') and img['full_match'] in line:
                        self._insert_image_inline(img)
                        break
                else:
                    # No matching image found, render as text
                    self.text_widget.insert(tk.END, line + '\n')
                continue
            
            # Headers
            if line.startswith('######'):
                self.text_widget.insert(tk.END, line[6:].strip() + '\n', "h6")
            elif line.startswith('#####'):
                self.text_widget.insert(tk.END, line[5:].strip() + '\n', "h5")
            elif line.startswith('####'):
                self.text_widget.insert(tk.END, line[4:].strip() + '\n', "h4")
            elif line.startswith('###'):
                self.text_widget.insert(tk.END, line[3:].strip() + '\n', "h3")
            elif line.startswith('##'):
                self.text_widget.insert(tk.END, line[2:].strip() + '\n', "h2")
            elif line.startswith('#'):
                self.text_widget.insert(tk.END, line[1:].strip() + '\n', "h1")
            
            # Horizontal rule
            elif line.strip() in ['---', '***', '___']:
                self.text_widget.insert(tk.END, '─' * 60 + '\n', "hr")
            
            # Blockquote
            elif line.startswith('>'):
                self.text_widget.insert(tk.END, '  │ ' + line[1:].strip() + '\n', "blockquote")
            
            # List items
            elif re.match(r'^[\*\-\+]\s', line.strip()):
                self.text_widget.insert(tk.END, '  • ' + line.strip()[2:] + '\n', "list_item")
            elif re.match(r'^\d+\.\s', line.strip()):
                self.text_widget.insert(tk.END, '  ' + line.strip() + '\n', "list_item")
            
            # Normal text
            else:
                self.text_widget.insert(tk.END, line + '\n')
        
        # Flush any remaining table at end of content
        if in_table and table_rows:
            self._render_table(table_rows)
    
    def _render_table(self, table_rows):
        """Render a markdown table with proper formatting"""
        if not table_rows:
            return
        
        # Parse table structure
        parsed_rows = []
        separator_idx = -1
        
        for i, row in enumerate(table_rows):
            # Split by | and strip whitespace
            cells = [c.strip() for c in row.split('|')]
            # Remove empty first/last cells from leading/trailing |
            if cells and cells[0] == '':
                cells = cells[1:]
            if cells and cells[-1] == '':
                cells = cells[:-1]
            
            # Check if this is the separator row (contains only -, :, and spaces)
            if all(re.match(r'^[-:]+$', c) for c in cells if c):
                separator_idx = i
            else:
                parsed_rows.append(cells)
        
        if not parsed_rows:
            return
        
        # Calculate column widths
        num_cols = max(len(row) for row in parsed_rows)
        col_widths = [0] * num_cols
        
        for row in parsed_rows:
            for j, cell in enumerate(row):
                if j < num_cols:
                    col_widths[j] = max(col_widths[j], len(cell))
        
        # Ensure minimum width
        col_widths = [max(w, 3) for w in col_widths]
        
        # Build the formatted table
        self.text_widget.insert(tk.END, '\n')
        
        # Top border
        border = '┌' + '┬'.join('─' * (w + 2) for w in col_widths) + '┐\n'
        self.text_widget.insert(tk.END, border, "table_border")
        
        for row_idx, row in enumerate(parsed_rows):
            # Pad row to have correct number of columns
            while len(row) < num_cols:
                row.append('')
            
            # Data row
            row_text = '│'
            for j, cell in enumerate(row):
                row_text += ' ' + cell.ljust(col_widths[j]) + ' │'
            row_text += '\n'
            
            # Use different tag for header row
            if row_idx == 0 and separator_idx == 1:
                self.text_widget.insert(tk.END, row_text, "table_header")
            else:
                self.text_widget.insert(tk.END, row_text, "table_cell")
            
            # Add separator after header or between rows
            if row_idx == 0 and separator_idx == 1:
                sep = '├' + '┼'.join('─' * (w + 2) for w in col_widths) + '┤\n'
                self.text_widget.insert(tk.END, sep, "table_border")
        
        # Bottom border
        border = '└' + '┴'.join('─' * (w + 2) for w in col_widths) + '┘\n'
        self.text_widget.insert(tk.END, border, "table_border")
        self.text_widget.insert(tk.END, '\n')
    
    def _insert_image_inline(self, img_info):
        """Insert an image inline in the text widget with optional sizing"""
        # Try to load the image
        image = self.image_handler.load_image(img_info)
        
        if image:
            # Determine display size
            # Priority: explicit size > default thumbnail
            display_width = img_info.get('width')
            display_height = img_info.get('height')
            
            if display_width or display_height:
                # Calculate size maintaining aspect ratio
                orig_w, orig_h = image.size
                
                if display_width and display_height:
                    max_size = (display_width, display_height)
                elif display_width:
                    # Calculate height to maintain aspect ratio
                    ratio = display_width / orig_w
                    max_size = (display_width, int(orig_h * ratio))
                else:
                    # Calculate width to maintain aspect ratio
                    ratio = display_height / orig_h
                    max_size = (int(orig_w * ratio), display_height)
                
                # Cap at reasonable maximum
                max_size = (min(max_size[0], 800), min(max_size[1], 600))
            else:
                # Default thumbnail size
                max_size = (200, 200)
            
            photo = self.image_handler.get_photo_image(image, thumbnail=True, max_size=max_size)
            
            if photo:
                # Insert image into text widget
                self.text_widget.image_create(tk.END, image=photo, padx=10, pady=5)
                
                # Store reference and image for click handling
                index = self.text_widget.index(tk.END + "-2c")
                self.embedded_images[index] = (photo, image, img_info.get('alt', ''))
                
                # Add caption with size info
                alt_text = img_info.get('alt', '')
                orig_size = f"{image.size[0]}×{image.size[1]}"
                display_size = f"{photo.width()}×{photo.height()}"
                
                if display_width or display_height:
                    caption = f"  {alt_text} ({display_size}, original: {orig_size})" if alt_text else f"  ({display_size})"
                else:
                    caption = f"  {alt_text} ({orig_size})" if alt_text else f"  ({orig_size})"
                
                self.text_widget.insert(tk.END, caption + '\n', "image_caption")
                
                # Bind click on images
                self.text_widget.tag_bind("image", "<Button-1>", self._on_image_click)
                return
        
        # Fallback: show placeholder text
        alt = img_info.get('alt', 'image')
        self.text_widget.insert(tk.END, f"\n  [🖼️ {alt}]\n", "image_placeholder")
    
    def _on_image_click(self, event):
        """Handle click on embedded image"""
        # Find which image was clicked
        index = self.text_widget.index(f"@{event.x},{event.y}")
        for img_index, (photo, full_image, alt) in self.embedded_images.items():
            # Check if click is near this image
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
        tw = self.text_widget
        
        # Heading styles - improved spacing for document flow
        tw.tag_configure("h1", font=("Segoe UI", 26, "bold"), spacing1=20, spacing3=12, foreground="#1a1a2e")
        tw.tag_configure("h2", font=("Segoe UI", 22, "bold"), spacing1=18, spacing3=10, foreground="#16213e")
        tw.tag_configure("h3", font=("Segoe UI", 18, "bold"), spacing1=14, spacing3=8, foreground="#1f4068")
        tw.tag_configure("h4", font=("Segoe UI", 15, "bold"), spacing1=10, spacing3=6, foreground="#1b1b2f")
        tw.tag_configure("h5", font=("Segoe UI", 13, "bold"), spacing1=8, spacing3=4, foreground="#1b1b2f")
        tw.tag_configure("h6", font=("Segoe UI", 12, "bold"), spacing1=6, spacing3=3, foreground="#1b1b2f")
        
        # Inline styles
        tw.tag_configure("bold", font=("Segoe UI", 11, "bold"))
        tw.tag_configure("italic", font=("Segoe UI", 11, "italic"))
        tw.tag_configure("code", font=("Consolas", 10), background="#f5f2f0", foreground="#c7254e",
                        relief=tk.FLAT)
        tw.tag_configure("code_block", font=("Consolas", 10), background="#282c34", foreground="#abb2bf", 
                        spacing1=10, spacing3=10, lmargin1=25, lmargin2=25, rmargin=25)
        
        # Block styles
        tw.tag_configure("blockquote", font=("Segoe UI", 11, "italic"), foreground="#6c757d", 
                        lmargin1=40, lmargin2=40, spacing1=5, spacing3=5,
                        background="#f8f9fa")
        tw.tag_configure("list_item", lmargin1=25, lmargin2=45, spacing1=2, spacing3=2)
        tw.tag_configure("link", foreground="#0066cc", underline=True)
        tw.tag_configure("hr", font=("Segoe UI", 2), foreground="#dee2e6", spacing1=15, spacing3=15)
        
        # Table styles
        tw.tag_configure("table", font=("Consolas", 10), background="#f8f9fa", lmargin1=15, 
                        spacing1=5, spacing3=5)
        tw.tag_configure("table_border", font=("Consolas", 10), foreground="#6c757d", 
                        lmargin1=20, lmargin2=20)
        tw.tag_configure("table_header", font=("Consolas", 10, "bold"), background="#e9ecef",
                        lmargin1=20, lmargin2=20)
        tw.tag_configure("table_cell", font=("Consolas", 10), background="#f8f9fa",
                        lmargin1=20, lmargin2=20)
        
        # Image styles
        tw.tag_configure("image_caption", font=("Segoe UI", 9, "italic"), foreground="#6c757d",
                        spacing1=3, spacing3=8, justify="center")
        tw.tag_configure("image_placeholder", font=("Segoe UI", 10), foreground="#868e96", 
                        background="#f1f3f5", lmargin1=25, spacing1=5, spacing3=5)
        
        # Paragraph spacing (normal text)
        tw.tag_configure("paragraph", spacing1=3, spacing3=6)
    
    def get_content(self):
        """Get the current content"""
        return self.content
    
    def apply_theme(self, theme):
        """Apply theme settings to the visual widget"""
        self.text_widget.config(
            bg=theme.get('visual_bg', '#ffffff'),
            fg=theme.get('visual_fg', '#1a1a1a'),
            font=(theme.get('visual_font', 'Segoe UI'), theme.get('visual_font_size', 11))
        )


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
        self.file_path = None
        self.is_modified = False
        self.current_mode = "source"
        self.word_wrap = True
        
        # Source mode editor
        self.source_frame = ttk.Frame(self)
        self.source_editor = scrolledtext.ScrolledText(
            self.source_frame,
            wrap=tk.WORD,
            font=(theme['source_font'], theme['source_font_size']),
            undo=True,
            padx=15,
            pady=12,
            bg=theme['source_bg'],
            fg=theme['source_fg'],
            insertbackground=theme['source_fg'],
            selectbackground="#b3d9ff",
            selectforeground="#000000",
            relief=tk.FLAT,
            borderwidth=1,
            spacing1=1,
            spacing3=1,
        )
        self.source_editor.pack(fill=tk.BOTH, expand=True)
        self.source_frame.pack(fill=tk.BOTH, expand=True)
        
        # Visual mode viewer
        self.visual_frame = ttk.Frame(self)
        self.visual_viewer = MarkdownVisualWidget(
            self.visual_frame,
            base_path=os.getcwd(),
            on_content_change=self._on_visual_edit
        )
        self.visual_viewer.pack(fill=tk.BOTH, expand=True)
        self.visual_viewer.apply_theme(theme)
        
        # Setup modification tracking
        self.source_editor.bind("<<Modified>>", self._on_text_modified)
    
    def _on_text_modified(self, event=None):
        """Handle text modification"""
        if self.source_editor.edit_modified():
            self.is_modified = True
            self.source_editor.edit_modified(False)
            if self.on_modified:
                self.on_modified()
    
    def _on_visual_edit(self, content):
        """Handle visual editor changes"""
        self.is_modified = True
        if self.on_modified:
            self.on_modified()
        if self.on_content_change_callback:
            self.on_content_change_callback(content)
    
    def get_display_name(self):
        """Get display title for tab"""
        if self.file_path:
            return os.path.basename(self.file_path)
        return "Untitled"
    
    def get_content(self):
        """Get current document content"""
        if self.current_mode == "visual":
            return self.visual_viewer.get_content()
        return self.source_editor.get(1.0, tk.END).rstrip()
    
    def set_content(self, content):
        """Set document content"""
        self.source_editor.delete(1.0, tk.END)
        self.source_editor.insert(1.0, content)
        self.source_editor.edit_modified(False)
        self.is_modified = False
    
    def apply_theme(self, theme):
        """Apply theme to this tab"""
        self.theme = theme
        self.source_editor.config(
            bg=theme['source_bg'],
            fg=theme['source_fg'],
            insertbackground=theme['source_fg'],
            font=(theme['source_font'], theme['source_font_size'])
        )
        self.visual_viewer.apply_theme(theme)


class MarkdownNotepad(tk.Tk):
    """Main application window"""
    
    # Default theme settings
    DEFAULT_THEME = {
        'source_bg': '#fefefe',
        'source_fg': '#1a1a1a',
        'source_font': 'Consolas',
        'source_font_size': 11,
        'visual_bg': '#ffffff',
        'visual_fg': '#1a1a1a',
        'visual_font': 'Segoe UI',
        'visual_font_size': 11,
    }
    
    def __init__(self):
        super().__init__()
        
        self.title("MarkItDown Notepad")
        self.geometry("1000x700")
        self.configure(bg="#f5f5f5")
        
        # Application state - now tab-based
        self.tabs = []  # List of DocumentTab instances
        self.current_tab = None  # Currently active tab
        self.word_wrap = True  # Word wrap toggle (global setting)
        self.extracted_images_dir = None  # Directory for extracted images
        
        # Session manager for document caching
        self.session_manager = get_session_manager() if SESSION_MANAGER_AVAILABLE else None
        self._auto_save_job = None
        
        # Focus mode state
        self.focus_mode_active = False
        self.focus_overlay = None
        self.pre_focus_geometry = None
        self.pre_focus_state = None
        self.focus_widget = None
        self.focus_original_parent = None
        self.focus_escape_binding = None
        
        # Recent files
        self.recent_files = []
        self.max_recent_files = 10
        self._load_recent_files()
        
        # Theme settings
        self.theme = self.DEFAULT_THEME.copy()
        self._load_theme()
        
        # Initialize MarkItDown
        if MARKITDOWN_AVAILABLE:
            self.md_converter = MarkItDown(enable_plugins=False)
        else:
            self.md_converter = None
        
        self._setup_ui()
        self._setup_bindings()
        
        # Restore session or create initial empty tab
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
                    self.theme.update(saved_theme)
            except:
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
        
        # Check if already open
        for tab in self.tabs:
            if tab.file_path == filepath:
                self.notebook.select(tab)
                self._set_status(f"File already open: {os.path.basename(filepath)}")
                return
        
        # Open the file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tab = self.new_tab(file_path=filepath, content=content)
            tab.is_modified = False
            self._update_tab_title(tab)
            self._set_status(f"Opened: {filepath}")
            tab.visual_viewer.set_base_path(os.path.dirname(filepath))
            
            # Move to top of recent files
            self._add_to_recent_files(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file:\n{e}")
    
    def _clear_recent_files(self):
        """Clear the recent files list"""
        self.recent_files = []
        self._save_recent_files()
        self._update_recent_files_menu()
        self._set_status("Recent files cleared")
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Menu bar
        self._create_menu()
        
        # Toolbar
        self._create_toolbar()
        
        # Main content area with PanedWindow for editor and AI sidebar
        self.main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left pane: Editor content with tabs
        self.content_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.content_frame, weight=1)
        
        # Notebook widget for tabs
        self.notebook = ttk.Notebook(self.content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        
        # Configure right-click menu for tabs
        self.notebook.bind("<Button-3>", self._show_tab_context_menu)
        
        # Right pane: AI Sidebar (initially hidden)
        self.ai_sidebar = None
        self.ai_sidebar_visible = False
        self._setup_ai_sidebar()
        
        # Status bar
        self._create_status_bar()
        
        # Font size tracking for zoom
        self.current_font_size = 11
        self.min_font_size = 6
        self.max_font_size = 48
    
    def _create_menu(self):
        """Create the menu bar"""
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)
        menubar = self.menubar  # Keep local reference for compatibility
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Tab", command=self.new_tab, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self.open_file, accelerator="Ctrl+O")
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
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Source Mode", command=self._show_source_mode, accelerator="Ctrl+1")
        view_menu.add_command(label="Visual Mode", command=self._show_visual_mode, accelerator="Ctrl+2")
        view_menu.add_separator()
        view_menu.add_command(label="Toggle Mode", command=self._toggle_mode, accelerator="Ctrl+E")
        view_menu.add_separator()
        
        # Word wrap toggle
        self.wrap_var = tk.BooleanVar(value=True)
        view_menu.add_checkbutton(label="Word Wrap", variable=self.wrap_var, 
                                   command=self._toggle_word_wrap, accelerator="Ctrl+Shift+W")
        
        # Large file mode
        self.large_file_var = tk.BooleanVar(value=False)
        view_menu.add_checkbutton(label="Large File Mode (faster)", variable=self.large_file_var,
                                   command=self._toggle_large_file_mode)
        
        view_menu.add_separator()
        view_menu.add_command(label="Zoom In", command=self.zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self.zoom_out, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset Zoom", command=self.zoom_reset, accelerator="Ctrl+0")
        
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
        view_menu.add_command(label="Focus Mode", command=self._toggle_focus_mode, accelerator="F11")
        
        view_menu.add_separator()
        
        # AI Sidebar
        view_menu.add_command(label="AI Sidebar", command=self._toggle_ai_sidebar, accelerator="Ctrl+Shift+A")
        view_menu.add_command(label="AI Settings...", command=self._open_ai_settings)
        
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
        format_menu.add_separator()
        format_menu.add_command(label="Insert Image...", command=self.insert_image, accelerator="Ctrl+Shift+I")
        format_menu.add_command(label="Manage Images...", command=self.manage_images)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="MarkItDown Info", command=self.show_markitdown_info)
    
    def _create_toolbar(self):
        """Create the toolbar"""
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
        toolbar = self.toolbar_frame  # Keep local reference for compatibility
        
        # Style for buttons
        style = ttk.Style()
        style.configure("Toolbar.TButton", padding=5)
        
        # File operations with tooltips
        new_btn = ttk.Button(toolbar, text="New", command=self.new_tab, style="Toolbar.TButton")
        new_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(new_btn, "New Tab (Ctrl+N)")
        
        open_btn = ttk.Button(toolbar, text="Open", command=self.open_file, style="Toolbar.TButton")
        open_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(open_btn, "Open File (Ctrl+O)")
        
        convert_btn = ttk.Button(toolbar, text="Convert", command=self.open_with_markitdown, style="Toolbar.TButton")
        convert_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(convert_btn, "Open with MarkItDown - Convert PDF, DOCX, etc. (Ctrl+Shift+O)")
        
        save_btn = ttk.Button(toolbar, text="Save", command=self.save_file, style="Toolbar.TButton")
        save_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(save_btn, "Save File (Ctrl+S)")
        
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
        ToolTip(self.source_btn, "Edit raw markdown (Ctrl+1)")
        
        self.visual_btn = ttk.Radiobutton(
            mode_frame, text="Visual", variable=self.mode_var,
            value="visual", command=self._show_visual_mode
        )
        self.visual_btn.pack(side=tk.LEFT)
        ToolTip(self.visual_btn, "Visual rendered view (Ctrl+2)")
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        # Format buttons (for source mode) with tooltips
        bold_btn = ttk.Button(toolbar, text="B", command=lambda: self._insert_format("**", "**"), width=3)
        bold_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(bold_btn, "Bold (Ctrl+B)")
        
        italic_btn = ttk.Button(toolbar, text="I", command=lambda: self._insert_format("*", "*"), width=3)
        italic_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(italic_btn, "Italic (Ctrl+I)")
        
        code_btn = ttk.Button(toolbar, text="</>", command=lambda: self._insert_format("`", "`"), width=3)
        code_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(code_btn, "Inline Code (Ctrl+`)")
        
        link_btn = ttk.Button(toolbar, text="Link", command=lambda: self._insert_format("[", "](url)"), width=4)
        link_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(link_btn, "Insert Link")
        
        img_btn = ttk.Button(toolbar, text="Image", command=self.insert_image, width=5)
        img_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(img_btn, "Insert Image (Ctrl+Shift+I)")
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        # Theme button
        theme_btn = ttk.Button(toolbar, text="Theme", command=self.open_theme_dialog, style="Toolbar.TButton")
        theme_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(theme_btn, "Change color theme")
        
        # AI sidebar toggle button (on the right side)
        self.ai_sidebar_btn = ttk.Button(toolbar, text="AI Chat", command=self._toggle_ai_sidebar, style="Toolbar.TButton")
        self.ai_sidebar_btn.pack(side=tk.RIGHT, padx=2)
        ToolTip(self.ai_sidebar_btn, "Toggle AI Assistant Sidebar (Ctrl+Shift+A)")
    
    def _create_status_bar(self):
        """Create the status bar"""
        self.status_bar = ttk.Frame(self)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.position_label = ttk.Label(self.status_bar, text="Line 1, Col 1")
        self.position_label.pack(side=tk.RIGHT, padx=10)
        
        # Word and character count
        self.word_count_label = ttk.Label(self.status_bar, text="Words: 0 | Chars: 0")
        self.word_count_label.pack(side=tk.RIGHT, padx=10)
        
        # Zoom indicator
        self.zoom_label = ttk.Label(self.status_bar, text="100%")
        self.zoom_label.pack(side=tk.RIGHT, padx=10)
        
        # Wrap indicator
        self.wrap_label = ttk.Label(self.status_bar, text="Wrap: On")
        self.wrap_label.pack(side=tk.RIGHT, padx=10)
        
        self.mode_label = ttk.Label(self.status_bar, text="Source Mode")
        self.mode_label.pack(side=tk.RIGHT, padx=10)
        
        # MarkItDown status
        md_status = "MarkItDown: ✓" if MARKITDOWN_AVAILABLE else "MarkItDown: ✗"
        self.md_status_label = ttk.Label(self.status_bar, text=md_status)
        self.md_status_label.pack(side=tk.RIGHT, padx=10)
    
    def _setup_bindings(self):
        """Setup keyboard bindings"""
        self.bind("<Control-n>", lambda e: self.new_tab())
        self.bind("<Control-o>", lambda e: self.open_file())
        self.bind("<Control-O>", lambda e: self.open_with_markitdown())  # Ctrl+Shift+O
        self.bind("<Control-s>", lambda e: self.save_file())
        self.bind("<Control-S>", lambda e: self.save_file_as())  # Ctrl+Shift+S
        self.bind("<Control-I>", lambda e: self.insert_image())  # Ctrl+Shift+I
        self.bind("<Control-w>", lambda e: self.close_tab())  # Close tab
        self.bind("<Control-W>", lambda e: self._toggle_word_wrap())  # Ctrl+Shift+W for word wrap
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-y>", lambda e: self.redo())
        self.bind("<Control-a>", lambda e: self.select_all())
        self.bind("<Control-f>", lambda e: self.find_text())
        self.bind("<Control-h>", lambda e: self.find_replace())
        self.bind("<Control-g>", lambda e: self.goto_line())
        self.bind("<Control-b>", lambda e: self._insert_format("**", "**"))
        self.bind("<Control-i>", lambda e: self._insert_format("*", "*"))
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
        
        # AI sidebar toggle
        self.bind("<Control-A>", lambda e: self._toggle_ai_sidebar())  # Ctrl+Shift+A
        
        # Focus mode toggle
        self.bind("<F11>", lambda e: self._toggle_focus_mode())
        
        # Position update state
        self._position_update_pending = False
        
        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
    
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
                    
                    # Add to notebook
                    tab_name = tab.get_display_name()
                    if tab.is_modified:
                        tab_name = f"*{tab_name}"
                    self.notebook.add(tab, text=tab_name)
                    self.tabs.append(tab)
                    
                    # Setup bindings
                    self._setup_tab_bindings(tab)
                    
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
                    
                    # Show correct mode
                    if tab_state.current_mode == "visual":
                        tab.visual_frame.pack(fill=tk.BOTH, expand=True)
                        tab.source_frame.pack_forget()
                    else:
                        tab.source_frame.pack(fill=tk.BOTH, expand=True)
                        tab.visual_frame.pack_forget()
                
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
            tab_state = TabState(
                file_path=tab.file_path,
                content=tab.get_content(),
                cursor_position=tab.source_editor.index(tk.INSERT),
                scroll_position=tab.source_editor.yview()[0],
                current_mode=tab.current_mode,
                is_modified=tab.is_modified
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
            get_document_images_callback=self._get_current_document_images
        )
        self.ai_sidebar.pack(fill=tk.BOTH, expand=True)
    
    def _toggle_ai_sidebar(self):
        """Toggle AI sidebar visibility"""
        if not AI_CHAT_AVAILABLE:
            messagebox.showwarning("AI Chat Unavailable", 
                "AI chat module not available.\nEnsure ai_chat.py is present and anthropic is installed.")
            return
        
        if self.ai_sidebar_visible:
            # Hide sidebar
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
        
        AISettingsDialog(self)
        
        # Reload settings in sidebar if it exists
        if self.ai_sidebar:
            self.ai_sidebar.reload_settings()

    # === Tab Management Methods ===
    
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
        
        # Select the new tab
        self.notebook.select(tab)
        self.current_tab = tab
        
        self._update_title()
        return tab
    
    def _setup_tab_bindings(self, tab):
        """Setup bindings for a specific tab"""
        # Mouse wheel zoom
        tab.source_editor.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        tab.visual_viewer.text_widget.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        
        # Position tracking
        tab.source_editor.bind("<KeyRelease>", self._schedule_position_update)
        tab.source_editor.bind("<ButtonRelease>", self._update_position)
    
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
                        self._update_title()
                        self._update_mode_display()
                        self._update_position()
                        self._update_word_count()
                        # Update AI sidebar context for new tab
                        self._update_ai_sidebar_context()
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
                name = f"*{name}"
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
                menu.add_command(label="New Tab", command=self.new_tab)
                menu.tk_popup(event.x_root, event.y_root)
        except:
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
    
    def _do_position_update(self):
        """Actually update position after debounce"""
        self._position_update_pending = False
        self._update_position()
        self._update_word_count()
    
    def _on_modified(self, event=None):
        """Handle text modification - called by tab's source editor"""
        if self.current_tab and self.current_tab.source_editor.edit_modified():
            self.current_tab.is_modified = True
            self._update_tab_title(self.current_tab)
            self._update_title()
            self.current_tab.source_editor.edit_modified(False)
    
    def _update_position(self, event=None):
        """Update cursor position in status bar"""
        try:
            if self.current_tab:
                pos = self.current_tab.source_editor.index(tk.INSERT)
                line, col = pos.split('.')
                self.position_label.config(text=f"Line {line}, Col {int(col)+1}")
        except:
            pass
    
    def _update_word_count(self):
        """Update word and character count in status bar"""
        try:
            if self.current_tab:
                content = self.current_tab.get_content()
                char_count = len(content)
                # Count words (split on whitespace)
                words = content.split()
                word_count = len(words)
                self.word_count_label.config(text=f"Words: {word_count} | Chars: {char_count}")
            else:
                self.word_count_label.config(text="Words: 0 | Chars: 0")
        except:
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
            mode = self.current_tab.current_mode
            self.mode_var.set(mode)
            self.mode_label.config(text=f"{mode.capitalize()} Mode")
    
    def _set_status(self, message):
        """Set status bar message"""
        self.status_label.config(text=message)
    
    # === View Mode Methods ===
    
    def _toggle_word_wrap(self):
        """Toggle word wrap in source editor"""
        self.word_wrap = not self.word_wrap
        self.wrap_var.set(self.word_wrap)
        
        # Apply to current tab
        if self.current_tab:
            wrap_mode = tk.WORD if self.word_wrap else tk.NONE
            self.current_tab.source_editor.config(wrap=wrap_mode)
        
        # Update status bar
        self.wrap_label.config(text=f"Wrap: {'On' if self.word_wrap else 'Off'}")
        status = "Word wrap enabled" if self.word_wrap else "Word wrap disabled (horizontal scroll)"
        self._set_status(status)
    
    def _toggle_large_file_mode(self):
        """Toggle large file mode for better performance with big files"""
        is_large = self.large_file_var.get()
        
        if self.current_tab:
            if is_large:
                # Disable expensive operations
                self.current_tab.source_editor.config(undo=False)
                self._set_status("Large file mode: undo disabled for performance")
            else:
                # Re-enable
                self.current_tab.source_editor.config(undo=True)
                self._set_status("Normal mode: full features enabled")
    
    def _on_ctrl_mousewheel(self, event):
        """Handle Ctrl+MouseWheel for zoom"""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        return "break"  # Prevent default scrolling
    
    def zoom_in(self):
        """Increase font size"""
        if self.current_font_size < self.max_font_size:
            self.current_font_size += 1
            self._apply_font_size()
    
    def zoom_out(self):
        """Decrease font size"""
        if self.current_font_size > self.min_font_size:
            self.current_font_size -= 1
            self._apply_font_size()
    
    def zoom_reset(self):
        """Reset font size to default"""
        self.current_font_size = self.theme['source_font_size']
        self._apply_font_size()
    
    def _apply_font_size(self):
        """Apply current font size to all tabs"""
        for tab in self.tabs:
            # Apply to source editor
            tab.source_editor.config(font=(self.theme['source_font'], self.current_font_size))
            
            # Apply proportionally to visual editor
            visual_base = self.theme['visual_font_size']
            scale = self.current_font_size / self.theme['source_font_size']
            visual_size = max(6, int(visual_base * scale))
            tab.visual_viewer.text_widget.config(
                font=(self.theme['visual_font'], visual_size)
            )
        
        # Update status bar with zoom level
        zoom_pct = int((self.current_font_size / self.theme['source_font_size']) * 100)
        self.zoom_label.config(text=f"{zoom_pct}%")
    
    def _show_source_mode(self):
        """Switch current tab to source editing mode"""
        if not self.current_tab:
            return
            
        # Sync any changes from visual mode first
        if self.current_tab.current_mode == "visual":
            visual_content = self.current_tab.visual_viewer.get_content()
            source_content = self.current_tab.source_editor.get(1.0, tk.END).rstrip()
            if visual_content != source_content:
                self.current_tab.source_editor.delete(1.0, tk.END)
                self.current_tab.source_editor.insert(1.0, visual_content)
        
        self.current_tab.visual_frame.pack_forget()
        self.current_tab.source_frame.pack(fill=tk.BOTH, expand=True)
        self.current_tab.current_mode = "source"
        self.mode_var.set("source")
        self.mode_label.config(text="Source Mode")
        self._set_status("Source mode - edit raw markdown")
    
    def _show_visual_mode(self):
        """Switch current tab to visual rendering mode"""
        if not self.current_tab:
            return
            
        # Update visual view with current content
        content = self.current_tab.source_editor.get(1.0, tk.END).rstrip()
        self.current_tab.visual_viewer.set_content(content)
        
        self.current_tab.source_frame.pack_forget()
        self.current_tab.visual_frame.pack(fill=tk.BOTH, expand=True)
        self.current_tab.current_mode = "visual"
        self.mode_var.set("visual")
        self.mode_label.config(text="Visual Mode")
        self._set_status("Visual mode - editable rendered preview")
    
    def _on_visual_edit(self, content):
        """Called when visual editor content changes"""
        if self.current_tab:
            self.current_tab.is_modified = True
            self._update_tab_title(self.current_tab)
            self._update_title()
    
    def _toggle_mode(self):
        """Toggle between source and visual mode"""
        if not self.current_tab:
            return
        if self.current_tab.current_mode == "source":
            self._show_visual_mode()
        else:
            self._show_source_mode()
    
    # === Focus Mode ===
    
    def _toggle_focus_mode(self):
        """Toggle focus mode on/off"""
        if self.focus_mode_active:
            self._exit_focus_mode()
        else:
            self._enter_focus_mode()
    
    def _enter_focus_mode(self):
        """Enter distraction-free focus mode"""
        if not self.current_tab:
            return
        
        # Store state for restoration
        self.focus_mode_active = True
        self.pre_focus_geometry = self.geometry()
        self.pre_focus_state = self.state()
        self.focus_mode_type = self.current_tab.current_mode  # "source" or "visual"
        
        # Get current content and cursor position
        if self.focus_mode_type == "source":
            self.focus_original_content = self.current_tab.source_editor.get(1.0, tk.END)
            try:
                self.focus_cursor_pos = self.current_tab.source_editor.index(tk.INSERT)
            except:
                self.focus_cursor_pos = "1.0"
        else:
            self.focus_original_content = self.current_tab.visual_viewer.get_content()
        
        # Hide all UI elements
        self.toolbar_frame.pack_forget()
        self.status_bar.pack_forget()
        self.main_paned.pack_forget()
        self.config(menu="")  # Hide menu bar
        
        # Configure root window background to black
        self.configure(bg="black")
        
        # Create the focus overlay
        self.focus_overlay = tk.Frame(self, bg="black")
        self.focus_overlay.pack(fill=tk.BOTH, expand=True)
        
        # Calculate side widths (each side is 1/6 of screen = total 1/3 for margins)
        screen_width = self.winfo_screenwidth()
        side_width = screen_width // 6
        
        # Left spacer (black)
        left_spacer = tk.Frame(self.focus_overlay, bg="black", width=side_width)
        left_spacer.pack(side=tk.LEFT, fill=tk.Y)
        left_spacer.pack_propagate(False)
        
        # Center content frame
        self.focus_center = tk.Frame(self.focus_overlay, bg="black")
        self.focus_center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right spacer (black)
        right_spacer = tk.Frame(self.focus_overlay, bg="black", width=side_width)
        right_spacer.pack(side=tk.RIGHT, fill=tk.Y)
        right_spacer.pack_propagate(False)
        
        # Create a new editor widget for focus mode
        if self.focus_mode_type == "source":
            self.focus_editor = scrolledtext.ScrolledText(
                self.focus_center,
                wrap=tk.WORD,
                font=(self.theme['source_font'], self.current_font_size),
                undo=True,
                padx=20,
                pady=15,
                bg=self.theme['source_bg'],
                fg=self.theme['source_fg'],
                insertbackground=self.theme['source_fg'],
                selectbackground="#b3d9ff",
                selectforeground="#000000",
                relief=tk.FLAT,
                borderwidth=0,
            )
            self.focus_editor.pack(fill=tk.BOTH, expand=True)
            self.focus_editor.insert(1.0, self.focus_original_content.rstrip('\n'))
            self.focus_editor.mark_set(tk.INSERT, self.focus_cursor_pos)
            self.focus_editor.see(self.focus_cursor_pos)
            self.focus_editor.focus_set()
        else:
            # Visual mode - create a visual viewer
            self.focus_editor = MarkdownVisualWidget(
                self.focus_center,
                base_path=os.path.dirname(self.current_tab.file_path) if self.current_tab.file_path else os.getcwd(),
                on_content_change=None
            )
            self.focus_editor.pack(fill=tk.BOTH, expand=True)
            self.focus_editor.apply_theme(self.theme)
            self.focus_editor.set_content(self.focus_original_content)
        
        # Go fullscreen
        self.attributes('-fullscreen', True)
        
        # Bind Escape to exit focus mode
        self.focus_escape_binding = self.bind("<Escape>", lambda e: self._exit_focus_mode())
    
    def _exit_focus_mode(self):
        """Exit focus mode and restore normal view"""
        if not self.focus_mode_active:
            return
        
        # Get content from focus editor and sync back
        if self.focus_mode_type == "source":
            new_content = self.focus_editor.get(1.0, tk.END).rstrip('\n')
            cursor_pos = self.focus_editor.index(tk.INSERT)
            
            # Update the original editor
            self.current_tab.source_editor.delete(1.0, tk.END)
            self.current_tab.source_editor.insert(1.0, new_content)
            self.current_tab.source_editor.mark_set(tk.INSERT, cursor_pos)
            self.current_tab.source_editor.see(cursor_pos)
            
            # Check if content changed
            if new_content != self.focus_original_content.rstrip('\n'):
                self.current_tab.is_modified = True
                self._update_tab_title(self.current_tab)
        else:
            # Visual mode - get content and update
            new_content = self.focus_editor.get_content()
            if new_content != self.focus_original_content:
                self.current_tab.source_editor.delete(1.0, tk.END)
                self.current_tab.source_editor.insert(1.0, new_content)
                self.current_tab.visual_viewer.set_content(new_content)
                self.current_tab.is_modified = True
                self._update_tab_title(self.current_tab)
        
        # Unbind Escape
        if self.focus_escape_binding:
            self.unbind("<Escape>", self.focus_escape_binding)
            self.focus_escape_binding = None
        
        # Exit fullscreen
        self.attributes('-fullscreen', False)
        
        # Destroy the overlay and focus editor
        if self.focus_overlay:
            self.focus_overlay.destroy()
            self.focus_overlay = None
            self.focus_editor = None
        
        # Restore root window background
        self.configure(bg="#f5f5f5")
        
        # Restore UI elements in correct order
        self.config(menu=self.menubar)
        self.toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Restore window state/geometry
        if self.pre_focus_state == 'zoomed':
            self.state('zoomed')
        else:
            self.geometry(self.pre_focus_geometry)
        
        # Clear state
        self.focus_mode_active = False
        self.focus_center = None
        
        self._set_status("Ready")
    
    # === File Operations ===

    def new_file(self):
        """Create a new file - now creates a new tab"""
        self.new_tab()
        self._set_status("New file created")
    
    def open_file(self):
        """Open a markdown file in a new tab"""
        filetypes = [
            ("Markdown files", "*.md *.markdown"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            # Check if already open in a tab
            for tab in self.tabs:
                if tab.file_path == filepath:
                    # Switch to that tab
                    self.notebook.select(tab)
                    self._set_status(f"File already open: {os.path.basename(filepath)}")
                    return
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create a new tab for this file
                tab = self.new_tab(file_path=filepath, content=content)
                tab.is_modified = False
                self._update_tab_title(tab)
                self._set_status(f"Opened: {filepath}")
                
                # Update visual viewer base path for relative image resolution
                tab.visual_viewer.set_base_path(os.path.dirname(filepath))
                
                # Add to recent files
                self._add_to_recent_files(filepath)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file:\n{e}")
    
    def open_with_markitdown(self):
        """Open any file and convert with MarkItDown, extracting images if possible"""
        if not MARKITDOWN_AVAILABLE:
            messagebox.showwarning(
                "MarkItDown Not Available",
                "MarkItDown is not installed.\n\nInstall with:\npip install 'markitdown[all]'"
            )
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
                
                # Convert with MarkItDown
                result = self.md_converter.convert(filepath)
                content = result.text_content
                
                # Try to extract images from Office documents
                extracted_images = {}
                if ImageExtractor.can_extract(filepath):
                    # Ask user if they want to extract images
                    extract = messagebox.askyesno(
                        "Extract Images?",
                        "This document may contain images. Would you like to extract them?\n\n"
                        "Images will be saved alongside your markdown file."
                    )
                    
                    if extract:
                        # Create output directory for images
                        base_name = Path(filepath).stem
                        output_dir = filedialog.askdirectory(
                            title="Select folder to save extracted images",
                            mustexist=True
                        )
                        
                        if output_dir:
                            self._set_status("Extracting images...")
                            self.update()
                            
                            extracted_images = ImageExtractor.extract_images(filepath, output_dir)
                            
                            if extracted_images:
                                # Append image references to content
                                content = ImageExtractor.insert_image_references(
                                    content, extracted_images, output_dir
                                )
                                self.extracted_images_dir = output_dir
                
                # Create a new tab for the converted content
                tab = self.new_tab(content=content)
                tab.is_modified = True  # Mark as modified since not yet saved
                self._update_tab_title(tab)
                
                # Set base path for visual viewer
                if extracted_images and output_dir:
                    tab.visual_viewer.set_base_path(output_dir)
                
                # Status message
                img_count = len(extracted_images)
                if img_count > 0:
                    self._set_status(f"Converted: {os.path.basename(filepath)} ({img_count} images extracted)")
                    messagebox.showinfo(
                        "Conversion Complete",
                        f"Successfully converted '{os.path.basename(filepath)}' to Markdown.\n\n"
                        f"Extracted {img_count} image(s) to: {self.extracted_images_dir}\n\n"
                        "Use 'Save As' to save the markdown output."
                    )
                else:
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
        """Save the current tab's file"""
        if not self.current_tab:
            return False
            
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
            try:
                self.current_tab.source_editor.edit_undo()
            except tk.TclError:
                pass
    
    def redo(self):
        """Redo last undone action"""
        if self.current_tab:
            try:
                self.current_tab.source_editor.edit_redo()
            except tk.TclError:
                pass
    
    def cut(self):
        """Cut selected text"""
        if self.current_tab:
            self.current_tab.source_editor.event_generate("<<Cut>>")
    
    def copy(self):
        """Copy selected text"""
        if self.current_tab:
            self.current_tab.source_editor.event_generate("<<Copy>>")
    
    def paste(self):
        """Paste from clipboard"""
        if self.current_tab:
            self.current_tab.source_editor.event_generate("<<Paste>>")
    
    def select_all(self):
        """Select all text"""
        if self.current_tab:
            self.current_tab.source_editor.tag_add(tk.SEL, "1.0", tk.END)
        return "break"
    
    def find_text(self):
        """Open find dialog"""
        if self.current_tab:
            FindDialog(self, self.current_tab.source_editor)
    
    # === Format Operations ===
    
    def _insert_format(self, prefix, suffix):
        """Insert formatting around selected text"""
        if not self.current_tab or self.current_tab.current_mode != "source":
            return
        
        editor = self.current_tab.source_editor
        try:
            selected = editor.get(tk.SEL_FIRST, tk.SEL_LAST)
            editor.delete(tk.SEL_FIRST, tk.SEL_LAST)
            editor.insert(tk.INSERT, f"{prefix}{selected}{suffix}")
        except tk.TclError:
            # No selection, just insert the formatting
            editor.insert(tk.INSERT, f"{prefix}{suffix}")
            # Move cursor between the markers
            pos = editor.index(tk.INSERT)
            line, col = pos.split('.')
            new_col = int(col) - len(suffix)
            editor.mark_set(tk.INSERT, f"{line}.{new_col}")
    
    def _insert_line_prefix(self, prefix):
        """Insert prefix at the beginning of the current line"""
        if not self.current_tab or self.current_tab.current_mode != "source":
            return
        
        editor = self.current_tab.source_editor
        pos = editor.index(tk.INSERT)
        line = pos.split('.')[0]
        editor.insert(f"{line}.0", prefix)
    
    def _insert_text(self, text):
        """Insert text at cursor position"""
        if not self.current_tab or self.current_tab.current_mode != "source":
            return
        self.current_tab.source_editor.insert(tk.INSERT, text)
    
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
        if not self.current_tab:
            return
        if self.current_tab.current_mode != "source":
            messagebox.showinfo("Info", "Please switch to Source mode to insert images.")
            return
        
        # Open image insert dialog
        InsertImageDialog(self, self.current_tab.source_editor, self.current_tab.file_path, self._get_assets_folder)
    
    def manage_images(self):
        """Open image manager dialog"""
        if not self.current_tab or not self.current_tab.file_path:
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
        """Open find & replace dialog"""
        if self.current_tab:
            FindReplaceDialog(self, self.current_tab.source_editor)
    
    def goto_line(self):
        """Open go to line dialog"""
        if self.current_tab:
            GoToLineDialog(self, self.current_tab.source_editor)
    
    def _find_next_f3(self):
        """Find next using F3 (uses last search if available)"""
        # If there's an active FindReplaceDialog, trigger its find_next
        for widget in self.winfo_children():
            if isinstance(widget, FindReplaceDialog):
                widget.find_next()
                return
        # Otherwise open find dialog
        self.find_text()
    
    # === Application Methods ===
    
    def quit_app(self):
        """Exit the application"""
        # Save session state before checking for unsaved changes
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
    
    def _apply_theme(self, new_theme):
        """Apply a new theme to the application"""
        self.theme = new_theme
        self._save_theme()
        
        # Apply to all tabs
        for tab in self.tabs:
            tab.source_editor.config(
                bg=self.theme['source_bg'],
                fg=self.theme['source_fg'],
                insertbackground=self.theme['source_fg'],
                font=(self.theme['source_font'], self.theme['source_font_size'])
            )
            tab.visual_viewer.apply_theme(self.theme)
        
        # Update current font size for zoom
        self.current_font_size = self.theme['source_font_size']
        self._apply_font_size()
        
        self._set_status("Theme applied")


class ThemeDialog(tk.Toplevel):
    """Dialog for customizing theme settings"""
    
    # Preset themes
    PRESETS = {
        'Light': {
            'source_bg': '#fefefe', 'source_fg': '#1a1a1a',
            'visual_bg': '#ffffff', 'visual_fg': '#1a1a1a',
        },
        'Dark': {
            'source_bg': '#1e1e1e', 'source_fg': '#d4d4d4',
            'visual_bg': '#252526', 'visual_fg': '#cccccc',
        },
        'Sepia': {
            'source_bg': '#f4ecd8', 'source_fg': '#5b4636',
            'visual_bg': '#f9f5eb', 'visual_fg': '#5b4636',
        },
        'Solarized Light': {
            'source_bg': '#fdf6e3', 'source_fg': '#657b83',
            'visual_bg': '#fdf6e3', 'visual_fg': '#657b83',
        },
        'Solarized Dark': {
            'source_bg': '#002b36', 'source_fg': '#839496',
            'visual_bg': '#073642', 'visual_fg': '#93a1a1',
        },
    }
    
    def __init__(self, parent, current_theme, on_apply):
        super().__init__(parent)
        self.parent = parent
        self.theme = current_theme.copy()
        self.on_apply = on_apply
        
        self.title("Theme Settings")
        self.geometry("450x480")
        self.resizable(False, False)
        self.transient(parent)
        
        self._setup_ui()
        
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda e: self.destroy())
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Presets section
        preset_frame = ttk.LabelFrame(main_frame, text="Presets", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        preset_btn_frame = ttk.Frame(preset_frame)
        preset_btn_frame.pack(fill=tk.X)
        
        for i, (name, _) in enumerate(self.PRESETS.items()):
            btn = ttk.Button(preset_btn_frame, text=name, width=12,
                           command=lambda n=name: self._apply_preset(n))
            btn.grid(row=i//3, column=i%3, padx=2, pady=2)
        
        # Source Editor section
        source_frame = ttk.LabelFrame(main_frame, text="Source Editor", padding=10)
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
        self.source_font_var = tk.StringVar(value=self.theme['source_font'])
        font_combo = ttk.Combobox(font_frame, textvariable=self.source_font_var, width=15,
                                  values=['Consolas', 'Courier New', 'Cascadia Code', 'Fira Code', 
                                         'Source Code Pro', 'Monaco', 'Menlo'])
        font_combo.pack(side=tk.LEFT, padx=2)
        
        # Font size
        ttk.Label(font_frame, text="Size:").pack(side=tk.LEFT, padx=(10, 2))
        self.source_size_var = tk.IntVar(value=self.theme['source_font_size'])
        size_spin = ttk.Spinbox(font_frame, from_=8, to=24, width=5, textvariable=self.source_size_var)
        size_spin.pack(side=tk.LEFT, padx=2)
        
        # Visual Mode section
        visual_frame = ttk.LabelFrame(main_frame, text="Visual Mode", padding=10)
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
        self.visual_font_var = tk.StringVar(value=self.theme['visual_font'])
        vfont_combo = ttk.Combobox(vfont_frame, textvariable=self.visual_font_var, width=15,
                                   values=['Segoe UI', 'Arial', 'Calibri', 'Georgia', 'Times New Roman',
                                          'Verdana', 'Trebuchet MS'])
        vfont_combo.pack(side=tk.LEFT, padx=2)
        
        # Visual Font size
        ttk.Label(vfont_frame, text="Size:").pack(side=tk.LEFT, padx=(10, 2))
        self.visual_size_var = tk.IntVar(value=self.theme['visual_font_size'])
        vsize_spin = ttk.Spinbox(vfont_frame, from_=8, to=24, width=5, textvariable=self.visual_size_var)
        vsize_spin.pack(side=tk.LEFT, padx=2)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(btn_frame, text="Apply", command=self._apply, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset to Default", command=self._reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy, width=10).pack(side=tk.RIGHT, padx=5)
    
    def _choose_color(self, key):
        """Open color chooser for the specified key"""
        # Get current color
        if key == 'source_bg':
            current = self.source_bg_var.get()
        elif key == 'source_fg':
            current = self.source_fg_var.get()
        elif key == 'visual_bg':
            current = self.visual_bg_var.get()
        else:
            current = self.visual_fg_var.get()
        
        color = colorchooser.askcolor(color=current, title=f"Choose {key.replace('_', ' ').title()}")
        if color[1]:
            if key == 'source_bg':
                self.source_bg_var.set(color[1])
                self.source_bg_preview.config(bg=color[1])
            elif key == 'source_fg':
                self.source_fg_var.set(color[1])
                self.source_fg_preview.config(bg=color[1])
            elif key == 'visual_bg':
                self.visual_bg_var.set(color[1])
                self.visual_bg_preview.config(bg=color[1])
            else:
                self.visual_fg_var.set(color[1])
                self.visual_fg_preview.config(bg=color[1])
    
    def _apply_preset(self, preset_name):
        """Apply a preset theme"""
        preset = self.PRESETS[preset_name]
        
        self.source_bg_var.set(preset['source_bg'])
        self.source_bg_preview.config(bg=preset['source_bg'])
        self.source_fg_var.set(preset['source_fg'])
        self.source_fg_preview.config(bg=preset['source_fg'])
        self.visual_bg_var.set(preset['visual_bg'])
        self.visual_bg_preview.config(bg=preset['visual_bg'])
        self.visual_fg_var.set(preset['visual_fg'])
        self.visual_fg_preview.config(bg=preset['visual_fg'])
    
    def _apply(self):
        """Apply the current settings"""
        self.theme = {
            'source_bg': self.source_bg_var.get(),
            'source_fg': self.source_fg_var.get(),
            'source_font': self.source_font_var.get(),
            'source_font_size': self.source_size_var.get(),
            'visual_bg': self.visual_bg_var.get(),
            'visual_fg': self.visual_fg_var.get(),
            'visual_font': self.visual_font_var.get(),
            'visual_font_size': self.visual_size_var.get(),
        }
        self.on_apply(self.theme)
        self.destroy()
    
    def _reset(self):
        """Reset to default theme"""
        default = MarkdownNotepad.DEFAULT_THEME
        
        self.source_bg_var.set(default['source_bg'])
        self.source_bg_preview.config(bg=default['source_bg'])
        self.source_fg_var.set(default['source_fg'])
        self.source_fg_preview.config(bg=default['source_fg'])
        self.source_font_var.set(default['source_font'])
        self.source_size_var.set(default['source_font_size'])
        self.visual_bg_var.set(default['visual_bg'])
        self.visual_bg_preview.config(bg=default['visual_bg'])
        self.visual_fg_var.set(default['visual_fg'])
        self.visual_fg_preview.config(bg=default['visual_fg'])
        self.visual_font_var.set(default['visual_font'])
        self.visual_size_var.set(default['visual_font_size'])


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
        self.geometry("500x400")
        self.resizable(True, False)
        self.transient(parent)
        
        self._setup_ui()
        
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
        self.geometry("500x220")
        self.resizable(True, False)
        self.transient(parent)
        
        # Track state
        self.search_pos = "1.0"
        self.match_count = 0
        self.all_matches = []
        
        self._setup_ui()
        self._setup_bindings()
        
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
        self.geometry("250x100")
        self.resizable(False, False)
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
        ttk.Button(btn_frame, text="Close", command=self._on_close).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter and Escape keys
        self.search_entry.bind("<Return>", lambda e: self.find_next())
        self.bind("<Escape>", lambda e: self._on_close())
        
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


def main():
    """Main entry point"""
    app = MarkdownNotepad()
    app.mainloop()


if __name__ == "__main__":
    main()
