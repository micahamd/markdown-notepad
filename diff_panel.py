"""
Diff Panel Module for MarkItDown Notepad

Provides a side-by-side diff panel for reviewing AI-suggested edits
before applying them to the document. Supports streaming AI responses,
manual editing of suggestions, and Accept/Reject workflow.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
import queue
import threading
from typing import Optional, Callable, Dict, Any

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


class DiffPanel(tk.Toplevel):
    """
    Side-by-side diff panel for reviewing AI-suggested edits.
    
    Shows original text on the left and AI suggestion on the right.
    The suggestion can be streamed in chunks. The user can Accept (apply),
    Edit (modify before applying), or Reject (discard).
    
    Parameters:
        parent: Parent window
        original_text: The original selected text
        start_index: tk.Text index where the selection started (e.g., "3.5")
        end_index: tk.Text index where the selection ended (e.g., "5.12")
        action_name: Name of the AI action (e.g., "Proofread", "Rewrite")
        on_accept: Callback(replacement_text, start_index, end_index, original_text)
        on_reject: Optional callback when rejected
        llm_client: LLM client for generating responses
        settings: AISettings instance
        messages: Pre-built message list for the LLM
        images: Optional image list for the LLM
    """
    
    def __init__(self, parent, original_text: str, start_index: str, end_index: str,
                 action_name: str = "AI Edit",
                 on_accept: Optional[Callable] = None,
                 on_reject: Optional[Callable] = None,
                 llm_client=None, settings=None,
                 messages=None, images=None):
        super().__init__(parent)
        
        self.parent = parent
        self.original_text = original_text
        self.start_index = start_index
        self.end_index = end_index
        self.action_name = action_name
        self.on_accept = on_accept
        self.on_reject = on_reject
        self.llm_client = llm_client
        self.settings = settings
        self.messages = messages or []
        self.images = images or []
        
        self.response_text = ""
        self.response_queue = queue.Queue()
        self.is_generating = False
        self.is_editable = False
        
        # Resolve fonts
        self._sans_font = _select_best_font(SANS_FONTS)
        self._mono_font = _select_best_font(MONO_FONTS)
        
        self.title(f"Review: {action_name}")
        self.geometry("800x500")
        self.minsize(600, 350)
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
    
    def _setup_ui(self):
        """Build the diff panel UI"""
        # Main container
        main = ttk.Frame(self, padding=8)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 6))
        
        ttk.Label(header, text=f"  {self.action_name}", 
                  font=(self._sans_font, 11, 'bold')).pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(header, text="Generating...", 
                                       font=(self._sans_font, 9))
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Paned window for side-by-side
        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        
        # Left pane — Original
        left_frame = ttk.LabelFrame(paned, text="Original", padding=4)
        paned.add(left_frame, weight=1)
        
        self.original_display = scrolledtext.ScrolledText(
            left_frame, wrap=tk.WORD,
            font=(self._mono_font, 10),
            bg='#fff8f0', fg='#1a1a1a',
            state=tk.NORMAL,
            relief=tk.FLAT, borderwidth=1,
            padx=8, pady=6
        )
        self.original_display.pack(fill=tk.BOTH, expand=True)
        self.original_display.insert('1.0', self.original_text)
        self.original_display.config(state=tk.DISABLED)
        
        # Right pane — Suggestion
        right_frame = ttk.LabelFrame(paned, text="Suggested", padding=4)
        paned.add(right_frame, weight=1)
        
        self.suggestion_display = scrolledtext.ScrolledText(
            right_frame, wrap=tk.WORD,
            font=(self._mono_font, 10),
            bg='#f0fff0', fg='#1a1a1a',
            state=tk.DISABLED,
            relief=tk.FLAT, borderwidth=1,
            padx=8, pady=6
        )
        self.suggestion_display.pack(fill=tk.BOTH, expand=True)
        
        # Button bar
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)
        
        self.reject_btn = ttk.Button(btn_frame, text="Reject", command=self._on_reject)
        self.reject_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.edit_btn = ttk.Button(btn_frame, text="Edit", command=self._toggle_edit)
        self.edit_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.accept_btn = ttk.Button(btn_frame, text="Accept", command=self._on_accept,
                                      state=tk.DISABLED)
        self.accept_btn.pack(side=tk.RIGHT)
        
        # Info label
        ttk.Label(btn_frame, text="Lines " + self._format_range(), 
                  font=(self._sans_font, 9), foreground='#666666').pack(side=tk.LEFT)
    
    def _format_range(self) -> str:
        """Format the line range for display"""
        try:
            start_line = self.start_index.split('.')[0]
            end_line = self.end_index.split('.')[0]
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
            for chunk in self.llm_client.send_message_stream(
                self.messages,
                system_prompt=self.settings.system_prompt if self.settings else "",
                max_tokens=self.settings.max_tokens if self.settings else 4096,
                temperature=self.settings.temperature if self.settings else 0.7,
                top_p=self.settings.top_p if self.settings else 1.0,
                top_k=self.settings.top_k if self.settings else 0,
                images=self.images if self.images else None,
                model=self.settings.model if self.settings else None
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
        self.response_text = text
        self.suggestion_display.config(state=tk.NORMAL)
        self.suggestion_display.delete('1.0', tk.END)
        self.suggestion_display.insert('1.0', text)
        self.suggestion_display.config(state=tk.DISABLED)
        self.status_label.config(text="Ready for review")
        self.accept_btn.config(state=tk.NORMAL)
    
    def _toggle_edit(self):
        """Toggle editability of the suggestion pane"""
        if self.is_generating:
            return
        
        self.is_editable = not self.is_editable
        if self.is_editable:
            self.suggestion_display.config(state=tk.NORMAL, bg='#ffffff')
            self.edit_btn.config(text="Lock")
            self.status_label.config(text="Editing suggestion...")
        else:
            self.suggestion_display.config(state=tk.DISABLED, bg='#f0fff0')
            self.edit_btn.config(text="Edit")
            self.status_label.config(text="Ready for review")
    
    def _on_accept(self):
        """Accept the suggestion and apply to document"""
        if self.is_generating:
            return
        
        # Get the (possibly edited) suggestion text
        replacement = self.suggestion_display.get('1.0', tk.END).rstrip('\n')
        
        if not replacement.strip():
            if not messagebox.askyesno("Empty Suggestion", 
                                        "The suggestion is empty. This will delete the selected text. Continue?",
                                        parent=self):
                return
        
        if self.on_accept:
            self.on_accept(replacement, self.start_index, self.end_index, self.original_text)
        
        self.destroy()
    
    def _on_reject(self):
        """Reject and close"""
        if self.is_generating:
            # Confirm if still generating
            if not messagebox.askyesno("Cancel Generation", 
                                        "AI is still generating. Close anyway?",
                                        parent=self):
                return
        
        if self.on_reject:
            self.on_reject()
        
        self.destroy()
