"""
AI Chat Module for MarkItDown Notepad

Provides AI chatbot functionality with support for multiple LLM providers.
Currently implements Anthropic Claude, Google Gemini, DeepSeek, and Ollama.
"""

import json
import os
import re
import base64
import hashlib
import sys
import shutil
import subprocess
import threading
import queue
import time
import difflib
import webbrowser
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Iterator
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, font

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try to import Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not installed. Install with: pip install anthropic")

# Try to import Google Gemini
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Install with: pip install google-genai")

# Try to import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama not installed. Install with: pip install ollama")

# Try to import OpenAI (for DeepSeek)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")

# Import requests for Ollama connectivity check
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =============================================================================
# Font Detection Utilities
# =============================================================================

def get_available_fonts():
    """Get list of available system fonts for tkinter"""
    try:
        return sorted(font.families())
    except Exception:
        return []

def select_best_font(preferred_fonts, fallback="TkDefaultFont"):
    """Select the first available font from a list of preferences"""
    available = get_available_fonts()
    if not available:  # If no fonts available, return fallback
        return fallback
    
    available_lower = [f.lower() for f in available]
    
    for font_name in preferred_fonts:
        if font_name.lower() in available_lower:
            idx = available_lower.index(font_name.lower())
            return available[idx]
    
    return fallback

# Define cross-platform font preferences
SANS_SERIF_FONTS = ["Segoe UI", "SF Pro Display", "Ubuntu", "DejaVu Sans", "Arial", "Helvetica"]
MONOSPACE_FONTS = ["Consolas", "SF Mono", "Monaco", "Ubuntu Mono", "DejaVu Sans Mono", "Courier New"]

# Font selection deferred until Tkinter root window exists
# Will use fallbacks if fonts can't be detected yet
SANS_FONT = select_best_font(SANS_SERIF_FONTS, "TkDefaultFont")
MONO_FONT = select_best_font(MONOSPACE_FONTS, "TkFixedFont")


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

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str  # "user", "assistant", "system"
    content: str
    images: List[str] = field(default_factory=list)  # Base64-encoded images
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        return cls(
            role=data.get('role', 'user'),
            content=data.get('content', ''),
            images=data.get('images', []),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )


# =============================================================================
# AI Settings Management
# =============================================================================

class AISettings:
    """Manages AI configuration settings with persistence"""
    
    CONFIG_FILE = Path.home() / '.markitdown_ai_config.json'
    
    # Default Anthropic models
    ANTHROPIC_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ]
    
    # Default Gemini models
    GEMINI_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]
    
    # Default Ollama models (populated dynamically)
    OLLAMA_MODELS = [
        "llama3.2",
        "llama3.1",
        "mistral",
        "codellama",
        "phi3",
    ]
    
    # Default DeepSeek models
    DEEPSEEK_MODELS = [
        "deepseek-chat",
        "deepseek-reasoner",
    ]
    
    # Deprecated AGY model slugs from older versions (used to sanitize stale caches)
    DEPRECATED_AGY_MODELS = {
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-pro",
        "claude-sonnet-4.6",
        "claude-opus-4.6",
        "gpt-oss-120b",
        "gemini-2.5-flash",
    }

    # Default AGY models & efforts
    AGY_MODELS = [
        "gemini-3.8-flash-low",
        "gemini-3.8-flash-medium",
        "gemini-3.8-flash-high",
        "gemini-4-argon-low",
        "gemini-4-argon-medium",
        "gemini-4-argon-high",
        "claude-sonnet-5-5-low",
        "claude-sonnet-5-5-medium",
        "claude-sonnet-5-5-high",
        "claude-opus-5-5-low",
        "claude-opus-5-5-medium",
        "claude-opus-5-5-high",
        "gpt-oss-120b-medium",
    ]
    AGY_EFFORTS = ["low", "medium", "high"]
    
    DEFAULT_SETTINGS = {
        'provider': 'agy',
        'api_key': '',
        'gemini_api_key': '',  # Separate API key for Gemini
        'deepseek_api_key': '',  # Separate API key for DeepSeek
        'ollama_url': 'http://localhost:11434',  # Ollama server URL
        'agy_path': '',  # Custom path to agy CLI executable
        'agy_model': 'gemini-3.8-flash-low',
        'agy_effort': 'low',
        'model': 'gemini-3.8-flash-low',
        'system_prompt': 'You are a helpful AI assistant. You help users with their markdown documents, answer questions, and provide writing assistance.',
        'max_tokens': 4096,
        'temperature': 0.7,
        'top_p': 1.0,  # Nucleus sampling: consider tokens with top_p cumulative probability
        'top_k': 0,    # Consider only top_k tokens (0 = disabled/use default)
        'image_max_size': [512, 512],
        # Cached model lists (populated when Test/Refresh is successful)
        'cached_anthropic_models': [],
        'cached_gemini_models': [],
        'cached_ollama_models': [],
        'cached_deepseek_models': [],
        'cached_agy_models': [],
        # CLI provider
        'cli_command': '',  # e.g. 'gh copilot chat' or 'claude' or 'gemini'
        # Context menu AI actions (right-click on selected text)
        'context_menu_actions': [
            {'name': 'Transfer to Chat', 'prompt': '', 'enabled': True},
            {'name': 'Rewrite', 'prompt': 'Rewrite the following, improving clarity and flow while preserving meaning. Return ONLY the rewritten text with no commentary:\n\n{selection}', 'enabled': True},
            {'name': 'Proofread', 'prompt': 'Proofread the following text. Return ONLY the corrected text with no commentary:\n\n{selection}', 'enabled': True},
            {'name': 'Spellcheck', 'prompt': 'Fix any spelling errors in the following text. Return ONLY the corrected text with no other output:\n\n{selection}', 'enabled': True},
            {'name': 'Summarize', 'prompt': 'Summarize the following text concisely:\n\n{selection}', 'enabled': True},
            {'name': 'Expand', 'prompt': 'Expand the following text with more detail while maintaining the same tone. Return ONLY the expanded text:\n\n{selection}', 'enabled': True},
        ],
    }
    
    def __init__(self):
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.load()
    
    def load(self) -> bool:
        """Load settings from config file"""
        try:
            if self.CONFIG_FILE.exists():
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    self.settings.update(saved)
                # Sanitize stale cached AGY models from older fallback lists
                cached_agy = self.settings.get('cached_agy_models', [])
                if cached_agy:
                    self.settings['cached_agy_models'] = [
                        m for m in cached_agy if m not in self.DEPRECATED_AGY_MODELS
                    ]
                if self.settings.get('agy_model') in self.DEPRECATED_AGY_MODELS:
                    self.settings['agy_model'] = self.AGY_MODELS[0]
                if (self.settings.get('provider') in ('agy', 'antigravity')
                        and self.settings.get('model') in self.DEPRECATED_AGY_MODELS):
                    self.settings['model'] = self.settings['agy_model']
                return True
        except Exception as e:
            print(f"Error loading AI settings: {e}")
        return False
    
    def save(self) -> bool:
        """Save settings to config file"""
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving AI settings: {e}")
            return False
    
    def get(self, key: str, default=None):
        """Get a setting value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value):
        """Set a setting value"""
        self.settings[key] = value
    
    @property
    def provider(self) -> str:
        return self.settings.get('provider', 'anthropic')
    
    @provider.setter
    def provider(self, value: str):
        self.settings['provider'] = value
    
    @property
    def api_key(self) -> str:
        return self.settings.get('api_key', '')
    
    @api_key.setter
    def api_key(self, value: str):
        self.settings['api_key'] = value
    
    @property
    def model(self) -> str:
        return self.settings.get('model', 'claude-3-5-haiku-20241022')
    
    @model.setter
    def model(self, value: str):
        self.settings['model'] = value
    
    @property
    def system_prompt(self) -> str:
        return self.settings.get('system_prompt', '')
    
    @system_prompt.setter
    def system_prompt(self, value: str):
        self.settings['system_prompt'] = value
    
    @property
    def max_tokens(self) -> int:
        val = self.settings.get('max_tokens', 4096)
        try:
            return max(256, min(8192, int(val)))
        except (ValueError, TypeError):
            return 4096
    
    @max_tokens.setter
    def max_tokens(self, value: int):
        self.settings['max_tokens'] = value
    
    @property
    def temperature(self) -> float:
        return self.settings.get('temperature', 0.7)
    
    @temperature.setter
    def temperature(self, value: float):
        self.settings['temperature'] = value
    
    @property
    def chat_font_size(self) -> int:
        try:
            return max(8, min(24, int(self.settings.get('chat_font_size', 10))))
        except (TypeError, ValueError):
            return 10
    
    @chat_font_size.setter
    def chat_font_size(self, value: int):
        self.settings['chat_font_size'] = max(8, min(24, int(value)))
    
    @property
    def history_limit(self) -> int:
        """Number of trailing chat messages sent as context (0 = all)."""
        try:
            return max(0, int(self.settings.get('history_limit', 10)))
        except (TypeError, ValueError):
            return 10
    
    @history_limit.setter
    def history_limit(self, value: int):
        self.settings['history_limit'] = max(0, int(value))
    
    @property
    def image_max_size(self) -> tuple:
        size = self.settings.get('image_max_size', [512, 512])
        return tuple(size)
    
    @image_max_size.setter
    def image_max_size(self, value: tuple):
        self.settings['image_max_size'] = list(value)
    
    @property
    def top_p(self) -> float:
        return self.settings.get('top_p', 1.0)
    
    @top_p.setter
    def top_p(self, value: float):
        self.settings['top_p'] = value
    
    @property
    def top_k(self) -> int:
        return self.settings.get('top_k', 0)
    
    @top_k.setter
    def top_k(self, value: int):
        self.settings['top_k'] = value
    
    @property
    def gemini_api_key(self) -> str:
        return self.settings.get('gemini_api_key', '')
    
    @gemini_api_key.setter
    def gemini_api_key(self, value: str):
        self.settings['gemini_api_key'] = value
    
    @property
    def deepseek_api_key(self) -> str:
        return self.settings.get('deepseek_api_key', '')
    
    @deepseek_api_key.setter
    def deepseek_api_key(self, value: str):
        self.settings['deepseek_api_key'] = value
    
    @property
    def ollama_url(self) -> str:
        return self.settings.get('ollama_url', 'http://localhost:11434')
    
    @ollama_url.setter
    def ollama_url(self, value: str):
        self.settings['ollama_url'] = value

    @property
    def cli_command(self) -> str:
        return self.settings.get('cli_command', '')

    @cli_command.setter
    def cli_command(self, value: str):
        self.settings['cli_command'] = value

    @property
    def agy_path(self) -> str:
        return self.settings.get('agy_path', '')

    @agy_path.setter
    def agy_path(self, value: str):
        self.settings['agy_path'] = value

    @property
    def agy_model(self) -> str:
        return self.settings.get('agy_model', self.AGY_MODELS[0])

    @agy_model.setter
    def agy_model(self, value: str):
        self.settings['agy_model'] = value

    @property
    def agy_effort(self) -> str:
        return self.settings.get('agy_effort', 'low')

    @agy_effort.setter
    def agy_effort(self, value: str):
        self.settings['agy_effort'] = value

    def is_configured(self) -> bool:
        """Check if API key or CLI backend is configured for current provider"""
        if self.provider in ('agy', 'antigravity'):
            cmd = self.agy_path or find_antigravity_cli_command()
            return bool(cmd)
        elif self.provider == 'gemini':
            return bool(self.gemini_api_key)
        elif self.provider == 'deepseek':
            return bool(self.deepseek_api_key)
        elif self.provider == 'ollama':
            return bool(self.ollama_url)
        elif self.provider == 'cli':
            return bool(self.cli_command)
        return bool(self.api_key)
    
    def get_available_models(self) -> List[str]:
        """Get available models for current provider"""
        if self.provider in ('agy', 'antigravity'):
            cached = self.settings.get('cached_agy_models', [])
            return cached if cached else self.AGY_MODELS.copy()
        elif self.provider == 'anthropic':
            return self.ANTHROPIC_MODELS.copy()
        elif self.provider == 'gemini':
            return self.GEMINI_MODELS.copy()
        elif self.provider == 'deepseek':
            return self.DEEPSEEK_MODELS.copy()
        elif self.provider == 'ollama':
            return self.OLLAMA_MODELS.copy()
        return []
    
    def get_current_api_key(self) -> str:
        """Get API key for current provider"""
        if self.provider == 'gemini':
            return self.gemini_api_key
        elif self.provider == 'deepseek':
            return self.deepseek_api_key
        elif self.provider == 'ollama':
            return ''  # Ollama doesn't use API keys
        return self.api_key

    @property
    def context_menu_actions(self) -> list:
        """Get context menu AI actions"""
        return self.settings.get('context_menu_actions', self.DEFAULT_SETTINGS['context_menu_actions'])

    @context_menu_actions.setter
    def context_menu_actions(self, value: list):
        self.settings['context_menu_actions'] = value

    def get_enabled_context_actions(self) -> list:
        """Get only enabled context menu actions"""
        return [a for a in self.context_menu_actions if a.get('enabled', True)]


# =============================================================================
# Chat History Management
# =============================================================================

class ChatHistoryManager:
    """Manages per-document chat history with persistence"""
    
    HISTORY_DIR = Path.home() / '.markitdown_chat_history'
    
    def __init__(self):
        # Ensure history directory exists
        self.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_document_hash(self, document_path: Optional[str], content: str = "") -> str:
        """Generate unique hash for a document"""
        if document_path:
            return hashlib.md5(document_path.encode()).hexdigest()[:16]
        elif content:
            return hashlib.md5(content.encode()).hexdigest()[:16]
        return "unsaved_default"
    
    def _get_history_file(self, doc_hash: str) -> Path:
        """Get the history file path for a document"""
        return self.HISTORY_DIR / f"{doc_hash}.json"
    
    def get_history(self, document_path: Optional[str] = None, content: str = "") -> List[ChatMessage]:
        """Get chat history for a document"""
        doc_hash = self._get_document_hash(document_path, content)
        history_file = self._get_history_file(doc_hash)
        
        try:
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [ChatMessage.from_dict(m) for m in data.get('messages', [])]
        except Exception as e:
            print(f"Error loading chat history: {e}")
        
        return []
    
    def save_history(self, messages: List[ChatMessage], document_path: Optional[str] = None, content: str = ""):
        """Save chat history for a document"""
        doc_hash = self._get_document_hash(document_path, content)
        history_file = self._get_history_file(doc_hash)
        
        try:
            data = {
                'document_path': document_path,
                'last_updated': datetime.now().isoformat(),
                'messages': [m.to_dict() for m in messages]
            }
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def add_message(self, message: ChatMessage, document_path: Optional[str] = None, content: str = ""):
        """Add a message to history"""
        messages = self.get_history(document_path, content)
        messages.append(message)
        self.save_history(messages, document_path, content)
    
    def clear_history(self, document_path: Optional[str] = None, content: str = ""):
        """Clear history for a document"""
        doc_hash = self._get_document_hash(document_path, content)
        history_file = self._get_history_file(doc_hash)
        
        try:
            if history_file.exists():
                history_file.unlink()
        except Exception as e:
            print(f"Error clearing chat history: {e}")


# =============================================================================
# Image Processing Utilities
# =============================================================================

class ImageProcessor:
    """Utility class for processing images for LLM APIs"""
    
    @staticmethod
    def encode_image_to_base64(image_path: str, max_size: tuple = (512, 512)) -> Optional[str]:
        """Load an image, resize it, and encode to base64"""
        if not PIL_AVAILABLE:
            return None
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Resize maintaining aspect ratio
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save to bytes
                from io import BytesIO
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    @staticmethod
    def resize_base64_image(base64_data: str, max_size: tuple = (512, 512)) -> Optional[str]:
        """Resize a base64-encoded image"""
        if not PIL_AVAILABLE:
            return base64_data
        
        try:
            from io import BytesIO
            
            # Decode base64
            image_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(image_data))
            
            # Check if resize is needed
            if img.size[0] <= max_size[0] and img.size[1] <= max_size[1]:
                return base64_data
            
            # Convert and resize
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Re-encode
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error resizing image: {e}")
            return base64_data


# =============================================================================
# Abstract LLM Client
# =============================================================================

class LLMClient(ABC):
    """Abstract base class for LLM API clients"""
    
    @abstractmethod
    def send_message(
        self,
        messages: List[ChatMessage],
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> str:
        """Send a message and get a response (optionally streaming)"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        pass
    
    @abstractmethod
    def test_connection(self) -> tuple[bool, str]:
        """Test API connection. Returns (success, message)"""
        pass


# =============================================================================
# Anthropic Client Implementation
# =============================================================================

class AnthropicClient(LLMClient):
    """Anthropic Claude API client"""
    
    # Fallback models if API fetch fails
    DEFAULT_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._cached_models = None
        self._client_api_key = None
        if ANTHROPIC_AVAILABLE and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
            self._client_api_key = api_key
    
    def _convert_messages_to_api_format(self, messages: List[ChatMessage]) -> List[Dict]:
        """Convert ChatMessage objects to Anthropic API format"""
        api_messages = []
        
        for msg in messages:
            if msg.role == "system":
                continue  # System messages handled separately
            
            content = []
            
            # Add text content
            if msg.content:
                content.append({
                    "type": "text",
                    "text": msg.content
                })
            
            # Add images (only for user messages)
            if msg.role == "user" and msg.images:
                for img_data in msg.images:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data
                        }
                    })
            
            api_messages.append({
                "role": msg.role,
                "content": content if len(content) > 1 or msg.images else msg.content
            })
        
        return api_messages
    
    def send_message(
        self,
        messages: List[ChatMessage],
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> str:
        """Send messages to Claude and get response"""
        if not self.client or getattr(self, '_client_api_key', None) != self.api_key:
            if ANTHROPIC_AVAILABLE and self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self._client_api_key = self.api_key
            else:
                raise ValueError("Anthropic client not initialized. Check API key.")
        
        api_messages = self._convert_messages_to_api_format(messages)
        
        try:
            if on_chunk:
                # Streaming response
                full_response = ""
                with self.client.messages.stream(
                    model=messages[0].role if hasattr(messages[0], 'model') else "claude-3-5-haiku-20241022",
                    messages=api_messages,
                    system=system_prompt if system_prompt else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        on_chunk(text)
                return full_response
            else:
                # Non-streaming response
                response = self.client.messages.create(
                    model="claude-3-5-haiku-20241022",  # Will be overridden by settings
                    messages=api_messages,
                    system=system_prompt if system_prompt else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.content[0].text
        except anthropic.APIConnectionError:
            raise ConnectionError("Failed to connect to Anthropic API. Check your internet connection.")
        except anthropic.AuthenticationError:
            raise ValueError("Invalid API key. Please check your Anthropic API key.")
        except anthropic.RateLimitError:
            raise RuntimeError("Rate limit exceeded. Please wait before sending more messages.")
        except Exception as e:
            raise RuntimeError(f"API error: {str(e)}")
    
    def send_message_with_model(
        self,
        messages: List[ChatMessage],
        model: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> str:
        """Send messages with specific model"""
        if not self.client:
            raise ValueError("Anthropic client not initialized. Check API key.")
        
        api_messages = self._convert_messages_to_api_format(messages)
        
        try:
            if on_chunk:
                # Streaming response
                full_response = ""
                with self.client.messages.stream(
                    model=model,
                    messages=api_messages,
                    system=system_prompt if system_prompt else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        on_chunk(text)
                return full_response
            else:
                # Non-streaming response
                response = self.client.messages.create(
                    model=model,
                    messages=api_messages,
                    system=system_prompt if system_prompt else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.content[0].text
        except anthropic.APIConnectionError:
            raise ConnectionError("Failed to connect to Anthropic API.")
        except anthropic.AuthenticationError:
            raise ValueError("Invalid API key.")
        except anthropic.RateLimitError:
            raise RuntimeError("Rate limit exceeded.")
        except Exception as e:
            raise RuntimeError(f"API error: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Claude models from API"""
        if self._cached_models:
            return self._cached_models.copy()
        
        if not self.client:
            return self.DEFAULT_MODELS.copy()
        
        try:
            # Fetch models from Anthropic API (beta endpoint)
            models = []
            page = self.client.beta.models.list(limit=100)
            for model in page.data:
                models.append(model.id)
            
            if models:
                # Sort to put newer/preferred models first
                models.sort(reverse=True)
                self._cached_models = models
                return models
        except Exception as e:
            print(f"Could not fetch models from API: {e}")
        
        return self.DEFAULT_MODELS.copy()
    
    def send_message_stream(
        self,
        messages: List[Dict],
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        images: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Iterator[str]:
        """Send messages and stream the response"""
        if not self.client or getattr(self, '_client_api_key', None) != self.api_key:
            # Reinitialize client if API key was updated
            if ANTHROPIC_AVAILABLE and self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self._client_api_key = self.api_key
            else:
                raise ValueError("Anthropic client not initialized. Check API key.")
        
        # Use provided model or default
        model_name = model or "claude-sonnet-4-20250514"
        
        # Convert messages format if needed (handle both dict and ChatMessage)
        api_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                api_messages.append(dict(msg))
            elif isinstance(msg, ChatMessage):
                api_messages.append({"role": msg.role, "content": msg.content})
        
        # Add images to the last user message if provided
        if images and api_messages:
            for i in range(len(api_messages) - 1, -1, -1):
                if api_messages[i].get('role') == 'user':
                    content = api_messages[i].get('content', '')
                    new_content = []
                    
                    # Add text content
                    if isinstance(content, str):
                        new_content.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        new_content.extend(content)
                    
                    # Add images
                    for img_data in images:
                        new_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_data
                            }
                        })
                    
                    api_messages[i]['content'] = new_content
                    break
        
        try:
            # Clamp max_tokens to valid Anthropic output range [1, 8192]
            safe_max_tokens = max(1, min(int(max_tokens or 4096), 8192))
            # Build kwargs for the API call
            stream_kwargs = {
                "model": model_name,
                "messages": api_messages,
                "system": system_prompt if system_prompt else anthropic.NOT_GIVEN,
                "max_tokens": safe_max_tokens,
                "temperature": temperature,
            }
            
            # Add top_p if not default (Anthropic uses 'top_p')
            if top_p < 1.0:
                stream_kwargs["top_p"] = top_p
            
            # Add top_k if set (Anthropic uses 'top_k')
            if top_k > 0:
                stream_kwargs["top_k"] = top_k
            
            with self.client.messages.stream(**stream_kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        except anthropic.APIConnectionError:
            raise ConnectionError("Failed to connect to Anthropic API.")
        except anthropic.AuthenticationError:
            raise ValueError("Invalid API key.")
        except anthropic.RateLimitError:
            raise RuntimeError("Rate limit exceeded.")
        except Exception as e:
            raise RuntimeError(f"API error: {str(e)}")
    
    def encode_image_for_api(self, image, max_size: int = 512) -> Optional[str]:
        """Encode a PIL Image for the API"""
        if not PIL_AVAILABLE:
            return None
        
        try:
            from io import BytesIO
            
            # Ensure RGB mode
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            
            # Resize if needed
            if image.size[0] > max_size or image.size[1] > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Encode to base64
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def test_connection(self) -> tuple[bool, str]:
        """Test API connection with a simple request"""
        if not ANTHROPIC_AVAILABLE:
            return False, "Anthropic library not installed"
        
        if not self.api_key:
            return False, "API key not configured"
        
        try:
            # Simple test message
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10
            )
            return True, "Connection successful!"
        except anthropic.AuthenticationError:
            return False, "Invalid API key"
        except anthropic.APIConnectionError:
            return False, "Could not connect to API"
        except Exception as e:
            return False, f"Error: {str(e)}"


# =============================================================================
# Google Gemini Client Implementation
# =============================================================================

class GeminiClient(LLMClient):
    """Google Gemini API client"""
    
    DEFAULT_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._cached_models = None
        self._client_api_key = None
        if GEMINI_AVAILABLE and api_key:
            self.client = genai.Client(api_key=api_key)
            self._client_api_key = api_key
    
    def _convert_messages_to_api_format(self, messages: List[ChatMessage]) -> List[Dict]:
        """Convert ChatMessage objects to Gemini API format"""
        api_messages = []
        
        for msg in messages:
            if msg.role == "system":
                continue  # System messages handled separately
            
            # Gemini uses "user" and "model" roles
            role = "model" if msg.role == "assistant" else "user"
            
            parts = []
            if msg.content:
                parts.append({"text": msg.content})
            
            # Add images if present
            if msg.role == "user" and msg.images:
                for img_data in msg.images:
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_data
                        }
                    })
            
            api_messages.append({
                "role": role,
                "parts": parts
            })
        
        return api_messages
    
    def send_message(
        self,
        messages: List[ChatMessage],
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> str:
        """Send messages to Gemini and get response"""
        if not self.client or getattr(self, '_client_api_key', None) != self.api_key:
            if GEMINI_AVAILABLE and self.api_key:
                self.client = genai.Client(api_key=self.api_key)
                self._client_api_key = self.api_key
            else:
                raise ValueError("Gemini client not initialized. Check API key.")
        
        api_messages = self._convert_messages_to_api_format(messages)
        
        # Build generation config
        gen_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p < 1.0:
            gen_config["top_p"] = top_p
        if top_k > 0:
            gen_config["top_k"] = top_k
        
        try:
            if on_chunk:
                # Streaming response
                full_response = ""
                response = self.client.models.generate_content_stream(
                    model="gemini-2.5-flash",  # Will be overridden
                    contents=api_messages,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt if system_prompt else None,
                        **gen_config
                    )
                )
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        on_chunk(chunk.text)
                return full_response
            else:
                # Non-streaming response
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=api_messages,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt if system_prompt else None,
                        **gen_config
                    )
                )
                return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models from API"""
        if self._cached_models:
            return self._cached_models.copy()
        
        if not self.client:
            return self.DEFAULT_MODELS.copy()
        
        try:
            # Fetch models from Gemini API
            models = []
            for m in self.client.models.list():
                for action in m.supported_actions:
                    if action == "generateContent":
                        # Extract just the model name (remove 'models/' prefix if present)
                        model_name = m.name
                        if model_name.startswith("models/"):
                            model_name = model_name[7:]
                        models.append(model_name)
                        break
            
            if models:
                # Sort to put newer/preferred models first
                models.sort(reverse=True)
                self._cached_models = models
                return models
        except Exception as e:
            print(f"Could not fetch models from Gemini API: {e}")
        
        return self.DEFAULT_MODELS.copy()
    
    def send_message_stream(
        self,
        messages: List[Dict],
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        images: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Iterator[str]:
        """Send messages and stream the response"""
        if not self.client or getattr(self, '_client_api_key', None) != self.api_key:
            # Reinitialize client if API key was updated
            if GEMINI_AVAILABLE and self.api_key:
                self.client = genai.Client(api_key=self.api_key)
                self._client_api_key = self.api_key
            else:
                raise ValueError("Gemini client not initialized. Check API key.")
        
        # Use provided model or default
        model_name = model or "gemini-2.5-flash"
        
        # Convert messages format
        api_contents = []
        for msg in messages:
            if isinstance(msg, dict):
                role = "model" if msg.get('role') == 'assistant' else "user"
                content = msg.get('content', '')
                parts = [{"text": content}] if isinstance(content, str) else content
                api_contents.append({"role": role, "parts": parts})
            elif isinstance(msg, ChatMessage):
                role = "model" if msg.role == "assistant" else "user"
                api_contents.append({"role": role, "parts": [{"text": msg.content}]})
        
        # Add images to the last user message if provided
        if images and api_contents:
            for i in range(len(api_contents) - 1, -1, -1):
                if api_contents[i].get('role') == 'user':
                    for img_data in images:
                        api_contents[i]['parts'].append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_data
                            }
                        })
                    break
        
        # Build generation config
        gen_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p < 1.0:
            gen_config["top_p"] = top_p
        if top_k > 0:
            gen_config["top_k"] = top_k
        
        try:
            # Ensure model name has correct format
            if not model_name.startswith("models/"):
                full_model_name = f"models/{model_name}"
            else:
                full_model_name = model_name
            
            response = self.client.models.generate_content_stream(
                model=full_model_name,
                contents=api_contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt if system_prompt else None,
                    **gen_config
                )
            )
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                elif hasattr(chunk, 'parts'):
                    for part in chunk.parts:
                        if hasattr(part, 'text') and part.text:
                            yield part.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
    
    def encode_image_for_api(self, image, max_size: int = 512) -> Optional[str]:
        """Encode a PIL Image for the API"""
        if not PIL_AVAILABLE:
            return None
        
        try:
            from io import BytesIO
            
            # Ensure RGB mode
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            
            # Resize if needed
            if image.size[0] > max_size or image.size[1] > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Encode to base64
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def test_connection(self) -> tuple[bool, str]:
        """Test API connection with a simple request"""
        if not GEMINI_AVAILABLE:
            return False, "Google GenAI library not installed"
        
        if not self.api_key:
            return False, "API key not configured"
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hi"
            )
            return True, "Connection successful!"
        except Exception as e:
            return False, f"Error: {str(e)}"


# =============================================================================
# Ollama Client Implementation
# =============================================================================

class OllamaClient(LLMClient):
    """Ollama local LLM client"""
    
    DEFAULT_MODELS = [
        "llama3.2",
        "llama3.1",
        "mistral",
        "codellama",
        "phi3",
    ]
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.api_key = ""  # Not used but kept for interface compatibility
        self._cached_models = None
        
        # Set the host for ollama library if available
        if OLLAMA_AVAILABLE:
            try:
                # Set custom host if not default
                if base_url != "http://localhost:11434":
                    os.environ['OLLAMA_HOST'] = base_url
            except:
                pass
    
    def _convert_messages_to_api_format(self, messages: List[ChatMessage]) -> List[Dict]:
        """Convert ChatMessage objects to Ollama API format"""
        api_messages = []
        
        for msg in messages:
            message_dict = {
                "role": msg.role,
                "content": msg.content
            }
            
            # Add images if present (Ollama supports images for vision models)
            if msg.role == "user" and msg.images:
                message_dict["images"] = msg.images
            
            api_messages.append(message_dict)
        
        return api_messages
    
    def send_message(
        self,
        messages: List[ChatMessage],
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> str:
        """Send messages to Ollama and get response"""
        if not OLLAMA_AVAILABLE:
            raise ValueError("Ollama library not installed. Install with: pip install ollama")
        
        api_messages = self._convert_messages_to_api_format(messages)
        
        # Add system message if provided
        if system_prompt:
            api_messages.insert(0, {"role": "system", "content": system_prompt})
        
        try:
            if on_chunk:
                # Streaming response
                full_response = ""
                stream = ollama.chat(
                    model="llama3.2",
                    messages=api_messages,
                    stream=True,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                )
                for chunk in stream:
                    text = chunk.get('message', {}).get('content', '')
                    if text:
                        full_response += text
                        on_chunk(text)
                return full_response
            else:
                # Non-streaming response
                response = ollama.chat(
                    model="llama3.2",
                    messages=api_messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                )
                return response.get('message', {}).get('content', '')
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models from local server"""
        if self._cached_models:
            return self._cached_models.copy()
        
        if not OLLAMA_AVAILABLE:
            return self.DEFAULT_MODELS.copy()
        
        try:
            # Fetch models from local Ollama server
            models_response = ollama.list()
            models = []
            
            # Handle different response formats
            if hasattr(models_response, 'models'):
                for model in models_response.models:
                    model_name = model.model if hasattr(model, 'model') else str(model)
                    # Filter out embedding models
                    if 'embed' not in model_name.lower():
                        models.append(model_name)
            elif isinstance(models_response, dict) and 'models' in models_response:
                for model in models_response['models']:
                    model_name = model.get('name', model.get('model', str(model)))
                    if 'embed' not in model_name.lower():
                        models.append(model_name)
            
            if models:
                models.sort()
                self._cached_models = models
                return models
        except Exception as e:
            print(f"Could not fetch models from Ollama: {e}")
        
        return self.DEFAULT_MODELS.copy()
    
    def send_message_stream(
        self,
        messages: List[Dict],
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        images: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Iterator[str]:
        """Send messages and stream the response"""
        if not OLLAMA_AVAILABLE:
            raise ValueError("Ollama library not installed. Install with: pip install ollama")
        
        # Use provided model or default
        model_name = model or "llama3.2"
        
        # Convert messages format
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            if isinstance(msg, dict):
                api_messages.append(msg)
            elif isinstance(msg, ChatMessage):
                api_messages.append({"role": msg.role, "content": msg.content})
        
        # Add images to the last user message if provided
        if images and api_messages:
            for i in range(len(api_messages) - 1, -1, -1):
                if api_messages[i].get('role') == 'user':
                    api_messages[i]['images'] = images
                    break
        
        # Build options
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if top_p < 1.0:
            options["top_p"] = top_p
        if top_k > 0:
            options["top_k"] = top_k
        
        try:
            stream = ollama.chat(
                model=model_name,
                messages=api_messages,
                stream=True,
                options=options
            )
            for chunk in stream:
                text = chunk.get('message', {}).get('content', '')
                if text:
                    yield text
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")
    
    def encode_image_for_api(self, image, max_size: int = 512) -> Optional[str]:
        """Encode a PIL Image for the API"""
        if not PIL_AVAILABLE:
            return None
        
        try:
            from io import BytesIO
            
            # Ensure RGB mode
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            
            # Resize if needed
            if image.size[0] > max_size or image.size[1] > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Encode to base64
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def test_connection(self) -> tuple[bool, str]:
        """Test connection to Ollama server"""
        if not OLLAMA_AVAILABLE:
            return False, "Ollama library not installed. Install with: pip install ollama"
        
        try:
            # Try to list models as a connection test
            models = ollama.list()
            model_count = 0
            if hasattr(models, 'models'):
                model_count = len(models.models)
            elif isinstance(models, dict) and 'models' in models:
                model_count = len(models['models'])
            
            if model_count > 0:
                return True, f"Connected! Found {model_count} model(s)."
            else:
                return True, "Connected, but no models installed. Run 'ollama pull <model>' to download models."
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                return False, "Could not connect to Ollama. Is it running? Start with 'ollama serve'"
            return False, f"Error: {error_msg}"


# =============================================================================
# DeepSeek Client
# =============================================================================

class DeepSeekClient(LLMClient):
    """Client for DeepSeek API (using OpenAI-compatible interface)"""
    
    DEFAULT_MODELS = [
        "deepseek-chat",
        "deepseek-reasoner",
    ]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._cached_models = []
        self._client_api_key = None
        
        if OPENAI_AVAILABLE and api_key:
            try:
                self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                self._client_api_key = api_key
            except Exception as e:
                print(f"Error initializing DeepSeek client: {e}")
    
    def _convert_messages_to_api_format(self, messages: List[ChatMessage]) -> List[Dict]:
        """Convert ChatMessage objects to DeepSeek API format"""
        api_messages = []
        for msg in messages:
            content = []
            
            # Add text content
            if msg.content:
                content.append({
                    "type": "text",
                    "text": msg.content
                })
            
            # Add images if present (DeepSeek supports vision)
            if msg.images:
                for img_b64 in msg.images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    })
            
            api_messages.append({
                "role": msg.role,
                "content": content if len(content) > 1 else msg.content
            })
        
        return api_messages
    
    def send_message(
        self,
        messages: List[ChatMessage],
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> str:
        """Send a message and get a response"""
        if not self.client or getattr(self, '_client_api_key', None) != self.api_key:
            if OPENAI_AVAILABLE and self.api_key:
                self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
                self._client_api_key = self.api_key
            else:
                raise ValueError("DeepSeek client not initialized. Check API key.")
        
        api_messages = self._convert_messages_to_api_format(messages)
        
        # Add system prompt as first message if provided
        if system_prompt:
            api_messages.insert(0, {"role": "system", "content": system_prompt})
        
        try:
            if on_chunk:
                # Streaming response
                full_response = ""
                stream = self.client.chat.completions.create(
                    model=self.settings.model if hasattr(self, 'settings') else "deepseek-chat",
                    messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        full_response += text
                        on_chunk(text)
                return full_response
            else:
                # Non-streaming response
                response = self.client.chat.completions.create(
                    model=self.settings.model if hasattr(self, 'settings') else "deepseek-chat",
                    messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                raise ValueError("Invalid API key. Please check your DeepSeek API key.")
            elif "rate_limit" in error_msg.lower():
                raise RuntimeError("Rate limit exceeded. Please wait before sending more messages.")
            else:
                raise RuntimeError(f"DeepSeek API error: {error_msg}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available DeepSeek models"""
        # DeepSeek doesn't provide a models endpoint, return static list
        return self.DEFAULT_MODELS.copy()
    
    def send_message_stream(
        self,
        messages: List[Dict],
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        images: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Iterator[str]:
        """Send messages and stream the response"""
        if not self.client or getattr(self, '_client_api_key', None) != self.api_key:
            # Reinitialize client if API key was updated
            if OPENAI_AVAILABLE and self.api_key:
                self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
                self._client_api_key = self.api_key
            else:
                raise ValueError("DeepSeek client not initialized.")
        
        model_name = model or "deepseek-chat"
        
        # Build messages
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, list):
                content = "\n".join(
                    part.get('text', '') for part in content
                    if isinstance(part, dict) and part.get('type') == 'text'
                )
            api_messages.append({
                'role': msg.get('role', 'user'),
                'content': str(content or '')
            })
        
        try:
            safe_max_tokens = max(1, min(int(max_tokens or 4096), 8192))
            stream = self.client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                max_tokens=safe_max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise RuntimeError(f"DeepSeek error: {str(e)}")
    
    def test_connection(self) -> tuple[bool, str]:
        """Test connection to DeepSeek API"""
        if not OPENAI_AVAILABLE:
            return False, "OpenAI library not installed. Install with: pip install openai"
        
        if not self.api_key:
            return False, "No API key provided."
        
        try:
            # Make a minimal test request
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True, "Connected successfully!"
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                return False, "Invalid API key."
            else:
                return False, f"Error: {error_msg}"


# =============================================================================
# ConPTY / PTY helpers for CliClient
# =============================================================================

if sys.platform == 'win32':
    import ctypes
    import ctypes.wintypes as _wt

    class _WinPTYStream:
        """File-like read/write wrapper around a Win32 pipe HANDLE.

        Intentionally chunk-based: read_chunk() returns whatever bytes
        arrive in a single ReadFile call, decoded to str.  This avoids
        blocking on \\n, which TUI tools (e.g. gh copilot chat) never
        emit for cursor-positioned response text.
        """

        _READ_BUF = 4096  # bytes per ReadFile call

        def __init__(self, k32, handle: '_wt.HANDLE', mode: str):
            self._k32    = k32
            self._handle = handle
            self._mode   = mode  # 'r' or 'w'

        def write(self, text: str) -> int:
            data    = text.encode('utf-8')
            written = _wt.DWORD(0)
            self._k32.WriteFile(self._handle, data, len(data),
                                ctypes.byref(written), None)
            return written.value

        def flush(self):
            pass  # WriteFile to a pipe is synchronous

        def read_chunk(self) -> str:
            """Block until data arrives; decode and return as str.
            Returns '' on EOF or error.
            """
            buf   = ctypes.create_string_buffer(self._READ_BUF)
            nread = _wt.DWORD(0)
            ok = self._k32.ReadFile(self._handle, buf, self._READ_BUF,
                                    ctypes.byref(nread), None)
            if not ok or nread.value == 0:
                return ''
            return buf.raw[:nread.value].decode('utf-8', errors='replace')

    class _WinPTY:
        """Minimal Windows ConPTY wrapper using ctypes only.

        Requires Windows 10 version 1809 (build 17763) or later.
        Gives the child process a real pseudo-terminal so that interactive
        TUI tools (e.g. 'gh copilot chat') see isatty(stdout)==True and
        remain line-buffered / streaming.
        """

        def __init__(self, command: str, cols: int = 220, rows: int = 50):
            import os as _os
            k32 = ctypes.WinDLL('kernel32', use_last_error=True)

            # ── Structures ────────────────────────────────────────────────
            class _STARTUPINFOW(ctypes.Structure):
                _fields_ = [
                    ('cb',             _wt.DWORD ), ('lpReserved',    _wt.LPWSTR),
                    ('lpDesktop',      _wt.LPWSTR), ('lpTitle',       _wt.LPWSTR),
                    ('dwX',            _wt.DWORD ), ('dwY',           _wt.DWORD ),
                    ('dwXSize',        _wt.DWORD ), ('dwYSize',       _wt.DWORD ),
                    ('dwXCountChars',  _wt.DWORD ), ('dwYCountChars', _wt.DWORD ),
                    ('dwFillAttribute',_wt.DWORD ), ('dwFlags',       _wt.DWORD ),
                    ('wShowWindow',    _wt.WORD  ), ('cbReserved2',   _wt.WORD  ),
                    ('lpReserved2',    _wt.LPBYTE), ('hStdInput',     _wt.HANDLE),
                    ('hStdOutput',     _wt.HANDLE), ('hStdError',     _wt.HANDLE),
                ]

            class _STARTUPINFOEXW(ctypes.Structure):
                _fields_ = [
                    ('StartupInfo',    _STARTUPINFOW),
                    ('lpAttributeList', ctypes.c_void_p),
                ]

            class _PROCESS_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ('hProcess',    _wt.HANDLE), ('hThread',    _wt.HANDLE),
                    ('dwProcessId', _wt.DWORD ), ('dwThreadId', _wt.DWORD ),
                ]

            # ── Create two pipe pairs ─────────────────────────────────────
            # stdin  path: pty_in_write → [pipe] → pty_in_read  (PTY reads)
            # stdout path: pty_out_write → [PTY writes] → pty_out_read (we read)
            pty_in_read   = _wt.HANDLE()
            pty_in_write  = _wt.HANDLE()
            pty_out_read  = _wt.HANDLE()
            pty_out_write = _wt.HANDLE()
            if not k32.CreatePipe(ctypes.byref(pty_in_read),
                                  ctypes.byref(pty_in_write), None, 0):
                raise OSError(ctypes.get_last_error(), 'CreatePipe(stdin) failed')
            if not k32.CreatePipe(ctypes.byref(pty_out_read),
                                  ctypes.byref(pty_out_write), None, 0):
                raise OSError(ctypes.get_last_error(), 'CreatePipe(stdout) failed')

            # ── CreatePseudoConsole ───────────────────────────────────────
            # COORD is packed as c_uint32: low 16 bits = X (cols), high 16 = Y (rows)
            k32.CreatePseudoConsole.argtypes = [
                ctypes.c_uint32,                          # COORD (by value)
                _wt.HANDLE, _wt.HANDLE, _wt.DWORD,        # hInput, hOutput, dwFlags
                ctypes.POINTER(_wt.HANDLE),                # phPC
            ]
            k32.CreatePseudoConsole.restype = ctypes.HRESULT
            hpc = _wt.HANDLE()
            size_coord = ctypes.c_uint32((rows << 16) | (cols & 0xFFFF))
            hr = k32.CreatePseudoConsole(
                size_coord, pty_in_read, pty_out_write, 0, ctypes.byref(hpc))
            if hr != 0:
                raise OSError(f'CreatePseudoConsole failed: hr={hr:#010x}')

            # Close the ends now owned by the PTY (must happen BEFORE CreateProcess)
            k32.CloseHandle(pty_in_read)
            k32.CloseHandle(pty_out_write)

            # ── PROC_THREAD_ATTRIBUTE_LIST ────────────────────────────────
            attr_size = ctypes.c_size_t(0)
            k32.InitializeProcThreadAttributeList(None, 1, 0,
                                                  ctypes.byref(attr_size))
            attr_buf = ctypes.create_string_buffer(attr_size.value)
            k32.InitializeProcThreadAttributeList(attr_buf, 1, 0,
                                                  ctypes.byref(attr_size))
            PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE = 0x00020016
            k32.UpdateProcThreadAttribute(
                attr_buf, 0,
                PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
                hpc,                          # lpValue = HPCON handle
                ctypes.sizeof(_wt.HANDLE),    # cbSize
                None, None,
            )

            # ── STARTUPINFOEXW ────────────────────────────────────────────
            si_ex = _STARTUPINFOEXW()
            si_ex.StartupInfo.cb = ctypes.sizeof(_STARTUPINFOEXW)
            si_ex.lpAttributeList = ctypes.cast(attr_buf, ctypes.c_void_p)

            # ── CreateProcessW ────────────────────────────────────────────
            EXTENDED_STARTUPINFO_PRESENT = 0x00080000
            CREATE_UNICODE_ENVIRONMENT   = 0x00000400
            env_dict = _os.environ.copy()
            env_dict['NO_COLOR'] = '1'
            env_dict['TERM']     = 'dumb'
            env_block = '\x00'.join(f'{k}={v}' for k, v in env_dict.items()) + '\x00\x00'
            env_wchar = ctypes.create_unicode_buffer(env_block)

            # Wrap in cmd.exe /c so that PATH resolution works for bare
            # commands like 'copilot' (CreateProcessW doesn't search PATH).
            cmd_line = f'cmd.exe /c {command}'

            pi = _PROCESS_INFORMATION()
            ok = k32.CreateProcessW(
                None, cmd_line,
                None, None,
                False,
                EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
                env_wchar, None,
                ctypes.byref(si_ex),
                ctypes.byref(pi),
            )
            k32.DeleteProcThreadAttributeList(attr_buf)
            if not ok:
                raise OSError(ctypes.get_last_error(),
                              f'CreateProcessW failed for: {command}')
            k32.CloseHandle(pi.hThread)

            self._k32       = k32
            self._hpc       = hpc
            self._hprocess  = pi.hProcess
            self._pid       = pi.dwProcessId
            self._in_write  = pty_in_write
            self._out_read  = pty_out_read
            self.stdin  = _WinPTYStream(k32, pty_in_write,  'w')
            self.stdout = _WinPTYStream(k32, pty_out_read,  'r')

        def poll(self):
            STILL_ACTIVE = 259
            ec = _wt.DWORD()
            if self._k32.GetExitCodeProcess(self._hprocess, ctypes.byref(ec)):
                return None if ec.value == STILL_ACTIVE else ec.value
            return 0

        def terminate(self):
            try:
                self._k32.TerminateProcess(self._hprocess, 1)
            except Exception:
                pass

        def wait(self, timeout=None):
            ms = int(timeout * 1000) if timeout is not None else 0xFFFFFFFF
            self._k32.WaitForSingleObject(self._hprocess, ms)

        def close(self):
            """Release all ConPTY and pipe handles."""
            for attr, closer in (
                ('_hpc',      self._k32.ClosePseudoConsole),
                ('_in_write', self._k32.CloseHandle),
                ('_out_read', self._k32.CloseHandle),
                ('_hprocess', self._k32.CloseHandle),
            ):
                h = getattr(self, attr, None)
                if h:
                    try:
                        closer(h)
                    except Exception:
                        pass


# =============================================================================
# Antigravity CLI (agy) Provider Client & Helpers
# =============================================================================

@lru_cache(maxsize=1)
def find_antigravity_cli_command() -> str:
    candidates = []

    # 1. Environment Variable Overrides
    env_path = os.getenv("ANTIGRAVITY_CLI_PATH", os.getenv("AGY_PATH", "")).strip()
    if env_path:
        candidates.append(env_path)

    # 2. Standard Global Paths (Windows & Cross-Platform)
    home = Path.home()
    local_agy = home / 'AppData' / 'Local' / 'agy' / 'bin' / 'agy.exe'
    npm_dir = home / 'AppData' / 'Roaming' / 'npm'
    
    candidates.extend([
        str(local_agy),
        str(npm_dir / 'agy.cmd'),
        str(npm_dir / 'antigravity.cmd'),
        'agy.cmd',
        'antigravity.cmd',
        'agy.exe',
        'antigravity.exe',
        'agy',
        'antigravity'
    ])

    for candidate in candidates:
        if not candidate:
            continue
        # CRITICAL SAFETY GUARD: Exclude deprecated gemini cli executables
        if 'gemini' in candidate.lower():
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)
        resolved = shutil.which(candidate)
        if resolved and 'gemini' not in resolved.lower():
            return resolved

    return ""


def launch_cli_auth_window():
    """Launch an independent terminal console window running agy signin."""
    try:
        cmd_executable = shutil.which("cmd.exe") or "cmd.exe"
        cli_bin = find_antigravity_cli_command() or "agy"
        auth_cmd = f'start "Antigravity CLI Signin" {cmd_executable} /k "{cli_bin} signin"'
        subprocess.Popen(auth_cmd, shell=True)
        return True, "Launched CLI authentication window. Follow prompts in external console."
    except Exception as e:
        return False, f"Failed to launch terminal window: {e}"


_AGY_EFFORT_SUFFIXES = ('-low', '-medium', '-high', '-xhigh', '-max')


def _parse_agy_models_output(stdout_text: str) -> List[str]:
    """Parse stdout from `agy models` (either TSV/text lines or JSON) into model slugs."""
    if not stdout_text or not stdout_text.strip():
        return []
    text = stdout_text.strip()
    # Try JSON first in case future CLI versions emit JSON
    if text.startswith('[') or text.startswith('{'):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(m.get('name', m)).strip() for m in data if m]
            if isinstance(data, dict):
                models = data.get('models', [])
                return [str(m.get('name', m)).strip() for m in models if m]
        except Exception:
            pass

    models: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if (lower.startswith('fetching ')
                or lower.startswith('available models')
                or lower.startswith('usage:')
                or lower.startswith('error:')
                or lower.startswith('flags:')
                or lower.startswith('---')):
            continue
        # `agy models` outputs: <model_slug>\t<Display Name>
        slug = line.split('\t')[0].split()[0].strip()
        if slug and re.match(r'^[a-zA-Z0-9._-]+$', slug):
            if slug not in models:
                models.append(slug)
    return models


def call_antigravity_cli(system_prompt: str, user_message: str, model_name: str = "gemini-3.8-flash-low", effort: str = "low", cli_command: str = "") -> str:
    """Execute non-interactive prompt call via agy CLI."""
    executable = str(cli_command or find_antigravity_cli_command()).strip()
    if not executable:
        raise RuntimeError("Antigravity CLI (agy) executable was not found. Verify installation or set AGY path in Settings.")

    prompt_input = f"{system_prompt.strip()}\n\n{user_message.strip()}".strip() if system_prompt and system_prompt.strip() else user_message.strip()
    cmd = [
        executable,
        '--dangerously-skip-permissions',
        '--output-format', 'json'
    ]

    clean_model = str(model_name or '').strip()
    if clean_model:
        cmd.extend(['--model', clean_model])
        # Only pass --effort if the model slug does not already end with an effort suffix
        if not clean_model.lower().endswith(_AGY_EFFORT_SUFFIXES):
            eff = str(effort or 'low').strip().lower()
            if eff in ('low', 'medium', 'high'):
                cmd.extend(['--effort', eff])
            else:
                cmd.extend(['--effort', 'low'])

    cmd.extend(['-p', prompt_input])
    run_kwargs = {
        'capture_output': True,
        'text': True,
        'encoding': 'utf-8',
        'errors': 'replace',
        'timeout': 180,
    }
    if sys.platform == 'win32':
        run_kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

    try:
        completed = subprocess.run(cmd, **run_kwargs)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Antigravity CLI executable not found: {executable}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Antigravity CLI timed out while generating a response.") from exc

    stdout_text = str(completed.stdout or '').strip()
    stderr_text = str(completed.stderr or '').strip()
    detail = stderr_text or stdout_text or f"exit code {completed.returncode}"

    # Fallback guard 1: if --effort conflicted or is unsupported, retry without --effort
    if completed.returncode != 0 and '--effort' in cmd and any(
        phrase in detail.lower()
        for phrase in ('effort is not supported', 'conflicts with --effort', 'invalid model selection')
    ):
        cmd_no_effort = [c for c in cmd]
        try:
            eff_idx = cmd_no_effort.index('--effort')
            del cmd_no_effort[eff_idx:eff_idx + 2]
        except ValueError:
            pass

        try:
            completed = subprocess.run(cmd_no_effort, **run_kwargs)
            stdout_text = str(completed.stdout or '').strip()
            stderr_text = str(completed.stderr or '').strip()
            detail = stderr_text or stdout_text or f"exit code {completed.returncode}"
        except Exception:
            pass

    # Fallback guard 2: if the model slug is unrecognized/stale, discover live models and retry once
    if completed.returncode != 0 and 'not recognized as a known model' in detail.lower():
        live_models = list_antigravity_models(executable)
        if live_models:
            fallback_model = live_models[0]
            for m in live_models:
                if 'flash-low' in m or 'flash' in m:
                    fallback_model = m
                    break
            retry_cmd = [
                executable,
                '--dangerously-skip-permissions',
                '--output-format', 'json',
                '--model', fallback_model,
                '-p', prompt_input,
            ]
            try:
                completed = subprocess.run(retry_cmd, **run_kwargs)
                stdout_text = str(completed.stdout or '').strip()
                stderr_text = str(completed.stderr or '').strip()
                detail = stderr_text or stdout_text or f"exit code {completed.returncode}"
            except Exception:
                pass

    if completed.returncode != 0:
        raise RuntimeError(f"Antigravity CLI failed: {detail}")

    if not stdout_text:
        raise RuntimeError("Antigravity CLI returned no output.")

    try:
        payload = json.loads(stdout_text)
        response_text = str(payload.get('response', payload.get('text', payload.get('output', '')))).strip()
        if not response_text:
            response_text = stdout_text
    except Exception:
        response_text = stdout_text

    return response_text


def list_antigravity_models(cli_command: str = "", raise_on_error: bool = False) -> List[str]:
    """List models available to the agy CLI session via `agy models`."""
    executable = str(cli_command or find_antigravity_cli_command()).strip()
    fallback = list(AISettings.AGY_MODELS)
    if not executable:
        if raise_on_error:
            raise RuntimeError("Antigravity CLI (agy) executable was not found.")
        return fallback

    run_kwargs = {
        'capture_output': True,
        'text': True,
        'encoding': 'utf-8',
        'errors': 'replace',
        'timeout': 30,
    }
    if sys.platform == 'win32':
        run_kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

    cmd = [executable, 'models']
    try:
        completed = subprocess.run(cmd, **run_kwargs)
        if completed.returncode == 0 and completed.stdout:
            parsed = _parse_agy_models_output(completed.stdout)
            if parsed:
                return parsed
        if raise_on_error:
            detail = (completed.stderr or completed.stdout or f"exit code {completed.returncode}").strip()
            raise RuntimeError(f"Failed to query AGY models: {detail}")
    except Exception as exc:
        if raise_on_error:
            raise
    return fallback


def test_antigravity_connection(model_name: str = "gemini-3.8-flash-low", effort: str = "low", cli_command: str = "") -> str:
    """Run lightweight test query ('hi') to verify CLI connection."""
    test_sys = "You are a test assistant. Reply with the single word HI."
    test_msg = "hi"
    res = call_antigravity_cli(test_sys, test_msg, model_name=model_name, effort=effort, cli_command=cli_command)
    return str(res or '').strip()


class AgyClient(LLMClient):
    """LLM client using Google Antigravity CLI (agy) backend."""

    def __init__(self, model: str = "gemini-3.8-flash-low", effort: str = "low", cli_command: str = ""):
        self.model = model or "gemini-3.8-flash-low"
        self.effort = effort or "low"
        self.cli_command = cli_command or find_antigravity_cli_command()

    def send_message(self, messages, system_prompt='', max_tokens=4096,
                     temperature=0.7, on_chunk=None, **kwargs) -> str:
        user_parts = []
        for msg in messages:
            role = msg.get('role', 'user') if isinstance(msg, dict) else getattr(msg, 'role', 'user')
            content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
            if role == 'system':
                system_prompt += f"\n{content}"
            else:
                user_parts.append(f"{role.capitalize()}: {content}")

        user_message = "\n".join(user_parts) if user_parts else ""
        model = kwargs.get('model') or self.model
        effort = kwargs.get('effort') or self.effort

        response = call_antigravity_cli(
            system_prompt=system_prompt,
            user_message=user_message,
            model_name=model,
            effort=effort,
            cli_command=self.cli_command
        )
        if on_chunk:
            on_chunk(response)
        return response

    def send_message_stream(self, messages, system_prompt='', max_tokens=4096,
                            temperature=0.7, top_p=1.0, top_k=0,
                            images=None, model=None, on_chunk=None,
                            **kwargs):
        res = self.send_message(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            on_chunk=on_chunk,
            model=model,
            **kwargs
        )
        yield res

    def get_available_models(self) -> List[str]:
        return list_antigravity_models(self.cli_command)

    def test_connection(self) -> tuple:
        try:
            models = list_antigravity_models(self.cli_command, raise_on_error=True)
            if models and self.model not in models:
                # Check if f"{self.model}-{self.effort}" matches a discovered model
                candidate = f"{self.model}-{self.effort}"
                if candidate in models:
                    self.model = candidate
                else:
                    self.model = models[0]
            res = test_antigravity_connection(
                model_name=self.model,
                effort=self.effort,
                cli_command=self.cli_command
            )
            return True, f"AGY CLI connection successful (model: {self.model})! Output: '{res}'"
        except Exception as e:
            return False, f"AGY CLI connection failed: {e}"


# =============================================================================
# CLI Provider Client
# =============================================================================

class CliClient(LLMClient):
    """LLM client that drives a persistent interactive CLI process.

    The CLI command (e.g. 'gh copilot chat', 'claude', 'gemini') is launched
    once as a subprocess and kept alive for the entire app session.  Each user
    message is written to its stdin; stdout is streamed back until a short
    silence indicates the response is complete.  CLI-native commands such as
    /clear, /model, /models, /help are passed through transparently.
    """

    # After first data arrives, wait this long for more before declaring done.
    SILENCE_TIMEOUT = 3.0
    # Wait up to 30 s for the very first byte after sending a message.
    FIRST_TOKEN_TIMEOUT = 30.0
    # Wait up to 10 s for the startup banner / loading spinner to finish.
    STARTUP_DRAIN_TIMEOUT = 10.0
    # Inter-chunk timeout used inside _drain (smaller than SILENCE_TIMEOUT so
    # the startup drain doesn't take forever between banner chunks).
    DRAIN_INTER_CHUNK_TIMEOUT = 1.0
    # Regex to strip ANSI/VT100/DEC/OSC escape sequences
    _ANSI_RE = re.compile(
        r'\x1b(?:'
        r'\[[0-?]*[ -/]*[@-~]'   # CSI sequences:  ESC [ ... <final>
        r'|[()][AB012]'           # Charset designations
        r'|\][^\x07\x1b]*(?:\x07|\x1b\\)'  # OSC sequences (window title etc.)
        r'|[@-Z\\-_]'            # Two-char sequences (ESC + single byte)
        r'|#[0-9]'               # DEC screen alignment
        r')'
    )
    # Also strip bare \r and non-printable control chars
    _CTRL_RE = re.compile(r'[\x00-\x08\x0b-\x1f\x7f]')

    def __init__(self, command: str):
        self.command = command.strip()
        self._process: Optional['subprocess.Popen'] = None
        self._out_queue: 'queue.Queue[Optional[str]]' = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # LLMClient ABC                                                        #
    # ------------------------------------------------------------------ #

    def send_message(self, messages, system_prompt='', max_tokens=4096,
                     temperature=0.7, on_chunk=None, **kwargs) -> str:
        full = ''
        for chunk in self.send_message_stream(messages, system_prompt=system_prompt, **kwargs):
            full += chunk
        return full

    def send_message_stream(self, messages, system_prompt='', max_tokens=4096,
                            temperature=0.7, top_p=1.0, top_k=0,
                            images=None, model=None, on_chunk=None,
                            **kwargs):
        self._ensure_alive()

        # Extract last user message content (the CLI session owns its own context)
        last = messages[-1] if messages else {}
        content = last.get('content', '') if isinstance(last, dict) else getattr(last, 'content', '')

        print(f"[DEBUG] CLI send_message_stream: sending '{content[:50]}...'")

        # Write to CLI stdin
        self._process.stdin.write(content + '\n')
        self._process.stdin.flush()

        print(f"[DEBUG] CLI message sent, waiting for response...")

        # Stream response until silence; yield each line
        yield from self._iter_response()

    def get_available_models(self) -> List[str]:
        return []  # model managed by the CLI session

    def test_connection(self) -> tuple:
        """Check that the CLI command is reachable via the system shell.

        Uses shell=True so PATH resolution matches what an interactive terminal
        sees — this handles .cmd/.bat wrappers on Windows and shell-PATH
        extensions that wouldn't be visible to a bare subprocess.run([...]) call.
        """
        import subprocess as _sp
        if not self.command:
            return False, "No CLI command configured."
        exe = self.command.split()[0]
        # Shell "not found" fingerprints (case-insensitive check)
        _NOT_FOUND = ('not recognized', 'not found', 'no such file',
                      'command not found', 'cannot find', 'is not recognized')
        for probe in (f'{exe} --version', f'{exe} --help'):
            try:
                r = _sp.run(probe, shell=True, capture_output=True, text=True, timeout=8)
                output = (r.stdout or r.stderr or '').strip()
                low = output.lower()
                is_not_found = any(p in low for p in _NOT_FOUND)
                if not is_not_found and (r.returncode == 0 or output):
                    first_line = output.split('\n')[0].strip() if output else ''
                    return True, f'"{exe}" is available.' + (f'  {first_line}' if first_line else '')
            except Exception:
                continue
        return False, (
            f'"{exe}" could not be verified from this environment. '
            f'Confirm it runs in your terminal first.'
        )

    # ------------------------------------------------------------------ #
    # Session management                                                   #
    # ------------------------------------------------------------------ #

    def _is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _ensure_alive(self):
        if not self._is_alive():
            self._start_session()

    def _start_session(self):
        # Fresh queue for new session
        self._out_queue = queue.Queue()
        if sys.platform == 'win32':
            # Use ConPTY so the child process sees isatty(stdout) == True,
            # which keeps interactive TUI tools (e.g. 'gh copilot chat')
            # line-buffered and streaming rather than fully-buffered/silent.
            try:
                self._process = _WinPTY(self.command)
                print('[DEBUG] CLI session started via ConPTY')
            except OSError as _e:
                print(f'[DEBUG] ConPTY unavailable ({_e}), falling back to Popen')
                import subprocess as _sp
                self._process = _sp.Popen(
                    self.command, shell=True,
                    stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.STDOUT,
                    text=True, bufsize=1,
                )
        else:
            # Non-Windows: use a pty master/slave pair for the same effect.
            import subprocess as _sp
            import pty as _pty
            import os as _os
            import io as _io
            master_fd, slave_fd = _pty.openpty()
            proc = _sp.Popen(
                self.command, shell=True,
                stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                close_fds=True,
                preexec_fn=_os.setsid,
            )
            _os.close(slave_fd)
            _master_raw = _io.open(master_fd, 'r+b', buffering=0, closefd=True)

            class _PTYStream:
                def write(self, text):
                    _master_raw.write(text.encode('utf-8'))
                def flush(self): pass
                def read(self, n=1):
                    try:
                        data = _master_raw.read(n)
                        return data.decode('utf-8', errors='replace') if data else ''
                    except OSError:
                        return ''
            _stream = _PTYStream()
            proc.stdin  = _stream
            proc.stdout = _stream
            proc._pty_master = _master_raw
            self._process = proc
            print('[DEBUG] CLI session started via POSIX pty')

        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name='CliReader'
        )
        self._reader_thread.start()
        # Swallow welcome banner / initial prompt
        self._drain(self.STARTUP_DRAIN_TIMEOUT)

    def close_session(self):
        """Terminate the CLI process cleanly."""
        if self._is_alive():
            try:
                self._process.stdin.write('/exit\n')
                self._process.stdin.flush()
                self._process.wait(timeout=3)
            except Exception:
                pass
            try:
                self._process.terminate()
            except Exception:
                pass
            # Release ConPTY handles (or pty master fd) if applicable
            if hasattr(self._process, 'close'):
                try:
                    self._process.close()
                except Exception:
                    pass
        self._process = None

    # ------------------------------------------------------------------ #
    # I/O helpers                                                          #
    # ------------------------------------------------------------------ #

    def _reader_loop(self):
        """Background: read raw chunks from the PTY and queue them as-is.

        ANSI sequences are preserved here so that _iter_response can feed
        everything through a VT100 screen buffer and reason about the final
        rendered layout rather than the raw byte stream.
        """
        try:
            while True:
                if self._process is None:
                    break
                try:
                    stdout = self._process.stdout
                    if stdout is None:
                        break
                    # ConPTY path: read a full chunk without blocking on \n
                    if hasattr(stdout, 'read_chunk'):
                        raw = stdout.read_chunk()
                    else:
                        raw = stdout.read(1)
                    if not raw:
                        break
                    self._out_queue.put(raw)  # raw (ANSI intact)
                except (AttributeError, ValueError, OSError):
                    break
        finally:
            self._out_queue.put(None)  # EOF sentinel

    def _drain(self, timeout: float) -> str:
        """Collect all pending output up to *timeout* seconds of silence."""
        buf: list = []
        t = timeout
        while True:
            try:
                ch = self._out_queue.get(timeout=t)
                if ch is None:
                    # EOF sentinel — re-queue so _iter_response() can also see it
                    self._out_queue.put(None)
                    break
                buf.append(ch)
                # Use the short drain inter-chunk timeout (not SILENCE_TIMEOUT)
                # so the welcome banner is consumed quickly.
                t = self.DRAIN_INTER_CHUNK_TIMEOUT
            except queue.Empty:
                break
        return self._strip_ansi(''.join(buf))

    # ---- VT100 screen buffer ------------------------------------------ #

    class _VT100Screen:
        """Minimal VT100/ANSI screen emulator.

        Interprets cursor-movement and erase sequences to maintain a 2-D
        character buffer.  After feeding all raw PTY output, get_rows()
        returns the final rendered text lines — correctly positioned, with
        no escape codes.
        """

        def __init__(self, cols: int = 220, rows: int = 50):
            self.cols = cols
            self.rows = rows
            self.buf  = [[' '] * cols for _ in range(rows)]
            self.cx   = 0   # cursor column (0-based)
            self.cy   = 0   # cursor row    (0-based)

        def feed(self, text: str) -> None:  # noqa: C901
            i = 0
            n = len(text)
            while i < n:
                c = text[i]
                if c == '\x1b' and i + 1 < n:
                    nc = text[i + 1]
                    if nc == '[':
                        # CSI sequence: ESC [ <params> <cmd>
                        j = i + 2
                        while j < n and (text[j].isdigit() or text[j] in ';?'):
                            j += 1
                        if j < n:
                            self._csi(text[j], text[i + 2:j])
                            i = j + 1
                            continue
                    elif nc == ']':
                        # OSC: skip to BEL or ST
                        j = i + 2
                        while j < n and text[j] != '\x07':
                            if text[j] == '\x1b' and j + 1 < n and text[j+1] == '\\':
                                j += 2
                                break
                            j += 1
                        else:
                            j += 1
                        i = j
                        continue
                    else:
                        i += 2
                        continue
                elif c == '\r':
                    self.cx = 0
                elif c == '\n':
                    self.cy = min(self.cy + 1, self.rows - 1)
                elif c == '\b':
                    self.cx = max(self.cx - 1, 0)
                elif ord(c) >= 32:
                    if 0 <= self.cy < self.rows and 0 <= self.cx < self.cols:
                        self.buf[self.cy][self.cx] = c
                    self.cx += 1
                    if self.cx >= self.cols:
                        self.cx = 0
                        self.cy = min(self.cy + 1, self.rows - 1)
                i += 1

        def _csi(self, cmd: str, params: str) -> None:
            raw = params.replace('?', '')
            nums: list = []
            for p in raw.split(';'):
                try:
                    nums.append(int(p))
                except ValueError:
                    nums.append(0)

            def _n(idx: int, default: int = 1) -> int:
                return nums[idx] if idx < len(nums) and nums[idx] else default

            if cmd in ('H', 'f'):
                row = (_n(0, 1)) - 1
                col = (_n(1, 1)) - 1
                self.cy = max(0, min(row, self.rows - 1))
                self.cx = max(0, min(col, self.cols - 1))
            elif cmd == 'A':
                self.cy = max(0, self.cy - _n(0))
            elif cmd == 'B':
                self.cy = min(self.rows - 1, self.cy + _n(0))
            elif cmd == 'C':
                self.cx = min(self.cols - 1, self.cx + _n(0))
            elif cmd == 'D':
                self.cx = max(0, self.cx - _n(0))
            elif cmd == 'G':
                self.cx = max(0, min(_n(0) - 1, self.cols - 1))
            elif cmd == 'd':
                self.cy = max(0, min(_n(0) - 1, self.rows - 1))
            elif cmd == 'J':
                mode = nums[0] if nums else 0
                if mode in (2, 3):
                    self.buf = [[' '] * self.cols for _ in range(self.rows)]
                    self.cx = self.cy = 0
                elif mode == 0:
                    for col in range(self.cx, self.cols):
                        self.buf[self.cy][col] = ' '
                    for row in range(self.cy + 1, self.rows):
                        self.buf[row] = [' '] * self.cols
                elif mode == 1:
                    for col in range(0, self.cx + 1):
                        self.buf[self.cy][col] = ' '
                    for row in range(0, self.cy):
                        self.buf[row] = [' '] * self.cols
            elif cmd == 'K':
                mode = nums[0] if nums else 0
                if mode == 0:
                    for col in range(self.cx, self.cols):
                        self.buf[self.cy][col] = ' '
                elif mode == 1:
                    for col in range(0, self.cx + 1):
                        self.buf[self.cy][col] = ' '
                elif mode == 2:
                    self.buf[self.cy] = [' '] * self.cols
            elif cmd == 'P':
                m = _n(0)
                row = self.buf[self.cy]
                del row[self.cx:self.cx + m]
                row.extend([' '] * m)
            elif cmd == 'L':
                for _ in range(_n(0)):
                    self.buf.insert(self.cy, [' '] * self.cols)
                    self.buf.pop()
            elif cmd == 'M':
                for _ in range(_n(0)):
                    self.buf.pop(self.cy)
                    self.buf.append([' '] * self.cols)
            # Ignore: SGR colour (m), cursor visibility (l/h), margins (r), etc.

        def get_rows(self) -> list:
            """Return trailing-space-stripped row strings."""
            return [''.join(row).rstrip() for row in self.buf]

    # ---- Chrome detection on rendered rows ----------------------------- #

    _BOX_RE      = re.compile(r'^[\u2500-\u257f\u2550-\u256c \u00b7\u200b\u2502]*$')
    _HINT_RE     = re.compile(r'shift\+tab|ctrl\+[a-z]|switch mode|run command|'
                              r'tab to select|enter to confirm|remaining req',
                              re.IGNORECASE)
    _PROMPT_RE   = re.compile(r'^\u276f')          # ❯  input echo
    _GITPATH_RE  = re.compile(r'\u2387|\[\u2387')  # ⎇  git branch
    _WINPATH_RE  = re.compile(r'^[A-Z]:\\')
    _MODEL_RE    = re.compile(r'claude-|gpt-|gemini-|copilot', re.IGNORECASE)

    @classmethod
    def _is_chrome_row(cls, row: str) -> bool:
        """Return True if *row* is TUI chrome rather than AI response text."""
        s = row.strip()
        if not s:
            return True
        if cls._BOX_RE.match(s):
            return True
        if cls._HINT_RE.search(s):
            return True
        if cls._PROMPT_RE.match(s):
            return True
        if cls._GITPATH_RE.search(s):
            return True
        if cls._WINPATH_RE.match(s):
            return True
        if cls._MODEL_RE.search(s) and len(s) < 80:
            return True
        return False

    def _iter_response(self):
        """Collect raw PTY output, render incrementally through VT100 screen.

        Timeout strategy:
        - FIRST_TOKEN_TIMEOUT (30 s) is used as long as the rendered screen
          contains only chrome — i.e. keyboard hints, dividers, input echo.
          This keeps us waiting even when the TUI re-renders its chrome frames
          between sending the message and the AI starting to reply.
        - SILENCE_TIMEOUT (3 s) kicks in the moment we detect actual AI content
          rows on the rendered screen.  Once content has appeared, a 3 s gap
          means the response is complete.
        """
        screen = self._VT100Screen(cols=220, rows=50)
        got_content = False
        t = self.FIRST_TOKEN_TIMEOUT
        raw_chunks: list = []

        while True:
            try:
                chunk = self._out_queue.get(timeout=t)
                if chunk is None:
                    print(f'[DEBUG] CLI process exited after {len(raw_chunks)} raw chunks')
                    break
                raw_chunks.append(chunk)
                screen.feed(chunk)

                if not got_content:
                    # Check rendered screen for any non-chrome content
                    rows = screen.get_rows()
                    if any(r.strip() and not self._is_chrome_row(r) for r in rows):
                        got_content = True
                        t = self.SILENCE_TIMEOUT
                        print('[DEBUG] Content detected on screen, switching to short timeout')
                    # else: keep FIRST_TOKEN_TIMEOUT — AI hasn't started responding yet
            except queue.Empty:
                print(f'[DEBUG] Silence after {len(raw_chunks)} raw chunks '
                      f'(got_content={got_content})')
                break

        if not raw_chunks:
            print('[DEBUG] No data from CLI provider')
            return

        # Final render and chrome extraction
        rows = screen.get_rows()

        # Debug: show non-empty rendered rows
        print('[DEBUG] Rendered screen rows (non-empty):')
        for i, r in enumerate(rows):
            if r.strip():
                print(f'  [{i:02d}] {repr(r[:120])}')

        content_lines = [r.strip() for r in rows if not self._is_chrome_row(r)]
        response = '\n'.join(content_lines).strip()

        if response:
            print(f'[DEBUG] Extracted response: {repr(response[:300])}')
            yield response
        else:
            print('[DEBUG] All rows were chrome — yielding stripped fallback')
            fallback = self._strip_ansi(''.join(raw_chunks)).strip()
            if fallback:
                yield fallback

    @classmethod
    def _strip_ansi(cls, text: str) -> str:
        cleaned = cls._ANSI_RE.sub('', text).replace('\r', '')
        return cls._CTRL_RE.sub('', cleaned)


# =============================================================================
# LLM Client Factory
# =============================================================================

def get_llm_client(settings: AISettings) -> Optional[LLMClient]:
    """Factory function to get appropriate LLM client based on settings"""
    provider = settings.provider
    
    if provider in ('agy', 'antigravity'):
        cli_cmd = settings.agy_path or find_antigravity_cli_command()
        model = settings.agy_model or AISettings.AGY_MODELS[0]
        effort = settings.agy_effort or 'low'
        return AgyClient(model=model, effort=effort, cli_command=cli_cmd)

    elif provider == 'anthropic':
        if not ANTHROPIC_AVAILABLE:
            return None
        return AnthropicClient(settings.api_key)
    
    elif provider == 'gemini':
        if not GEMINI_AVAILABLE:
            return None
        return GeminiClient(settings.gemini_api_key)
    
    elif provider == 'ollama':
        if not OLLAMA_AVAILABLE:
            return None
        return OllamaClient(settings.ollama_url)
    
    elif provider == 'deepseek':
        if not OPENAI_AVAILABLE:
            return None
        return DeepSeekClient(settings.deepseek_api_key)

    elif provider == 'cli':
        if not settings.cli_command:
            return None
        return CliClient(settings.cli_command)

    return None


# =============================================================================
# AI Settings Dialog
# =============================================================================

class AISettingsDialog(tk.Toplevel):
    """Dialog for configuring AI settings"""
    
    def __init__(self, parent, settings: AISettings = None, on_save: Optional[Callable] = None):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings if settings else get_ai_settings()
        self.on_save = on_save
        self.result = None
        
        self.title("AI Settings")
        self.geometry("580x640")
        self.resizable(True, True)
        self.minsize(540, 580)
        self.transient(parent)
        self.grab_set()
        
        self._setup_ui()
        self._load_current_settings()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def _create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            label = ttk.Label(tooltip, text=text, background="#ffffe0", 
                             relief="solid", borderwidth=1, padding=5,
                             wraplength=300)
            label.pack()
            widget._tooltip = tooltip
            
            def hide_tooltip(event=None):
                if hasattr(widget, '_tooltip') and widget._tooltip:
                    widget._tooltip.destroy()
                    widget._tooltip = None
            
            widget.bind('<Leave>', hide_tooltip)
            tooltip.bind('<Leave>', hide_tooltip)
            widget.after(5000, hide_tooltip)  # Auto-hide after 5 seconds
        
        widget.bind('<Enter>', show_tooltip)
    
    def _setup_ui(self):
        """Setup the dialog UI with a compact, tabbed layout."""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Notebook with 3 tabs
        nb = ttk.Notebook(main_frame)
        nb.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # ── Tab 1: Connection ─────────────────────────────────────────── #
        t_conn = ttk.Frame(nb, padding=8)
        nb.add(t_conn, text="  Connection  ")

        prov_frame = ttk.LabelFrame(t_conn, text="Provider", padding=8)
        prov_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(prov_frame, text="AI Provider:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.provider_var = tk.StringVar(value="agy")
        provider_combo = ttk.Combobox(
            prov_frame, textvariable=self.provider_var,
            values=["agy", "anthropic", "gemini", "deepseek", "ollama", "cli"],
            state="readonly", width=28)
        provider_combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 0))
        provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)

        # Dynamic credential panel — one sub-frame per provider
        cred_outer = ttk.LabelFrame(t_conn, text="API Keys / Connection", padding=8)
        cred_outer.pack(fill=tk.X, pady=(0, 8))
        self._cred_frames = {}

        # AGY (Google Antigravity CLI)
        f_agy = ttk.Frame(cred_outer)
        f_agy.columnconfigure(1, weight=1)
        
        ttk.Label(f_agy, text="AGY Path:", width=11, anchor=tk.W).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.agy_path_var = tk.StringVar()
        self.agy_path_entry = ttk.Entry(f_agy, textvariable=self.agy_path_var)
        self.agy_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=(6, 0), pady=2)
        
        btn_frame = ttk.Frame(f_agy)
        btn_frame.grid(row=0, column=2, padx=(4, 0), pady=2)
        ttk.Button(btn_frame, text="Auto", command=self._auto_detect_agy_path).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Browse", command=self._browse_agy_path).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Test", command=lambda: self._test_connection('agy')).pack(side=tk.LEFT, padx=1)
        
        row2 = ttk.Frame(f_agy)
        row2.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(4, 0))
        ttk.Label(row2, text="Reasoning Effort:").pack(side=tk.LEFT)
        self.agy_effort_var = tk.StringVar(value='low')
        effort_combo = ttk.Combobox(row2, textvariable=self.agy_effort_var, values=AISettings.AGY_EFFORTS, state="readonly", width=8)
        effort_combo.pack(side=tk.LEFT, padx=(4, 16))
        
        signin_btn = ttk.Button(row2, text="🔑 Launch Signin Console", command=launch_cli_auth_window)
        signin_btn.pack(side=tk.LEFT)
        self._create_tooltip(signin_btn, "Open external terminal window running 'agy signin' for interactive authentication.")
        
        self._cred_frames['agy'] = f_agy

        # Anthropic
        f = ttk.Frame(cred_outer)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="API Key:", width=11, anchor=tk.W).grid(row=0, column=0, sticky=tk.W)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(f, textvariable=self.api_key_var, show="•")
        self.api_key_entry.grid(row=0, column=1, sticky=tk.EW, padx=(6, 0))
        self.show_key_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Show", variable=self.show_key_var,
                        command=self._toggle_key_visibility).grid(row=0, column=2, padx=(4, 0))
        ttk.Button(f, text="Test", width=6,
                   command=lambda: self._test_connection('anthropic')).grid(row=0, column=3, padx=(4, 0))
        self._cred_frames['anthropic'] = f

        # Gemini
        f = ttk.Frame(cred_outer)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="API Key:", width=11, anchor=tk.W).grid(row=0, column=0, sticky=tk.W)
        self.gemini_key_var = tk.StringVar()
        self.gemini_key_entry = ttk.Entry(f, textvariable=self.gemini_key_var, show="•")
        self.gemini_key_entry.grid(row=0, column=1, sticky=tk.EW, padx=(6, 0))
        self.show_gemini_key_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Show", variable=self.show_gemini_key_var,
                        command=self._toggle_gemini_key_visibility).grid(row=0, column=2, padx=(4, 0))
        ttk.Button(f, text="Test", width=6,
                   command=lambda: self._test_connection('gemini')).grid(row=0, column=3, padx=(4, 0))
        self._cred_frames['gemini'] = f

        # DeepSeek
        f = ttk.Frame(cred_outer)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="API Key:", width=11, anchor=tk.W).grid(row=0, column=0, sticky=tk.W)
        self.deepseek_key_var = tk.StringVar()
        self.deepseek_key_entry = ttk.Entry(f, textvariable=self.deepseek_key_var, show="•")
        self.deepseek_key_entry.grid(row=0, column=1, sticky=tk.EW, padx=(6, 0))
        self.show_deepseek_key_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Show", variable=self.show_deepseek_key_var,
                        command=self._toggle_deepseek_key_visibility).grid(row=0, column=2, padx=(4, 0))
        ttk.Button(f, text="Test", width=6,
                   command=lambda: self._test_connection('deepseek')).grid(row=0, column=3, padx=(4, 0))
        self._cred_frames['deepseek'] = f

        # Ollama
        f = ttk.Frame(cred_outer)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="URL:", width=11, anchor=tk.W).grid(row=0, column=0, sticky=tk.W)
        self.ollama_url_var = tk.StringVar(value="http://localhost:11434")
        self.ollama_url_entry = ttk.Entry(f, textvariable=self.ollama_url_var)
        self.ollama_url_entry.grid(row=0, column=1, sticky=tk.EW, padx=(6, 0))
        ttk.Button(f, text="Test", width=6,
                   command=lambda: self._test_connection('ollama')).grid(row=0, column=2, padx=(4, 0))
        self._cred_frames['ollama'] = f

        # CLI
        f = ttk.Frame(cred_outer)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Command:", width=11, anchor=tk.W).grid(row=0, column=0, sticky=tk.W)
        self.cli_cmd_var = tk.StringVar()
        self.cli_cmd_entry = ttk.Entry(f, textvariable=self.cli_cmd_var)
        self.cli_cmd_entry.grid(row=0, column=1, sticky=tk.EW, padx=(6, 0))
        ttk.Button(f, text="Test", width=6,
                   command=lambda: self._test_connection('cli')).grid(row=0, column=2, padx=(4, 0))
        self._create_tooltip(self.cli_cmd_entry,
            "Enter the command that starts an interactive CLI session,\n"
            "e.g.  gh copilot chat  or  claude  or  gemini\n"
            "The session stays alive while the app is open.\n"
            "CLI commands like /clear, /model, /help are passed through.")
        self._cred_frames['cli'] = f

        # ── Tab 2: Parameters ─────────────────────────────────────────── #
        t_params = ttk.Frame(nb, padding=8)
        nb.add(t_params, text="  Parameters  ")

        model_frame = ttk.LabelFrame(t_params, text="Model", padding=8)
        model_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                         values=AISettings.ANTHROPIC_MODELS, width=32)
        self.model_combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 0))
        refresh_btn = ttk.Button(model_frame, text="🔄", width=3, command=self._refresh_models)
        refresh_btn.grid(row=0, column=2, padx=(4, 0))
        self._create_tooltip(refresh_btn,
            "Refresh model list: Fetches the latest available models from the API. "
            "Use this if you have access to new models that aren't showing in the dropdown.")

        params_frame = ttk.LabelFrame(t_params, text="Generation Parameters", padding=8)
        params_frame.pack(fill=tk.X, pady=(0, 8))

        # Max Tokens
        ttk.Label(params_frame, text="Max Tokens:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.max_tokens_var = tk.IntVar(value=4096)
        ttk.Spinbox(params_frame, from_=256, to=8192,
                    textvariable=self.max_tokens_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=(8, 0))
        max_t_tip = ttk.Label(params_frame, text="ℹ️", font=('Segoe UI', 10))
        max_t_tip.grid(row=0, column=2, padx=(4, 0))
        self._create_tooltip(max_t_tip,
            "Maximum number of tokens to generate. Higher = longer responses but may cost more.\n"
            "Common values: 1024 (short), 2048 (medium), 4096 (long), 8192 (very long).")

        # Temperature
        ttk.Label(params_frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, pady=4)
        temp_inner = ttk.Frame(params_frame)
        temp_inner.grid(row=1, column=1, sticky=tk.W, padx=(8, 0))
        self.temp_var = tk.DoubleVar(value=0.7)
        self.temp_scale = ttk.Scale(temp_inner, from_=0.0, to=2.0, variable=self.temp_var,
                                     orient=tk.HORIZONTAL, length=140, command=self._update_temp_label)
        self.temp_scale.pack(side=tk.LEFT)
        self.temp_label = ttk.Label(temp_inner, text="0.70", width=5)
        self.temp_label.pack(side=tk.LEFT, padx=(4, 0))
        temp_tip = ttk.Label(params_frame, text="ℹ️", font=('Segoe UI', 10))
        temp_tip.grid(row=1, column=2, padx=(4, 0))
        self._create_tooltip(temp_tip,
            "Controls randomness. Lower = more focused & deterministic. Higher = more creative.\n"
            "Range 0-2. Recommended: 0.3-0.5 (factual), 0.7-0.9 (creative), 1.0+ (brainstorm).")

        # Top-p
        ttk.Label(params_frame, text="Top-p:").grid(row=2, column=0, sticky=tk.W, pady=4)
        top_p_inner = ttk.Frame(params_frame)
        top_p_inner.grid(row=2, column=1, sticky=tk.W, padx=(8, 0))
        self.top_p_var = tk.DoubleVar(value=1.0)
        self.top_p_scale = ttk.Scale(top_p_inner, from_=0.0, to=1.0, variable=self.top_p_var,
                                      orient=tk.HORIZONTAL, length=140, command=self._update_top_p_label)
        self.top_p_scale.pack(side=tk.LEFT)
        self.top_p_label = ttk.Label(top_p_inner, text="1.00", width=5)
        self.top_p_label.pack(side=tk.LEFT, padx=(4, 0))
        top_p_tip = ttk.Label(params_frame, text="ℹ️", font=('Segoe UI', 10))
        top_p_tip.grid(row=2, column=2, padx=(4, 0))
        self._create_tooltip(top_p_tip,
            "Nucleus sampling: considers only tokens whose cumulative probability exceeds this value.\n"
            "1.0 = all tokens; lower = more focused. Most users should leave at 1.0.")

        # Top-k
        ttk.Label(params_frame, text="Top-k:").grid(row=3, column=0, sticky=tk.W, pady=4)
        self.top_k_var = tk.IntVar(value=0)
        ttk.Spinbox(params_frame, from_=0, to=100,
                    textvariable=self.top_k_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=(8, 0))
        top_k_tip = ttk.Label(params_frame, text="ℹ️", font=('Segoe UI', 10))
        top_k_tip.grid(row=3, column=2, padx=(4, 0))
        self._create_tooltip(top_k_tip,
            "Considers only the top-k most likely tokens. 0 = disabled (model default).\n"
            "Lower = more focused. Leave at 0 for most use cases.")

        # Image Max Size
        ttk.Label(params_frame, text="Image Max Size:").grid(row=4, column=0, sticky=tk.W, pady=4)
        img_inner = ttk.Frame(params_frame)
        img_inner.grid(row=4, column=1, sticky=tk.W, padx=(8, 0))
        self.img_size_var = tk.IntVar(value=512)
        ttk.Spinbox(img_inner, from_=256, to=1024, increment=128,
                    textvariable=self.img_size_var, width=10).pack(side=tk.LEFT)
        ttk.Label(img_inner, text="px").pack(side=tk.LEFT, padx=(4, 0))
        img_tip = ttk.Label(params_frame, text="ℹ️", font=('Segoe UI', 10))
        img_tip.grid(row=4, column=2, padx=(4, 0))
        self._create_tooltip(img_tip,
            "Maximum dimension for images sent to the AI.\n"
            "Larger = more detail but slower and costlier.\n"
            "Recommended: 512px (standard), 768px (detailed), 1024px (maximum).")

        # ── Tab 3: Prompt & Actions ───────────────────────────────────── #
        t_prompt = ttk.Frame(nb, padding=8)
        nb.add(t_prompt, text="  Prompt & Actions  ")

        prompt_frame = ttk.LabelFrame(t_prompt, text="System Prompt", padding=8)
        prompt_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.system_prompt_text = tk.Text(prompt_frame, height=6, wrap=tk.WORD)
        prompt_sb = ttk.Scrollbar(prompt_frame, orient=tk.VERTICAL,
                                   command=self.system_prompt_text.yview)
        self.system_prompt_text.configure(yscrollcommand=prompt_sb.set)
        prompt_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.system_prompt_text.pack(fill=tk.BOTH, expand=True)

        ctx_frame = ttk.LabelFrame(t_prompt, text="Context Menu AI Actions", padding=8)
        ctx_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        ctx_top = ttk.Frame(ctx_frame)
        ctx_top.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(ctx_top, text="Right-click actions for selected text. Use {selection} as placeholder.",
                  font=('Segoe UI', 8), foreground='#666666').pack(side=tk.LEFT)
        ttk.Button(ctx_top, text="+ Add",
                   command=self._add_context_action).pack(side=tk.RIGHT)

        self.ctx_list_frame = ttk.Frame(ctx_frame)
        self.ctx_list_frame.pack(fill=tk.BOTH, expand=True)

        self.ctx_canvas = tk.Canvas(self.ctx_list_frame, height=120, bg='#ffffff', highlightthickness=0)
        ctx_scrollbar = ttk.Scrollbar(self.ctx_list_frame, orient=tk.VERTICAL,
                                       command=self.ctx_canvas.yview)
        self.ctx_inner = ttk.Frame(self.ctx_canvas)
        self.ctx_inner.bind('<Configure>',
                             lambda e: self.ctx_canvas.configure(
                                 scrollregion=self.ctx_canvas.bbox('all')))
        self.ctx_canvas.create_window((0, 0), window=self.ctx_inner, anchor='nw')
        self.ctx_canvas.configure(yscrollcommand=ctx_scrollbar.set)
        self.ctx_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ctx_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.ctx_action_widgets = []

        # ── Buttons (below notebook) ──────────────────────────────────── #
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Discard Changes",
                   command=self.destroy).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(btn_frame, text="Save Changes",
                   command=self._save_settings).pack(side=tk.RIGHT)
    
    def _auto_detect_agy_path(self):
        cmd = find_antigravity_cli_command()
        if cmd:
            self.agy_path_var.set(cmd)
            messagebox.showinfo("Auto-detect", f"Found AGY CLI executable at:\n{cmd}", parent=self)
        else:
            messagebox.showwarning("Auto-detect", "Could not locate agy executable automatically. Please browse manually.", parent=self)

    def _browse_agy_path(self):
        filename = filedialog.askopenfilename(
            title="Select agy executable",
            filetypes=[("Executables", "*.exe *.cmd *.bat"), ("All Files", "*.*")],
            parent=self
        )
        if filename:
            self.agy_path_var.set(filename)

    def _load_current_settings(self):
        """Load current settings into the dialog"""
        self.provider_var.set(self.settings.provider)
        self.api_key_var.set(self.settings.api_key)
        self.gemini_key_var.set(self.settings.gemini_api_key)
        self.deepseek_key_var.set(self.settings.deepseek_api_key)
        self.ollama_url_var.set(self.settings.ollama_url)
        self.cli_cmd_var.set(self.settings.cli_command)
        self.agy_path_var.set(self.settings.agy_path or find_antigravity_cli_command())
        self.agy_effort_var.set(self.settings.agy_effort or 'low')
        self.model_var.set(self.settings.model)
        self.max_tokens_var.set(self.settings.max_tokens)
        self.temp_var.set(self.settings.temperature)
        self.top_p_var.set(self.settings.top_p)
        self.top_k_var.set(self.settings.top_k)
        self.img_size_var.set(self.settings.image_max_size[0])
        self.system_prompt_text.insert(1.0, self.settings.system_prompt)
        self._update_temp_label()
        self._update_top_p_label()
        self._on_provider_change()  # Update model list for current provider
        
        # Hide API key if already set
        if self.settings.api_key:
            self.api_key_entry.config(show="•")
        if self.settings.gemini_api_key:
            self.gemini_key_entry.config(show="•")
        if self.settings.deepseek_api_key:
            self.deepseek_key_entry.config(show="•")
        
        # Load context menu actions
        self._load_context_actions()

    def _load_context_actions(self):
        """Populate context menu actions list from settings"""
        # Clear existing
        for w in self.ctx_action_widgets:
            w['frame'].destroy()
        self.ctx_action_widgets.clear()
        
        actions = self.settings.context_menu_actions
        for action in actions:
            self._add_context_action_row(action.get('name', ''), action.get('prompt', ''),
                                          action.get('enabled', True))

    def _add_context_action(self):
        """Add a new empty context menu action row"""
        self._add_context_action_row('New Action', 'Instruction for AI about {selection}', True)

    def _add_context_action_row(self, name: str, prompt: str, enabled: bool):
        """Add a single context action row to the config UI"""
        row_frame = ttk.Frame(self.ctx_inner)
        row_frame.pack(fill=tk.X, pady=2, padx=2)
        
        enabled_var = tk.BooleanVar(value=enabled)
        ttk.Checkbutton(row_frame, variable=enabled_var).pack(side=tk.LEFT, padx=(0, 3))
        
        name_var = tk.StringVar(value=name)
        name_entry = ttk.Entry(row_frame, textvariable=name_var, width=14)
        name_entry.pack(side=tk.LEFT, padx=(0, 3))
        
        prompt_var = tk.StringVar(value=prompt)
        prompt_entry = ttk.Entry(row_frame, textvariable=prompt_var, width=35)
        prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        
        # Special handling: "Transfer to Chat" has no prompt field
        is_transfer = (name == 'Transfer to Chat')
        if is_transfer:
            prompt_entry.config(state=tk.DISABLED)
        
        remove_btn = ttk.Button(row_frame, text="\u2212", width=2,
                                 command=lambda: self._remove_context_action(row_frame))
        remove_btn.pack(side=tk.RIGHT)
        
        self.ctx_action_widgets.append({
            'frame': row_frame,
            'enabled_var': enabled_var,
            'name_var': name_var,
            'prompt_var': prompt_var,
        })

    def _remove_context_action(self, frame):
        """Remove a context action row"""
        for w in self.ctx_action_widgets:
            if w['frame'] is frame:
                self.ctx_action_widgets.remove(w)
                break
        frame.destroy()
    
    def _toggle_key_visibility(self):
        """Toggle Anthropic API key visibility"""
        if self.show_key_var.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="•")
    
    def _toggle_gemini_key_visibility(self):
        """Toggle Gemini API key visibility"""
        if self.show_gemini_key_var.get():
            self.gemini_key_entry.config(show="")
        else:
            self.gemini_key_entry.config(show="•")
    
    def _toggle_deepseek_key_visibility(self):
        """Toggle DeepSeek API key visibility"""
        if self.show_deepseek_key_var.get():
            self.deepseek_key_entry.config(show="")
        else:
            self.deepseek_key_entry.config(show="•")
    
    def _update_temp_label(self, *args):
        """Update temperature label"""
        self.temp_label.config(text=f"{self.temp_var.get():.2f}")
    
    def _update_top_p_label(self, *args):
        """Update top-p label"""
        self.top_p_label.config(text=f"{self.top_p_var.get():.2f}")
    
    def _on_provider_change(self, event=None):
        """Handle provider change: update credential panel and model list."""
        provider = self.provider_var.get()
        current_model = self.model_var.get()

        # Show only the relevant credential sub-frame
        for p, f in self._cred_frames.items():
            if p == provider:
                f.pack(fill=tk.X)
            else:
                f.pack_forget()

        if provider in ('agy', 'antigravity'):
            cached = self.settings.get('cached_agy_models', [])
            models = cached if cached else AISettings.AGY_MODELS
            self.model_combo['values'] = models
            current_agy = self.settings.agy_model
            self.model_var.set(current_agy if current_agy in models else (models[0] if models else AISettings.AGY_MODELS[0]))
        elif provider == 'cli':
            # CLI manages its own model; disable model selection
            self.model_combo['values'] = []
            self.model_var.set('(managed by CLI)')
            self.model_combo.config(state='disabled')
            return

        self.model_combo.config(state='normal')

        if provider == 'anthropic':
            cached = self.settings.get('cached_anthropic_models', [])
            models = cached if cached else AISettings.ANTHROPIC_MODELS
            self.model_combo['values'] = models
            if current_model not in models:
                self.model_var.set(models[0] if models else '')
        elif provider == 'gemini':
            cached = self.settings.get('cached_gemini_models', [])
            models = cached if cached else AISettings.GEMINI_MODELS
            self.model_combo['values'] = models
            if current_model not in models:
                self.model_var.set(models[0] if models else '')
        elif provider == 'deepseek':
            cached = self.settings.get('cached_deepseek_models', [])
            models = cached if cached else AISettings.DEEPSEEK_MODELS
            self.model_combo['values'] = models
            if current_model not in models:
                self.model_var.set(models[0] if models else '')
        elif provider == 'ollama':
            cached = self.settings.get('cached_ollama_models', [])
            models = cached if cached else AISettings.OLLAMA_MODELS
            self.model_combo['values'] = models
            if current_model not in models:
                self.model_var.set(models[0] if models else '')
    
    def _refresh_models(self):
        """Fetch available models from the API and cache them"""
        provider = self.provider_var.get()
        
        if provider in ('agy', 'antigravity'):
            cli_cmd = self.agy_path_var.get().strip()
            try:
                models = list_antigravity_models(cli_cmd, raise_on_error=True)
                if models:
                    self.model_combo['values'] = models
                    if self.model_var.get().strip() not in models:
                        self.model_var.set(models[0])
                    self.settings.set('cached_agy_models', models)
                    self.settings.save()
                    messagebox.showinfo("Success", f"Found {len(models)} AGY models (cached).", parent=self)
                else:
                    messagebox.showwarning("Warning", "No AGY models found.", parent=self)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to fetch AGY models: {str(e)}", parent=self)

        elif provider == 'anthropic':
            api_key = self.api_key_var.get()
            if not api_key:
                messagebox.showwarning("Warning", "Please enter an Anthropic API key first.", parent=self)
                return
            try:
                client = AnthropicClient(api_key)
                models = client.get_available_models()
                if models:
                    self.model_combo['values'] = models
                    # Cache the models to settings
                    self.settings.set('cached_anthropic_models', models)
                    self.settings.save()
                    messagebox.showinfo("Success", f"Found {len(models)} Anthropic models (cached).", parent=self)
                else:
                    messagebox.showwarning("Warning", "No models found.", parent=self)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to fetch models: {str(e)}", parent=self)
        
        elif provider == 'gemini':
            api_key = self.gemini_key_var.get()
            if not api_key:
                messagebox.showwarning("Warning", "Please enter a Gemini API key first.", parent=self)
                return
            try:
                client = GeminiClient(api_key)
                models = client.get_available_models()
                if models:
                    self.model_combo['values'] = models
                    # Cache the models to settings
                    self.settings.set('cached_gemini_models', models)
                    self.settings.save()
                    messagebox.showinfo("Success", f"Found {len(models)} Gemini models (cached).", parent=self)
                else:
                    messagebox.showwarning("Warning", "No models found.", parent=self)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to fetch models: {str(e)}", parent=self)
        
        elif provider == 'deepseek':
            api_key = self.deepseek_key_var.get()
            if not api_key:
                messagebox.showwarning("Warning", "Please enter a DeepSeek API key first.", parent=self)
                return
            try:
                client = DeepSeekClient(api_key)
                models = client.get_available_models()
                if models:
                    self.model_combo['values'] = models
                    # Cache the models to settings
                    self.settings.set('cached_deepseek_models', models)
                    self.settings.save()
                    messagebox.showinfo("Success", f"Found {len(models)} DeepSeek models (cached).", parent=self)
                else:
                    messagebox.showwarning("Warning", "No models found.", parent=self)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to fetch models: {str(e)}", parent=self)
        
        elif provider == 'ollama':
            ollama_url = self.ollama_url_var.get()
            try:
                client = OllamaClient(ollama_url)
                models = client.get_available_models()
                if models:
                    self.model_combo['values'] = models
                    # Cache the models to settings
                    self.settings.set('cached_ollama_models', models)
                    self.settings.save()
                    messagebox.showinfo("Success", f"Found {len(models)} Ollama models (cached).", parent=self)
                else:
                    messagebox.showwarning("Warning", "No models found. Run 'ollama pull <model>' to download models.", parent=self)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to fetch models: {str(e)}", parent=self)
    
    def _test_connection(self, provider: str = None):
        """Test API connection for specified provider and cache models on success"""
        if provider is None:
            provider = self.provider_var.get()
        
        client = None
        if provider in ('agy', 'antigravity'):
            cli_cmd = self.agy_path_var.get().strip()
            effort = self.agy_effort_var.get().strip() or 'low'
            model = self.model_var.get().strip() or AISettings.AGY_MODELS[0]
            # Discover and validate available AGY models first
            try:
                discovered = list_antigravity_models(cli_cmd, raise_on_error=True)
                if discovered:
                    self.model_combo['values'] = discovered
                    self.settings.set('cached_agy_models', discovered)
                    if model not in discovered:
                        candidate = f"{model}-{effort}"
                        model = candidate if candidate in discovered else discovered[0]
                        self.model_var.set(model)
                    self.settings.agy_model = model
                    self.settings.save()
            except Exception as e:
                messagebox.showerror("Error", f"AGY CLI model check failed: {e}", parent=self)
                return
            client = AgyClient(model=model, effort=effort, cli_command=cli_cmd)
            success, message = client.test_connection()
        elif provider == 'anthropic':
            api_key = self.api_key_var.get()
            if not api_key:
                messagebox.showwarning("Warning", "Please enter an Anthropic API key first.", parent=self)
                return
            client = AnthropicClient(api_key)
            success, message = client.test_connection()
        elif provider == 'gemini':
            api_key = self.gemini_key_var.get()
            if not api_key:
                messagebox.showwarning("Warning", "Please enter a Gemini API key first.", parent=self)
                return
            client = GeminiClient(api_key)
            success, message = client.test_connection()
        elif provider == 'deepseek':
            api_key = self.deepseek_key_var.get()
            if not api_key:
                messagebox.showwarning("Warning", "Please enter a DeepSeek API key first.", parent=self)
                return
            client = DeepSeekClient(api_key)
            success, message = client.test_connection()
        elif provider == 'ollama':
            ollama_url = self.ollama_url_var.get()
            client = OllamaClient(ollama_url)
            success, message = client.test_connection()
        elif provider == 'cli':
            cmd = self.cli_cmd_var.get().strip()
            if not cmd:
                messagebox.showwarning("Warning", "Please enter a CLI command first.", parent=self)
                return
            client = CliClient(cmd)
            success, message = client.test_connection()
            if success:
                messagebox.showinfo("Success", message, parent=self)
            else:
                messagebox.showerror("Error", message, parent=self)
            return
        else:
            messagebox.showwarning("Warning", "Unknown provider.", parent=self)
            return
        
        if success:
            # Also fetch and cache models on successful connection
            try:
                models = client.get_available_models()
                if models:
                    self.model_combo['values'] = models
                    # Cache models based on provider
                    if provider in ('agy', 'antigravity'):
                        self.settings.set('cached_agy_models', models)
                    elif provider == 'anthropic':
                        self.settings.set('cached_anthropic_models', models)
                    elif provider == 'gemini':
                        self.settings.set('cached_gemini_models', models)
                    elif provider == 'ollama':
                        self.settings.set('cached_ollama_models', models)
                    self.settings.save()
                    message += f"\nLoaded {len(models)} models."
            except Exception as e:
                message += f"\n(Could not fetch models: {e})"
            
            messagebox.showinfo("Success", message, parent=self)
        else:
            messagebox.showerror("Error", message, parent=self)
    
    def _save_settings(self):
        """Save settings and close dialog"""
        self.settings.provider = self.provider_var.get()
        self.settings.api_key = self.api_key_var.get()
        self.settings.gemini_api_key = self.gemini_key_var.get()
        self.settings.deepseek_api_key = self.deepseek_key_var.get()
        self.settings.ollama_url = self.ollama_url_var.get()
        self.settings.cli_command = self.cli_cmd_var.get().strip()
        self.settings.agy_path = self.agy_path_var.get().strip()
        self.settings.agy_effort = self.agy_effort_var.get().strip()
        if self.provider_var.get() in ('agy', 'antigravity'):
            self.settings.agy_model = self.model_var.get().strip()

        # Don't save '(managed by CLI)' placeholder as the model
        saved_model = self.model_var.get()
        if saved_model != '(managed by CLI)':
            self.settings.model = saved_model
        self.settings.max_tokens = self.max_tokens_var.get()
        self.settings.temperature = self.temp_var.get()
        self.settings.top_p = self.top_p_var.get()
        self.settings.top_k = self.top_k_var.get()
        self.settings.image_max_size = (self.img_size_var.get(), self.img_size_var.get())
        self.settings.system_prompt = self.system_prompt_text.get(1.0, tk.END).strip()
        
        # Save context menu actions
        actions = []
        for w in self.ctx_action_widgets:
            actions.append({
                'name': w['name_var'].get().strip(),
                'prompt': w['prompt_var'].get().strip(),
                'enabled': w['enabled_var'].get(),
            })
        self.settings.context_menu_actions = actions
        
        if self.settings.save():
            self.result = True
            if self.on_save:
                self.on_save()
            self.destroy()
        else:
            messagebox.showerror("Error", "Failed to save settings.", parent=self)


# =============================================================================
# Singleton instances
# =============================================================================

_ai_settings: Optional[AISettings] = None
_chat_history_manager: Optional[ChatHistoryManager] = None


def get_ai_settings() -> AISettings:
    """Get singleton AISettings instance"""
    global _ai_settings
    if _ai_settings is None:
        _ai_settings = AISettings()
    return _ai_settings


def get_chat_history_manager() -> ChatHistoryManager:
    """Get singleton ChatHistoryManager instance"""
    global _chat_history_manager
    if _chat_history_manager is None:
        _chat_history_manager = ChatHistoryManager()
    return _chat_history_manager


# =============================================================================
# Chat Sidebar Widget
# =============================================================================

CHAT_FONT_MIN, CHAT_FONT_MAX, CHAT_FONT_DEFAULT = 8, 24, 10
_SPINNER_FRAMES = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

# Message separator used by "Edit Chat" mode, e.g.  ──── [user] ────
_HISTORY_DELIM_RE = re.compile(
    r'^\s*[─=\-]{3,}\s*\[\s*(user|assistant|system)\s*\]\s*[─=\-]{3,}\s*$', re.IGNORECASE)


def serialize_history(messages) -> str:
    """Render chat messages as one raw-markdown document for Edit Chat mode."""
    return '\n\n'.join(f'──── [{m.role}] ────\n{m.content}' for m in messages)


def parse_history(text: str, original=None) -> List[ChatMessage]:
    """Parse Edit Chat text back into messages.

    Metadata (timestamp, images) of unchanged messages is preserved by diffing
    against *original*; edited messages keep the metadata of the message they
    replaced; brand-new messages receive a fresh timestamp.
    """
    blocks = []
    role, buf = None, []

    def flush():
        body = '\n'.join(buf).strip('\n').rstrip()
        if body.strip():
            blocks.append((role or 'user', body))

    for line in text.split('\n'):
        m = _HISTORY_DELIM_RE.match(line)
        if m:
            flush()
            role, buf = m.group(1).lower(), []
        else:
            buf.append(line)
    flush()

    orig = list(original or [])
    a = [(m.role, m.content.strip()) for m in orig]
    b = [(r, c.strip()) for r, c in blocks]
    result = [None] * len(blocks)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                result[j1 + k] = orig[i1 + k]
        elif tag == 'replace':
            for k in range(min(i2 - i1, j2 - j1)):
                o = orig[i1 + k]
                result[j1 + k] = ChatMessage(role=blocks[j1 + k][0], content=blocks[j1 + k][1],
                                             images=o.images, timestamp=o.timestamp)
    for j, (r, c) in enumerate(blocks):
        if result[j] is None:
            result[j] = ChatMessage(role=r, content=c)
    return result


def _hex_luminance(hex_color: str) -> float:
    c = (hex_color or '').lstrip('#')
    if len(c) == 6:
        try:
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        except ValueError:
            pass
    return 1.0


def make_chat_palette(theme: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Derive the chat colour palette (light or dark) from the application theme."""
    theme = theme or {}
    bg = theme.get('visual_bg') or '#ffffff'
    dark = _hex_luminance(bg) < 0.5
    if dark:
        pal = dict(
            dark=True, panel='#1e1136', display_bg=bg, text=theme.get('visual_fg') or '#e6e8ee',
            muted='#94a3b8', border='#3b1d6e', input_bg='#2a1a40', input_fg='#f5f3ff',
            hdr_user_bg='#3b1d6e', hdr_ai_bg='#231a38', hdr_sys_bg='#2a1a40',
            acc_user='#c084fc', acc_ai='#a78bfa', acc_sys='#c4b5fd', err='#f87171',
            btn_bg='#3b1d6e', btn_fg='#f5f3ff', btn_hover='#4c1d95',
            accent='#9333ea', accent_fg='#ffffff', accent_hover='#a855f7',
            code_bg='#120b1a', code_fg='#e9d5ff', icode_bg='#2e1065', icode_fg='#f472b6',
            hl_bg='#713f12', hl_fg='#fef9c3', quote_fg='#c4b5fd', quote_bg='#231a38',
            link='#c084fc', table_bg='#231a38')
    else:
        pal = dict(
            dark=False, panel='#f3effa', display_bg=bg, text=theme.get('visual_fg') or '#1a1a1a',
            muted='#6b7280', border='#dcd6e8', input_bg='#ffffff', input_fg='#1a1a1a',
            hdr_user_bg='#ede9fe', hdr_ai_bg='#f5f3ff', hdr_sys_bg='#f3effa',
            acc_user='#5b21b6', acc_ai='#4c1d95', acc_sys='#6b21a8', err='#dc2626',
            btn_bg='#ffffff', btn_fg='#2e1065', btn_hover='#ede9fe',
            accent='#5b21b6', accent_fg='#ffffff', accent_hover='#4c1d95',
            code_bg='#2e1065', code_fg='#faf5ff', icode_bg='#ede9fe', icode_fg='#9333ea',
            hl_bg='#fef08a', hl_fg='#1a1a1a', quote_fg='#4c1d95', quote_bg='#f8f6fc',
            link='#6d28d9', table_bg='#f8f6fc')
    return pal


# Inline markdown for the chat renderer (single line; spans never cross lines)
_CHAT_INLINE_RE = re.compile(
    r'(?P<code>`[^`\n]+`)'
    r'|\*\*\*(?P<bi>[^*\n]+?)\*\*\*'
    r'|\*\*(?!\s)(?P<b>.+?)(?<!\s)\*\*'
    r'|(?<!\w)__(?!\s)(?P<b2>.+?)(?<!\s)__(?!\w)'
    r'|(?<!\*)\*(?![\s*])(?P<i>[^*\n]+?)(?<![\s*])\*(?!\*)'
    r'|(?<!\w)_(?![\s_])(?P<i2>[^_\n]+?)(?<![\s_])_(?!\w)'
    r'|==(?P<hl>[^=\n]+?)=='
    r'|\[(?P<ll>[^\]\n]+)\]\((?P<lu>[^)\s]+)\)'
    r'|(?P<url>https?://[^\s<>)\]]+)'
)
_TABLE_ROW_RE = re.compile(r'^\s*\|.*\|\s*$')
_TABLE_SEP_RE = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$')


class ChatSidebar(tk.Frame):
    """
    Collapsible AI chat sidebar widget.

    Provides:
    - Chat input (Shift+Enter / Ctrl+Enter to send)
    - Options to include the current document and/or chat history
    - Animated progress indicator, Stop button and line-buffered live markdown rendering
    - "Edit Chat" mode for editing the raw markdown history
    - Font size controls and a theme-aware message layout
    - File attachment support
    """

    ROLE_LABELS = {'user': 'You', 'assistant': 'AI', 'system': 'System'}

    def __init__(self, parent, get_document_content_callback: Optional[Callable] = None,
                 get_document_images_callback: Optional[Callable] = None,
                 apply_edit_callback: Optional[Callable] = None):
        self._pal = make_chat_palette(None)
        super().__init__(parent, bg=self._pal['panel'])

        self.get_document_content = get_document_content_callback
        self.get_document_images = get_document_images_callback
        self.apply_edit_callback = apply_edit_callback  # Callable(original, replacement, start_idx, end_idx)

        # Settings and state
        self.settings = get_ai_settings()
        self.history_manager = get_chat_history_manager()
        self.current_document_id: Optional[str] = None
        self.llm_client: Optional[LLMClient] = None
        self.is_generating = False
        self.response_queue = queue.Queue()

        # Pending selection for "Transfer to Chat" (keys: text, start_index, end_index)
        self.pending_selection: Optional[Dict[str, str]] = None

        # Attached files storage: list of dicts with 'path', 'name', 'type', 'content'
        self.attached_files: List[Dict[str, Any]] = []

        # Message model (source of truth for display / edit mode / copy buttons)
        self._messages: List[ChatMessage] = []
        self._link_counter = 0

        # Generation state
        self._gen_id = 0
        self._gen_doc_id: Optional[str] = None
        self._gen_visible = False
        self._gen_t0 = 0.0
        self._cancel = threading.Event()
        self._spin = 0
        self._tick_id = None
        self._pending_active = False
        self._stream_msg: Optional[ChatMessage] = None
        self._stream_buf = ''
        self._stream_text = ''
        self._stream_state: Dict[str, Any] = {}

        # UI state
        self._fs = self.settings.chat_font_size
        self._editing = False
        self._edit_original = ''
        self._themed: list = []   # (widget, {option: palette_key})
        self._font_save_id = None

        # Initialize LLM client
        self._init_llm_client()

        self._setup_ui()
        self._setup_bindings()
        self._configure_tags()
        self._render_all()

        # Start response queue processor
        self._process_queue()

    # ------------------------------------------------------------------ setup

    def _init_llm_client(self):
        """Initialize or reinitialize the LLM client"""
        new_client = get_llm_client(self.settings)
        # For CLI provider: if the command hasn't changed, keep the live session
        if (isinstance(self.llm_client, CliClient)
                and isinstance(new_client, CliClient)
                and self.llm_client.command == new_client.command
                and self.llm_client._is_alive()):
            return
        # Close any existing CLI session before replacing
        if isinstance(self.llm_client, CliClient):
            self.llm_client.close_session()
        self.llm_client = new_client

    def cleanup(self):
        """Release resources (call when the sidebar is destroyed)."""
        try:
            self.commit_edit_mode()
        except Exception:
            pass
        self._cancel.set()
        if isinstance(self.llm_client, CliClient):
            self.llm_client.close_session()

    def _reg(self, widget, **opts):
        """Register a widget whose options track palette keys, and apply them now."""
        self._themed.append((widget, opts))
        self._apply_widget_colors(widget, opts)
        return widget

    def _apply_widget_colors(self, widget, opts):
        try:
            widget.configure(**{k: self._pal[v] for k, v in opts.items()})
        except tk.TclError:
            pass

    def _recolor(self):
        alive = []
        for widget, opts in self._themed:
            try:
                if widget.winfo_exists():
                    self._apply_widget_colors(widget, opts)
                    alive.append((widget, opts))
            except tk.TclError:
                pass
        self._themed = alive

    def _flat_button(self, parent, text, command, width=None, accent=False, font_size=9):
        kw = dict(text=text, command=command, relief=tk.FLAT, bd=0, cursor='hand2',
                  font=(SANS_FONT, font_size), padx=8, pady=3, takefocus=0)
        if width:
            kw['width'] = width
        btn = tk.Button(parent, **kw)
        if accent:
            self._reg(btn, bg='accent', fg='accent_fg', activebackground='accent_hover',
                      activeforeground='accent_fg', disabledforeground='accent_fg')
        else:
            self._reg(btn, bg='btn_bg', fg='btn_fg', activebackground='btn_hover',
                      activeforeground='btn_fg', disabledforeground='muted')
        return btn

    def _setup_ui(self):
        """Setup the sidebar UI"""
        P = self._pal
        self._reg(self, bg='panel')

        # --- Header: title + model badge --------------------------------------
        header = self._reg(tk.Frame(self), bg='panel')
        header.pack(fill=tk.X, padx=8, pady=(8, 2))
        self._reg(tk.Label(header, text="AI Assistant", font=(SANS_FONT, 12, 'bold')),
                  bg='panel', fg='text').pack(side=tk.LEFT)
        self.badge_label = self._reg(tk.Label(header, text="", font=(SANS_FONT, 8), anchor=tk.E),
                                     bg='panel', fg='muted')
        self.badge_label.pack(side=tk.RIGHT)

        # --- Toolbar: Edit Chat, font size, Clear, Config ---------------------
        bar = self._reg(tk.Frame(self), bg='panel')
        bar.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.edit_var = tk.BooleanVar(value=False)
        self.edit_btn = tk.Checkbutton(
            bar, text="Edit Chat", variable=self.edit_var, command=self._on_edit_toggle,
            indicatoron=False, relief=tk.FLAT, bd=0, cursor='hand2', takefocus=0,
            font=(SANS_FONT, 9), padx=8, pady=3)
        self._reg(self.edit_btn, bg='btn_bg', fg='btn_fg', activebackground='btn_hover',
                  activeforeground='btn_fg', selectcolor='accent')
        self.edit_btn.pack(side=tk.LEFT)
        ToolTip(self.edit_btn, "Toggle raw markdown editing mode to directly inspect, edit, or clean up conversation turns.")

        self.font_minus_btn = self._flat_button(bar, "A\u2212", lambda: self._bump_font(-1), width=3)
        self.font_minus_btn.pack(side=tk.LEFT, padx=(8, 1))
        ToolTip(self.font_minus_btn, "Decrease chat typography text size. You can also hold Ctrl and scroll your mouse wheel down.")
        self.font_plus_btn = self._flat_button(bar, "A+", lambda: self._bump_font(1), width=3)
        self.font_plus_btn.pack(side=tk.LEFT, padx=1)
        ToolTip(self.font_plus_btn, "Increase chat typography text size. You can also hold Ctrl and scroll your mouse wheel up.")

        settings_btn = self._flat_button(bar, "Config", self._open_settings)
        settings_btn.pack(side=tk.RIGHT)
        ToolTip(settings_btn, "Open AI Settings to configure providers (AGY, Gemini, Anthropic, DeepSeek, Ollama), API keys, and model parameters.")
        self.clear_btn = self._flat_button(bar, "Clear", self._clear_history)
        self.clear_btn.pack(side=tk.RIGHT, padx=(0, 4))
        ToolTip(self.clear_btn, "Clear conversation history for this document tab to begin a fresh conversation.")

        # --- Display container: chat view <-> raw edit view -------------------
        self.display_container = self._reg(tk.Frame(self), bg='panel')
        self.display_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=0)

        self.chat_display = scrolledtext.ScrolledText(
            self.display_container, wrap=tk.WORD, state=tk.DISABLED, height=15,
            relief=tk.FLAT, bd=0, highlightthickness=1, padx=0, pady=4, cursor='arrow')
        self._reg(self.chat_display, bg='display_bg', fg='text', highlightbackground='border',
                  highlightcolor='border', insertbackground='text')
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        self.edit_banner = self._reg(tk.Frame(self.display_container), bg='hdr_user_bg')
        self.edit_banner_label = self._reg(
            tk.Label(self.edit_banner, text="", font=(SANS_FONT, 8), justify=tk.LEFT,
                     anchor=tk.W, wraplength=250),
            bg='hdr_user_bg', fg='text')
        self.edit_banner_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6, pady=3)
        self.edit_discard_btn = self._flat_button(self.edit_banner, "Discard", self._discard_edit_mode,
                                                  font_size=8)
        self.edit_discard_btn.pack(side=tk.RIGHT, padx=4, pady=3)

        self.edit_text = scrolledtext.ScrolledText(
            self.display_container, wrap=tk.WORD, undo=True, maxundo=-1, autoseparators=True,
            relief=tk.FLAT, bd=0, highlightthickness=1, padx=8, pady=6)
        self._reg(self.edit_text, bg='input_bg', fg='input_fg', highlightbackground='border',
                  highlightcolor='accent', insertbackground='input_fg')

        # --- Progress bar (indeterminate, shown while generating) -------------
        self.progress = ttk.Progressbar(self, mode='indeterminate', length=100)

        # --- Options ----------------------------------------------------------
        self.options_frame = self._reg(tk.Frame(self), bg='panel')
        self.options_frame.pack(fill=tk.X, padx=8, pady=(6, 0))

        def check(text, var, tip):
            cb = tk.Checkbutton(self.options_frame, text=text, variable=var, font=(SANS_FONT, 9),
                                anchor=tk.W, bd=0, highlightthickness=0, takefocus=0)
            self._reg(cb, bg='panel', fg='text', activebackground='panel', activeforeground='text',
                      selectcolor='input_bg')
            ToolTip(cb, tip)
            return cb

        self.include_doc_var = tk.BooleanVar(value=False)
        self.include_doc_cb = check("Include Document", self.include_doc_var,
                                    "Include current document content with your message as contextual grounding for the AI assistant.")
        self.include_doc_cb.pack(side=tk.LEFT, padx=(0, 8))
        self.include_history_var = tk.BooleanVar(value=True)
        self.include_history_cb = check("Include Chat History", self.include_history_var,
                                        "Include prior messages from this chat session so the AI maintains multi-turn conversation context.")
        self.include_history_cb.pack(side=tk.LEFT)

        limit_row = self._reg(tk.Frame(self), bg='panel')
        limit_row.pack(fill=tk.X, padx=8, pady=(0, 2))
        self._reg(tk.Label(limit_row, text="History sent to the model:", font=(SANS_FONT, 8)),
                  bg='panel', fg='muted').pack(side=tk.LEFT, padx=(2, 4))
        self.history_limit_var = tk.StringVar(value=self._limit_to_text(self.settings.history_limit))
        self.history_limit_cb = ttk.Combobox(limit_row, textvariable=self.history_limit_var, width=5,
                                             state='readonly',
                                             values=('2', '4', '6', '10', '20', '40', 'All'))
        self.history_limit_cb.pack(side=tk.LEFT)
        self.history_limit_cb.bind('<<ComboboxSelected>>', self._on_history_limit_change)
        self._reg(tk.Label(limit_row, text="messages", font=(SANS_FONT, 8)),
                  bg='panel', fg='muted').pack(side=tk.LEFT, padx=4)
        ToolTip(self.history_limit_cb, "Select how many recent conversational turns are included in the prompt context sent to the AI.")

        # --- Attachments ------------------------------------------------------
        self.attachments_frame = self._reg(tk.Frame(self), bg='panel')
        self.attachments_frame.pack(fill=tk.X, padx=8, pady=(0, 2))
        self.attachments_label = self._reg(
            tk.Label(self.attachments_frame, text="", font=(SANS_FONT, 9), anchor=tk.W),
            bg='panel', fg='muted')
        self.attachments_label.pack(fill=tk.X)

        # --- Input ------------------------------------------------------------
        input_frame = self._reg(tk.Frame(self), bg='panel')
        input_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        btn_frame = self._reg(tk.Frame(input_frame), bg='panel')
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.input_text = tk.Text(input_frame, height=3, wrap=tk.WORD, relief=tk.FLAT, bd=0,
                                  highlightthickness=1, padx=8, pady=6, undo=True)
        self._reg(self.input_text, bg='input_bg', fg='input_fg', highlightbackground='border',
                  highlightcolor='accent', insertbackground='input_fg')
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.input_hint = tk.Label(self.input_text, text="Message\u2026   Shift+Enter to send",
                                   anchor=tk.W, cursor='xterm')
        self._reg(self.input_hint, bg='input_bg', fg='muted')
        self.input_hint.place(x=9, y=6)
        self.input_hint.bind('<Button-1>', lambda e: self.input_text.focus_set())

        self.attach_btn = self._flat_button(btn_frame, "Attach", self._attach_file, width=6)
        self.attach_btn.pack(side=tk.TOP, padx=(6, 0), pady=(0, 3))
        ToolTip(self.attach_btn, "Attach an image or text file from your computer for multimodal vision analysis or document comprehension.")
        self.send_btn = self._flat_button(btn_frame, "Send", self._send_message, width=6, accent=True)
        self.send_btn.pack(side=tk.TOP, padx=(6, 0))
        self.send_btn_tip = ToolTip(self.send_btn, "Submit your prompt message to the AI assistant (Shift+Enter or Ctrl+Enter). Click while generating to stop.")

        # --- Selection indicator (shown when text is transferred from editor) --
        self.selection_indicator_frame = tk.Frame(self, bg='#e8f0fe')
        self.selection_indicator_label = tk.Label(
            self.selection_indicator_frame, text="", bg='#e8f0fe', fg='#1a73e8',
            font=(SANS_FONT, 9), anchor=tk.W, cursor='hand2')
        self.selection_indicator_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
        self.selection_clear_btn = ttk.Button(
            self.selection_indicator_frame, text="\u2715", width=2,
            command=self._clear_pending_selection)
        self.selection_clear_btn.pack(side=tk.RIGHT, padx=(0, 5), pady=2)

        # --- Status -----------------------------------------------------------
        self.status_label = self._reg(
            tk.Label(self, text="Ready", font=(SANS_FONT, 9), anchor=tk.W),
            bg='panel', fg='muted')
        self.status_label.pack(fill=tk.X, padx=10, pady=(0, 6))

        self._update_badge()
        self._apply_input_fonts()

    def _setup_bindings(self):
        """Setup keyboard bindings"""
        # Shift+Enter to send, regular Enter for newline
        self.input_text.bind('<Shift-Return>', self._on_shift_enter)
        # Ctrl+Enter also sends for convenience
        self.input_text.bind('<Control-Return>', self._on_shift_enter)
        for seq in ('<KeyRelease>', '<FocusIn>', '<FocusOut>'):
            self.input_text.bind(seq, lambda e: self._update_hint(), add='+')
        for widget in (self.chat_display, self.edit_text, self.input_text):
            widget.bind('<Control-MouseWheel>', self._on_ctrl_wheel)

    # ---------------------------------------------------------------- theming

    def apply_theme(self, theme: Optional[Dict[str, Any]]):
        """Re-style the chat from the application theme (light/dark aware)."""
        self._pal = make_chat_palette(theme)
        self._recolor()
        self._configure_tags()
        if not self.is_generating and not self._editing:
            self._render_all()

    def _configure_tags(self):
        """(Re)configure all chat text tags for the current palette and font size."""
        w, P, S = self.chat_display, self._pal, self._fs
        sans, mono = SANS_FONT, MONO_FONT
        pad = 12

        def f(size, *style):
            return (sans, size, *style) if style else (sans, size)

        w.configure(font=f(S))
        # Base text (all message text carries 'body' for margins / spacing)
        w.tag_configure('body', font=f(S), foreground=P['text'], lmargin1=pad, lmargin2=pad,
                        rmargin=pad, spacing1=1, spacing3=2)
        w.tag_configure('md_blank', font=f(max(4, S // 2)))
        w.tag_configure('gap', font=f(-6), background=P['display_bg'])
        # Role header bands (solid, full width)
        for role, bg_key, acc_key in (('user', 'hdr_user_bg', 'acc_user'),
                                      ('assistant', 'hdr_ai_bg', 'acc_ai'),
                                      ('system', 'hdr_sys_bg', 'acc_sys')):
            w.tag_configure(f'hdr_{role}', font=f(max(7, S - 1), 'bold'), foreground=P[acc_key],
                            background=P[bg_key], lmargin1=pad, lmargin2=pad, rmargin=pad,
                            lmargincolor=P[bg_key], rmargincolor=P[bg_key],
                            spacing1=2, spacing3=2)
            w.tag_configure(f'hdrtime_{role}', font=f(max(6, S - 2)), foreground=P['muted'],
                            background=P[bg_key])
        w.tag_configure('empty', font=f(S), foreground=P['muted'], justify='center',
                        spacing1=40, lmargin1=20, lmargin2=20, rmargin=20)
        w.tag_configure('pending', font=f(S, 'italic'), foreground=P['muted'])
        w.tag_configure('md_tail', font=f(S), foreground=P['text'])
        w.tag_configure('md_sys', font=f(max(7, S - 1), 'italic'), foreground=P['muted'])
        w.tag_configure('md_err', font=f(S), foreground=P['err'])
        # Headings
        for n, delta in ((1, 5), (2, 3), (3, 1)):
            w.tag_configure(f'md_h{n}', font=f(S + delta, 'bold'), foreground=P['text'],
                            spacing1=8 if n == 1 else 6, spacing3=3)
        # Lists / quotes / rules / tables
        for d in range(4):
            first = pad + 6 + d * 18
            w.tag_configure(f'md_list{d}', lmargin1=first, lmargin2=first + 16, spacing1=1, spacing3=1)
        w.tag_configure('md_quote', font=f(S, 'italic'), foreground=P['quote_fg'], background=P['quote_bg'],
                        lmargin1=pad + 8, lmargin2=pad + 8)
        w.tag_configure('md_hr', foreground=P['muted'], font=f(max(6, S - 3)))
        w.tag_configure('md_table', font=(mono, max(7, S - 1)), background=P['table_bg'],
                        foreground=P['text'], lmargin1=pad + 4, lmargin2=pad + 4, spacing1=0, spacing3=0)
        w.tag_configure('md_table_head', font=(mono, max(7, S - 1), 'bold'))
        # Code
        w.tag_configure('md_code_block', font=(mono, max(7, S - 1)), background=P['code_bg'],
                        foreground=P['code_fg'], lmargin1=pad + 6, lmargin2=pad + 6, rmargin=pad,
                        spacing1=0, spacing3=0)
        w.tag_configure('md_code_inline', font=(mono, max(7, S - 1)), background=P['icode_bg'],
                        foreground=P['icode_fg'])
        # Inline styles (created last so they take priority over block tags)
        w.tag_configure('md_bold', font=f(S, 'bold'))
        w.tag_configure('md_italic', font=f(S, 'italic'))
        w.tag_configure('md_bolditalic', font=f(S, 'bold', 'italic'))
        w.tag_configure('md_highlight', background=P['hl_bg'], foreground=P['hl_fg'])
        w.tag_configure('md_link', foreground=P['link'], underline=True)
        # Copy buttons embedded in headers
        for name in w.window_names():
            try:
                w.nametowidget(name).configure(font=f(max(7, S - 2)))
            except (tk.TclError, KeyError):
                pass
        self._apply_input_fonts()

    def _apply_input_fonts(self):
        S = self._fs
        self.input_text.configure(font=(SANS_FONT, S))
        self.input_hint.configure(font=(SANS_FONT, S))
        self.edit_text.configure(font=(MONO_FONT, max(8, S - 1)))
        self.edit_banner_label.configure(font=(SANS_FONT, max(7, S - 2)))

    def _update_hint(self):
        empty = not self.input_text.get('1.0', 'end-1c').strip()
        if empty and self.focus_get() is not self.input_text:
            self.input_hint.place(x=9, y=6)
        elif empty:
            self.input_hint.place(x=9, y=6)
        else:
            self.input_hint.place_forget()

    def _update_badge(self):
        names = {'anthropic': 'Claude', 'gemini': 'Gemini', 'ollama': 'Ollama', 'deepseek': 'DeepSeek',
                 'cli': 'CLI', 'agy': 'Antigravity', 'antigravity': 'Antigravity'}
        prov = names.get(self.settings.provider, self.settings.provider)
        if self.settings.provider == 'cli':
            model = ''
        elif self.settings.provider in ('agy', 'antigravity'):
            model = self.settings.agy_model or self.settings.model or ''
        else:
            model = self.settings.model or ''
        self.badge_label.config(text=f"{prov} \u00b7 {model}" if model else prov)

    # ------------------------------------------------------------- font size

    def _bump_font(self, delta: int):
        self._set_font_size(self._fs + delta)

    def _on_ctrl_wheel(self, event):
        self._bump_font(1 if event.delta > 0 else -1)
        return 'break'

    def _set_font_size(self, size: int):
        size = max(CHAT_FONT_MIN, min(CHAT_FONT_MAX, int(size)))
        if size == self._fs:
            return
        self._fs = size
        self._configure_tags()
        self._flash_status(f"Chat text size: {size} pt")
        # Persist (debounced) without blocking typing/scrolling
        if self._font_save_id:
            self.after_cancel(self._font_save_id)
        self._font_save_id = self.after(600, self._save_font_size)

    def _save_font_size(self):
        self._font_save_id = None
        self.settings.chat_font_size = self._fs
        self.settings.save()

    # -------------------------------------------------------- history options

    @staticmethod
    def _limit_to_text(limit: int) -> str:
        return 'All' if limit <= 0 else str(limit)

    def _on_history_limit_change(self, event=None):
        text = self.history_limit_var.get()
        self.settings.history_limit = 0 if text == 'All' else int(text)
        self.settings.save()
        if self._editing:
            self._update_edit_banner()

    def _history_for_api(self) -> List[Dict[str, str]]:
        """Trailing history (user/assistant only) honouring the configured limit."""
        hist = [m for m in self._messages if m.role in ('user', 'assistant') and m.content.strip()]
        limit = self.settings.history_limit
        if limit > 0:
            hist = hist[-limit:]
        while hist and hist[0].role != 'user':   # providers require a leading user turn
            hist.pop(0)
        return [{"role": m.role, "content": m.content} for m in hist]

    # ------------------------------------------------------- display helpers

    @contextmanager
    def _writable(self):
        w = self.chat_display
        w.configure(state=tk.NORMAL)
        try:
            yield w
        finally:
            w.configure(state=tk.DISABLED)

    def _flash_status(self, text: str):
        self.status_label.config(text=text)

        def restore():
            if not self.is_generating:
                self.status_label.config(text="Ready")
        self.after(1800, restore)

    def _copy_message(self, idx: int):
        if 0 <= idx < len(self._messages):
            self.clipboard_clear()
            self.clipboard_append(self._messages[idx].content)
            self._flash_status("Copied!")

    def _clear_display(self):
        """Clear the chat display (does not touch stored history)."""
        with self._writable() as w:
            w.delete('1.0', tk.END)

    def _render_all(self):
        """Redraw every stored message from the model (history load, theme, edit exit)."""
        y = self.chat_display.yview()[0]
        self._clear_display()
        if not self._messages:
            self._draw_empty_state()
        for idx, msg in enumerate(self._messages):
            self._draw_message(msg.role, msg.content, msg.timestamp, idx)
        self.chat_display.update_idletasks()
        self.chat_display.yview_moveto(1.0 if y > 0.98 or y == 0.0 else y)

    def _draw_empty_state(self):
        with self._writable() as w:
            w.insert(tk.END, "Ask about this document, or select text in the editor and use "
                             "\u201cTransfer to Chat\u201d.\n", ('empty',))

    def _remove_empty_state(self):
        rng = self.chat_display.tag_ranges('empty')
        if rng:
            with self._writable() as w:
                w.delete(rng[0], rng[1])

    @staticmethod
    def _format_time(ts: str) -> str:
        try:
            return datetime.fromisoformat(ts).strftime('%H:%M')
        except Exception:
            return ''

    def _draw_header(self, role: str, ts: str, idx: Optional[int]):
        """Solid role band: label, time and a Copy button; carries 'msg_start' for streaming."""
        P = self._pal
        bg_key = {'user': 'hdr_user_bg', 'assistant': 'hdr_ai_bg'}.get(role, 'hdr_sys_bg')
        with self._writable() as w:
            if w.compare('end-1c', '>', '1.0'):
                w.insert(tk.END, '\n', ('gap',))
            start = w.index('end-1c')
            w.insert(tk.END, f"{self.ROLE_LABELS.get(role, role)}")
            label_end = w.index('end-1c')
            tstr = self._format_time(ts)
            if tstr:
                w.insert(tk.END, f"   {tstr}")
            time_end = w.index('end-1c')
            w.insert(tk.END, "  ")
            if idx is not None and role != 'system':
                btn = tk.Button(w, text="\u29c9 Copy", relief=tk.FLAT, bd=0, cursor='hand2', takefocus=0,
                                font=(SANS_FONT, max(7, self._fs - 2)), padx=4, pady=0,
                                bg=P[bg_key], fg=P['muted'], activebackground=P[bg_key],
                                activeforeground=P['accent'],
                                command=lambda i=idx: self._copy_message(i))
                w.window_create(tk.END, window=btn)
                if role == 'assistant' and self.apply_edit_callback:
                    w.insert(tk.END, " ")
                    rev_btn = tk.Button(
                        w, text="\u21c4 Review & Apply", relief=tk.FLAT, bd=0, cursor='hand2', takefocus=0,
                        font=(SANS_FONT, max(7, self._fs - 2)), padx=4, pady=0,
                        bg=P[bg_key], fg=P['accent'], activebackground=P[bg_key],
                        activeforeground=P['text'],
                        command=lambda i=idx: self._review_and_apply_message(i)
                    )
                    w.window_create(tk.END, window=rev_btn)
            w.insert(tk.END, '\n')
            end = w.index('end-1c')
            w.tag_add(f'hdr_{role}', start, end)
            w.tag_add(f'hdrtime_{role}', label_end, time_end)

    def _draw_message(self, role: str, text: str, ts: str, idx: Optional[int]):
        self._draw_header(role, ts, idx)
        if role == 'system':
            with self._writable() as w:
                tag = 'md_err' if 'error' in text.lower() else 'md_sys'
                w.insert(tk.END, text + '\n', ('body', tag))
        else:
            self._render_markdown(text, buffer_tables=True)

    def _add_chat_message(self, role: str, content: str, streaming: bool = False):
        """Display a transient note (e.g. errors). Not stored in history."""
        self._remove_empty_state()
        self._draw_message(role, content, datetime.now().isoformat(), None)
        self.chat_display.see(tk.END)

    # -------------------------------------------------------- markdown render

    def _render_markdown(self, text: str, buffer_tables: bool):
        state: Dict[str, Any] = {'in_code': False, 'table': []}
        with self._writable():
            for line in text.split('\n'):
                self._render_line(line, state, buffer_tables)
            self._flush_table(state)

    def _render_line(self, line: str, st: Dict[str, Any], buffer_tables: bool = False):
        """Render one complete markdown line at the end of the display (caller holds _writable)."""
        w = self.chat_display
        s = line.strip()

        if buffer_tables and not st['in_code'] and _TABLE_ROW_RE.match(line):
            st.setdefault('table', []).append(line)
            return
        if st.get('table'):
            self._flush_table(st)

        if s.startswith('```'):
            st['in_code'] = not st['in_code']
            return
        if st['in_code']:
            w.insert(tk.END, (line if line else ' ') + '\n', ('body', 'md_code_block'))
            return
        if not s:
            w.insert(tk.END, '\n', ('body', 'md_blank'))
            return

        hm = re.match(r'^(#{1,6})\s+(.*)$', s)
        if hm:
            tag = f'md_h{min(len(hm.group(1)), 3)}'
            self._inline(hm.group(2), ('body', tag), heading=True)
            w.insert(tk.END, '\n', ('body', tag))
            return
        if re.match(r'^(-{3,}|\*{3,}|_{3,})$', s):
            w.insert(tk.END, '\u2500' * 22 + '\n', ('body', 'md_hr'))
            return
        if s.startswith('>'):
            base = ('body', 'md_quote')
            self._inline(s.lstrip('>').strip(), base)
            w.insert(tk.END, '\n', base)
            return
        if _TABLE_ROW_RE.match(line):          # streaming: unaligned monospace row
            if not _TABLE_SEP_RE.match(line):
                w.insert(tk.END, s + '\n', ('body', 'md_table'))
            return
        lm = re.match(r'^(\s*)([-*+]|\d+[.)])\s+(.*)$', line)
        if lm:
            expanded = lm.group(1).expandtabs(4)
            depth = max(0, min(3, len(expanded) // 2))
            marker = lm.group(2)
            if marker[0].isdigit():
                shown = marker
            else:
                shown = ('\u2022', '\u25e6', '\u25aa', '\u25aa')[depth]
            base = ('body', f'md_list{depth}')
            w.insert(tk.END, shown + ' ', base)
            self._inline(lm.group(3), base)
            w.insert(tk.END, '\n', base)
            return

        base = ('body',)
        self._inline(line.strip() if line.startswith(' ') else line, base)
        w.insert(tk.END, '\n', base)

    def _flush_table(self, st: Dict[str, Any]):
        """Render buffered pipe-table rows as an aligned monospace block."""
        rows = st.get('table') or []
        st['table'] = []
        if not rows:
            return
        w = self.chat_display
        parsed, header_done = [], False
        for r in rows:
            if _TABLE_SEP_RE.match(r):
                header_done = True
                parsed.append(None)
                continue
            cells = [re.sub(r'(\*\*|__|`)', '', c.strip()) for c in r.strip().strip('|').split('|')]
            parsed.append(cells)
        data = [c for c in parsed if c is not None]
        if not data:
            return
        ncols = max(len(c) for c in data)
        widths = [0] * ncols
        for cells in data:
            for i, c in enumerate(cells):
                widths[i] = max(widths[i], len(c))
        first = True
        for cells in parsed:
            if cells is None:
                rule = '\u2500\u253c\u2500'.join('\u2500' * wd for wd in widths)
                w.insert(tk.END, '\u2500' + rule + '\u2500\n', ('body', 'md_table'))
                continue
            padded = [(cells[i] if i < len(cells) else '').ljust(widths[i]) for i in range(ncols)]
            tags = ('body', 'md_table', 'md_table_head') if (first and header_done) else ('body', 'md_table')
            w.insert(tk.END, ' ' + ' \u2502 '.join(padded) + ' \n', tags)
            first = False

    def _inline(self, text: str, base: tuple, heading: bool = False):
        """Insert one line of text with inline markdown styling (caller holds _writable)."""
        w = self.chat_display
        pos = 0

        def put(s, *extra):
            if s:
                w.insert(tk.END, s, base + tuple(extra))

        for m in _CHAT_INLINE_RE.finditer(text):
            put(text[pos:m.start()])
            kind = m.lastgroup
            if m.group('code'):
                put(m.group('code')[1:-1], 'md_code_inline')
            elif m.group('bi') is not None:
                put(m.group('bi'), *(() if heading else ('md_bolditalic',)))
            elif m.group('b') is not None or m.group('b2') is not None:
                put(m.group('b') or m.group('b2'), *(() if heading else ('md_bold',)))
            elif m.group('i') is not None or m.group('i2') is not None:
                put(m.group('i') or m.group('i2'), *(() if heading else ('md_italic',)))
            elif m.group('hl') is not None:
                put(m.group('hl'), 'md_highlight')
            elif m.group('ll') is not None:
                self._insert_link(m.group('ll'), m.group('lu'), base)
            elif m.group('url') is not None:
                url = m.group('url')
                trail = ''
                while url and url[-1] in '.,;:!?':
                    trail = url[-1] + trail
                    url = url[:-1]
                self._insert_link(url, url, base)
                put(trail)
            pos = m.end()
        put(text[pos:])

    def _insert_link(self, label: str, url: str, base: tuple):
        w = self.chat_display
        self._link_counter += 1
        tag = f'lnk{self._link_counter}'
        w.insert(tk.END, label, base + ('md_link', tag))
        w.tag_bind(tag, '<Button-1>', lambda e, u=url: webbrowser.open(u))
        w.tag_bind(tag, '<Enter>', lambda e: w.configure(cursor='hand2'))
        w.tag_bind(tag, '<Leave>', lambda e: w.configure(cursor='arrow'))

    # ----------------------------------------------------------- send / stream

    def _on_shift_enter(self, event):
        """Handle Shift+Enter - send message"""
        if not self.is_generating and not self._editing:
            self._send_message()
        return 'break'  # Prevent default newline

    def _provider_ready(self) -> bool:
        """Validate provider configuration, reporting problems as system notes."""
        if not self.llm_client:
            self._init_llm_client()
            if not self.llm_client:
                self._add_chat_message("system", "AI not configured. Please set up your API key in Settings.")
                return False
        p, s = self.settings.provider, self.settings
        missing = {
            'ollama': (not s.ollama_url, "Ollama URL not configured. Please configure in Settings."),
            'gemini': (not s.gemini_api_key, "Gemini API key not configured. Please add in Settings."),
            'cli': (not s.cli_command, "CLI command not configured. Please add a CLI command in Settings."),
            'deepseek': (not s.deepseek_api_key, "DeepSeek API key not configured. Please add in Settings."),
        }
        if p in ('agy', 'antigravity'):
            return True
        if p in missing:
            bad, msg = missing[p]
        else:
            bad, msg = (not s.api_key), "API key not configured. Please add your API key in Settings."
        if bad:
            self._add_chat_message("system", msg)
            return False
        return True

    def _send_message(self):
        """Send message to AI"""
        if self.is_generating or self._editing:
            return

        message = self.input_text.get(1.0, tk.END).strip()
        if not message and not self.attached_files:
            return
        if not self._provider_ready():
            return

        # Immediate debounce and state lock
        self.is_generating = True
        self._gen_id += 1
        gen_id = self._gen_id
        self._cancel = threading.Event()
        cancel = self._cancel
        self._gen_doc_id = self.current_document_id
        self._gen_visible = True
        self._gen_t0 = time.time()
        self._spin = 0
        self._set_generating_ui(True)

        self.input_text.delete(1.0, tk.END)
        self._update_hint()

        try:
            # Build display message (include attachment info)
            display_msg = message
            if self.attached_files:
                names = ', '.join(f['name'] for f in self.attached_files)
                display_msg = f"{message}\n\U0001f4ce Attached: {names}" if message else f"\U0001f4ce Attached: {names}"

            # History for the API is captured BEFORE the new message is added
            messages = self._history_for_api() if self.include_history_var.get() else []

            # Show + store the user message
            self._remove_empty_state()
            user_msg = ChatMessage(role="user", content=message)
            self._messages.append(user_msg)
            self._draw_message("user", display_msg, user_msg.timestamp, len(self._messages) - 1)
            if self._gen_doc_id:
                self.history_manager.add_message(user_msg, self._gen_doc_id)

            # Build current message content
            content_parts: List[str] = []
            images: List[str] = []
            if self.include_doc_var.get() and self.get_document_content:
                doc_content = self.get_document_content()
                if doc_content:
                    content_parts.append(f"[Current Document]\n{doc_content}\n[End Document]\n\n")
            if self.get_document_images and self.include_doc_var.get():
                for img_data in self.get_document_images()[:3]:  # Limit to 3 images
                    try:
                        img = img_data.get('image')
                        if img and PIL_AVAILABLE:
                            encoded = self.llm_client.encode_image_for_api(img, self.settings.image_max_size[0])
                            if encoded:
                                images.append(encoded)
                    except Exception:
                        pass
            for attachment in self.attached_files:
                if attachment['type'] == 'image':
                    images.append(attachment['content'])
                elif attachment['type'] == 'text':
                    content_parts.append(
                        f"\n[Attached File: {attachment['name']}]\n{attachment['content']}\n[End File]\n")
            self.attached_files.clear()
            self._update_attachments_display()

            content_parts.append(message)
            messages.append({"role": "user", "content": "".join(content_parts)})

            # Assistant card with live progress indicator
            self._start_assistant_stream()

            thread = threading.Thread(target=self._generate_response,
                                      args=(messages, images, gen_id, cancel), daemon=True)
            thread.start()
        except Exception as e:
            self._reset_generating_state()
            self.status_label.config(text="Error")
            self._add_chat_message("system", f"Error: failed to dispatch message: {e}")

    def _generate_response(self, messages: List[Dict], images: List[str], gen_id: int,
                           cancel: threading.Event):
        """Generate response in background thread"""
        try:
            active_model = self.settings.model
            if self.settings.provider == 'gemini':
                self.llm_client.api_key = self.settings.gemini_api_key
            elif self.settings.provider == 'anthropic':
                self.llm_client.api_key = self.settings.api_key
            elif self.settings.provider == 'deepseek':
                self.llm_client.api_key = self.settings.deepseek_api_key
            elif self.settings.provider in ('agy', 'antigravity'):
                active_model = self.settings.agy_model or self.settings.model
                if isinstance(self.llm_client, AgyClient):
                    self.llm_client.model = active_model
                    self.llm_client.effort = self.settings.agy_effort or 'low'
                    self.llm_client.cli_command = self.settings.agy_path or find_antigravity_cli_command()

            text = ""
            gen = self.llm_client.send_message_stream(
                messages,
                system_prompt=self.settings.system_prompt,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                top_k=self.settings.top_k,
                images=images if images else None,
                model=active_model
            )
            try:
                for chunk in gen:
                    if cancel.is_set():
                        break
                    if not chunk:
                        continue
                    text += chunk
                    self.response_queue.put(('chunk', gen_id, chunk))
            finally:
                close = getattr(gen, 'close', None)
                if close:
                    try:
                        close()
                    except Exception:
                        pass
            if not cancel.is_set():
                self.response_queue.put(('done', gen_id, text))
        except Exception as e:
            import traceback
            print(f"[ERROR] _generate_response: {e}\n{traceback.format_exc()}")
            if not cancel.is_set():
                self.response_queue.put(('error', gen_id, str(e)))

    def _process_queue(self):
        """Process response queue (runs on main thread). Chunks are coalesced per tick."""
        try:
            chunks: List[str] = []
            terminal = None
            while True:
                kind, gid, payload = self.response_queue.get_nowait()
                if gid != self._gen_id:
                    continue                       # stale (stopped / superseded)
                if kind == 'chunk':
                    chunks.append(payload)
                else:
                    terminal = (kind, payload)
                    break
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[ERROR] _process_queue: {e}")
            chunks, terminal = [], None
        try:
            if chunks:
                self._on_chunk(''.join(chunks))
            if terminal:
                kind, payload = terminal
                if kind == 'done':
                    self._on_done()
                elif kind == 'error':
                    self._on_error(payload)
        except Exception as e:
            import traceback
            print(f"[ERROR] chat UI update: {e}\n{traceback.format_exc()}")
            self._reset_generating_state()
        self.after(50, self._process_queue)

    # ----------------------------------------------------- generation UI state

    def _set_generating_ui(self, on: bool):
        if on:
            self.send_btn.configure(text="Stop", command=self._stop_generation)
            self.progress.pack(fill=tk.X, padx=8, pady=(4, 0), before=self.options_frame)
            self.progress.start(12)
            self.edit_btn.configure(state=tk.DISABLED)
            self.clear_btn.configure(state=tk.DISABLED)
            if hasattr(self, 'send_btn_tip') and self.send_btn_tip: self.send_btn_tip.set_text("Stop generating AI response and return to ready state.")
            self._tick()
        else:
            self.send_btn.configure(text="Send", command=self._send_message, state=tk.NORMAL)
            ToolTip(self.send_btn, "Send message (Shift+Enter)")
            self.progress.stop()
            self.progress.pack_forget()
            self.edit_btn.configure(state=tk.NORMAL)
            self.clear_btn.configure(state=tk.NORMAL)
            if self._tick_id:
                self.after_cancel(self._tick_id)
                self._tick_id = None

    def _reset_generating_state(self):
        self.is_generating = False
        self._pending_active = False
        self._set_generating_ui(False)

    def _pending_text(self, frame: str, elapsed: int) -> str:
        if self.settings.provider in ('agy', 'antigravity', 'cli'):
            return f"{frame} Waiting for the CLI\u2026 {elapsed}s (this can take a while)"
        return f"{frame} Thinking\u2026 {elapsed}s"

    def _tick(self):
        """Animate spinner + elapsed time while a reply is pending (100 ms cadence)."""
        self._tick_id = None
        if not self.is_generating:
            return
        elapsed = int(time.time() - self._gen_t0)
        frame = _SPINNER_FRAMES[self._spin % len(_SPINNER_FRAMES)]
        self._spin += 1
        self.status_label.config(text=f"{frame} Generating\u2026 {elapsed}s")
        if self._pending_active and self._gen_visible:
            w = self.chat_display
            try:
                with self._writable():
                    w.delete('pending_start', 'end-1c')
                    w.insert(tk.END, self._pending_text(frame, elapsed), ('body', 'pending'))
            except tk.TclError:
                self._pending_active = False
        self._tick_id = self.after(100, self._tick)

    def _start_assistant_stream(self):
        """Create the assistant card (header + animated pending line) and reset stream state."""
        self._stream_msg = ChatMessage(role="assistant", content="")
        self._messages.append(self._stream_msg)
        idx = len(self._messages) - 1
        self._stream_buf = ''
        self._stream_text = ''
        self._stream_state = {'in_code': False, 'table': []}
        w = self.chat_display
        with self._writable():
            w.mark_set('msg_start', 'end-1c')
            w.mark_gravity('msg_start', tk.LEFT)
        self._draw_header("assistant", self._stream_msg.timestamp, idx)
        with self._writable():
            w.mark_set('body_start', 'end-1c')
            w.mark_gravity('body_start', tk.LEFT)
            w.mark_set('pending_start', 'end-1c')
            w.mark_gravity('pending_start', tk.LEFT)
            w.mark_set('tail_start', 'end-1c')
            w.mark_gravity('tail_start', tk.LEFT)
            w.insert(tk.END, self._pending_text(_SPINNER_FRAMES[0], 0), ('body', 'pending'))
        self._pending_active = True
        w.see(tk.END)

    def _clear_pending(self):
        if self._pending_active:
            self._pending_active = False
            with self._writable() as w:
                w.delete('pending_start', 'end-1c')

    def _on_chunk(self, text: str):
        """Render newly received text: complete lines as markdown, unfinished tail raw."""
        self._stream_text += text
        if self._stream_msg is not None:
            self._stream_msg.content = self._stream_text
        if not self._gen_visible:
            return
        w = self.chat_display
        at_bottom = w.yview()[1] >= 0.98
        self._stream_buf += text
        parts = self._stream_buf.split('\n')
        self._stream_buf = parts[-1]
        complete = parts[:-1]
        self._clear_pending()
        with self._writable():
            w.delete('tail_start', 'end-1c')
            for ln in complete:
                self._render_line(ln, self._stream_state, buffer_tables=False)
            w.mark_set('tail_start', 'end-1c')
            if self._stream_buf:
                w.insert(tk.END, self._stream_buf, ('body', 'md_tail'))
        if at_bottom:
            w.see(tk.END)

    def _finalize_stream(self, stopped: bool = False):
        """Flush the unfinished tail as markdown and persist the reply."""
        msg = self._stream_msg
        text = self._stream_text
        w = self.chat_display
        if self._gen_visible:
            self._clear_pending()
            has_table = bool(re.search(r'^\s*\|.*\|\s*$\n\s*\|?\s*:?-{2,}', text, re.MULTILINE))
            if not text.strip():
                # Nothing was produced: drop the empty card
                with self._writable():
                    w.delete('msg_start', 'end-1c')
                if msg in self._messages:
                    self._messages.remove(msg)
            elif has_table:
                with self._writable():
                    w.delete('body_start', 'end-1c')
                self._render_markdown(text, buffer_tables=True)
            else:
                with self._writable():
                    w.delete('tail_start', 'end-1c')
                    if self._stream_buf:
                        self._render_line(self._stream_buf, self._stream_state, buffer_tables=False)
                    self._flush_table(self._stream_state)
        if msg is not None and text.strip():
            msg.content = text
            if self._gen_doc_id:
                self.history_manager.add_message(msg, self._gen_doc_id)
        self._stream_msg = None
        self._stream_buf = ''

    def _after_generation(self, status: str):
        was_visible = self._gen_visible
        self._reset_generating_state()
        self.status_label.config(text=status)
        if not was_visible and self.current_document_id and self.current_document_id == self._gen_doc_id:
            # User switched away and back mid-generation: reload to show the reply
            self._messages = list(self.history_manager.get_history(self.current_document_id))
            self._render_all()
        elif was_visible:
            self.chat_display.see(tk.END)
            if self.pending_selection and status == "Ready":
                self._show_apply_button()
        self._gen_visible = False

    def _on_done(self):
        self._finalize_stream()
        self._after_generation("Ready")

    def _on_error(self, message: str):
        self._finalize_stream(stopped=True)
        if self._gen_visible:
            self._add_chat_message("system", f"Error: {message}")
        self._after_generation("Error")

    def _stop_generation(self):
        """Stop button: keep whatever has been generated so far."""
        if not self.is_generating:
            return
        self._cancel.set()
        self._gen_id += 1                      # invalidate late results from the worker
        if isinstance(self.llm_client, CliClient):
            client = self.llm_client
            threading.Thread(target=client.close_session, daemon=True).start()
        self._finalize_stream(stopped=True)
        self._after_generation("Stopped")

    def _attach_file(self):
        """Open file dialog to attach a file"""
        filetypes = [
            ("All supported", "*.png *.jpg *.jpeg *.gif *.bmp *.webp *.txt *.md *.py *.json *.xml *.csv *.html *.css *.js"),
            ("Images", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
            ("Text files", "*.txt *.md *.py *.json *.xml *.csv *.html *.css *.js"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Attach File",
            filetypes=filetypes,
            parent=self.winfo_toplevel()
        )
        
        if not file_path:
            return
        
        try:
            file_name = Path(file_path).name
            file_ext = Path(file_path).suffix.lower()
            
            # Determine file type and read content
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
            
            if file_ext in image_extensions:
                # Handle image file
                if not PIL_AVAILABLE:
                    messagebox.showwarning("Warning", "PIL not available. Cannot attach images.", parent=self)
                    return
                
                with Image.open(file_path) as img:
                    # Convert and resize
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    
                    max_size = self.settings.image_max_size
                    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                        img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # Encode to base64
                    from io import BytesIO
                    buffer = BytesIO()
                    img.save(buffer, format='JPEG', quality=85)
                    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    self.attached_files.append({
                        'path': file_path,
                        'name': file_name,
                        'type': 'image',
                        'content': encoded
                    })
            else:
                # Handle text file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Limit text content size (max 50KB)
                    if len(content) > 50000:
                        content = content[:50000] + "\n... [truncated]"
                    
                    self.attached_files.append({
                        'path': file_path,
                        'name': file_name,
                        'type': 'text',
                        'content': content
                    })
                except UnicodeDecodeError:
                    messagebox.showwarning("Warning", "Cannot read file. It may be binary or use an unsupported encoding.", parent=self)
                    return
            
            # Update attachments display
            self._update_attachments_display()
            self.status_label.config(text=f"Attached: {file_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to attach file: {str(e)}", parent=self)
    
    def _update_attachments_display(self):
        """Update the attachments label to show attached files"""
        if not self.attached_files:
            self.attachments_label.config(text="")
            return
        
        # Build display text
        names = [f['name'] for f in self.attached_files]
        if len(names) == 1:
            display_text = f"📎 {names[0]}"
        else:
            display_text = f"📎 {len(names)} files: {', '.join(names[:3])}"
            if len(names) > 3:
                display_text += f" (+{len(names) - 3} more)"
        
        # Add clear link
        display_text += " [Clear]"
        self.attachments_label.config(text=display_text)
        
        # Bind click to clear attachments
        self.attachments_label.bind('<Button-1>', self._on_attachments_click)
        self.attachments_label.config(cursor="hand2")
    
    def _on_attachments_click(self, event):
        """Handle click on attachments label to clear"""
        if self.attached_files:
            if messagebox.askyesno("Clear Attachments", "Remove all attached files?", parent=self):
                self.attached_files.clear()
                self._update_attachments_display()
                self.attachments_label.unbind('<Button-1>')
                self.attachments_label.config(cursor="")
                self.status_label.config(text="Attachments cleared")

    # ------------------------------------------------------------ Edit Chat

    def _update_edit_banner(self):
        limit = self.settings.history_limit
        sent = "all messages are" if limit <= 0 else f"only the last {limit} messages are"
        self.edit_banner_label.config(
            text=("Editing raw markdown. Messages are separated by lines like \u2500\u2500\u2500\u2500 [user] "
                  f"\u2500\u2500\u2500\u2500 or [assistant]; {sent} sent to the model."))

    def _on_edit_toggle(self):
        if self.edit_var.get():
            self._enter_edit_mode()
        else:
            self._exit_edit_mode(save=True)

    def _enter_edit_mode(self):
        if self.is_generating:
            self.edit_var.set(False)
            self._flash_status("Wait for the reply (or press Stop) before editing")
            return
        self._editing = True
        self._edit_original = serialize_history(self._messages)
        self.edit_text.delete('1.0', tk.END)
        self.edit_text.insert('1.0', self._edit_original)
        self.edit_text.edit_reset()
        self.chat_display.pack_forget()
        self._update_edit_banner()
        self.edit_banner.pack(fill=tk.X, pady=(0, 4))
        self.edit_text.pack(fill=tk.BOTH, expand=True)
        for btn in (self.send_btn, self.attach_btn, self.clear_btn):
            btn.configure(state=tk.DISABLED)
        self.edit_text.focus_set()
        self.status_label.config(text="Editing chat history")

    def _exit_edit_mode(self, save: bool = True):
        if not self._editing:
            return
        self._editing = False
        self.edit_var.set(False)
        new_text = self.edit_text.get('1.0', 'end-1c')
        self.edit_text.pack_forget()
        self.edit_banner.pack_forget()
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        for btn in (self.send_btn, self.attach_btn, self.clear_btn):
            btn.configure(state=tk.NORMAL)
        status = "Ready"
        if save and new_text.strip() != self._edit_original.strip():
            new_msgs = parse_history(new_text, self._messages)
            old_keys = [(m.role, m.content) for m in self._messages]
            if [(m.role, m.content) for m in new_msgs] != old_keys:
                self._messages = new_msgs
                if self.current_document_id:
                    self.history_manager.save_history(new_msgs, self.current_document_id)
                status = "Chat history updated"
        self._render_all()
        self.status_label.config(text=status)

    def _discard_edit_mode(self):
        self._exit_edit_mode(save=False)

    def commit_edit_mode(self):
        """Save and leave Edit Chat mode if it is active (document switch, sidebar hide, exit)."""
        if self._editing:
            self._exit_edit_mode(save=True)

    # ------------------------------------------------------------- history

    def _clear_history(self):
        """Clear chat history for current document"""
        if self.is_generating:
            self._flash_status("Wait for the reply (or press Stop) first")
            return
        if self.current_document_id:
            if messagebox.askyesno("Clear History", "Clear chat history for this document?", parent=self):
                self.history_manager.clear_history(self.current_document_id)
                self._messages = []
                self._render_all()
                self.status_label.config(text="History cleared")
        else:
            self._messages = []
            self._render_all()

    def _open_settings(self):
        """Open settings dialog"""
        AISettingsDialog(self.winfo_toplevel(), self.settings, on_save=self._on_settings_saved)

    def _on_settings_saved(self):
        """Called when settings are saved"""
        self._init_llm_client()
        self._update_badge()
        if self._editing:
            self._update_edit_banner()
        self.history_limit_var.set(self._limit_to_text(self.settings.history_limit))
        self.status_label.config(text="Settings saved")

    def set_document_id(self, doc_id: Optional[str]):
        """Set current document ID and load its chat history"""
        if doc_id == self.current_document_id:
            return  # Same document, keep current chat display
        self.commit_edit_mode()
        self.current_document_id = doc_id
        if self.is_generating:
            self._gen_visible = False   # reply continues in background and is saved to its own document
        self._messages = list(self.history_manager.get_history(doc_id)) if doc_id else []
        self._render_all()

    def reload_settings(self):
        """Reload settings (e.g., after settings dialog)"""
        self.settings = get_ai_settings()
        self._init_llm_client()
        self._update_badge()
        self.history_limit_var.set(self._limit_to_text(self.settings.history_limit))

    # === Transfer to Chat / Selection Methods ===

    def set_pending_selection(self, text: str, start_index: str, end_index: str):
        """Set a pending selection from the editor for 'Transfer to Chat'"""
        self.pending_selection = {
            'text': text,
            'start_index': start_index,
            'end_index': end_index,
        }
        try:
            start_line = start_index.split('.')[0]
            end_line = end_index.split('.')[0]
            if start_line == end_line:
                loc = f"line {start_line}"
            else:
                loc = f"lines {start_line}\u2013{end_line}"
        except Exception:
            loc = "selection"
        preview = text[:60].replace('\n', ' ')
        if len(text) > 60:
            preview += '\u2026'
        self.selection_indicator_label.config(text=f"\U0001f4cc Selection ({loc}): {preview}")
        self.selection_indicator_frame.pack(fill=tk.X, padx=8, pady=(0, 3), before=self.status_label)
        # Pre-fill input with quoted selection
        self.input_text.delete('1.0', tk.END)
        quoted = '\n'.join(f'> {l}' for l in text.split('\n'))
        self.input_text.insert('1.0', quoted + '\n\n')
        self.input_text.mark_set(tk.INSERT, tk.END)
        self.input_text.focus_set()
        self._update_hint()

    def _clear_pending_selection(self):
        """Clear pending selection state"""
        self.pending_selection = None
        self.selection_indicator_frame.pack_forget()
        self._remove_apply_buttons()

    def _review_and_apply_message(self, idx: int):
        """Open DiffPanel to review and selectively apply any assistant message to the document."""
        if not self.apply_edit_callback or idx is None or idx < 0 or idx >= len(self._messages):
            return
        msg = self._messages[idx]
        if not msg.content or not msg.content.strip():
            return
        self._apply_response_to_document(msg.content.strip(), self.pending_selection)

    def _show_apply_button(self):
        """Show an 'Apply to Document' button below the last AI message in the chat"""
        if not self.pending_selection or not self.apply_edit_callback:
            return
        last = next((m for m in reversed(self._messages) if m.role == 'assistant'), None)
        if last is None or not last.content.strip():
            return
        response_text = last.content.strip()
        sel = self.pending_selection

        w = self.chat_display
        with self._writable():
            w.insert(tk.END, '\n', ('body',))
            apply_btn = ttk.Button(
                w, text="\u2714 Review & Apply to Document",
                command=lambda: self._apply_response_to_document(response_text, sel))
            w.window_create(tk.END, window=apply_btn, padx=12)
            w.insert(tk.END, '  ', ('body',))
            discard_btn = ttk.Button(w, text="\u2718 Discard", command=self._clear_pending_selection)
            w.window_create(tk.END, window=discard_btn)
            w.insert(tk.END, '\n', ('body',))
        w.see(tk.END)

    def _apply_response_to_document(self, response_text: str, selection: Optional[Dict] = None):
        """Open the diff panel to apply AI response to document"""
        if self.apply_edit_callback:
            if selection:
                self.apply_edit_callback(
                    selection['text'],
                    response_text,
                    selection['start_index'],
                    selection['end_index']
                )
            else:
                self.apply_edit_callback(
                    None,
                    response_text,
                    None,
                    None
                )
        self._clear_pending_selection()

    def _remove_apply_buttons(self):
        """Remove any embedded Apply/Discard buttons from chat display"""
        # The buttons are embedded windows referencing the old selection; they are
        # dropped on the next redraw. Clearing pending selection is sufficient.
        pass

