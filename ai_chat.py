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
            font=(SANS_FONT, 9), padx=6, pady=3
        )
        label.pack()
    
    def _hide_tooltip(self):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

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
    
    # Default AGY models & efforts
    AGY_MODELS = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-pro",
        "claude-sonnet-4.6",
        "claude-opus-4.6",
        "gpt-oss-120b",
    ]
    AGY_EFFORTS = ["low", "medium", "high"]
    
    DEFAULT_SETTINGS = {
        'provider': 'agy',
        'api_key': '',
        'gemini_api_key': '',  # Separate API key for Gemini
        'deepseek_api_key': '',  # Separate API key for DeepSeek
        'ollama_url': 'http://localhost:11434',  # Ollama server URL
        'agy_path': '',  # Custom path to agy CLI executable
        'agy_model': 'gemini-3.6-flash',
        'agy_effort': 'low',
        'model': 'gemini-3.6-flash',
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
        return self.settings.get('max_tokens', 4096)
    
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
        return self.settings.get('agy_model', 'gemini-2.5-flash')

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
        if ANTHROPIC_AVAILABLE and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
    
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
        if not self.client:
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
        if not self.client:
            # Reinitialize client if API key was updated
            if ANTHROPIC_AVAILABLE and self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            else:
                raise ValueError("Anthropic client not initialized. Check API key.")
        
        # Use provided model or default
        model_name = model or "claude-sonnet-4-20250514"
        
        # Convert messages format if needed (handle both dict and ChatMessage)
        api_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                api_messages.append(msg)
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
            # Build kwargs for the API call
            stream_kwargs = {
                "model": model_name,
                "messages": api_messages,
                "system": system_prompt if system_prompt else anthropic.NOT_GIVEN,
                "max_tokens": max_tokens,
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
        if GEMINI_AVAILABLE and api_key:
            self.client = genai.Client(api_key=api_key)
    
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
        if not self.client:
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
        if not self.client:
            # Reinitialize client if API key was updated
            if GEMINI_AVAILABLE and self.api_key:
                self.client = genai.Client(api_key=self.api_key)
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
        
        if OPENAI_AVAILABLE and api_key:
            try:
                self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
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
        if not self.client:
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
        if not self.client:
            # Reinitialize client if API key was updated
            if OPENAI_AVAILABLE and self.api_key:
                self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            else:
                raise ValueError("DeepSeek client not initialized.")
        
        model_name = model or "deepseek-chat"
        
        # Build messages
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            content = []
            
            # Add text
            if msg.get('content'):
                content.append({
                    "type": "text",
                    "text": msg['content']
                })
            
            # Add images
            if images:
                for img_b64 in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    })
            
            api_messages.append({
                "role": msg.get('role', 'user'),
                "content": content if len(content) > 1 else msg.get('content', '')
            })
        
        try:
            stream = self.client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                max_tokens=max_tokens,
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


def call_antigravity_cli(system_prompt: str, user_message: str, model_name: str = "gemini-3.6-flash", effort: str = "low", cli_command: str = "") -> str:
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

    if model_name and str(model_name).strip():
        cmd.extend(['--model', str(model_name).strip()])
        eff = str(effort or 'low').strip().lower()
        if eff in ('low', 'medium', 'high'):
            cmd.extend(['--effort', eff])
        else:
            cmd.extend(['--effort', 'low'])

    cmd.extend(['-p', prompt_input])

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=180
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Antigravity CLI executable not found: {executable}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Antigravity CLI timed out while generating a response.") from exc

    stdout_text = str(completed.stdout or '').strip()
    stderr_text = str(completed.stderr or '').strip()
    detail = stderr_text or stdout_text or f"exit code {completed.returncode}"

    # Fallback guard: if --effort was passed but model doesn't support it, retry without --effort
    if completed.returncode != 0 and '--effort' in cmd and 'effort is not supported' in detail.lower():
        cmd_no_effort = [c for c in cmd]
        try:
            eff_idx = cmd_no_effort.index('--effort')
            del cmd_no_effort[eff_idx:eff_idx + 2]
        except ValueError:
            pass
        
        try:
            completed = subprocess.run(
                cmd_no_effort,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=180
            )
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


def list_antigravity_models(cli_command: str = "") -> List[str]:
    """List models available to the agy CLI session."""
    executable = str(cli_command or find_antigravity_cli_command()).strip()
    fallback = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-pro", "claude-sonnet-4.6", "claude-opus-4.6", "gpt-oss-120b"]
    if not executable:
        return fallback

    cmd = [executable, 'models', '--output-format', 'json']
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=30)
        if completed.returncode == 0 and completed.stdout:
            data = json.loads(completed.stdout)
            if isinstance(data, list):
                res = [str(m.get('name', m)) for m in data if m]
                return res if res else fallback
            elif isinstance(data, dict):
                models = data.get('models', [])
                res = [str(m.get('name', m)) for m in models if m]
                return res if res else fallback
    except Exception:
        pass
    return fallback


def test_antigravity_connection(model_name: str = "gemini-3.6-flash", effort: str = "low", cli_command: str = "") -> str:
    """Run lightweight test query ('hi') to verify CLI connection."""
    test_sys = "You are a test assistant. Reply with the single word HI."
    test_msg = "hi"
    res = call_antigravity_cli(test_sys, test_msg, model_name=model_name, effort=effort, cli_command=cli_command)
    return str(res or '').strip()


class AgyClient(LLMClient):
    """LLM client using Google Antigravity CLI (agy) backend."""

    def __init__(self, model: str = "gemini-3.6-flash", effort: str = "low", cli_command: str = ""):
        self.model = model or "gemini-3.6-flash"
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
            res = test_antigravity_connection(
                model_name=self.model,
                effort=self.effort,
                cli_command=self.cli_command
            )
            return True, f"AGY CLI connection successful! Output: '{res}'"
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
        model = settings.agy_model or 'gemini-2.5-flash'
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
        self.geometry("560x560")
        self.resizable(True, True)
        self.minsize(520, 500)
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
        ttk.Button(btn_frame, text="Auto", width=5, command=self._auto_detect_agy_path).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Browse", width=6, command=self._browse_agy_path).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Test", width=5, command=lambda: self._test_connection('agy')).pack(side=tk.LEFT, padx=1)
        
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
        ttk.Button(ctx_top, text="+ Add", width=6,
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
            self.model_var.set(current_agy if current_agy in models else (models[0] if models else 'gemini-2.5-flash'))
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
                models = list_antigravity_models(cli_cmd)
                if models:
                    self.model_combo['values'] = models
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
            effort = self.agy_effort_var.get().strip()
            model = self.model_var.get().strip() or 'gemini-2.5-flash'
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

def _insert_inline(widget, text: str):
    """Insert one line of text into widget with bold/italic/code tags applied."""
    pattern = re.compile(
        r'(`[^`]+`)'
        r'|(\*\*[^*]+\*\*)'
        r'|(__[^_]+__)'
        r'|(\*[^*]+\*)'
        r'|(_[^_]+_)'
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


class ChatSidebar(tk.Frame):
    """
    Collapsible AI chat sidebar widget.
    
    Integrates with the main application to provide:
    - Chat input with Enter to send, Ctrl+Enter for newline
    - Option to include current document content
    - Option to include chat history
    - Clear history button
    - Settings access
    - File attachment support
    """
    
    def __init__(self, parent, get_document_content_callback: Optional[Callable] = None,
                 get_document_images_callback: Optional[Callable] = None,
                 apply_edit_callback: Optional[Callable] = None):
        super().__init__(parent, bg='#f0f0f0')
        
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
        
        # Pending selection for "Transfer to Chat" — stores selection context
        # Dict with keys: 'text', 'start_index', 'end_index' or None
        self.pending_selection: Optional[Dict[str, str]] = None
        
        # Attached files storage: list of dicts with 'path', 'name', 'type', 'content'
        self.attached_files: List[Dict[str, Any]] = []
        
        # Streaming / markdown rendering state
        self._streaming_buffer: str = ''
        self._streaming_start_index: str = ''
        
        # Copy-button message store
        self._message_texts: list = []
        
        # Initialize LLM client
        self._init_llm_client()
        
        self._setup_ui()
        self._setup_bindings()
        
        # Start response queue processor
        self._process_queue()
    
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
        if isinstance(self.llm_client, CliClient):
            self.llm_client.close_session()
    
    def _setup_ui(self):
        """Setup the sidebar UI"""
        # Header
        header_frame = tk.Frame(self, bg='#e0e0e0')
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(header_frame, text="AI Assistant", font=(SANS_FONT, 11, 'bold'),
                 bg='#e0e0e0').pack(side=tk.LEFT, padx=5)
        
        # Settings button
        settings_btn = ttk.Button(header_frame, text="Config", width=6, 
                                   command=self._open_settings)
        settings_btn.pack(side=tk.RIGHT, padx=2)
        ToolTip(settings_btn, "Configure AI provider, API keys, and model settings")
        
        # Clear history button
        clear_btn = ttk.Button(header_frame, text="Clear", width=6,
                                command=self._clear_history)
        clear_btn.pack(side=tk.RIGHT, padx=2)
        ToolTip(clear_btn, "Clear chat history for this document")
        
        # Chat display area
        chat_frame = tk.Frame(self, bg='#f0f0f0')
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=(SANS_FONT, 10),
            bg='#ffffff',
            fg='#1a1a1a',
            state=tk.DISABLED,
            height=15
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for styling
        self.chat_display.tag_configure('user', foreground='#0066cc', font=(SANS_FONT, 10, 'bold'))
        self.chat_display.tag_configure('assistant', foreground='#006600', font=(SANS_FONT, 10, 'bold'))
        self.chat_display.tag_configure('system', foreground='#666666', font=(SANS_FONT, 9, 'italic'))
        self.chat_display.tag_configure('error', foreground='#cc0000')
        self.chat_display.tag_configure('message', font=(SANS_FONT, 10))
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
        
        # Options frame
        options_frame = tk.Frame(self, bg='#f0f0f0')
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Include document checkbox
        self.include_doc_var = tk.BooleanVar(value=False)
        self.include_doc_cb = ttk.Checkbutton(
            options_frame, 
            text="Include Document",
            variable=self.include_doc_var
        )
        self.include_doc_cb.pack(side=tk.LEFT, padx=(0, 10))
        ToolTip(self.include_doc_cb, "Send current document content with your message")
        
        # Include history checkbox
        self.include_history_var = tk.BooleanVar(value=True)
        self.include_history_cb = ttk.Checkbutton(
            options_frame,
            text="Include Chat History",
            variable=self.include_history_var
        )
        self.include_history_cb.pack(side=tk.LEFT)
        ToolTip(self.include_history_cb, "Include previous messages for context")
        
        # Attachments frame (shows attached files)
        self.attachments_frame = tk.Frame(self, bg='#f0f0f0')
        self.attachments_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Attachments label (hidden when no attachments)
        self.attachments_label = tk.Label(
            self.attachments_frame, 
            text="", 
            bg='#f0f0f0', 
            fg='#666666',
            font=('Segoe UI', 9),
            anchor=tk.W
        )
        self.attachments_label.pack(fill=tk.X)
        
        # Input area
        input_frame = tk.Frame(self, bg='#f0f0f0')
        input_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Button frame for attach and send
        btn_frame = tk.Frame(input_frame, bg='#f0f0f0')
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.input_text = tk.Text(
            input_frame,
            height=3,
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            bg='#ffffff',
            fg='#1a1a1a'
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Attach button
        self.attach_btn = ttk.Button(btn_frame, text="Attach", command=self._attach_file, width=6)
        self.attach_btn.pack(side=tk.TOP, padx=(5, 0), pady=2)
        ToolTip(self.attach_btn, "Attach image or text file")
        
        # Send button
        self.send_btn = ttk.Button(btn_frame, text="Send", command=self._send_message, width=6)
        self.send_btn.pack(side=tk.TOP, padx=(5, 0), pady=2)
        ToolTip(self.send_btn, "Send message (Shift+Enter)")
        
        # Selection indicator (shown when text is transferred from editor)
        self.selection_indicator_frame = tk.Frame(self, bg='#e8f0fe')
        # Not packed until needed
        self.selection_indicator_label = tk.Label(
            self.selection_indicator_frame, text="", bg='#e8f0fe', fg='#1a73e8',
            font=(SANS_FONT, 9), anchor=tk.W, cursor='hand2'
        )
        self.selection_indicator_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
        self.selection_clear_btn = ttk.Button(
            self.selection_indicator_frame, text="\u2715", width=2,
            command=self._clear_pending_selection
        )
        self.selection_clear_btn.pack(side=tk.RIGHT, padx=(0, 5), pady=2)

        # Status label
        self.status_label = tk.Label(self, text="Ready", bg='#f0f0f0', fg='#666666',
                                      font=('Segoe UI', 9))
        self.status_label.pack(fill=tk.X, padx=5, pady=(0, 5))
    
    def _setup_bindings(self):
        """Setup keyboard bindings"""
        # Shift+Enter to send, regular Enter for newline
        self.input_text.bind('<Shift-Return>', self._on_shift_enter)
        # Ctrl+Enter also sends for convenience
        self.input_text.bind('<Control-Return>', self._on_shift_enter)
    
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
    
    def _on_shift_enter(self, event):
        """Handle Shift+Enter - send message"""
        if not self.is_generating:
            self._send_message()
        return 'break'  # Prevent default newline
    
    def _send_message(self):
        """Send message to AI"""
        if self.is_generating:
            return
        
        # Get user message
        message = self.input_text.get(1.0, tk.END).strip()
        if not message and not self.attached_files:
            return
        
        # Check if client is available
        if not self.llm_client:
            self._init_llm_client()
            if not self.llm_client:
                self._add_chat_message("system", "AI not configured. Please set up your API key in Settings.")
                return
        
        # Check configuration based on provider
        if self.settings.provider == 'ollama':
            if not self.settings.ollama_url:
                self._add_chat_message("system", "Ollama URL not configured. Please configure in Settings.")
                return
        elif self.settings.provider == 'gemini':
            if not self.settings.gemini_api_key:
                self._add_chat_message("system", "Gemini API key not configured. Please add in Settings.")
                return
        elif self.settings.provider == 'cli':
            if not self.settings.cli_command:
                self._add_chat_message("system", "CLI command not configured. Please add a CLI command in Settings.")
                return
        else:
            if not self.settings.api_key:
                self._add_chat_message("system", "API key not configured. Please add your API key in Settings.")
                return
        
        # Clear input
        self.input_text.delete(1.0, tk.END)
        
        # Build display message (include attachment info)
        display_msg = message
        if self.attached_files:
            attachment_names = [f['name'] for f in self.attached_files]
            display_msg = f"{message}\n📎 Attached: {', '.join(attachment_names)}" if message else f"📎 Attached: {', '.join(attachment_names)}"
        
        # Add user message to display
        self._add_chat_message("user", display_msg)
        
        # Prepare messages for API
        messages = []
        
        # Include chat history if enabled
        if self.include_history_var.get() and self.current_document_id:
            history = self.history_manager.get_history(self.current_document_id)
            for msg in history[-10:]:  # Last 10 messages
                messages.append({"role": msg.role, "content": msg.content})
        
        # Build current message content
        content_parts = []
        images = []
        
        # Include document content if enabled
        if self.include_doc_var.get() and self.get_document_content:
            doc_content = self.get_document_content()
            if doc_content:
                content_parts.append(f"[Current Document]\n{doc_content}\n[End Document]\n\n")
        
        # Get images from document if available
        if self.get_document_images and self.include_doc_var.get():
            doc_images = self.get_document_images()
            for img_data in doc_images[:3]:  # Limit to 3 images
                try:
                    img = img_data.get('image')
                    if img and PIL_AVAILABLE:
                        encoded = self.llm_client.encode_image_for_api(
                            img, 
                            self.settings.image_max_size[0]
                        )
                        if encoded:
                            images.append(encoded)
                except:
                    pass
        
        # Process attached files
        for attachment in self.attached_files:
            if attachment['type'] == 'image':
                images.append(attachment['content'])
            elif attachment['type'] == 'text':
                content_parts.append(f"\n[Attached File: {attachment['name']}]\n{attachment['content']}\n[End File]\n")
        
        # Clear attachments after sending
        self.attached_files.clear()
        self._update_attachments_display()
        
        content_parts.append(message)
        full_content = "".join(content_parts)
        
        messages.append({"role": "user", "content": full_content})
        
        # Save user message to history
        user_msg = ChatMessage(role="user", content=message)
        if self.current_document_id:
            self.history_manager.add_message(user_msg, self.current_document_id)
        
        # Start async generation
        self.is_generating = True
        self.send_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Generating...")
        
        # Add placeholder for assistant response
        self._add_chat_message("assistant", "", streaming=True)
        
        # Start generation thread
        thread = threading.Thread(
            target=self._generate_response,
            args=(messages, images),
            daemon=True
        )
        thread.start()
    
    def _generate_response(self, messages: List[Dict], images: List[str]):
        """Generate response in background thread"""
        try:
            # Update client settings based on provider
            if self.settings.provider == 'gemini':
                self.llm_client.api_key = self.settings.gemini_api_key
            elif self.settings.provider == 'anthropic':
                self.llm_client.api_key = self.settings.api_key
            # Ollama and CLI don't need API key updates

            # Stream response with all parameters
            response_text = ""
            chunk_count = 0
            for chunk in self.llm_client.send_message_stream(
                messages,
                system_prompt=self.settings.system_prompt,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                top_k=self.settings.top_k,
                images=images if images else None,
                model=self.settings.model
            ):
                chunk_count += 1
                response_text += chunk
                self._streaming_buffer += chunk
                self.response_queue.put(('chunk', chunk))

            # Debug: log if no chunks were received
            if chunk_count == 0:
                print(f"[DEBUG] No chunks received from {self.settings.provider} provider")
            else:
                print(f"[DEBUG] Received {chunk_count} chunks from {self.settings.provider} provider")

            # Save assistant message to history
            assistant_msg = ChatMessage(role="assistant", content=response_text)
            if self.current_document_id:
                self.history_manager.add_message(assistant_msg, self.current_document_id)

            self.response_queue.put(('done', None))

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] _generate_response: {error_msg}")
            self.response_queue.put(('error', str(e)))
    
    def _process_queue(self):
        """Process response queue (runs on main thread)"""
        try:
            while True:
                msg_type, content = self.response_queue.get_nowait()
                
                if msg_type == 'chunk':
                    self._append_to_last_message(content)
                elif msg_type == 'done':
                    self.is_generating = False
                    self.send_btn.config(state=tk.NORMAL)
                    self.status_label.config(text="Ready")
                    # Re-render last assistant message with markdown formatting
                    if self._streaming_buffer:
                        self.chat_display.config(state=tk.NORMAL)
                        self.chat_display.delete(self._streaming_start_index, tk.END)
                        self._insert_markdown_text(self._streaming_buffer)
                        self.chat_display.config(state=tk.DISABLED)
                        self.chat_display.see(tk.END)
                    # Copy button for streaming message
                    if self._streaming_buffer:
                        streamed_text = self._streaming_buffer

                        def _make_copy_streamed(text):
                            def _do():
                                self.clipboard_clear()
                                self.clipboard_append(text)
                                self.status_label.config(text='Copied!')
                                self.after(1500, lambda: self.status_label.config(text='Ready'))
                            return _do

                        self.chat_display.config(state=tk.NORMAL)
                        btn = tk.Button(
                            self.chat_display,
                            text='⧉', font=(SANS_FONT, 8),
                            relief=tk.FLAT, bd=0,
                            bg=self.chat_display.cget('bg'), fg='#888888',
                            activeforeground='#0066cc',
                            cursor='hand2', padx=2, pady=0,
                            command=_make_copy_streamed(streamed_text)
                        )
                        self.chat_display.window_create(tk.END, window=btn, padx=2)
                        self.chat_display.insert(tk.END, '\n')
                        self.chat_display.config(state=tk.DISABLED)
                        self._streaming_buffer = ''
                    # If we have a pending selection, show Apply to Document button
                    if self.pending_selection:
                        self._show_apply_button()
                elif msg_type == 'error':
                    self._add_chat_message("system", f"Error: {content}")
                    self.is_generating = False
                    self.send_btn.config(state=tk.NORMAL)
                    self.status_label.config(text="Error")
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[ERROR] _process_queue: {e}")
            self.is_generating = False
            self.send_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Error")
        
        # Schedule next check
        self.after(50, self._process_queue)
    
    def _add_chat_message(self, role: str, content: str, streaming: bool = False):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add separator if not first message
        current_content = self.chat_display.get(1.0, tk.END).strip()
        if current_content:
            self.chat_display.insert(tk.END, "\n\n")
        
        # Add role label
        if role == "user":
            self.chat_display.insert(tk.END, "You: ", 'user')
        elif role == "assistant":
            self.chat_display.insert(tk.END, "AI: ", 'assistant')
        elif role == "system":
            self.chat_display.insert(tk.END, "System: ", 'system')
        
        # Add content
        if content:
            if role == "system" and "Error" in content:
                self.chat_display.insert(tk.END, content, 'error')
            else:
                self.chat_display.insert(tk.END, content, 'message')
        
        if role == 'assistant' and streaming:
            # Use a named Tkinter mark so the position tracks correctly even
            # when embedded widgets (copy buttons) are inserted before it.
            self.chat_display.mark_set('streaming_start', tk.END)
            self.chat_display.mark_gravity('streaming_start', tk.LEFT)
            self._streaming_start_index = 'streaming_start'
            self._streaming_buffer = ''
        
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
            self.chat_display.insert(tk.END, '\n')
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _append_to_last_message(self, content: str):
        """Append content to the last message (for streaming)"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, content, 'message')
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _clear_history(self):
        """Clear chat history for current document"""
        if self.current_document_id:
            if messagebox.askyesno("Clear History", 
                                    "Clear chat history for this document?",
                                    parent=self):
                self.history_manager.clear_history(self.current_document_id)
                self._clear_display()
                self.status_label.config(text="History cleared")
        else:
            self._clear_display()
    
    def _insert_markdown_text(self, text: str):
        """Insert markdown-formatted assistant response into chat_display."""
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

        flush_code()

    def _clear_display(self):
        """Clear the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self._message_texts.clear()
    
    def _open_settings(self):
        """Open settings dialog"""
        dialog = AISettingsDialog(self.winfo_toplevel(), self.settings, 
                                   on_save=self._on_settings_saved)
    
    def _on_settings_saved(self):
        """Called when settings are saved"""
        self._init_llm_client()
        self.status_label.config(text="Settings saved")
    
    def set_document_id(self, doc_id: Optional[str]):
        """Set current document ID and load its chat history"""
        # Only reload if document changed
        if doc_id == self.current_document_id:
            return  # Same document, keep current chat display
        
        self.current_document_id = doc_id
        self._clear_display()
        
        # Load history for this document
        if doc_id:
            history = self.history_manager.get_history(doc_id)
            for msg in history:
                self._add_chat_message(msg.role, msg.content)
    
    def reload_settings(self):
        """Reload settings (e.g., after settings dialog)"""
        self.settings = get_ai_settings()
        self._init_llm_client()

    # === Transfer to Chat / Selection Methods ===

    def set_pending_selection(self, text: str, start_index: str, end_index: str):
        """Set a pending selection from the editor for 'Transfer to Chat'"""
        self.pending_selection = {
            'text': text,
            'start_index': start_index,
            'end_index': end_index,
        }
        # Show indicator
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
        self.selection_indicator_frame.pack(fill=tk.X, padx=5, pady=(0, 3), before=self.status_label)
        # Pre-fill input with quoted selection
        self.input_text.delete('1.0', tk.END)
        quoted = '\n'.join(f'> {l}' for l in text.split('\n'))
        self.input_text.insert('1.0', quoted + '\n\n')
        self.input_text.mark_set(tk.INSERT, tk.END)
        self.input_text.focus_set()

    def _clear_pending_selection(self):
        """Clear pending selection state"""
        self.pending_selection = None
        self.selection_indicator_frame.pack_forget()
        # Remove any Apply to Document buttons
        self._remove_apply_buttons()

    def _show_apply_button(self):
        """Show an 'Apply to Document' button below the last AI message in the chat"""
        if not self.pending_selection or not self.apply_edit_callback:
            return

        # Get the AI response text (last assistant message)
        # We parse from the chat display
        full_text = self.chat_display.get('1.0', tk.END)
        # Find the last "AI: " marker
        last_ai_idx = full_text.rfind('AI: ')
        if last_ai_idx < 0:
            return
        response_text = full_text[last_ai_idx + 4:].strip()

        sel = self.pending_selection

        # Create an inline button in the chat display
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, '\n')

        apply_btn = ttk.Button(
            self.chat_display, text="\u2714 Apply to Document",
            command=lambda: self._apply_response_to_document(response_text, sel)
        )
        self.chat_display.window_create(tk.END, window=apply_btn)
        self.chat_display.insert(tk.END, '  ')

        discard_btn = ttk.Button(
            self.chat_display, text="\u2718 Discard",
            command=self._clear_pending_selection
        )
        self.chat_display.window_create(tk.END, window=discard_btn)

        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _apply_response_to_document(self, response_text: str, selection: Dict):
        """Open the diff panel to apply AI response to document"""
        if self.apply_edit_callback:
            self.apply_edit_callback(
                selection['text'],
                response_text,
                selection['start_index'],
                selection['end_index']
            )
        self._clear_pending_selection()

    def _remove_apply_buttons(self):
        """Remove any embedded Apply/Discard buttons from chat display"""
        # The buttons are embedded windows; clearing pending selection is enough
        # since the buttons reference the old selection
        pass
