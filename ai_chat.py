"""
AI Chat Module for MarkItDown Notepad

Provides AI chatbot functionality with support for multiple LLM providers.
Currently implements Anthropic Claude, Google Gemini, DeepSeek, and Ollama.
"""

import json
import os
import base64
import hashlib
import threading
import queue
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
    
    DEFAULT_SETTINGS = {
        'provider': 'anthropic',
        'api_key': '',
        'gemini_api_key': '',  # Separate API key for Gemini
        'deepseek_api_key': '',  # Separate API key for DeepSeek
        'ollama_url': 'http://localhost:11434',  # Ollama server URL
        'model': 'claude-sonnet-4-20250514',
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
    
    def is_configured(self) -> bool:
        """Check if API key is configured for current provider"""
        if self.provider == 'gemini':
            return bool(self.gemini_api_key)
        elif self.provider == 'deepseek':
            return bool(self.deepseek_api_key)
        elif self.provider == 'ollama':
            return bool(self.ollama_url)  # Ollama doesn't need API key
        return bool(self.api_key)
    
    def get_available_models(self) -> List[str]:
        """Get available models for current provider"""
        if self.provider == 'anthropic':
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
# LLM Client Factory
# =============================================================================

def get_llm_client(settings: AISettings) -> Optional[LLMClient]:
    """Factory function to get appropriate LLM client based on settings"""
    provider = settings.provider
    
    if provider == 'anthropic':
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
        self.geometry("560x780")
        self.resizable(True, True)
        self.minsize(520, 720)
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
        """Setup the dialog UI"""
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Provider selection
        provider_frame = ttk.LabelFrame(main_frame, text="Provider", padding=10)
        provider_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(provider_frame, text="AI Provider:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.provider_var = tk.StringVar(value="anthropic")
        provider_combo = ttk.Combobox(provider_frame, textvariable=self.provider_var, 
                                       values=["anthropic", "gemini", "deepseek", "ollama"], state="readonly", width=30)
        provider_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)
        
        # API Keys Frame
        api_frame = ttk.LabelFrame(main_frame, text="API Keys / Connection", padding=10)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Anthropic API Key
        ttk.Label(api_frame, text="Anthropic:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=40, show="•")
        self.api_key_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        self.show_key_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(api_frame, text="Show", variable=self.show_key_var,
                        command=self._toggle_key_visibility).grid(row=0, column=2, padx=(5, 0))
        ttk.Button(api_frame, text="Test", command=lambda: self._test_connection('anthropic'), 
                   width=6).grid(row=0, column=3, padx=(5, 0))
        
        # Gemini API Key
        ttk.Label(api_frame, text="Gemini:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.gemini_key_var = tk.StringVar()
        self.gemini_key_entry = ttk.Entry(api_frame, textvariable=self.gemini_key_var, width=40, show="•")
        self.gemini_key_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        self.show_gemini_key_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(api_frame, text="Show", variable=self.show_gemini_key_var,
                        command=self._toggle_gemini_key_visibility).grid(row=1, column=2, padx=(5, 0))
        ttk.Button(api_frame, text="Test", command=lambda: self._test_connection('gemini'),
                   width=6).grid(row=1, column=3, padx=(5, 0))
        
        # DeepSeek API Key
        ttk.Label(api_frame, text="DeepSeek:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.deepseek_key_var = tk.StringVar()
        self.deepseek_key_entry = ttk.Entry(api_frame, textvariable=self.deepseek_key_var, width=40, show="•")
        self.deepseek_key_entry.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        self.show_deepseek_key_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(api_frame, text="Show", variable=self.show_deepseek_key_var,
                        command=self._toggle_deepseek_key_visibility).grid(row=2, column=2, padx=(5, 0))
        ttk.Button(api_frame, text="Test", command=lambda: self._test_connection('deepseek'),
                   width=6).grid(row=2, column=3, padx=(5, 0))
        
        # Ollama URL
        ttk.Label(api_frame, text="Ollama URL:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.ollama_url_var = tk.StringVar(value="http://localhost:11434")
        self.ollama_url_entry = ttk.Entry(api_frame, textvariable=self.ollama_url_var, width=40)
        self.ollama_url_entry.grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Button(api_frame, text="Test", command=lambda: self._test_connection('ollama'),
                   width=6).grid(row=3, column=3, padx=(5, 0))
        
        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                         values=AISettings.ANTHROPIC_MODELS, width=35)
        self.model_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Refresh models button
        refresh_btn = ttk.Button(model_frame, text="🔄", width=3, command=self._refresh_models)
        refresh_btn.grid(row=0, column=2, padx=(5, 0))
        self._create_tooltip(refresh_btn, "Refresh model list: Fetches the latest available models from the API. Use this if you have access to new models that aren't showing in the dropdown.")
        
        # Parameters
        params_frame = ttk.LabelFrame(main_frame, text="Generation Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Max tokens
        ttk.Label(params_frame, text="Max Tokens:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.max_tokens_var = tk.IntVar(value=4096)
        max_tokens_spin = ttk.Spinbox(params_frame, from_=256, to=8192, 
                                       textvariable=self.max_tokens_var, width=10)
        max_tokens_spin.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        max_tokens_tip = ttk.Label(params_frame, text="ℹ️", font=('Segoe UI', 10))
        max_tokens_tip.grid(row=0, column=2, padx=(5, 0))
        self._create_tooltip(max_tokens_tip, "Maximum number of tokens to generate in the response. Higher values allow longer responses but may cost more. A typical page of text is ~500-800 tokens. Common values: 1024 (short), 2048 (medium), 4096 (long), 8192 (very long).")
        
        # Temperature
        ttk.Label(params_frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, pady=5)
        temp_frame = ttk.Frame(params_frame)
        temp_frame.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        self.temp_var = tk.DoubleVar(value=0.7)
        self.temp_scale = ttk.Scale(temp_frame, from_=0.0, to=2.0, variable=self.temp_var,
                                     orient=tk.HORIZONTAL, length=150, command=self._update_temp_label)
        self.temp_scale.pack(side=tk.LEFT)
        self.temp_label = ttk.Label(temp_frame, text="0.70", width=5)
        self.temp_label.pack(side=tk.LEFT, padx=(5, 0))
        temp_tip = ttk.Label(params_frame, text="ℹ️", font=('Segoe UI', 10))
        temp_tip.grid(row=1, column=2, padx=(5, 0))
        self._create_tooltip(temp_tip, "Controls randomness in responses. Lower = more focused, consistent, and deterministic. Higher = more creative, varied, and unpredictable. Range: 0.0-2.0. Recommended: 0.3-0.5 for factual tasks, 0.7-0.9 for creative writing, 1.0+ for brainstorming.")
        
        # Top-p (nucleus sampling)
        ttk.Label(params_frame, text="Top-p:").grid(row=2, column=0, sticky=tk.W, pady=5)
        top_p_frame = ttk.Frame(params_frame)
        top_p_frame.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        self.top_p_var = tk.DoubleVar(value=1.0)
        self.top_p_scale = ttk.Scale(top_p_frame, from_=0.0, to=1.0, variable=self.top_p_var,
                                      orient=tk.HORIZONTAL, length=150, command=self._update_top_p_label)
        self.top_p_scale.pack(side=tk.LEFT)
        self.top_p_label = ttk.Label(top_p_frame, text="1.00", width=5)
        self.top_p_label.pack(side=tk.LEFT, padx=(5, 0))
        top_p_tip = ttk.Label(params_frame, text="ℹ️", font=('Segoe UI', 10))
        top_p_tip.grid(row=2, column=2, padx=(5, 0))
        self._create_tooltip(top_p_tip, "Nucleus sampling: only considers tokens whose cumulative probability exceeds this threshold. 1.0 = consider all possible tokens. Lower values (e.g., 0.9) = more focused, coherent output by eliminating unlikely words. Often used together with temperature. Most users should leave at 1.0.")
        
        # Top-k
        ttk.Label(params_frame, text="Top-k:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.top_k_var = tk.IntVar(value=0)
        top_k_spin = ttk.Spinbox(params_frame, from_=0, to=100, 
                                  textvariable=self.top_k_var, width=10)
        top_k_spin.grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
        top_k_tip = ttk.Label(params_frame, text="ℹ️", font=('Segoe UI', 10))
        top_k_tip.grid(row=3, column=2, padx=(5, 0))
        self._create_tooltip(top_k_tip, "Only considers the top-k most likely tokens at each step. 0 = disabled (use model's default behavior). Lower values (e.g., 10-20) = more focused and deterministic. Higher values (e.g., 40-50) = more diverse but potentially less coherent. Leave at 0 for most use cases.")
        
        # Image size
        ttk.Label(params_frame, text="Image Max Size:").grid(row=4, column=0, sticky=tk.W, pady=5)
        img_frame = ttk.Frame(params_frame)
        img_frame.grid(row=4, column=1, sticky=tk.W, padx=(10, 0))
        self.img_size_var = tk.IntVar(value=512)
        img_size_spin = ttk.Spinbox(img_frame, from_=256, to=1024, increment=128,
                                     textvariable=self.img_size_var, width=10)
        img_size_spin.pack(side=tk.LEFT)
        ttk.Label(img_frame, text="px").pack(side=tk.LEFT, padx=(5, 0))
        img_tip = ttk.Label(params_frame, text="ℹ️", font=('Segoe UI', 10))
        img_tip.grid(row=4, column=2, padx=(5, 0))
        self._create_tooltip(img_tip, "Maximum dimension (width or height) for images sent to the AI. Images are resized proportionally to fit within this size while maintaining aspect ratio. Larger sizes = more detail visible to AI but slower processing and higher API costs. Recommended: 512px (standard), 768px (detailed), 1024px (maximum detail).")
        
        # System prompt
        prompt_frame = ttk.LabelFrame(main_frame, text="System Prompt", padding=10)
        prompt_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.system_prompt_text = tk.Text(prompt_frame, height=5, wrap=tk.WORD)
        self.system_prompt_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Discard Changes", command=self.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="Save Changes", command=self._save_settings).pack(side=tk.RIGHT)
    
    def _load_current_settings(self):
        """Load current settings into the dialog"""
        self.provider_var.set(self.settings.provider)
        self.api_key_var.set(self.settings.api_key)
        self.gemini_key_var.set(self.settings.gemini_api_key)
        self.deepseek_key_var.set(self.settings.deepseek_api_key)
        self.ollama_url_var.set(self.settings.ollama_url)
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
        """Handle provider change - use cached models if available"""
        provider = self.provider_var.get()
        current_model = self.model_var.get()
        
        if provider == 'anthropic':
            # Use cached models if available, otherwise defaults
            cached = self.settings.get('cached_anthropic_models', [])
            models = cached if cached else AISettings.ANTHROPIC_MODELS
            self.model_combo['values'] = models
            # Only change model if current isn't in the list
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
        
        if provider == 'anthropic':
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
        if provider == 'anthropic':
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
                    if provider == 'anthropic':
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
        self.settings.model = self.model_var.get()
        self.settings.max_tokens = self.max_tokens_var.get()
        self.settings.temperature = self.temp_var.get()
        self.settings.top_p = self.top_p_var.get()
        self.settings.top_k = self.top_k_var.get()
        self.settings.image_max_size = (self.img_size_var.get(), self.img_size_var.get())
        self.settings.system_prompt = self.system_prompt_text.get(1.0, tk.END).strip()
        
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
                 get_document_images_callback: Optional[Callable] = None):
        super().__init__(parent, bg='#f0f0f0')
        
        self.get_document_content = get_document_content_callback
        self.get_document_images = get_document_images_callback
        
        # Settings and state
        self.settings = get_ai_settings()
        self.history_manager = get_chat_history_manager()
        self.current_document_id: Optional[str] = None
        self.llm_client: Optional[LLMClient] = None
        self.is_generating = False
        self.response_queue = queue.Queue()
        
        # Attached files storage: list of dicts with 'path', 'name', 'type', 'content'
        self.attached_files: List[Dict[str, Any]] = []
        
        # Initialize LLM client
        self._init_llm_client()
        
        self._setup_ui()
        self._setup_bindings()
        
        # Start response queue processor
        self._process_queue()
    
    def _init_llm_client(self):
        """Initialize or reinitialize the LLM client"""
        self.llm_client = get_llm_client(self.settings)
    
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
            # Ollama doesn't need API key update
            
            # Stream response with all parameters
            response_text = ""
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
                response_text += chunk
                self.response_queue.put(('chunk', chunk))
            
            # Save assistant message to history
            assistant_msg = ChatMessage(role="assistant", content=response_text)
            if self.current_document_id:
                self.history_manager.add_message(assistant_msg, self.current_document_id)
            
            self.response_queue.put(('done', None))
            
        except Exception as e:
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
                elif msg_type == 'error':
                    self._add_chat_message("system", f"Error: {content}")
                    self.is_generating = False
                    self.send_btn.config(state=tk.NORMAL)
                    self.status_label.config(text="Error")
        except queue.Empty:
            pass
        
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
    
    def _clear_display(self):
        """Clear the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
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
