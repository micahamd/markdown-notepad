"""
AI Chat Module for MarkItDown Notepad

Provides AI chatbot functionality with support for multiple LLM providers.
Currently implements Anthropic Claude, with architecture ready for Google, Deepseek, etc.
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
from tkinter import ttk, messagebox, scrolledtext

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
        "claude-3-5-haiku-20241022",
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ]
    
    DEFAULT_SETTINGS = {
        'provider': 'anthropic',
        'api_key': '',
        'model': 'claude-3-5-haiku-20241022',
        'system_prompt': 'You are a helpful AI assistant. You help users with their markdown documents, answer questions, and provide writing assistance.',
        'max_tokens': 4096,
        'temperature': 0.7,
        'image_max_size': [512, 512],
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
    
    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)
    
    def get_available_models(self) -> List[str]:
        """Get available models for current provider"""
        if self.provider == 'anthropic':
            return self.ANTHROPIC_MODELS.copy()
        return []


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
            with self.client.messages.stream(
                model=model_name,
                messages=api_messages,
                system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
                max_tokens=max_tokens,
                temperature=temperature,
            ) as stream:
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
# LLM Client Factory
# =============================================================================

def get_llm_client(settings: AISettings) -> Optional[LLMClient]:
    """Factory function to get appropriate LLM client based on settings"""
    provider = settings.provider
    api_key = settings.api_key
    
    if provider == 'anthropic':
        if not ANTHROPIC_AVAILABLE:
            return None
        return AnthropicClient(api_key)
    
    # Future providers can be added here:
    # elif provider == 'google':
    #     return GoogleClient(api_key)
    # elif provider == 'deepseek':
    #     return DeepseekClient(api_key)
    
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
        self.geometry("520x650")
        self.resizable(True, True)
        self.minsize(480, 600)
        self.transient(parent)
        self.grab_set()
        
        self._setup_ui()
        self._load_current_settings()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
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
                                       values=["anthropic"], state="readonly", width=30)
        provider_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)
        
        # API Key
        api_frame = ttk.LabelFrame(main_frame, text="API Key", padding=10)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="•")
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.show_key_var = tk.BooleanVar(value=False)
        self.show_key_btn = ttk.Checkbutton(api_frame, text="Show", variable=self.show_key_var,
                                             command=self._toggle_key_visibility)
        self.show_key_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Button(api_frame, text="Test", command=self._test_connection, width=8).pack(side=tk.LEFT, padx=(5, 0))
        
        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                         values=AISettings.ANTHROPIC_MODELS, width=35)
        self.model_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Refresh models button
        ttk.Button(model_frame, text="🔄", width=3, 
                   command=self._refresh_models).grid(row=0, column=2, padx=(5, 0))
        
        # Parameters
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Max tokens
        ttk.Label(params_frame, text="Max Tokens:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.max_tokens_var = tk.IntVar(value=4096)
        max_tokens_spin = ttk.Spinbox(params_frame, from_=256, to=8192, 
                                       textvariable=self.max_tokens_var, width=10)
        max_tokens_spin.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Temperature
        ttk.Label(params_frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.temp_var = tk.DoubleVar(value=0.7)
        temp_frame = ttk.Frame(params_frame)
        temp_frame.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        self.temp_scale = ttk.Scale(temp_frame, from_=0.0, to=1.0, variable=self.temp_var,
                                     orient=tk.HORIZONTAL, length=150, command=self._update_temp_label)
        self.temp_scale.pack(side=tk.LEFT)
        self.temp_label = ttk.Label(temp_frame, text="0.7", width=5)
        self.temp_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Image size
        ttk.Label(params_frame, text="Image Max Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.img_size_var = tk.IntVar(value=512)
        img_size_spin = ttk.Spinbox(params_frame, from_=256, to=1024, increment=128,
                                     textvariable=self.img_size_var, width=10)
        img_size_spin.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Label(params_frame, text="px").grid(row=2, column=2, sticky=tk.W)
        
        # System prompt
        prompt_frame = ttk.LabelFrame(main_frame, text="System Prompt", padding=10)
        prompt_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.system_prompt_text = tk.Text(prompt_frame, height=6, wrap=tk.WORD)
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
        self.model_var.set(self.settings.model)
        self.max_tokens_var.set(self.settings.max_tokens)
        self.temp_var.set(self.settings.temperature)
        self.img_size_var.set(self.settings.image_max_size[0])
        self.system_prompt_text.insert(1.0, self.settings.system_prompt)
        self._update_temp_label()
        
        # Hide API key if already set
        if self.settings.api_key:
            self.api_key_entry.config(show="•")
    
    def _toggle_key_visibility(self):
        """Toggle API key visibility"""
        if self.show_key_var.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="•")
    
    def _update_temp_label(self, *args):
        """Update temperature label"""
        self.temp_label.config(text=f"{self.temp_var.get():.2f}")
    
    def _on_provider_change(self, event=None):
        """Handle provider change"""
        provider = self.provider_var.get()
        if provider == 'anthropic':
            self.model_combo['values'] = AISettings.ANTHROPIC_MODELS
            if self.model_var.get() not in AISettings.ANTHROPIC_MODELS:
                self.model_var.set(AISettings.ANTHROPIC_MODELS[0])
    
    def _refresh_models(self):
        """Fetch available models from the API"""
        api_key = self.api_key_var.get()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key first.", parent=self)
            return
        
        provider = self.provider_var.get()
        if provider == 'anthropic':
            try:
                client = AnthropicClient(api_key)
                models = client.get_available_models()
                if models:
                    self.model_combo['values'] = models
                    messagebox.showinfo("Success", f"Found {len(models)} models.", parent=self)
                else:
                    messagebox.showwarning("Warning", "No models found.", parent=self)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to fetch models: {str(e)}", parent=self)
    
    def _test_connection(self):
        """Test API connection"""
        api_key = self.api_key_var.get()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key first.", parent=self)
            return
        
        provider = self.provider_var.get()
        if provider == 'anthropic':
            client = AnthropicClient(api_key)
            success, message = client.test_connection()
            
            if success:
                messagebox.showinfo("Success", message, parent=self)
            else:
                messagebox.showerror("Error", message, parent=self)
    
    def _save_settings(self):
        """Save settings and close dialog"""
        self.settings.provider = self.provider_var.get()
        self.settings.api_key = self.api_key_var.get()
        self.settings.model = self.model_var.get()
        self.settings.max_tokens = self.max_tokens_var.get()
        self.settings.temperature = self.temp_var.get()
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
        
        tk.Label(header_frame, text="💬 AI Assistant", font=('Segoe UI', 11, 'bold'),
                 bg='#e0e0e0').pack(side=tk.LEFT, padx=5)
        
        # Settings button
        settings_btn = ttk.Button(header_frame, text="⚙️", width=3, 
                                   command=self._open_settings)
        settings_btn.pack(side=tk.RIGHT, padx=2)
        
        # Clear history button
        clear_btn = ttk.Button(header_frame, text="🗑️", width=3,
                                command=self._clear_history)
        clear_btn.pack(side=tk.RIGHT, padx=2)
        
        # Chat display area
        chat_frame = tk.Frame(self, bg='#f0f0f0')
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            bg='#ffffff',
            fg='#1a1a1a',
            state=tk.DISABLED,
            height=15
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for styling
        self.chat_display.tag_configure('user', foreground='#0066cc', font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_configure('assistant', foreground='#006600', font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_configure('system', foreground='#666666', font=('Segoe UI', 9, 'italic'))
        self.chat_display.tag_configure('error', foreground='#cc0000')
        self.chat_display.tag_configure('message', font=('Segoe UI', 10))
        
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
        
        # Include history checkbox
        self.include_history_var = tk.BooleanVar(value=True)
        self.include_history_cb = ttk.Checkbutton(
            options_frame,
            text="Include Chat History",
            variable=self.include_history_var
        )
        self.include_history_cb.pack(side=tk.LEFT)
        
        # Input area
        input_frame = tk.Frame(self, bg='#f0f0f0')
        input_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.input_text = tk.Text(
            input_frame,
            height=3,
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            bg='#ffffff',
            fg='#1a1a1a'
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Send button
        self.send_btn = ttk.Button(input_frame, text="Send", command=self._send_message)
        self.send_btn.pack(side=tk.RIGHT, padx=(5, 0), pady=2)
        
        # Status label
        self.status_label = tk.Label(self, text="Ready", bg='#f0f0f0', fg='#666666',
                                      font=('Segoe UI', 9))
        self.status_label.pack(fill=tk.X, padx=5, pady=(0, 5))
    
    def _setup_bindings(self):
        """Setup keyboard bindings"""
        # Enter to send (but Ctrl+Enter for newline)
        self.input_text.bind('<Return>', self._on_enter)
        self.input_text.bind('<Control-Return>', self._on_ctrl_enter)
    
    def _on_enter(self, event):
        """Handle Enter key - send message"""
        if not self.is_generating:
            self._send_message()
        return 'break'  # Prevent default newline
    
    def _on_ctrl_enter(self, event):
        """Handle Ctrl+Enter - insert newline"""
        self.input_text.insert(tk.INSERT, '\n')
        return 'break'
    
    def _send_message(self):
        """Send message to AI"""
        if self.is_generating:
            return
        
        # Get user message
        message = self.input_text.get(1.0, tk.END).strip()
        if not message:
            return
        
        # Check if client is available
        if not self.llm_client:
            self._init_llm_client()
            if not self.llm_client:
                self._add_chat_message("system", "AI not configured. Please set up your API key in Settings.")
                return
        
        if not self.settings.api_key:
            self._add_chat_message("system", "API key not configured. Please add your API key in Settings.")
            return
        
        # Clear input
        self.input_text.delete(1.0, tk.END)
        
        # Add user message to display
        self._add_chat_message("user", message)
        
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
        
        # Get images if available
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
        
        content_parts.append(message)
        full_content = "".join(content_parts)
        
        messages.append({"role": "user", "content": full_content})
        
        # Save user message to history
        user_msg = ChatMessage(role="user", content=message)
        if self.current_document_id:
            self.history_manager.add_message(self.current_document_id, user_msg)
        
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
            # Update client settings
            self.llm_client.api_key = self.settings.api_key
            
            # Stream response
            response_text = ""
            for chunk in self.llm_client.send_message_stream(
                messages,
                system_prompt=self.settings.system_prompt,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                images=images if images else None
            ):
                response_text += chunk
                self.response_queue.put(('chunk', chunk))
            
            # Save assistant message to history
            assistant_msg = ChatMessage(role="assistant", content=response_text)
            if self.current_document_id:
                self.history_manager.add_message(self.current_document_id, assistant_msg)
            
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
