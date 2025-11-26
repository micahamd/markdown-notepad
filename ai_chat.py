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

# Try to import Google Gemini
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Install with: pip install google-genai")


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
    
    DEFAULT_SETTINGS = {
        'provider': 'anthropic',
        'api_key': '',
        'gemini_api_key': '',  # Separate API key for Gemini
        'model': 'claude-sonnet-4-20250514',
        'system_prompt': 'You are a helpful AI assistant. You help users with their markdown documents, answer questions, and provide writing assistance.',
        'max_tokens': 4096,
        'temperature': 0.7,
        'top_p': 1.0,  # Nucleus sampling: consider tokens with top_p cumulative probability
        'top_k': 0,    # Consider only top_k tokens (0 = disabled/use default)
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
    
    def is_configured(self) -> bool:
        """Check if API key is configured for current provider"""
        if self.provider == 'gemini':
            return bool(self.gemini_api_key)
        return bool(self.api_key)
    
    def get_available_models(self) -> List[str]:
        """Get available models for current provider"""
        if self.provider == 'anthropic':
            return self.ANTHROPIC_MODELS.copy()
        elif self.provider == 'gemini':
            return self.GEMINI_MODELS.copy()
        return []
    
    def get_current_api_key(self) -> str:
        """Get API key for current provider"""
        if self.provider == 'gemini':
            return self.gemini_api_key
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
            response = self.client.models.generate_content_stream(
                model=model_name,
                contents=api_contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt if system_prompt else None,
                    **gen_config
                )
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
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
                                       values=["anthropic", "gemini"], state="readonly", width=30)
        provider_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)
        
        # API Keys Frame
        api_frame = ttk.LabelFrame(main_frame, text="API Keys", padding=10)
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
        self._create_tooltip(refresh_btn, "Fetch available models from the API")
        
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
        self._create_tooltip(max_tokens_tip, "Maximum number of tokens to generate in the response.")
        
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
        self._create_tooltip(temp_tip, "Controls randomness. Lower = more focused and deterministic. Higher = more creative and varied. Range: 0.0-2.0")
        
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
        self._create_tooltip(top_p_tip, "Nucleus sampling: only consider tokens whose cumulative probability exceeds this threshold. 1.0 = consider all tokens. Lower values = more focused output.")
        
        # Top-k
        ttk.Label(params_frame, text="Top-k:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.top_k_var = tk.IntVar(value=0)
        top_k_spin = ttk.Spinbox(params_frame, from_=0, to=100, 
                                  textvariable=self.top_k_var, width=10)
        top_k_spin.grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
        top_k_tip = ttk.Label(params_frame, text="ℹ️", font=('Segoe UI', 10))
        top_k_tip.grid(row=3, column=2, padx=(5, 0))
        self._create_tooltip(top_k_tip, "Only consider the top-k most likely tokens. 0 = disabled (use model default). Lower values = more focused, higher = more diverse.")
        
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
        self._create_tooltip(img_tip, "Maximum dimension for images sent to the AI. Larger = more detail but slower/costlier.")
        
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
    
    def _update_temp_label(self, *args):
        """Update temperature label"""
        self.temp_label.config(text=f"{self.temp_var.get():.2f}")
    
    def _update_top_p_label(self, *args):
        """Update top-p label"""
        self.top_p_label.config(text=f"{self.top_p_var.get():.2f}")
    
    def _on_provider_change(self, event=None):
        """Handle provider change"""
        provider = self.provider_var.get()
        if provider == 'anthropic':
            self.model_combo['values'] = AISettings.ANTHROPIC_MODELS
            if self.model_var.get() not in AISettings.ANTHROPIC_MODELS:
                self.model_var.set(AISettings.ANTHROPIC_MODELS[0])
        elif provider == 'gemini':
            self.model_combo['values'] = AISettings.GEMINI_MODELS
            if self.model_var.get() not in AISettings.GEMINI_MODELS:
                self.model_var.set(AISettings.GEMINI_MODELS[0])
    
    def _refresh_models(self):
        """Fetch available models from the API"""
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
                    messagebox.showinfo("Success", f"Found {len(models)} Anthropic models.", parent=self)
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
                    messagebox.showinfo("Success", f"Found {len(models)} Gemini models.", parent=self)
                else:
                    messagebox.showwarning("Warning", "No models found.", parent=self)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to fetch models: {str(e)}", parent=self)
    
    def _test_connection(self, provider: str = None):
        """Test API connection for specified provider"""
        if provider is None:
            provider = self.provider_var.get()
        
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
        else:
            messagebox.showwarning("Warning", "Unknown provider.", parent=self)
            return
        
        if success:
            messagebox.showinfo("Success", message, parent=self)
        else:
            messagebox.showerror("Error", message, parent=self)
    
    def _save_settings(self):
        """Save settings and close dialog"""
        self.settings.provider = self.provider_var.get()
        self.settings.api_key = self.api_key_var.get()
        self.settings.gemini_api_key = self.gemini_key_var.get()
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
            # Update client settings based on provider
            if self.settings.provider == 'gemini':
                self.llm_client.api_key = self.settings.gemini_api_key
            else:
                self.llm_client.api_key = self.settings.api_key
            
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
