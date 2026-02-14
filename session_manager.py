"""
Session Manager for MarkItDown Notepad

Handles document state caching and restoration across application sessions.
This allows documents to persist even if not explicitly saved.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib


class TabState:
    """Represents the state of a single document tab"""
    
    def __init__(
        self,
        file_path: Optional[str] = None,
        content: str = "",
        cursor_position: str = "1.0",
        scroll_position: float = 0.0,
        current_mode: str = "source",
        is_modified: bool = False
    ):
        self.file_path = file_path
        self.content = content
        self.cursor_position = cursor_position
        self.scroll_position = scroll_position
        self.current_mode = current_mode
        self.is_modified = is_modified
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'file_path': self.file_path,
            'content': self.content,
            'cursor_position': self.cursor_position,
            'scroll_position': self.scroll_position,
            'current_mode': self.current_mode,
            'is_modified': self.is_modified
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TabState':
        """Create TabState from dictionary"""
        return cls(
            file_path=data.get('file_path'),
            content=data.get('content', ''),
            cursor_position=data.get('cursor_position', '1.0'),
            scroll_position=data.get('scroll_position', 0.0),
            current_mode=data.get('current_mode', 'source'),
            is_modified=data.get('is_modified', False)
        )
    
    def get_document_hash(self) -> str:
        """Generate a unique hash for this document (used for chat history linking)"""
        if self.file_path:
            # Use file path for saved documents
            return hashlib.md5(self.file_path.encode()).hexdigest()[:16]
        else:
            # Use content hash for unsaved documents
            return hashlib.md5(self.content.encode()).hexdigest()[:16]


class SessionState:
    """Represents the complete application session state"""
    
    def __init__(self):
        self.caching_enabled: bool = True
        self.sidebar_expanded: bool = False
        self.active_tab_index: int = 0
        self.tabs: List[TabState] = []
        self.window_geometry: Optional[str] = None
        self.last_saved: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'caching_enabled': self.caching_enabled,
            'sidebar_expanded': self.sidebar_expanded,
            'active_tab_index': self.active_tab_index,
            'tabs': [tab.to_dict() for tab in self.tabs],
            'window_geometry': self.window_geometry,
            'last_saved': datetime.now().isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create SessionState from dictionary"""
        state = cls()
        state.caching_enabled = data.get('caching_enabled', True)
        state.sidebar_expanded = data.get('sidebar_expanded', False)
        state.active_tab_index = data.get('active_tab_index', 0)
        state.tabs = [TabState.from_dict(t) for t in data.get('tabs', [])]
        state.window_geometry = data.get('window_geometry')
        state.last_saved = data.get('last_saved')
        return state


class SessionManager:
    """
    Manages document state caching and restoration across application sessions.
    
    Features:
    - Auto-saves all open tabs (content, cursor position, scroll, mode)
    - Restores session on application start
    - Works with both saved and unsaved documents
    - Can be disabled by user preference
    """
    
    # Session file location
    SESSION_FILE = Path.home() / '.markitdown_session.json'
    
    def __init__(self):
        self.state = SessionState()
        self._auto_save_interval = 30000  # 30 seconds in milliseconds
        self._auto_save_job = None
    
    def load_session(self) -> SessionState:
        """
        Load session state from file.
        Returns the loaded state, or a default state if file doesn't exist.
        """
        try:
            if self.SESSION_FILE.exists():
                with open(self.SESSION_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.state = SessionState.from_dict(data)
                    
                    # Validate loaded tabs - check if file paths still exist
                    valid_tabs = []
                    for tab in self.state.tabs:
                        if tab.file_path:
                            # For saved files, check if file still exists
                            if os.path.exists(tab.file_path):
                                # Optionally reload content from file if not modified
                                if not tab.is_modified:
                                    try:
                                        with open(tab.file_path, 'r', encoding='utf-8') as doc:
                                            tab.content = doc.read()
                                    except:
                                        pass  # Keep cached content if file read fails
                                valid_tabs.append(tab)
                            else:
                                # File no longer exists - keep tab but mark as modified/unsaved
                                tab.is_modified = True
                                valid_tabs.append(tab)
                        else:
                            # Unsaved documents are always kept
                            valid_tabs.append(tab)
                    
                    self.state.tabs = valid_tabs
                    return self.state
        except Exception as e:
            print(f"Error loading session: {e}")
        
        # Return default state if loading fails
        self.state = SessionState()
        return self.state
    
    def save_session(self) -> bool:
        """
        Save current session state to file.
        Returns True if successful, False otherwise.
        """
        if not self.state.caching_enabled:
            # If caching is disabled, don't save and remove existing session file
            self.clear_session()
            return True
        
        try:
            with open(self.SESSION_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.state.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    def clear_session(self) -> bool:
        """
        Clear the saved session file.
        Called when caching is disabled or when user explicitly clears.
        """
        try:
            if self.SESSION_FILE.exists():
                self.SESSION_FILE.unlink()
            self.state = SessionState()
            return True
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False
    
    def update_tab_state(
        self,
        index: int,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        cursor_position: Optional[str] = None,
        scroll_position: Optional[float] = None,
        current_mode: Optional[str] = None,
        is_modified: Optional[bool] = None
    ):
        """Update the state of a specific tab"""
        # Ensure we have enough tabs
        while len(self.state.tabs) <= index:
            self.state.tabs.append(TabState())
        
        tab = self.state.tabs[index]
        
        if file_path is not None:
            tab.file_path = file_path
        if content is not None:
            tab.content = content
        if cursor_position is not None:
            tab.cursor_position = cursor_position
        if scroll_position is not None:
            tab.scroll_position = scroll_position
        if current_mode is not None:
            tab.current_mode = current_mode
        if is_modified is not None:
            tab.is_modified = is_modified
    
    def add_tab(self, tab_state: Optional[TabState] = None) -> int:
        """Add a new tab and return its index"""
        if tab_state is None:
            tab_state = TabState()
        self.state.tabs.append(tab_state)
        return len(self.state.tabs) - 1
    
    def remove_tab(self, index: int):
        """Remove a tab at the given index"""
        if 0 <= index < len(self.state.tabs):
            self.state.tabs.pop(index)
            # Adjust active tab index if needed
            if self.state.active_tab_index >= len(self.state.tabs):
                self.state.active_tab_index = max(0, len(self.state.tabs) - 1)
    
    def get_tab_state(self, index: int) -> Optional[TabState]:
        """Get the state of a specific tab"""
        if 0 <= index < len(self.state.tabs):
            return self.state.tabs[index]
        return None
    
    def set_active_tab(self, index: int):
        """Set the active tab index"""
        self.state.active_tab_index = index
    
    def set_sidebar_expanded(self, expanded: bool):
        """Set sidebar expanded state"""
        self.state.sidebar_expanded = expanded
    
    def set_window_geometry(self, geometry: str):
        """Set window geometry string"""
        self.state.window_geometry = geometry
    
    def set_caching_enabled(self, enabled: bool):
        """Enable or disable caching"""
        self.state.caching_enabled = enabled
        if not enabled:
            self.clear_session()
    
    def is_caching_enabled(self) -> bool:
        """Check if caching is enabled"""
        return self.state.caching_enabled
    
    def has_saved_session(self) -> bool:
        """Check if a saved session exists"""
        return self.SESSION_FILE.exists()
    
    def get_document_hash(self, tab_index: int) -> Optional[str]:
        """Get the document hash for a specific tab (used for chat history)"""
        tab = self.get_tab_state(tab_index)
        if tab:
            return tab.get_document_hash()
        return None


# Singleton instance for easy access
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the singleton SessionManager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
