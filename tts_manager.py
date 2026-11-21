"""
Local Background TTS Dictation Engine & Dictation Properties Dialog

Integrates Software Mansion's Reverb-TTS (`software-mansion/Reverb-TTS`, default `int8` variant)
with pipelined chunked streaming playback, markdown preprocessing, persistent zero-shot custom
voice cloning (.npz codec cache), and the `Dictation Properties` configuration interface.
"""

from __future__ import annotations

import os
import re
import json
import time
import wave
import queue
import shutil
import subprocess
import importlib.util
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

REVERB_TTS_AVAILABLE = importlib.util.find_spec("reverb_tts") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
WHISPER_AVAILABLE = importlib.util.find_spec("whisper") is not None

TTS: Any = None
Voice: Any = None

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None  # type: ignore
    SOUNDDEVICE_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None  # type: ignore
    SOUNDFILE_AVAILABLE = False

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    winsound = None  # type: ignore
    WINSOUND_AVAILABLE = False


# =============================================================================
# Constants, Supported Languages, Variants & Preset Voices
# =============================================================================

REPO_ID = "software-mansion/Reverb-TTS"

SUPPORTED_VARIANTS = ("int8", "int4", "fp16", "fp32")
VARIANT_LABELS: Dict[str, str] = {
    "int8": "int8 (307 MB — Recommended)",
    "int4": "int4 (191 MB — Compact)",
    "fp16": "fp16 (594 MB — Half Precision)",
    "fp32": "fp32 (1.19 GB — Full Precision)",
}

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pl": "Polish",
    "pt": "Portuguese",
}

LANGUAGE_SAMPLES: Dict[str, str] = {
    "en": "Hello! This is local dictation powered by Reverb text to speech.",
    "de": "Hallo! Dies ist eine lokale Sprachausgabe mit Reverb Text-to-Speech.",
    "fr": "Bonjour! Ceci est une lecture vocale locale avec Reverb.",
    "es": "¡Hola! Esta es una lectura de voz local con Reverb.",
    "it": "Ciao! Questa è una lettura vocale locale con Reverb.",
    "pl": "Cześć! To jest lokalne czytanie tekstu z użyciem Reverb.",
    "pt": "Olá! Esta é uma leitura de voz local com Reverb.",
}

DEFAULT_PRESET_VOICES: Tuple[str, ...] = (
    "en_female_narrator",
    "en_female_whisper",
    "en_male_narrator_1",
    "en_male_narrator_2",
    "en_male_angry",
    "de_female_standard",
    "de_male_standard",
    "es_male_standard_1",
    "es_male_standard_2",
    "fr_female_standard",
    "fr_male_standard",
    "it_male_standard",
    "it_male_expressive",
    "pl_male_standard",
    "pt_female_standard",
    "pt_male_standard",
)

TTS_DIR = Path.home() / ".markitdown_tts"
VOICES_DIR = TTS_DIR / "voices"
SETTINGS_FILE = TTS_DIR / "settings.json"
MAX_REFERENCE_AUDIO_BYTES = 100 * 1024 * 1024  # 100 MB safety cap


def _restrict_permissions(path: Path, is_dir: bool = False) -> None:
    """Best-effort restriction of local TTS config and voice cache paths to owner access."""
    try:
        if path and Path(path).exists():
            os.chmod(path, 0o700 if is_dir else 0o600)
    except Exception:
        pass


def _sanitize_voice_id(name: str) -> str:
    """Convert an arbitrary display name into a safe filesystem stem."""
    cleaned = re.sub(r"[^a-z0-9_-]+", "_", (name or "").strip().lower()).strip("_-")
    if not cleaned:
        cleaned = "custom_voice"
    return cleaned[:48]


def _resolve_safe_voice_path(voices_dir: Path, filename: str) -> Path:
    """Resolve a voice cache filename strictly inside `voices_dir`, blocking traversal."""
    if not filename or not isinstance(filename, str):
        raise ValueError("Invalid voice cache filename.")
    candidate_name = filename.strip()
    if Path(candidate_name).name != candidate_name or ".." in candidate_name:
        raise ValueError(f"Unsafe voice cache path rejected: {filename!r}")
    base_resolved = voices_dir.resolve()
    target_resolved = (base_resolved / candidate_name).resolve()
    if os.path.commonpath([str(base_resolved), str(target_resolved)]) != str(base_resolved):
        raise ValueError(f"Path traversal outside voice directory rejected: {filename!r}")
    return target_resolved


def format_preset_label(voice_name: str) -> str:
    """Format a preset voice identifier like 'en_female_narrator' into a readable label."""
    parts = (voice_name or "").split("_")
    lang_code = parts[0] if parts else ""
    lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())
    desc = " ".join(p.capitalize() for p in parts[1:]) if len(parts) > 1 else voice_name
    if lang_name and desc:
        return f"{lang_name} — {desc} ({voice_name})"
    return voice_name


def variant_to_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, VARIANT_LABELS["int8"])


def label_to_variant(label: str) -> str:
    for code, lbl in VARIANT_LABELS.items():
        if label == lbl or label == code:
            return code
    prefix = (label or "").split()[0].strip().lower()
    return prefix if prefix in SUPPORTED_VARIANTS else "int8"


def language_to_label(code: str) -> str:
    name = SUPPORTED_LANGUAGES.get(code, "English")
    return f"{name} ({code})"


def label_to_language(label: str) -> str:
    m = re.search(r"\(([a-z]{2})\)\s*$", (label or "").strip())
    if m and m.group(1) in SUPPORTED_LANGUAGES:
        return m.group(1)
    low = (label or "").strip().lower()
    if low in SUPPORTED_LANGUAGES:
        return low
    for code, name in SUPPORTED_LANGUAGES.items():
        if name.lower() == low:
            return code
    return "en"


def _is_soundfile_readable(audio_path: str) -> bool:
    """Return True when soundfile/libsndfile can open and inspect `audio_path` directly."""
    if not SOUNDFILE_AVAILABLE or sf is None:
        return False
    try:
        info = sf.info(audio_path)
        return bool(info.frames > 0 and info.samplerate > 0)
    except Exception:
        return False


def _transcode_to_temp_wav(audio_path: Path) -> Path:
    """Transcode compressed/container audio (e.g. .m4a) to a mono 24 kHz temporary WAV via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"Format '{audio_path.suffix}' requires ffmpeg on PATH (or convert to WAV/MP3/FLAC/OGG)."
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_wav = Path(tf.name)
    try:
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", str(audio_path), "-ac", "1", "-ar", "24000", "-f", "wav", str(tmp_wav)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode != 0 or not tmp_wav.is_file() or tmp_wav.stat().st_size == 0:
            err_lines = (proc.stderr or b"").decode("utf-8", errors="replace").strip().splitlines()
            tail = err_lines[-1] if err_lines else f"exit code {proc.returncode}"
            raise RuntimeError(f"Failed to decode '{audio_path.name}' via ffmpeg: {tail}")
        _restrict_permissions(tmp_wav, is_dir=False)
        return tmp_wav
    except Exception:
        try:
            if tmp_wav.exists():
                tmp_wav.unlink()
        except Exception:
            pass
        raise


def get_audio_duration(audio_path: str) -> Optional[float]:
    """Return duration in seconds for an audio file, or None if unavailable."""
    if not audio_path or not os.path.isfile(audio_path):
        return None
    if SOUNDFILE_AVAILABLE:
        try:
            info = sf.info(audio_path)
            if info.frames > 0 and info.samplerate > 0:
                return float(info.frames) / float(info.samplerate)
        except Exception:
            pass
    try:
        with wave.open(audio_path, "rb") as wf:
            fr = wf.getframerate()
            if fr > 0:
                return float(wf.getnframes()) / float(fr)
    except Exception:
        pass
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            proc = subprocess.run(
                [
                    ffprobe, "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if proc.returncode == 0 and (proc.stdout or "").strip():
                val = float(proc.stdout.strip())
                if val > 0:
                    return val
        except Exception:
            pass
    return None


# =============================================================================
# Screen-Clamped Tooltip Helper
# =============================================================================

class _DialogToolTip:
    """Screen-edge-clamped tooltip for Dictation Properties controls."""

    def __init__(self, widget: tk.Widget, text: str, delay: int = 380):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_win: Optional[tk.Toplevel] = None
        self.after_id: Optional[str] = None
        widget.bind("<Enter>", self._on_enter, add=True)
        widget.bind("<Leave>", self._on_leave, add=True)
        widget.bind("<Button>", self._on_leave, add=True)

    def _on_enter(self, _e=None):
        self._cancel()
        self.after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, _e=None):
        self._cancel()
        if self.tip_win:
            try:
                self.tip_win.destroy()
            except Exception:
                pass
            self.tip_win = None

    def _cancel(self):
        if self.after_id:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def _show(self):
        if self.tip_win or not self.widget.winfo_exists():
            return
        try:
            wx = self.widget.winfo_rootx()
            wy = self.widget.winfo_rooty()
            ww = self.widget.winfo_width()
            wh = self.widget.winfo_height()
        except Exception:
            return
        self.tip_win = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        lbl = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#2e1065", foreground="#faf5ff",
            relief=tk.SOLID, borderwidth=1,
            font=("Segoe UI", 9), padx=8, pady=5, wraplength=320,
        )
        lbl.pack()
        tw.update_idletasks()
        tip_w, tip_h = tw.winfo_reqwidth(), tw.winfo_reqheight()
        x = max(12, min(wx + (ww - tip_w) // 2, tw.winfo_screenwidth() - tip_w - 12))
        y = wy + wh + 6
        if y + tip_h > tw.winfo_screenheight() - 12:
            y = max(12, wy - tip_h - 6)
        tw.wm_geometry(f"+{x}+{y}")


# =============================================================================
# Persistent TTS Settings & Custom Voice Registry
# =============================================================================

@dataclass
class TTSSettings:
    """User-configurable Reverb-TTS settings & saved custom voice records."""

    variant: str = "int8"
    device: str = "auto"
    voice_type: str = "preset"       # "preset" | "custom"
    voice_id: str = "en_female_narrator"
    language: str = "en"
    speed: float = 1.0
    pace: str = "speech"             # "speech" | "recording"
    steps: int = 8
    seed: Optional[int] = None
    max_chars_per_chunk: int = 320
    max_input_chars: int = 0         # 0 = unlimited
    skip_code_blocks: bool = True
    auto_read_ai_replies: bool = False
    custom_voices: List[Dict[str, Any]] = field(default_factory=list)

    def normalize(self) -> None:
        if self.variant not in SUPPORTED_VARIANTS:
            self.variant = "int8"
        if self.device not in ("auto", "cuda", "cpu"):
            self.device = "auto"
        if self.voice_type not in ("preset", "custom"):
            self.voice_type = "preset"
        if self.language not in SUPPORTED_LANGUAGES:
            self.language = "en"
        try:
            self.speed = max(0.6, min(1.8, float(self.speed)))
        except (TypeError, ValueError):
            self.speed = 1.0
        if self.pace not in ("speech", "recording"):
            self.pace = "speech"
        try:
            self.steps = max(4, min(16, int(self.steps)))
        except (TypeError, ValueError):
            self.steps = 8
        if self.seed is not None:
            try:
                self.seed = max(0, int(self.seed))
            except (TypeError, ValueError):
                self.seed = None
        try:
            self.max_chars_per_chunk = max(100, min(2000, int(self.max_chars_per_chunk)))
        except (TypeError, ValueError):
            self.max_chars_per_chunk = 320
        try:
            self.max_input_chars = max(0, min(100000, int(self.max_input_chars)))
        except (TypeError, ValueError):
            self.max_input_chars = 0
        self.skip_code_blocks = bool(self.skip_code_blocks)
        self.auto_read_ai_replies = bool(self.auto_read_ai_replies)

        clean_voices: List[Dict[str, Any]] = []
        seen_ids = set()
        if isinstance(self.custom_voices, list):
            for entry in self.custom_voices:
                if not isinstance(entry, dict):
                    continue
                vid = _sanitize_voice_id(str(entry.get("id") or entry.get("name") or ""))
                npz_file = str(entry.get("npz_file") or "")
                if not npz_file or Path(npz_file).name != npz_file or ".." in npz_file:
                    continue
                if vid in seen_ids:
                    continue
                seen_ids.add(vid)
                clean_voices.append({
                    "id": vid,
                    "name": str(entry.get("name") or vid),
                    "language": entry.get("language") if entry.get("language") in SUPPORTED_LANGUAGES else "en",
                    "transcript": str(entry.get("transcript") or ""),
                    "npz_file": npz_file,
                    "source_audio": str(entry.get("source_audio") or ""),
                    "duration_sec": float(entry.get("duration_sec") or 0.0),
                    "created_at": str(entry.get("created_at") or ""),
                })
        self.custom_voices = clean_voices

        if self.voice_type == "custom" and self.get_custom_voice(self.voice_id) is None:
            self.voice_type = "preset"
            self.voice_id = "en_female_narrator"
        elif self.voice_type == "preset" and not self.voice_id:
            self.voice_id = "en_female_narrator"

    def get_custom_voice(self, voice_id: str) -> Optional[Dict[str, Any]]:
        for entry in self.custom_voices:
            if entry.get("id") == voice_id:
                return entry
        return None

    def add_or_update_custom_voice(self, record: Dict[str, Any]) -> None:
        vid = _sanitize_voice_id(str(record.get("id") or record.get("name") or ""))
        record = dict(record)
        record["id"] = vid
        for i, existing in enumerate(self.custom_voices):
            if existing.get("id") == vid:
                self.custom_voices[i] = record
                self.normalize()
                return
        self.custom_voices.append(record)
        self.normalize()

    def remove_custom_voice(self, voice_id: str, voices_dir: Optional[Path] = None) -> bool:
        target = self.get_custom_voice(voice_id)
        if target is None:
            return False
        vdir = voices_dir or VOICES_DIR
        npz_name = target.get("npz_file", "")
        if npz_name:
            try:
                safe_path = _resolve_safe_voice_path(vdir, npz_name)
                if safe_path.is_file():
                    safe_path.unlink()
            except Exception:
                pass
        self.custom_voices = [v for v in self.custom_voices if v.get("id") != voice_id]
        if self.voice_type == "custom" and self.voice_id == voice_id:
            self.voice_type = "preset"
            self.voice_id = "en_female_narrator"
        self.normalize()
        return True

    def save(self, settings_file: Optional[Path] = None) -> None:
        self.normalize()
        target = Path(settings_file) if settings_file else SETTINGS_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        _restrict_permissions(target.parent, is_dir=True)

        data = asdict(self)
        tmp_path = target.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, target)
        _restrict_permissions(target, is_dir=False)

    @classmethod
    def load(cls, settings_file: Optional[Path] = None) -> "TTSSettings":
        target = Path(settings_file) if settings_file else SETTINGS_FILE
        if target.is_file():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    valid_fields = cls.__dataclass_fields__.keys()
                    filtered = {k: v for k, v in raw.items() if k in valid_fields}
                    obj = cls(**filtered)
                    obj.normalize()
                    return obj
            except Exception as e:
                print(f"Warning: could not load TTS settings ({e}), using defaults.")
        obj = cls()
        obj.normalize()
        return obj


# =============================================================================
# Markdown Preprocessing & Sentence-Aware Chunking
# =============================================================================

_FENCED_CODE_RE = re.compile(r"```[^\n]*\n([\s\S]*?)```|~~~[^\n]*\n([\s\S]*?)~~~")
_TABLE_SEP_ROW_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+")


def clean_markdown_for_speech(text: str, skip_code_blocks: bool = True) -> str:
    """Convert Markdown prose into natural spoken text for TTS dictation."""
    if not text or not text.strip():
        return ""

    s = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1. Handle fenced code blocks
    def _code_repl(m):
        if skip_code_blocks:
            return "\nCode block omitted.\n"
        body = m.group(1) if m.group(1) is not None else (m.group(2) or "")
        return f"\n{body.strip()}\n"

    s = _FENCED_CODE_RE.sub(_code_repl, s)

    # 2. Strip footnote definitions & inline footnote markers
    s = re.sub(r"^\s*\[\^[^\]\n]+\]:\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\[\^[^\]\n]+\]", "", s)

    # 3. Images: keep alt text if descriptive, else drop
    s = re.sub(r"!\[([^\]]*)\]\([^)\n]+\)", lambda m: m.group(1).strip() if m.group(1).strip() else "", s)

    # 4. Links: keep label, discard URL
    s = re.sub(r"\[([^\]]+)\]\([^)\n]+\)", r"\1", s)

    # 5. Unified cross-references & anchors ([@anchor: Foo] / [@Foo])
    s = re.sub(r"\[@(?:anchor|def|ref)\s*:\s*([^\]\n]+)\]", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\[@([^\]\n]+)\]", r"\1", s)

    # 6. Strip bare HTTP/HTTPS URLs so TTS does not spell them out
    s = re.sub(r"https?://\S+", "", s)

    # 6b. Strip alignment, font-color, and highlight HTML tags while preserving inner text
    s = re.sub(r"<div\s+align=[^>]*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</div>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<span\s+style=[^>]*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</span>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</?mark>", "", s, flags=re.IGNORECASE)

    # 7. Strip inline code ticks & emphasis markers cleanly
    s = re.sub(r"`([^`\n]+)`", r"\1", s)
    s = re.sub(r"\*\*\*([^*\n]+)\*\*\*", r"\1", s)
    s = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", s)
    s = re.sub(r"__([^_\n]+)__", r"\1", s)
    s = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", s)
    s = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", s)
    s = re.sub(r"~~([^~\n]+)~~", r"\1", s)
    s = re.sub(r"==([^=\n]+)==", r"\1", s)

    # 8. Strip inline math $ delimiters without altering symbols/words inside
    s = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: " " + m.group(1).strip() + " ", s)
    s = re.sub(r"\$([^$\n]+)\$", r"\1", s)

    # 9. Line-by-line structural cleanup (headings, quotes, rules, lists, tables)
    out_lines: List[str] = []
    for raw_line in s.split("\n"):
        line = raw_line.strip()
        if not line:
            out_lines.append("")
            continue
        if re.match(r"^(?:-{3,}|\*{3,}|_{3,})$", line):
            continue
        if _TABLE_SEP_ROW_RE.match(line):
            continue
        if line.startswith("|") and "|" in line[1:]:
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            if cells:
                row_prose = ", ".join(cells)
                if not row_prose.endswith((".", "!", "?", ";")):
                    row_prose += "."
                out_lines.append(row_prose)
            continue
        hm = re.match(r"^#{1,6}\s+(.*)$", line)
        if hm:
            heading = hm.group(1).strip()
            if heading and not heading.endswith((".", "!", "?", ":")):
                heading += "."
            out_lines.append(heading)
            continue
        line = re.sub(r"^>+\s*", "", line)
        line = re.sub(r"^(?:[-*•+]|\d+[.)])\s+", "", line)
        line = line.strip()
        if line:
            out_lines.append(line)

    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_into_dictation_chunks(
    text: str,
    max_chars_per_chunk: int = 320,
    max_input_chars: int = 0,
) -> List[str]:
    """Split text into sentence/clause chunks bounded by `max_chars_per_chunk` and optional total cap."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    chunk_limit = max(80, min(2000, int(max_chars_per_chunk)))
    if max_input_chars and max_input_chars > 0 and len(cleaned) > max_input_chars:
        sliced = cleaned[:max_input_chars]
        # Prefer ending at a clean sentence or word boundary
        m_sent = list(re.finditer(r"[.!?…](?:\s|$)", sliced))
        if m_sent and m_sent[-1].end() >= int(max_input_chars * 0.55):
            cleaned = sliced[:m_sent[-1].end()].strip()
        else:
            last_sp = sliced.rfind(" ")
            cleaned = (sliced[:last_sp] if last_sp > int(max_input_chars * 0.5) else sliced).strip()

    if not cleaned:
        return []

    def _split_oversized_sentence(sent: str) -> List[str]:
        if len(sent) <= chunk_limit:
            return [sent]
        parts: List[str] = []
        # First try clause boundaries (, ; :)
        clauses = re.split(r"(?<=[,;:])\s+", sent)
        buf = ""
        for cl in clauses:
            if len(cl) > chunk_limit:
                if buf:
                    parts.append(buf.strip())
                    buf = ""
                words = cl.split()
                wbuf = ""
                for w in words:
                    cand = f"{wbuf} {w}".strip() if wbuf else w
                    if len(cand) <= chunk_limit:
                        wbuf = cand
                    else:
                        if wbuf:
                            parts.append(wbuf)
                        wbuf = w
                if wbuf:
                    parts.append(wbuf)
            else:
                cand = f"{buf} {cl}".strip() if buf else cl
                if len(cand) <= chunk_limit:
                    buf = cand
                else:
                    if buf:
                        parts.append(buf)
                    buf = cl
        if buf:
            parts.append(buf.strip())
        return parts

    chunks: List[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", cleaned) if p.strip()]
    for para in paragraphs:
        para_single = re.sub(r"\s*\n+\s*", " ", para).strip()
        sentences = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(para_single) if s.strip()]
        buf = ""
        for sent in sentences:
            for piece in _split_oversized_sentence(sent):
                cand = f"{buf} {piece}".strip() if buf else piece
                if len(cand) <= chunk_limit:
                    buf = cand
                else:
                    if buf:
                        chunks.append(buf)
                    buf = piece
        if buf:
            chunks.append(buf)
    return chunks


# =============================================================================
# Background Reverb-TTS Manager (Streaming Producer-Consumer)
# =============================================================================

class ReverbTTSManager:
    """Thread-safe controller for local Reverb-TTS loading, voice cloning, and background dictation."""

    def __init__(
        self,
        settings_file: Optional[Path] = None,
        voices_dir: Optional[Path] = None,
    ):
        self.settings_file = Path(settings_file) if settings_file else SETTINGS_FILE
        self.voices_dir = Path(voices_dir) if voices_dir else VOICES_DIR
        self.settings = TTSSettings.load(self.settings_file)

        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._playback_lock = threading.Lock()
        self._voice_cache_lock = threading.Lock()
        self._settings_lock = threading.RLock()
        self._tts: Any = None
        self._loaded_variant: Optional[str] = None
        self._loaded_device: Optional[str] = None
        self._loaded_actual_device: Optional[str] = None
        self._cached_preset_voices: Optional[List[str]] = None
        self._use_anonymous_hf: bool = False
        self._voice_cache: Dict[str, Any] = {}

        self._state_lock = threading.Lock()
        self._is_speaking = False
        self._is_loading = False
        self._session_id = 0
        self._cancel_event: Optional[threading.Event] = None
        self._current_tag: Optional[Any] = None

        self._state_listeners: List[Callable[[bool], None]] = []
        self._status_listeners: List[Callable[[str], None]] = []

    @property
    def is_speaking(self) -> bool:
        with self._state_lock:
            return self._is_speaking

    @property
    def is_loading(self) -> bool:
        with self._state_lock:
            return self._is_loading

    @property
    def current_tag(self) -> Optional[Any]:
        with self._state_lock:
            return self._current_tag

    @property
    def loaded_info(self) -> Optional[Tuple[str, str]]:
        with self._state_lock:
            if self._loaded_variant is None:
                return None
            dev = self._loaded_actual_device or self._loaded_device or "unknown"
            return (self._loaded_variant, str(dev))

    def add_state_listener(self, cb: Callable[[bool], None]) -> None:
        if cb not in self._state_listeners:
            self._state_listeners.append(cb)

    def remove_state_listener(self, cb: Callable[[bool], None]) -> None:
        if cb in self._state_listeners:
            self._state_listeners.remove(cb)

    def add_status_listener(self, cb: Callable[[str], None]) -> None:
        if cb not in self._status_listeners:
            self._status_listeners.append(cb)

    def remove_status_listener(self, cb: Callable[[str], None]) -> None:
        if cb in self._status_listeners:
            self._status_listeners.remove(cb)

    def _emit_state(self, speaking: bool) -> None:
        for cb in list(self._state_listeners):
            try:
                cb(speaking)
            except Exception:
                pass

    def _emit_status(self, message: str) -> None:
        for cb in list(self._status_listeners):
            try:
                cb(message)
            except Exception:
                pass

    def _set_speaking(self, value: bool, tag: Optional[Any] = None) -> None:
        with self._state_lock:
            self._is_speaking = value
            self._current_tag = tag if value else None
        self._emit_state(value)

    @staticmethod
    def _ensure_reverb_imported() -> Tuple[Any, Any]:
        """Lazily import reverb_tts classes so app startup remains lightweight."""
        global TTS, Voice
        # Guard legacy brotli < 1.1 environments against httpx2 output_buffer_limit TypeError
        try:
            import brotli as _brotli
            _brotli.Decompressor().process(b"", output_buffer_limit=1024)
        except TypeError:
            try:
                import httpx2._decoders as _hx_dec
                _hx_dec.SUPPORTED_DECODERS.pop("br", None)
            except Exception:
                pass
        except Exception:
            pass
        if TTS is not None and Voice is not None:
            return TTS, Voice
        from reverb_tts import TTS as _TTS, Voice as _Voice
        TTS = _TTS
        Voice = _Voice
        return TTS, Voice

    @staticmethod
    @contextmanager
    def _anonymous_hf_hub():
        """Temporarily disable implicit local HF tokens for anonymous public model/voice fetches."""
        try:
            import huggingface_hub.constants as hf_const
        except Exception:
            yield
            return
        prev_flag = hf_const.HF_HUB_DISABLE_IMPLICIT_TOKEN
        hf_const.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
        try:
            yield
        finally:
            hf_const.HF_HUB_DISABLE_IMPLICIT_TOKEN = prev_flag

    def ensure_model_loaded(
        self,
        status_cb: Optional[Callable[[str], None]] = None,
        force_reload: bool = False,
    ) -> Any:
        """Load Reverb-TTS lazily with the configured variant and compute device."""
        if not REVERB_TTS_AVAILABLE:
            raise RuntimeError(
                "The 'reverb-tts' package is not installed. Run: pip install reverb-tts"
            )

        with self._settings_lock:
            desired_variant = self.settings.variant
            desired_device = self.settings.device

        with self._model_lock:
            with self._state_lock:
                needs_load = (
                    force_reload
                    or self._tts is None
                    or self._loaded_variant != desired_variant
                    or self._loaded_device != desired_device
                )
            if not needs_load:
                return self._tts

            tts_cls, _voice_cls = self._ensure_reverb_imported()

            with self._state_lock:
                self._is_loading = True
            msg = f"Loading Reverb-TTS ({desired_variant}, device={desired_device})..."
            self._emit_status(msg)
            if status_cb:
                status_cb(msg)

            try:
                dev_arg = None if desired_device == "auto" else desired_device
                if self._use_anonymous_hf:
                    with self._anonymous_hf_hub():
                        new_tts = tts_cls.from_pretrained(REPO_ID, variant=desired_variant, device=dev_arg, token=False)
                else:
                    try:
                        new_tts = tts_cls.from_pretrained(REPO_ID, variant=desired_variant, device=dev_arg)
                    except Exception as exc:
                        err_txt = str(exc).lower()
                        if any(k in err_txt for k in ("401", "expired", "unauthorized", "token", "repository not found")):
                            self._use_anonymous_hf = True
                            with self._anonymous_hf_hub():
                                new_tts = tts_cls.from_pretrained(REPO_ID, variant=desired_variant, device=dev_arg, token=False)
                        else:
                            raise

                cached_presets: Optional[List[str]] = None
                try:
                    if self._use_anonymous_hf:
                        with self._anonymous_hf_hub():
                            names = new_tts.preset_voices()
                    else:
                        names = new_tts.preset_voices()
                    if names:
                        cached_presets = list(names)
                except Exception:
                    cached_presets = None

                actual_dev = str(getattr(new_tts, "device", desired_device))
                with self._inference_lock:
                    old_tts = self._tts
                    self._tts = new_tts
                    with self._voice_cache_lock:
                        self._voice_cache.clear()
                    with self._state_lock:
                        self._loaded_variant = desired_variant
                        self._loaded_device = desired_device
                        self._loaded_actual_device = actual_dev
                        if cached_presets is not None:
                            self._cached_preset_voices = cached_presets

                if old_tts is not None:
                    del old_tts
                    if TORCH_AVAILABLE:
                        try:
                            import torch as _torch
                            if _torch.cuda.is_available():
                                _torch.cuda.empty_cache()
                        except Exception:
                            pass

                ready_msg = f"Reverb-TTS ({desired_variant}) ready on {actual_dev.upper()}"
                self._emit_status(ready_msg)
                if status_cb:
                    status_cb(ready_msg)
                return new_tts
            finally:
                with self._state_lock:
                    self._is_loading = False

    def list_preset_voices(self) -> List[str]:
        """Return available built-in Reverb-TTS speaker names (cached, never blocks UI)."""
        with self._state_lock:
            if self._cached_preset_voices:
                return list(self._cached_preset_voices)
        return list(DEFAULT_PRESET_VOICES)

    def get_active_voice(self) -> Any:
        """Resolve the currently configured preset or persistent custom Voice object."""
        with self._settings_lock:
            voice_type = self.settings.voice_type
            voice_id = self.settings.voice_id
            rec = self.settings.get_custom_voice(voice_id) if voice_type == "custom" else None

        if voice_type == "custom" and rec is not None:
            cache_key = f"custom:{rec['id']}:{rec['npz_file']}"
            with self._voice_cache_lock:
                if cache_key in self._voice_cache:
                    return self._voice_cache[cache_key]
            safe_path = _resolve_safe_voice_path(self.voices_dir, rec["npz_file"])
            if safe_path.is_file() and REVERB_TTS_AVAILABLE:
                try:
                    _tts_cls, voice_cls = self._ensure_reverb_imported()
                    voice_obj = voice_cls.load(safe_path)
                    with self._voice_cache_lock:
                        self._voice_cache[cache_key] = voice_obj
                    return voice_obj
                except Exception:
                    pass

        preset_name = voice_id if voice_type == "preset" and voice_id else "en_female_narrator"
        cache_key = f"preset:{preset_name}"
        with self._voice_cache_lock:
            if cache_key in self._voice_cache:
                return self._voice_cache[cache_key]

        if self._tts is None:
            self.ensure_model_loaded()

        def _fetch_preset(pname: str):
            with self._inference_lock:
                tts = self._tts
                if tts is None:
                    raise RuntimeError("Reverb-TTS model is not loaded.")
                if self._use_anonymous_hf:
                    with self._anonymous_hf_hub():
                        return tts.preset_voice(pname)
                try:
                    return tts.preset_voice(pname)
                except Exception as exc:
                    err_txt = str(exc).lower()
                    if any(k in err_txt for k in ("401", "expired", "unauthorized", "token", "repository not found")):
                        self._use_anonymous_hf = True
                        tts.token = False
                        with self._anonymous_hf_hub():
                            return tts.preset_voice(pname)
                    raise

        try:
            voice_obj = _fetch_preset(preset_name)
        except Exception:
            voice_obj = _fetch_preset("en_female_narrator")
            cache_key = "preset:en_female_narrator"
        with self._voice_cache_lock:
            self._voice_cache[cache_key] = voice_obj
        return voice_obj

    def train_custom_voice(
        self,
        name: str,
        audio_path: str,
        transcript: str,
        language: str = "en",
        status_cb: Optional[Callable[[str], None]] = None,
        set_active: bool = True,
    ) -> Dict[str, Any]:
        """Encode reference speech into a persistent `.npz` voice profile saved across restarts."""
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("Please provide a name for the custom voice.")
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language {language!r}.")
        clean_transcript = (transcript or "").strip()
        if not clean_transcript:
            raise ValueError("Please enter the verbatim transcript matching the audio recording.")

        src_path = Path(audio_path)
        if not src_path.is_file():
            raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
        if src_path.stat().st_size > MAX_REFERENCE_AUDIO_BYTES:
            raise ValueError("Reference audio file exceeds the 100 MB limit.")

        self.ensure_model_loaded(status_cb=status_cb)

        msg = f"Encoding custom voice '{clean_name}'..."
        self._emit_status(msg)
        if status_cb:
            status_cb(msg)

        temp_wav: Optional[Path] = None
        needs_transcode = (not _is_soundfile_readable(str(src_path))) or (
            src_path.suffix.lower() in {".m4a", ".aac", ".wma", ".webm", ".mp4", ".opus"}
        )
        encode_path = src_path
        if needs_transcode:
            temp_wav = _transcode_to_temp_wav(src_path)
            encode_path = temp_wav

        try:
            with self._inference_lock:
                if self._tts is None:
                    raise RuntimeError("Reverb-TTS model is not loaded.")
                voice_obj = self._tts.voice(str(encode_path), clean_transcript, language=language)
        finally:
            if temp_wav is not None and temp_wav.exists():
                try:
                    temp_wav.unlink()
                except Exception:
                    pass

        self.voices_dir.mkdir(parents=True, exist_ok=True)
        _restrict_permissions(self.voices_dir, is_dir=True)

        with self._settings_lock:
            base_id = _sanitize_voice_id(clean_name)
            voice_id = base_id
            suffix = 1
            while self.settings.get_custom_voice(voice_id) is not None:
                sfx = f"_{suffix}"
                voice_id = f"{base_id[:48 - len(sfx)]}{sfx}"
                suffix += 1

            npz_name = f"{voice_id}.npz"
            safe_npz = _resolve_safe_voice_path(self.voices_dir, npz_name)
            voice_obj.save(safe_npz)
            _restrict_permissions(safe_npz, is_dir=False)

            record = {
                "id": voice_id,
                "name": clean_name,
                "language": language,
                "transcript": clean_transcript,
                "npz_file": npz_name,
                "source_audio": str(src_path),
                "duration_sec": round(float(getattr(voice_obj, "duration", 0.0) or 0.0), 2),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            self.settings.add_or_update_custom_voice(record)
            if set_active:
                self.settings.voice_type = "custom"
                self.settings.voice_id = voice_id
            self.settings.save(self.settings_file)
        with self._voice_cache_lock:
            self._voice_cache[f"custom:{voice_id}:{npz_name}"] = voice_obj

        done_msg = f"Saved custom voice '{clean_name}' ({record['duration_sec']}s)"
        self._emit_status(done_msg)
        if status_cb:
            status_cb(done_msg)
        return record

    def delete_custom_voice(self, voice_id: str) -> bool:
        """Delete a custom voice record and its cached .npz file."""
        with self._settings_lock:
            removed = self.settings.remove_custom_voice(voice_id, voices_dir=self.voices_dir)
            if removed:
                self.settings.save(self.settings_file)
        if removed:
            with self._voice_cache_lock:
                for k in [k for k in self._voice_cache if k.startswith(f"custom:{voice_id}:")]:
                    self._voice_cache.pop(k, None)
        return removed

    def record_microphone_sample(
        self,
        duration_sec: float = 8.0,
        sample_rate: int = 24000,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[float], None]] = None,
        abort_event: Optional[threading.Event] = None,
    ) -> Optional[Path]:
        """Record mono microphone speech into a persistent WAV sample for voice cloning."""
        if not SOUNDDEVICE_AVAILABLE or not SOUNDFILE_AVAILABLE:
            raise RuntimeError("Microphone recording requires 'sounddevice' and 'soundfile'.")

        self.voices_dir.mkdir(parents=True, exist_ok=True)
        _restrict_permissions(self.voices_dir, is_dir=True)
        fname = f"mic_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        out_path = _resolve_safe_voice_path(self.voices_dir, fname)

        frames: List[np.ndarray] = []
        block = int(sample_rate * 0.1)
        total_steps = max(1, int(round(duration_sec / 0.1)))

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            for step in range(total_steps):
                if abort_event is not None and abort_event.is_set():
                    return None
                if cancel_event is not None and cancel_event.is_set():
                    break
                data, _overflowed = stream.read(block)
                frames.append(data.copy())
                if progress_cb:
                    progress_cb(min(duration_sec, (step + 1) * 0.1))

        if abort_event is not None and abort_event.is_set():
            return None
        if not frames:
            raise RuntimeError("No audio captured from microphone.")
        audio = np.concatenate(frames, axis=0).reshape(-1)
        sf.write(str(out_path), audio, sample_rate)
        _restrict_permissions(out_path, is_dir=False)
        return out_path

    def transcribe_audio_whisper(self, audio_path: str, language: Optional[str] = None) -> str:
        """Auto-transcribe a reference audio clip using local Whisper if installed."""
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Local Whisper package is not installed for auto-transcription.")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        import whisper as _whisper
        use_fp16 = False
        if TORCH_AVAILABLE:
            import torch as _torch
            use_fp16 = bool(_torch.cuda.is_available())
        model = _whisper.load_model("base")
        kwargs: Dict[str, Any] = {"fp16": use_fp16}
        if language in SUPPORTED_LANGUAGES:
            kwargs["language"] = language
        result = model.transcribe(audio_path, **kwargs)
        return str(result.get("text") or "").strip()

    def _play_audio_array(
        self,
        audio_arr: np.ndarray,
        sample_rate: int,
        cancel_event: threading.Event,
    ) -> None:
        """Play float32 mono waveform synchronously while polling cancel_event every ~80ms."""
        if cancel_event.is_set() or audio_arr is None or len(audio_arr) == 0:
            return

        with self._playback_lock:
            if cancel_event.is_set():
                return

            if SOUNDDEVICE_AVAILABLE and sd is not None:
                try:
                    pcm = np.asarray(audio_arr, dtype=np.float32).reshape(-1, 1)
                    block = max(512, int(sample_rate * 0.08))
                    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
                        for offset in range(0, len(pcm), block):
                            if cancel_event.is_set():
                                try:
                                    stream.abort()
                                except Exception:
                                    pass
                                return
                            stream.write(pcm[offset : offset + block])
                    return
                except Exception as e:
                    if not WINSOUND_AVAILABLE:
                        raise RuntimeError(f"Audio playback failed: {e}") from e

            if WINSOUND_AVAILABLE and winsound is not None:
                tmp_wav: Optional[str] = None
                try:
                    clipped = np.clip(np.asarray(audio_arr, dtype=np.float32), -1.0, 1.0)
                    pcm16 = (clipped * 32767.0).astype(np.int16)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tmp_wav = tf.name
                    with wave.open(tmp_wav, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(int(sample_rate))
                        wf.writeframes(pcm16.tobytes())

                    duration_s = len(pcm16) / float(max(1, sample_rate))
                    winsound.PlaySound(tmp_wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    deadline = time.monotonic() + duration_s + 0.05
                    while time.monotonic() < deadline:
                        if cancel_event.is_set():
                            winsound.PlaySound(None, winsound.SND_PURGE)
                            return
                        time.sleep(0.06)
                finally:
                    if tmp_wav and os.path.exists(tmp_wav):
                        try:
                            os.unlink(tmp_wav)
                        except Exception:
                            pass
                return

            raise RuntimeError("No audio playback backend available (install sounddevice).")

    def _synthesize_chunk(self, chunk: str, voice_obj: Any, chunk_idx: int) -> Tuple[np.ndarray, int]:
        """Synthesize one text segment using active Reverb-TTS settings."""
        with self._settings_lock:
            s = self.settings
            lang = s.language
            speed = s.speed
            pace = s.pace
            steps = s.steps
            seed = s.seed
        chunk_seed = (seed + chunk_idx) if seed is not None else None
        with self._inference_lock:
            if self._tts is None:
                raise RuntimeError("Reverb-TTS model is not loaded.")
            audio = self._tts.generate(
                chunk,
                voice_obj,
                language=lang,
                speed=speed,
                pace=pace,
                steps=steps,
                seed=chunk_seed,
            )
            sr = int(getattr(self._tts, "sample_rate", 24000))
        return np.asarray(audio, dtype=np.float32), sr

    def speak(
        self,
        text: str,
        clean_markdown: bool = True,
        tag: Optional[Any] = None,
        status_cb: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[bool], None]] = None,
    ) -> bool:
        """Start non-blocking background dictation for `text`. Returns False if nothing to speak."""
        self.stop()

        raw = text or ""
        spoken = clean_markdown_for_speech(raw, skip_code_blocks=self.settings.skip_code_blocks) if clean_markdown else raw.strip()
        chunks = split_into_dictation_chunks(
            spoken,
            max_chars_per_chunk=self.settings.max_chars_per_chunk,
            max_input_chars=self.settings.max_input_chars,
        )
        if not chunks:
            msg = "Nothing readable to dictate."
            self._emit_status(msg)
            if status_cb:
                status_cb(msg)
            return False

        cancel_event = threading.Event()
        with self._state_lock:
            self._session_id += 1
            session_id = self._session_id
            self._cancel_event = cancel_event

        self._set_speaking(True, tag=tag)
        audio_queue: "queue.Queue[Optional[Tuple[Any, ...]]]" = queue.Queue(maxsize=2)
        consumer_done = threading.Event()

        def _put_with_cancel(item: Optional[Tuple[Any, ...]]) -> bool:
            while True:
                if cancel_event.is_set() or consumer_done.is_set():
                    return False
                try:
                    audio_queue.put(item, timeout=0.08)
                    return True
                except queue.Full:
                    continue

        def _producer():
            try:
                self.ensure_model_loaded(status_cb=status_cb)
                if cancel_event.is_set():
                    return
                voice_obj = self.get_active_voice()
                total = len(chunks)
                for i, chunk in enumerate(chunks):
                    if cancel_event.is_set():
                        return
                    s_msg = f"Synthesizing segment {i + 1}/{total}..."
                    self._emit_status(s_msg)
                    if status_cb:
                        status_cb(s_msg)
                    arr, sr = self._synthesize_chunk(chunk, voice_obj, i)
                    if not _put_with_cancel(("CHUNK", i + 1, total, arr, sr)):
                        return
            except Exception as e:
                if not cancel_event.is_set():
                    _put_with_cancel(("ERROR", str(e)))
            finally:
                _put_with_cancel(None)

        def _consumer():
            errored = False
            try:
                while not cancel_event.is_set():
                    try:
                        item = audio_queue.get(timeout=0.08)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    kind = item[0]
                    if kind == "ERROR":
                        err_msg = f"TTS Error: {item[1]}"
                        errored = True
                        self._emit_status(err_msg)
                        if status_cb:
                            status_cb(err_msg)
                        break
                    if kind == "CHUNK":
                        _tag, idx_num, total, arr, sr = item
                        p_msg = f"Dictating segment {idx_num}/{total}..."
                        self._emit_status(p_msg)
                        if status_cb:
                            status_cb(p_msg)
                        self._play_audio_array(arr, sr, cancel_event)
            except Exception as e:
                if not cancel_event.is_set():
                    errored = True
                    err_msg = f"Audio playback error: {e}"
                    self._emit_status(err_msg)
                    if status_cb:
                        status_cb(err_msg)
            finally:
                consumer_done.set()
                cancelled = cancel_event.is_set()
                should_clear = False
                with self._state_lock:
                    if self._session_id == session_id:
                        should_clear = True
                if should_clear:
                    self._set_speaking(False, tag=None)
                    if not errored:
                        end_msg = "Dictation stopped" if cancelled else "Ready"
                        self._emit_status(end_msg)
                        if status_cb:
                            status_cb(end_msg)
                if on_complete:
                    try:
                        on_complete(cancelled)
                    except Exception:
                        pass

        t_prod = threading.Thread(target=_producer, daemon=True, name="reverb-tts-synth")
        t_cons = threading.Thread(target=_consumer, daemon=True, name="reverb-tts-play")
        t_prod.start()
        t_cons.start()
        return True

    def speak_preview(
        self,
        sample_text: Optional[str] = None,
        status_cb: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[bool], None]] = None,
    ) -> bool:
        """Speak a short preview line in the currently selected voice and language."""
        txt = (sample_text or "").strip() or LANGUAGE_SAMPLES.get(self.settings.language, LANGUAGE_SAMPLES["en"])
        return self.speak(txt, clean_markdown=False, tag="preview", status_cb=status_cb, on_complete=on_complete)

    def stop(self) -> None:
        """Immediately halt ongoing dictation synthesis and audio playback."""
        with self._state_lock:
            ev = self._cancel_event
            self._cancel_event = None
            was_speaking = self._is_speaking
        if ev is not None:
            ev.set()
        if WINSOUND_AVAILABLE and winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        if was_speaking:
            self._set_speaking(False, tag=None)
            self._emit_status("Dictation stopped")

    def shutdown(self) -> None:
        """Stop playback cleanly on application exit."""
        self.stop()


_tts_manager_singleton: Optional[ReverbTTSManager] = None


def get_tts_manager() -> ReverbTTSManager:
    """Return the global ReverbTTSManager singleton instance."""
    global _tts_manager_singleton
    if _tts_manager_singleton is None:
        _tts_manager_singleton = ReverbTTSManager()
    return _tts_manager_singleton


# =============================================================================
# Dictation Properties Dialog
# =============================================================================

class DictationPropertiesDialog(tk.Toplevel):
    """Configuration dialog for Reverb-TTS speakers, languages, voice cloning, and parameters."""

    def __init__(
        self,
        parent: tk.Misc,
        manager: Optional[ReverbTTSManager] = None,
        theme: Optional[Dict[str, Any]] = None,
        on_save: Optional[Callable[[TTSSettings], None]] = None,
    ):
        super().__init__(parent)
        self.parent = parent
        self.manager = manager or get_tts_manager()
        self.theme = theme or {}
        self.on_save = on_save
        self.result: Optional[TTSSettings] = None

        self._record_cancel: Optional[threading.Event] = None
        self._record_abort: Optional[threading.Event] = None
        self._recording = False
        self._busy_training = False
        self._saved_committed = False
        self._destroyed = False
        self._ui_queue: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._ui_poll_id: Optional[str] = None
        with self.manager._settings_lock:
            self.manager.settings.normalize()
            self._initial_snapshot = {
                k: v for k, v in asdict(self.manager.settings).items() if k != "custom_voices"
            }

        self.title("Dictation Properties")
        self.geometry("680x650")
        self.resizable(True, True)
        self.minsize(620, 580)
        self.transient(parent)
        self.grab_set()
        self.bind("<Escape>", lambda e: self._cancel_and_close())
        self.protocol("WM_DELETE_WINDOW", self._cancel_and_close)

        self._build_ui()
        self._load_settings_into_ui()

        self._status_cb_ref = self._on_manager_status
        self._state_cb_ref = self._on_manager_state
        self.manager.add_status_listener(self._status_cb_ref)
        self.manager.add_state_listener(self._state_cb_ref)
        self._ui_poll_id = self.after(30, self._drain_ui_queue)

        self.update_idletasks()
        try:
            x = max(10, parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2)
            y = max(10, parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2)
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _cancel_and_close(self) -> None:
        """Revert unconfirmed parameter changes while keeping trained/deleted custom voices."""
        if self.manager.current_tag == "preview":
            self.manager.stop()
        if not getattr(self, "_saved_committed", False):
            try:
                with self.manager._settings_lock:
                    for k, v in self._initial_snapshot.items():
                        setattr(self.manager.settings, k, v)
                    self.manager.settings.normalize()
                    self.manager.settings.save(self.manager.settings_file)
            except Exception:
                pass
        self.destroy()

    def destroy(self):
        self._destroyed = True
        if getattr(self, "_ui_poll_id", None) is not None:
            try:
                self.after_cancel(self._ui_poll_id)
            except Exception:
                pass
            self._ui_poll_id = None
        try:
            if self._record_abort is not None:
                self._record_abort.set()
            if self._record_cancel is not None:
                self._record_cancel.set()
            self.manager.remove_status_listener(self._status_cb_ref)
            self.manager.remove_state_listener(self._state_cb_ref)
        except Exception:
            pass
        super().destroy()

    def _safe_after(self, fn: Callable[[], None]) -> None:
        if not getattr(self, "_destroyed", False):
            self._ui_queue.put(fn)

    def _drain_ui_queue(self) -> None:
        self._ui_poll_id = None
        if getattr(self, "_destroyed", False):
            return
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return
        while True:
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                if not getattr(self, "_destroyed", False) and self.winfo_exists():
                    fn()
            except Exception:
                pass
        if not getattr(self, "_destroyed", False):
            try:
                if self.winfo_exists():
                    self._ui_poll_id = self.after(30, self._drain_ui_queue)
            except Exception:
                pass

    def _on_manager_status(self, msg: str) -> None:
        self._safe_after(lambda: self._update_status_displays(msg))

    def _on_manager_state(self, speaking: bool) -> None:
        def _upd():
            self.test_btn.configure(state=tk.DISABLED if speaking else tk.NORMAL)
            self.stop_test_btn.configure(state=tk.NORMAL if speaking else tk.DISABLED)
            self._refresh_header_badge()
        self._safe_after(_upd)

    def _update_status_displays(self, msg: str) -> None:
        if hasattr(self, "status_lbl"):
            self.status_lbl.configure(text=msg)
        self._refresh_header_badge()

    def _refresh_header_badge(self) -> None:
        if not REVERB_TTS_AVAILABLE:
            text, fg = "reverb-tts missing", "#fecaca"
        elif self.manager.is_loading:
            text, fg = "Loading model...", "#fde68a"
        elif self.manager.is_speaking:
            text, fg = "Speaking...", "#bbf7d0"
        else:
            info = self.manager.loaded_info
            if info is not None:
                text, fg = f"{info[0]} · {info[1].upper()}", "#bbf7d0"
            else:
                text, fg = f"{self.variant_var.get().split()[0]} · Standby", "#ddd6fe"
        if hasattr(self, "header_badge"):
            self.header_badge.configure(text=text, fg=fg)

    # ------------------------------------------------------------------ UI Build

    def _build_ui(self) -> None:
        banner_bg = self.theme.get("status_bg", "#3b1d6e")
        banner_fg = self.theme.get("status_fg", "#f5f3ff")
        accent_bg = self.theme.get("accent_purple", "#5b21b6")

        # Header Banner
        banner = tk.Frame(self, bg=banner_bg, padx=14, pady=10)
        banner.pack(fill=tk.X, side=tk.TOP)

        title_col = tk.Frame(banner, bg=banner_bg)
        title_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            title_col, text="🔊 Dictation Properties — Reverb-TTS",
            font=("Segoe UI", 12, "bold"), fg=banner_fg, bg=banner_bg, anchor="w"
        ).pack(fill=tk.X)
        tk.Label(
            title_col,
            text="Local neural speech synthesis (int8 · 307 MB) with preset speakers, 7 languages, and persistent voice cloning",
            font=("Segoe UI", 8), fg="#ddd6fe", bg=banner_bg, anchor="w"
        ).pack(fill=tk.X)

        self.header_badge = tk.Label(
            banner, text="Standby", font=("Segoe UI", 8, "bold"),
            bg=accent_bg, fg="#ddd6fe", padx=9, pady=4
        )
        self.header_badge.pack(side=tk.RIGHT, padx=(8, 0))

        main = ttk.Frame(self, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        self.nb = ttk.Notebook(main)
        self.nb.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self._build_speaker_tab()
        self._build_cloning_tab()
        self._build_model_params_tab()

        # Status & Progress row
        status_row = ttk.Frame(main)
        status_row.pack(fill=tk.X, pady=(0, 6))
        self.status_lbl = ttk.Label(status_row, text="Ready", font=("Segoe UI", 8, "italic"), foreground="#5b21b6")
        self.status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.op_progress = ttk.Progressbar(status_row, mode="indeterminate", length=140)

        # Footer action bar
        footer = ttk.Frame(main)
        footer.pack(fill=tk.X)
        ttk.Button(footer, text="Reset Defaults", command=self._reset_defaults).pack(side=tk.LEFT)
        ttk.Button(footer, text="Save & Apply", command=self._save_and_close).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Cancel", command=self._cancel_and_close).pack(side=tk.RIGHT)

    def _build_speaker_tab(self) -> None:
        t1 = ttk.Frame(self.nb, padding=12)
        self.nb.add(t1, text="  🎙 Speaker & Language  ")

        # Voice source mode
        src_frame = ttk.LabelFrame(t1, text="Active Speaker Profile", padding=10)
        src_frame.pack(fill=tk.X, pady=(0, 10))
        src_frame.columnconfigure(1, weight=1)

        self.voice_type_var = tk.StringVar(value="preset")
        rb_preset = ttk.Radiobutton(
            src_frame, text="Built-in Preset Speaker (16 voices)",
            variable=self.voice_type_var, value="preset", command=self._on_voice_type_changed
        )
        rb_preset.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))

        ttk.Label(src_frame, text="Preset Voice:", width=16, anchor=tk.W).grid(row=1, column=0, sticky=tk.W, padx=(18, 6), pady=3)
        self.preset_var = tk.StringVar()
        preset_values = [format_preset_label(v) for v in self.manager.list_preset_voices()]
        self.preset_cb = ttk.Combobox(src_frame, textvariable=self.preset_var, values=preset_values, state="readonly")
        self.preset_cb.grid(row=1, column=1, sticky=tk.EW, pady=3)
        _DialogToolTip(self.preset_cb, "Select one of Reverb-TTS's 16 built-in studio speaker voices across 7 languages.")

        rb_custom = ttk.Radiobutton(
            src_frame, text="Saved Custom Cloned Voice (persists across sessions)",
            variable=self.voice_type_var, value="custom", command=self._on_voice_type_changed
        )
        rb_custom.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 6))

        ttk.Label(src_frame, text="Custom Voice:", width=16, anchor=tk.W).grid(row=3, column=0, sticky=tk.W, padx=(18, 6), pady=3)
        self.custom_select_var = tk.StringVar()
        self.custom_cb = ttk.Combobox(src_frame, textvariable=self.custom_select_var, state="readonly")
        self.custom_cb.grid(row=3, column=1, sticky=tk.EW, pady=3)
        _DialogToolTip(self.custom_cb, "Choose a custom cloned voice trained in the Custom Voice Training tab.")

        # Language frame
        lang_frame = ttk.LabelFrame(t1, text="Spoken Language", padding=10)
        lang_frame.pack(fill=tk.X, pady=(0, 10))
        lang_frame.columnconfigure(1, weight=1)

        ttk.Label(lang_frame, text="Output Language:", width=16, anchor=tk.W).grid(row=0, column=0, sticky=tk.W, pady=3)
        self.lang_var = tk.StringVar(value=language_to_label("en"))
        lang_labels = [language_to_label(c) for c in SUPPORTED_LANGUAGES]
        self.lang_cb = ttk.Combobox(lang_frame, textvariable=self.lang_var, values=lang_labels, state="readonly")
        self.lang_cb.grid(row=0, column=1, sticky=tk.W, padx=(6, 0), pady=3)
        self.lang_cb.bind("<<ComboboxSelected>>", self._on_language_changed)

        ttk.Label(
            lang_frame,
            text="Cross-lingual cloning is supported: any speaker can dictate text in any of the 7 supported languages.",
            font=("Segoe UI", 8, "italic"), foreground="#5b21b6", wraplength=580
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

        # Audition frame
        test_frame = ttk.LabelFrame(t1, text="Voice Audition", padding=10)
        test_frame.pack(fill=tk.X)
        test_frame.columnconfigure(0, weight=1)

        self.test_text_var = tk.StringVar(value=LANGUAGE_SAMPLES["en"])
        self.test_entry = ttk.Entry(test_frame, textvariable=self.test_text_var)
        self.test_entry.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))

        btn_row = ttk.Frame(test_frame)
        btn_row.grid(row=1, column=0, sticky=tk.W)
        self.test_btn = ttk.Button(btn_row, text="▶ Test Voice", command=self._test_voice)
        self.test_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_test_btn = ttk.Button(btn_row, text="⏹ Stop", command=self.manager.stop, state=tk.DISABLED)
        self.stop_test_btn.pack(side=tk.LEFT)

    def _build_cloning_tab(self) -> None:
        t2 = ttk.Frame(self.nb, padding=12)
        self.nb.add(t2, text="  🧬 Custom Voice Training  ")

        # New custom voice card
        train_frame = ttk.LabelFrame(t2, text="Train Persistent Zero-Shot Voice Profile", padding=10)
        train_frame.pack(fill=tk.X, pady=(0, 10))
        train_frame.columnconfigure(1, weight=1)

        ttk.Label(train_frame, text="Voice Name:", width=16, anchor=tk.W).grid(row=0, column=0, sticky=tk.W, pady=3)
        name_lang_row = ttk.Frame(train_frame)
        name_lang_row.grid(row=0, column=1, sticky=tk.EW, pady=3)
        self.clone_name_var = tk.StringVar()
        ttk.Entry(name_lang_row, textvariable=self.clone_name_var, width=24).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(name_lang_row, text="Language:").pack(side=tk.LEFT, padx=(10, 4))
        self.clone_lang_var = tk.StringVar(value=language_to_label("en"))
        ttk.Combobox(
            name_lang_row, textvariable=self.clone_lang_var,
            values=[language_to_label(c) for c in SUPPORTED_LANGUAGES],
            state="readonly", width=16
        ).pack(side=tk.LEFT)

        ttk.Label(train_frame, text="Reference Audio:", width=16, anchor=tk.W).grid(row=1, column=0, sticky=tk.W, pady=3)
        audio_row = ttk.Frame(train_frame)
        audio_row.grid(row=1, column=1, sticky=tk.EW, pady=3)
        self.clone_audio_var = tk.StringVar()
        self.clone_audio_var.trace_add("write", lambda *_: self._update_audio_duration_hint())
        ttk.Entry(audio_row, textvariable=self.clone_audio_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(audio_row, text="📂 Browse...", width=11, command=self._browse_audio).pack(side=tk.LEFT, padx=(0, 4))
        self.rec_btn = ttk.Button(audio_row, text="🎙 Record 8s", width=12, command=self._toggle_record_mic)
        self.rec_btn.pack(side=tk.LEFT)
        _DialogToolTip(self.rec_btn, "Record 3 to 8 seconds of clean microphone speech directly for zero-shot voice cloning.")

        self.audio_info_lbl = ttk.Label(
            train_frame, text="Recommended: 3–10 seconds of clean speech from a single speaker.",
            font=("Segoe UI", 8, "italic"), foreground="#6b7280"
        )
        self.audio_info_lbl.grid(row=2, column=1, sticky=tk.W, pady=(0, 4))

        ttk.Label(train_frame, text="Verbatim Transcript:", width=16, anchor=tk.NW).grid(row=3, column=0, sticky=tk.NW, pady=3)
        tx_col = ttk.Frame(train_frame)
        tx_col.grid(row=3, column=1, sticky=tk.EW, pady=3)
        self.clone_transcript_text = tk.Text(tx_col, height=3, wrap=tk.WORD, font=("Segoe UI", 9))
        self.clone_transcript_text.pack(fill=tk.X, expand=True)

        action_row = ttk.Frame(train_frame)
        action_row.grid(row=4, column=1, sticky=tk.EW, pady=(6, 0))
        self.auto_tx_btn = ttk.Button(
            action_row, text="✨ Auto-Transcribe",
            command=self._auto_transcribe_audio,
            state=tk.NORMAL if WHISPER_AVAILABLE else tk.DISABLED,
        )
        self.auto_tx_btn.pack(side=tk.LEFT)
        _DialogToolTip(self.auto_tx_btn, "Use local Whisper speech recognition to transcribe the reference clip automatically.")

        self.train_btn = ttk.Button(action_row, text="⚡ Train & Save Custom Voice", command=self._train_custom_voice)
        self.train_btn.pack(side=tk.RIGHT)

        # Saved custom voices list
        saved_frame = ttk.LabelFrame(t2, text="Saved Custom Voices (Persisted in ~/.markitdown_tts/voices)", padding=8)
        saved_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("name", "language", "duration", "created")
        self.voices_tree = ttk.Treeview(saved_frame, columns=cols, show="headings", height=5)
        self.voices_tree.heading("name", text="Voice Name")
        self.voices_tree.heading("language", text="Language")
        self.voices_tree.heading("duration", text="Ref Duration")
        self.voices_tree.heading("created", text="Created")
        self.voices_tree.column("name", width=180, anchor=tk.W)
        self.voices_tree.column("language", width=95, anchor=tk.CENTER)
        self.voices_tree.column("duration", width=90, anchor=tk.CENTER)
        self.voices_tree.column("created", width=130, anchor=tk.CENTER)
        self.voices_tree.pack(fill=tk.BOTH, expand=True, side=tk.TOP, pady=(0, 6))

        tv_btns = ttk.Frame(saved_frame)
        tv_btns.pack(fill=tk.X)
        ttk.Button(tv_btns, text="Use Selected Voice", command=self._use_selected_custom_voice).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(tv_btns, text="▶ Test Selected", command=self._test_selected_custom_voice).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(tv_btns, text="Delete Selected", command=self._delete_selected_custom_voice).pack(side=tk.RIGHT)

    def _build_model_params_tab(self) -> None:
        t3 = ttk.Frame(self.nb, padding=12)
        self.nb.add(t3, text="  ⚙ Model & Parameters  ")

        # Model & Runtime frame
        mdl_frame = ttk.LabelFrame(t3, text="Reverb-TTS Model & Hardware", padding=10)
        mdl_frame.pack(fill=tk.X, pady=(0, 10))
        mdl_frame.columnconfigure(1, weight=1)

        ttk.Label(mdl_frame, text="Model Variant:", width=18, anchor=tk.W).grid(row=0, column=0, sticky=tk.W, pady=3)
        self.variant_var = tk.StringVar(value=variant_to_label("int8"))
        variant_cb = ttk.Combobox(
            mdl_frame, textvariable=self.variant_var,
            values=[variant_to_label(v) for v in SUPPORTED_VARIANTS],
            state="readonly"
        )
        variant_cb.grid(row=0, column=1, sticky=tk.EW, pady=3)
        _DialogToolTip(variant_cb, "Select quantized weight precision. int8 (307 MB) matches fp32 speech quality cleanly on local GPUs/CPUs.")

        ttk.Label(mdl_frame, text="Compute Device:", width=18, anchor=tk.W).grid(row=1, column=0, sticky=tk.W, pady=3)
        dev_row = ttk.Frame(mdl_frame)
        dev_row.grid(row=1, column=1, sticky=tk.EW, pady=3)
        self.device_var = tk.StringVar(value="auto")
        ttk.Combobox(dev_row, textvariable=self.device_var, values=("auto", "cuda", "cpu"), state="readonly", width=12).pack(side=tk.LEFT)
        self.load_now_btn = ttk.Button(dev_row, text="⬇ Load / Download Model Now", command=self._preload_model)
        self.load_now_btn.pack(side=tk.RIGHT)

        # Synthesis & Chunking parameters
        param_frame = ttk.LabelFrame(t3, text="Dictation Synthesis Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))
        param_frame.columnconfigure(1, weight=1)

        ttk.Label(param_frame, text="Speech Speed:", width=18, anchor=tk.W).grid(row=0, column=0, sticky=tk.W, pady=3)
        spd_row = ttk.Frame(param_frame)
        spd_row.grid(row=0, column=1, sticky=tk.EW, pady=3)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_readout = ttk.Label(spd_row, text="1.00×", width=7)
        self.speed_readout.pack(side=tk.RIGHT)
        spd_scale = ttk.Scale(
            spd_row, from_=0.6, to=1.8, orient=tk.HORIZONTAL, variable=self.speed_var,
            command=lambda _v: self.speed_readout.configure(text=f"{self.speed_var.get():.2f}×")
        )
        spd_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        ttk.Label(param_frame, text="Sampler Steps (4–16):", width=18, anchor=tk.W).grid(row=1, column=0, sticky=tk.W, pady=3)
        row_steps = ttk.Frame(param_frame)
        row_steps.grid(row=1, column=1, sticky=tk.EW, pady=3)
        self.steps_var = tk.IntVar(value=8)
        ttk.Spinbox(row_steps, from_=4, to=16, textvariable=self.steps_var, width=6).pack(side=tk.LEFT)

        ttk.Label(row_steps, text="Pace Mode:").pack(side=tk.LEFT, padx=(16, 4))
        self.pace_var = tk.StringVar(value="speech")
        ttk.Combobox(row_steps, textvariable=self.pace_var, values=("speech", "recording"), state="readonly", width=11).pack(side=tk.LEFT)

        ttk.Label(row_steps, text="Seed:").pack(side=tk.LEFT, padx=(16, 4))
        self.seed_var = tk.StringVar(value="")
        seed_ent = ttk.Entry(row_steps, textvariable=self.seed_var, width=8)
        seed_ent.pack(side=tk.LEFT)
        _DialogToolTip(seed_ent, "Optional integer seed for reproducible speech takes; leave blank for natural variation.")

        ttk.Label(param_frame, text="Chunk Length (chars):", width=18, anchor=tk.W).grid(row=2, column=0, sticky=tk.W, pady=3)
        len_row = ttk.Frame(param_frame)
        len_row.grid(row=2, column=1, sticky=tk.EW, pady=3)
        self.chunk_chars_var = tk.IntVar(value=320)
        chunk_sb = ttk.Spinbox(len_row, from_=100, to=2000, increment=40, textvariable=self.chunk_chars_var, width=8)
        chunk_sb.pack(side=tk.LEFT)
        _DialogToolTip(chunk_sb, "Maximum characters synthesized per streaming audio chunk (lower starts playback faster).")

        ttk.Label(len_row, text="Max Input Length:").pack(side=tk.LEFT, padx=(14, 4))
        self.max_input_var = tk.IntVar(value=0)
        max_sb = ttk.Spinbox(len_row, from_=0, to=100000, increment=500, textvariable=self.max_input_var, width=8)
        max_sb.pack(side=tk.LEFT)
        _DialogToolTip(max_sb, "Cap total dictated characters per request (0 = unlimited).")

        # Behavior options
        beh_frame = ttk.LabelFrame(t3, text="Behavior", padding=10)
        beh_frame.pack(fill=tk.X)
        self.skip_code_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(beh_frame, text="Skip fenced code blocks (```...```) during document dictation", variable=self.skip_code_var).pack(anchor=tk.W, pady=2)
        self.auto_read_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(beh_frame, text="Automatically read completed AI Chat responses aloud", variable=self.auto_read_var).pack(anchor=tk.W, pady=2)

    # ------------------------------------------------------------------ Data Sync

    def _load_settings_into_ui(self) -> None:
        s = self.manager.settings
        s.normalize()
        self.voice_type_var.set(s.voice_type)
        if s.voice_type == "preset":
            self.preset_var.set(format_preset_label(s.voice_id))
        else:
            self.preset_var.set(format_preset_label("en_female_narrator"))

        self._refresh_custom_voice_widgets()
        self.lang_var.set(language_to_label(s.language))
        self.test_text_var.set(LANGUAGE_SAMPLES.get(s.language, LANGUAGE_SAMPLES["en"]))

        self.variant_var.set(variant_to_label(s.variant))
        self.device_var.set(s.device)
        self.speed_var.set(s.speed)
        self.speed_readout.configure(text=f"{s.speed:.2f}×")
        self.steps_var.set(s.steps)
        self.pace_var.set(s.pace)
        self.seed_var.set("" if s.seed is None else str(s.seed))
        self.chunk_chars_var.set(s.max_chars_per_chunk)
        self.max_input_var.set(s.max_input_chars)
        self.skip_code_var.set(s.skip_code_blocks)
        self.auto_read_var.set(s.auto_read_ai_replies)

        self._on_voice_type_changed()
        speaking = self.manager.is_speaking
        self.test_btn.configure(state=tk.DISABLED if speaking else tk.NORMAL)
        self.stop_test_btn.configure(state=tk.NORMAL if speaking else tk.DISABLED)
        self._refresh_header_badge()

    def _refresh_custom_voice_widgets(self) -> None:
        s = self.manager.settings
        customs = s.custom_voices
        display_items = [f"{v['name']} ({SUPPORTED_LANGUAGES.get(v['language'], v['language'])}) [{v['id']}]" for v in customs]
        self.custom_cb.configure(values=display_items)
        if display_items:
            matched = next((lbl for lbl in display_items if lbl.endswith(f"[{s.voice_id}]")), display_items[0])
            self.custom_select_var.set(matched)
        else:
            self.custom_select_var.set("")

        for row in self.voices_tree.get_children():
            self.voices_tree.delete(row)
        for v in customs:
            lang_disp = SUPPORTED_LANGUAGES.get(v["language"], v["language"])
            dur_disp = f"{v.get('duration_sec', 0.0):.1f}s"
            self.voices_tree.insert("", tk.END, iid=v["id"], values=(v["name"], lang_disp, dur_disp, v.get("created_at", "")))

    def _collect_settings_from_ui(self) -> TTSSettings:
        seed_raw = self.seed_var.get().strip()
        seed_val: Optional[int] = None
        if seed_raw:
            try:
                seed_val = int(seed_raw)
            except ValueError:
                raise ValueError("Seed must be an integer or left blank.")

        vtype = self.voice_type_var.get()
        if vtype == "custom":
            sel_custom = self.custom_select_var.get()
            m = re.search(r"\[([a-z0-9_-]+)\]\s*$", sel_custom)
            vid = m.group(1) if m else (self.manager.settings.custom_voices[0]["id"] if self.manager.settings.custom_voices else "en_female_narrator")
            if not self.manager.settings.get_custom_voice(vid):
                vtype = "preset"
                vid = "en_female_narrator"
        else:
            sel_preset = self.preset_var.get()
            m = re.search(r"\(([a-z0-9_]+)\)\s*$", sel_preset)
            vid = m.group(1) if m else "en_female_narrator"

        updated = TTSSettings(
            variant=label_to_variant(self.variant_var.get()),
            device=self.device_var.get(),
            voice_type=vtype,
            voice_id=vid,
            language=label_to_language(self.lang_var.get()),
            speed=float(self.speed_var.get()),
            pace=self.pace_var.get(),
            steps=int(self.steps_var.get()),
            seed=seed_val,
            max_chars_per_chunk=int(self.chunk_chars_var.get()),
            max_input_chars=int(self.max_input_var.get()),
            skip_code_blocks=bool(self.skip_code_var.get()),
            auto_read_ai_replies=bool(self.auto_read_var.get()),
            custom_voices=[dict(v) for v in self.manager.settings.custom_voices],
        )
        updated.normalize()
        return updated

    def _on_voice_type_changed(self) -> None:
        is_custom = self.voice_type_var.get() == "custom"
        has_customs = bool(self.manager.settings.custom_voices)
        self.preset_cb.configure(state=tk.DISABLED if is_custom else "readonly")
        if is_custom and not has_customs:
            self.custom_cb.configure(state=tk.DISABLED)
            self.status_lbl.configure(text="No custom voices trained yet — add one in Custom Voice Training tab.")
        else:
            self.custom_cb.configure(state="readonly" if is_custom else tk.DISABLED)

    def _on_language_changed(self, _event=None) -> None:
        code = label_to_language(self.lang_var.get())
        cur = self.test_text_var.get().strip()
        if not cur or cur in LANGUAGE_SAMPLES.values():
            self.test_text_var.set(LANGUAGE_SAMPLES.get(code, LANGUAGE_SAMPLES["en"]))

    def _apply_to_manager(self) -> TTSSettings:
        with self.manager._settings_lock:
            new_settings = self._collect_settings_from_ui()
            new_settings.custom_voices = [dict(v) for v in self.manager.settings.custom_voices]
            new_settings.normalize()
            self.manager.settings = new_settings
            new_settings.save(self.manager.settings_file)
            return new_settings

    # ------------------------------------------------------------------ Actions

    def _test_voice(self) -> None:
        try:
            self._apply_to_manager()
        except Exception as e:
            messagebox.showwarning("Invalid Parameter", str(e), parent=self)
            return
        sample = self.test_text_var.get().strip()
        if not self.manager.speak_preview(sample_text=sample):
            messagebox.showinfo("Audition", "Enter sample text to test voice.", parent=self)

    def _preload_model(self) -> None:
        try:
            self._apply_to_manager()
        except Exception as e:
            messagebox.showwarning("Invalid Parameter", str(e), parent=self)
            return
        if self.manager.is_loading:
            return
        self.load_now_btn.configure(state=tk.DISABLED)
        self.op_progress.pack(side=tk.RIGHT, padx=(8, 0))
        self.op_progress.start(12)

        def _worker():
            err_msg = None
            try:
                self.manager.ensure_model_loaded(force_reload=True)
            except Exception as exc:
                err_msg = str(exc)

            def _done():
                self.op_progress.stop()
                self.op_progress.pack_forget()
                self.load_now_btn.configure(state=tk.NORMAL)
                self._refresh_header_badge()
                if err_msg:
                    messagebox.showerror("Model Load Error", err_msg, parent=self)
            self._safe_after(_done)

        threading.Thread(target=_worker, daemon=True).start()

    def _browse_audio(self) -> None:
        fp = filedialog.askopenfilename(
            title="Select Reference Speech Audio (3–10 seconds)",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All Files", "*.*")],
            parent=self,
        )
        if fp:
            self.clone_audio_var.set(fp)
            if not self.clone_name_var.get().strip():
                self.clone_name_var.set(Path(fp).stem.replace("_", " ").title())

    def _update_audio_duration_hint(self) -> None:
        path = self.clone_audio_var.get().strip()
        if not path:
            self.audio_info_lbl.configure(
                text="Recommended: 3–10 seconds of clean speech from a single speaker.",
                foreground="#6b7280",
            )
            return
        dur = get_audio_duration(path)
        if dur is None:
            self.audio_info_lbl.configure(text="Ready (duration detected during encoding).", foreground="#6b7280")
        elif dur < 2.5:
            self.audio_info_lbl.configure(
                text=f"Duration: {dur:.1f}s (Too short — 3 to 10s works best).",
                foreground="#dc2626",
            )
        elif dur > 15.0:
            self.audio_info_lbl.configure(
                text=f"Duration: {dur:.1f}s (>15s can degrade voice similarity).",
                foreground="#d97706",
            )
        else:
            self.audio_info_lbl.configure(
                text=f"Duration: {dur:.1f}s (Optimal 3–10s range).",
                foreground="#059669",
            )

    def _toggle_record_mic(self) -> None:
        if self._recording:
            if self._record_cancel is not None:
                self._record_cancel.set()
            return
        if not SOUNDDEVICE_AVAILABLE or not SOUNDFILE_AVAILABLE:
            messagebox.showwarning(
                "Microphone Unavailable",
                "Recording requires 'sounddevice' and 'soundfile'.",
                parent=self,
            )
            return

        self._recording = True
        self._record_cancel = threading.Event()
        self._record_abort = threading.Event()
        rec_cancel = self._record_cancel
        rec_abort = self._record_abort
        self.rec_btn.configure(text="⏹ Stop (0.0s)")
        self.status_lbl.configure(text="Recording microphone reference (speak clearly for 3–8 seconds)...")

        def _prog(elapsed: float):
            self._safe_after(lambda: self.rec_btn.configure(text=f"⏹ Stop ({elapsed:.1f}s)"))

        def _rec_worker():
            wav_path: Optional[Path] = None
            err_msg: Optional[str] = None
            try:
                wav_path = self.manager.record_microphone_sample(
                    duration_sec=8.0,
                    cancel_event=rec_cancel,
                    progress_cb=_prog,
                    abort_event=rec_abort,
                )
            except Exception as exc:
                err_msg = str(exc)

            def _done():
                self._recording = False
                self._record_cancel = None
                self._record_abort = None
                self.rec_btn.configure(text="🎙 Record 8s")
                if rec_abort.is_set():
                    return
                if err_msg:
                    self.status_lbl.configure(text=f"Recording failed: {err_msg}")
                    messagebox.showerror("Recording Error", err_msg, parent=self)
                elif wav_path:
                    self.clone_audio_var.set(str(wav_path))
                    if not self.clone_name_var.get().strip():
                        self.clone_name_var.set("My Voice")
                    self.status_lbl.configure(text=f"Captured microphone clip: {wav_path.name}")
            self._safe_after(_done)

        threading.Thread(target=_rec_worker, daemon=True).start()

    def _auto_transcribe_audio(self) -> None:
        audio_path = self.clone_audio_var.get().strip()
        if not audio_path or not os.path.isfile(audio_path):
            messagebox.showwarning("No Audio", "Please choose or record an audio file first.", parent=self)
            return
        lang = label_to_language(self.clone_lang_var.get())
        self.auto_tx_btn.configure(state=tk.DISABLED)
        self.op_progress.pack(side=tk.RIGHT, padx=(8, 0))
        self.op_progress.start(12)
        self.status_lbl.configure(text="Transcribing reference audio with local Whisper...")

        def _worker():
            text_out, err_msg = "", None
            try:
                text_out = self.manager.transcribe_audio_whisper(audio_path, language=lang)
            except Exception as exc:
                err_msg = str(exc)

            def _done():
                self.op_progress.stop()
                self.op_progress.pack_forget()
                self.auto_tx_btn.configure(state=tk.NORMAL if WHISPER_AVAILABLE else tk.DISABLED)
                if err_msg:
                    self.status_lbl.configure(text="Transcription failed")
                    messagebox.showerror("Transcription Error", err_msg, parent=self)
                else:
                    self.clone_transcript_text.delete("1.0", tk.END)
                    self.clone_transcript_text.insert("1.0", text_out)
                    self.status_lbl.configure(text="Transcript filled automatically.")
            self._safe_after(_done)

        threading.Thread(target=_worker, daemon=True).start()

    def _train_custom_voice(self) -> None:
        if self._busy_training:
            return
        name = self.clone_name_var.get().strip()
        audio_path = self.clone_audio_var.get().strip()
        transcript = self.clone_transcript_text.get("1.0", "end-1c").strip()
        lang = label_to_language(self.clone_lang_var.get())

        if not name or not audio_path or not transcript:
            messagebox.showwarning(
                "Missing Information",
                "Please enter a voice name, choose or record reference audio, and provide its verbatim transcript.",
                parent=self,
            )
            return

        try:
            self._apply_to_manager()
        except Exception as e:
            messagebox.showwarning("Invalid Settings", str(e), parent=self)
            return

        self._busy_training = True
        self.train_btn.configure(state=tk.DISABLED)
        self.op_progress.pack(side=tk.RIGHT, padx=(8, 0))
        self.op_progress.start(12)

        def _worker():
            record = None
            err_msg = None
            try:
                record = self.manager.train_custom_voice(
                    name=name, audio_path=audio_path, transcript=transcript, language=lang, set_active=True
                )
            except Exception as exc:
                err_msg = str(exc)

            def _done():
                self._busy_training = False
                self.op_progress.stop()
                self.op_progress.pack_forget()
                self.train_btn.configure(state=tk.NORMAL)
                self._refresh_header_badge()
                if err_msg:
                    messagebox.showerror("Voice Training Error", err_msg, parent=self)
                elif record:
                    self.voice_type_var.set("custom")
                    self._refresh_custom_voice_widgets()
                    self._on_voice_type_changed()
                    self.lang_var.set(language_to_label(record["language"]))
                    self._on_language_changed()
                    messagebox.showinfo(
                        "Custom Voice Saved",
                        f"Custom voice '{record['name']}' has been trained and saved. "
                        "Click 'Save & Apply' to confirm your active dictation settings.",
                        parent=self,
                    )
            self._safe_after(_done)

        threading.Thread(target=_worker, daemon=True).start()

    def _use_selected_custom_voice(self) -> Optional[Dict[str, Any]]:
        sel = self.voices_tree.selection()
        if not sel:
            return None
        vid = sel[0]
        with self.manager._settings_lock:
            rec = self.manager.settings.get_custom_voice(vid)
            if not rec:
                return None
            self.manager.settings.voice_type = "custom"
            self.manager.settings.voice_id = vid
            self.manager.settings.save(self.manager.settings_file)
        self.voice_type_var.set("custom")
        self._refresh_custom_voice_widgets()
        self._on_voice_type_changed()
        self.status_lbl.configure(text=f"Selected custom voice: {rec['name']}")
        return rec

    def _test_selected_custom_voice(self) -> None:
        rec = self._use_selected_custom_voice()
        if not rec:
            return
        lang = rec.get("language", "en")
        self.lang_var.set(language_to_label(lang))
        self._on_language_changed()
        with self.manager._settings_lock:
            self.manager.settings.language = lang
            self.manager.settings.save(self.manager.settings_file)
        sample = self.test_text_var.get().strip() or LANGUAGE_SAMPLES.get(lang, LANGUAGE_SAMPLES["en"])
        self.manager.speak_preview(sample_text=sample)

    def _delete_selected_custom_voice(self) -> None:
        sel = self.voices_tree.selection()
        if not sel:
            return
        vid = sel[0]
        rec = self.manager.settings.get_custom_voice(vid)
        if not rec:
            return
        if not messagebox.askyesno(
            "Delete Custom Voice",
            f"Permanently delete custom voice '{rec['name']}' from disk?",
            parent=self,
        ):
            return
        if self.manager.delete_custom_voice(vid):
            self.voice_type_var.set(self.manager.settings.voice_type)
            self._refresh_custom_voice_widgets()
            self._on_voice_type_changed()
            self.status_lbl.configure(text=f"Deleted custom voice '{rec['name']}'.")

    def _reset_defaults(self) -> None:
        with self.manager._settings_lock:
            kept_custom = [dict(v) for v in self.manager.settings.custom_voices]
            self.manager.settings = TTSSettings(custom_voices=kept_custom)
            self.manager.settings.save(self.manager.settings_file)
        self._load_settings_into_ui()
        self.status_lbl.configure(text="Restored default Dictation Properties (custom voices kept).")

    def _save_and_close(self) -> None:
        try:
            saved = self._apply_to_manager()
        except Exception as e:
            messagebox.showwarning("Invalid Parameter", str(e), parent=self)
            return
        self._saved_committed = True
        self.result = saved
        if self.on_save:
            try:
                self.on_save(saved)
            except Exception:
                pass
        self.destroy()
