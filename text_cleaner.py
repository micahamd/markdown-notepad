"""
Text Cleaner & Reflow Module for MarkItDown Notepad

Provides high-performance, deterministic algorithmic text normalization:
- De-hyphenation across line breaks
- Statistical soft-wrap paragraph reflow
- Artifact stripping (form-feeds, page numbers, repeating headers/footers)
- Markdown semantic preservation (code blocks, tables, lists, blockquotes, headings)
- Whitespace and punctuation normalization
"""

import re
import statistics
from typing import List, Dict, Set, Optional, Tuple


class TextCleanerOptions:
    """Configuration options for TextCleaner."""
    def __init__(
        self,
        dehyphenate: bool = True,
        reflow_paragraphs: bool = True,
        strip_page_numbers: bool = True,
        strip_headers_footers: bool = True,
        auto_promote_headings: bool = True,
        normalize_whitespace: bool = True,
        preserve_code_blocks: bool = True,
        preserve_tables: bool = True,
        preserve_lists: bool = True,
        preserve_blockquotes: bool = True,
    ):
        self.dehyphenate = dehyphenate
        self.reflow_paragraphs = reflow_paragraphs
        self.strip_page_numbers = strip_page_numbers
        self.strip_headers_footers = strip_headers_footers
        self.auto_promote_headings = auto_promote_headings
        self.normalize_whitespace = normalize_whitespace
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_tables = preserve_tables
        self.preserve_lists = preserve_lists
        self.preserve_blockquotes = preserve_blockquotes

    @classmethod
    def default(cls) -> 'TextCleanerOptions':
        return cls()


class TextCleaner:
    """
    Deterministic NLP & layout-aware document normalizer.
    """

    # Common abbreviations that end in a period but do not terminate a sentence
    ABBREVIATIONS: Set[str] = {
        'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'st.',
        'vs.', 'etc.', 'e.g.', 'i.e.', 'fig.', 'figs.', 'vol.', 'vols.',
        'pp.', 'p.', 'no.', 'nos.', 'ed.', 'eds.', 'al.', 'dept.',
        'est.', 'approx.', 'inc.', 'ltd.', 'co.', 'corp.', 'gen.',
        'col.', 'capt.', 'lieut.', 'rev.', 'hon.', 'gov.', 'pres.',
        'sen.', 'rep.', 'jan.', 'feb.', 'mar.', 'apr.', 'jun.', 'jul.',
        'aug.', 'sep.', 'sept.', 'oct.', 'nov.', 'dec.', 'cf.', 'ibid.',
        'op.', 'cit.', 'loc.', 'ca.'
    }

    # Prefixes that typically keep hyphens when split (e.g., self-evident)
    KEEP_HYPHEN_PREFIXES: Set[str] = {
        'self', 'all', 'quasi', 'ex', 'cross', 'half', 'well', 'ill',
        'co', 'pre', 'post', 'non', 'anti', 'multi', 'sub', 'ultra',
        'semi', 'mid', 'great', 'step'
    }

    # Regex patterns
    RE_CODE_FENCE = re.compile(r'^\s*(`{3,}|~{3,})')
    RE_TABLE_ROW = re.compile(r'^\s*\|.*\|\s*$')
    RE_MD_HEADING = re.compile(r'^\s*#{1,6}\s+')
    RE_LIST_ITEM = re.compile(r'^\s*(?:[-*+]|\d+[.)]|\([0-9a-zA-Z]+\))\s+')
    RE_BLOCKQUOTE = re.compile(r'^\s*>\s*')
    RE_HORIZONTAL_RULE = re.compile(r'^\s*(?:[-*_]\s*){3,}\s*$')
    
    # Page numbers and form feeds
    RE_FORM_FEED = re.compile(r'\x0c')
    RE_STANDALONE_PAGE_NUM = re.compile(
        r'^\s*(?:(?:Page|p\.|pp\.|-)\s*)?\d+(?:\s*-)?\s*$',
        re.IGNORECASE
    )
    RE_ROMAN_PAGE_NUM = re.compile(
        r'^\s*(?:'
        r'(?:Page|p\.|pp\.)\s+[-—]?\s*(?=[ivxlcdm]+\b)m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})(?:\s*[-—])?'
        r'|[-—]\s*(?=[ivxlcdm]+\b)m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})\s*[-—]'
        r')\s*$',
        re.IGNORECASE
    )
    RE_BRACKETED_PAGE_NUM = re.compile(r'^\s*\[\s*\d+\s*\]\s*$')

    # De-hyphenation across line breaks: "infor-\nmation" -> "information"
    RE_HYPHEN_LINEBREAK = re.compile(r'(\b[A-Za-z]+)[-\u2010\u00ad]\s*\n\s*([A-Za-z]+\b)')

    @classmethod
    def _update_fence_state(
        cls, line: str, active_fence: Optional[Tuple[str, int]]
    ) -> Tuple[bool, Optional[Tuple[str, int]]]:
        """Track fenced code block state across lines. Returns (is_fence_line, next_fence)."""
        if active_fence is None:
            m = cls.RE_CODE_FENCE.match(line)
            if m:
                fence = m.group(1)
                return True, (fence[0], len(fence))
            return False, None
        fence_char, fence_len = active_fence
        if re.match(rf'^\s*{re.escape(fence_char)}{{{fence_len},}}\s*$', line):
            return True, None
        return False, active_fence

    @classmethod
    def clean(cls, text: str, options: Optional[TextCleanerOptions] = None) -> str:
        """
        Execute full cleaning pipeline on text.
        """
        if not text or not text.strip():
            return text or ""

        if options is None:
            options = TextCleanerOptions.default()

        # Step 1: Normalize unicode characters (spaces, smart quotes, etc.)
        if options.normalize_whitespace:
            text = cls._normalize_unicode_characters(text)

        # Step 2: Strip page artifacts (form feeds, isolated page numbers)
        if options.strip_page_numbers:
            text = cls._strip_page_artifacts(text)

        # Step 3: Strip periodic repeating headers & footers
        if options.strip_headers_footers:
            text = cls._strip_running_headers_footers(text)

        # Step 4: De-hyphenate words broken across lines
        if options.dehyphenate:
            text = cls._dehyphenate(text)

        # Step 5: Statistical & structural paragraph reflow
        if options.reflow_paragraphs:
            text = cls._reflow_paragraphs(text, options)

        # Step 6: Final whitespace cleanup
        if options.normalize_whitespace:
            text = cls._finalize_whitespace(text)

        return text

    @classmethod
    def _normalize_unicode_characters(cls, text: str) -> str:
        """Normalize unusual whitespace, zero-width spaces, and control characters."""
        replacements = {
            '\u00a0': ' ',  # Non-breaking space
            '\u200b': '',   # Zero-width space
            '\u200c': '',   # Zero-width non-joiner
            '\u200d': '',   # Zero-width joiner
            '\ufeff': '',   # Byte order mark
            '\u2002': ' ',  # En space
            '\u2003': ' ',  # Em space
            '\u2009': ' ',  # Thin space
            '\r\n': '\n',   # CRLF to LF
            '\r': '\n',     # CR to LF
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    @classmethod
    def _strip_page_artifacts(cls, text: str) -> str:
        """Remove form feeds and standalone page number lines outside code blocks."""
        text = cls.RE_FORM_FEED.sub('\n\n', text)

        lines = text.split('\n')
        cleaned_lines = []
        active_fence: Optional[Tuple[str, int]] = None
        for i, line in enumerate(lines):
            is_fence, next_fence = cls._update_fence_state(line, active_fence)
            in_code = is_fence or (active_fence is not None)
            active_fence = next_fence
            if in_code:
                cleaned_lines.append(line)
                continue

            stripped = line.strip()
            if stripped and (
                cls.RE_STANDALONE_PAGE_NUM.match(stripped)
                or cls.RE_ROMAN_PAGE_NUM.match(stripped)
                or cls.RE_BRACKETED_PAGE_NUM.match(stripped)
            ):
                if len(stripped) <= 15:
                    prev_empty = (i == 0 or not lines[i - 1].strip())
                    next_empty = (i == len(lines) - 1 or not lines[i + 1].strip())
                    if prev_empty or next_empty:
                        continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    @classmethod
    def _strip_running_headers_footers(cls, text: str) -> str:
        """
        Detect and remove running headers/footers that repeat across page boundaries.
        """
        lines = text.split('\n')
        if len(lines) < 60:
            return text

        candidate_counts: Dict[str, int] = {}
        active_fence: Optional[Tuple[str, int]] = None
        for line in lines:
            is_fence, next_fence = cls._update_fence_state(line, active_fence)
            in_code = is_fence or (active_fence is not None)
            active_fence = next_fence
            if in_code:
                continue
            s = line.strip()
            if 4 <= len(s) <= 75 and not s.startswith(('#', '```', '~~~', '|', '>', '-', '*', '1.')):
                candidate_counts[s] = candidate_counts.get(s, 0) + 1

        repeating_headers = {
            s for s, count in candidate_counts.items()
            if count >= 3 and count > (len(lines) / 300)
        }

        if not repeating_headers:
            return text

        cleaned_lines = []
        active_fence = None
        for line in lines:
            is_fence, next_fence = cls._update_fence_state(line, active_fence)
            in_code = is_fence or (active_fence is not None)
            active_fence = next_fence
            if not in_code and line.strip() in repeating_headers:
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    @classmethod
    def _dehyphenate(cls, text: str) -> str:
        """
        Merge words broken across line breaks with hyphens (outside code blocks).
        'infor-\\nmation' -> 'information'
        'self-\\nevident' -> 'self-evident'
        """
        def replace_hyphen(match: re.Match) -> str:
            prefix = match.group(1)
            suffix = match.group(2)
            lower_prefix = prefix.lower()

            if lower_prefix in cls.KEEP_HYPHEN_PREFIXES:
                return f"{prefix}-{suffix}"

            if suffix[0].isupper():
                return f"{prefix}-{suffix}"

            return f"{prefix}{suffix}"

        segments: List[Tuple[bool, List[str]]] = []
        active_fence: Optional[Tuple[str, int]] = None
        for line in text.splitlines(keepends=True):
            is_fence, next_fence = cls._update_fence_state(line, active_fence)
            in_code = is_fence or (active_fence is not None)
            active_fence = next_fence
            if segments and segments[-1][0] == in_code:
                segments[-1][1].append(line)
            else:
                segments.append((in_code, [line]))

        out_parts = []
        for in_code, seg_lines in segments:
            seg_text = ''.join(seg_lines)
            if in_code:
                out_parts.append(seg_text)
            else:
                out_parts.append(cls.RE_HYPHEN_LINEBREAK.sub(replace_hyphen, seg_text))
        return ''.join(out_parts)

    @classmethod
    def _reflow_paragraphs(cls, text: str, options: TextCleanerOptions) -> str:
        """
        Intelligently reflow soft line breaks into cohesive paragraphs
        while preserving code blocks, tables, lists, blockquotes, and headings.
        """
        raw_lines = text.split('\n')
        if not raw_lines:
            return text

        non_empty_lengths = []
        active_fence: Optional[Tuple[str, int]] = None
        for l in raw_lines:
            is_fence, next_fence = cls._update_fence_state(l, active_fence)
            in_code = is_fence or (active_fence is not None)
            active_fence = next_fence
            if not in_code and l.strip() and not l.lstrip().startswith(('#', '```', '~~~', '|')):
                non_empty_lengths.append(len(l.strip()))
        median_length = statistics.median(non_empty_lengths) if non_empty_lengths else 65
        min_continuation_length = max(35, int(median_length * 0.60))

        output_paragraphs: List[str] = []
        current_paragraph: List[str] = []

        def flush_paragraph():
            if current_paragraph:
                combined = ' '.join(p.strip() for p in current_paragraph if p.strip())
                if combined:
                    output_paragraphs.append(combined)
                current_paragraph.clear()

        idx = 0
        n = len(raw_lines)
        while idx < n:
            line = raw_lines[idx]
            stripped = line.strip()

            # Fenced code block (accumulated as a single newline-joined block)
            if options.preserve_code_blocks:
                is_fence, fence_state = cls._update_fence_state(line, None)
                if is_fence and fence_state is not None:
                    flush_paragraph()
                    code_lines = [line]
                    idx += 1
                    while idx < n:
                        code_lines.append(raw_lines[idx])
                        _, next_state = cls._update_fence_state(raw_lines[idx], fence_state)
                        idx += 1
                        if next_state is None:
                            break
                    output_paragraphs.append('\n'.join(code_lines))
                    continue

            if not stripped:
                flush_paragraph()
                idx += 1
                continue

            # Markdown table block (accumulated as a single newline-joined block)
            if options.preserve_tables and cls.RE_TABLE_ROW.match(line):
                flush_paragraph()
                table_lines = [line]
                idx += 1
                while idx < n and cls.RE_TABLE_ROW.match(raw_lines[idx]):
                    table_lines.append(raw_lines[idx])
                    idx += 1
                output_paragraphs.append('\n'.join(table_lines))
                continue

            if cls.RE_MD_HEADING.match(line):
                flush_paragraph()
                output_paragraphs.append(line)
                idx += 1
                continue

            if options.auto_promote_headings and cls._is_all_caps_heading(stripped):
                flush_paragraph()
                output_paragraphs.append(f"## {stripped.title()}")
                idx += 1
                continue

            if cls.RE_HORIZONTAL_RULE.match(line):
                flush_paragraph()
                output_paragraphs.append(line)
                idx += 1
                continue

            # List block (accumulated as a single newline-joined block)
            if options.preserve_lists and cls.RE_LIST_ITEM.match(line):
                flush_paragraph()
                list_lines = [line]
                idx += 1
                while idx < n:
                    nxt = raw_lines[idx]
                    if not nxt.strip():
                        break
                    if cls.RE_LIST_ITEM.match(nxt):
                        list_lines.append(nxt)
                        idx += 1
                    elif nxt[0] in (' ', '\t') and not (
                        cls.RE_MD_HEADING.match(nxt)
                        or cls.RE_HORIZONTAL_RULE.match(nxt)
                        or cls.RE_TABLE_ROW.match(nxt)
                        or cls.RE_CODE_FENCE.match(nxt)
                        or cls.RE_BLOCKQUOTE.match(nxt)
                    ):
                        list_lines.append(nxt)
                        idx += 1
                    else:
                        break
                output_paragraphs.append('\n'.join(list_lines))
                continue

            # Blockquote block (accumulated as a single newline-joined block)
            if options.preserve_blockquotes and cls.RE_BLOCKQUOTE.match(line):
                flush_paragraph()
                quote_lines = [line]
                idx += 1
                while idx < n and cls.RE_BLOCKQUOTE.match(raw_lines[idx]):
                    quote_lines.append(raw_lines[idx])
                    idx += 1
                output_paragraphs.append('\n'.join(quote_lines))
                continue

            # Preserve standalone image tags
            if re.match(r'^\s*(?:!\[.*?\]\(.*?\)|<img\s+[^>]*>)\s*$', stripped, re.IGNORECASE):
                flush_paragraph()
                output_paragraphs.append(stripped)
                idx += 1
                continue

            if not current_paragraph:
                current_paragraph.append(stripped)
            else:
                prev_line = current_paragraph[-1]
                is_prev_sentence_end = cls._ends_sentence(prev_line)
                is_prev_short = len(prev_line) < min_continuation_length
                
                if is_prev_sentence_end and is_prev_short:
                    flush_paragraph()
                    current_paragraph.append(stripped)
                else:
                    current_paragraph.append(stripped)

            idx += 1

        flush_paragraph()
        return '\n\n'.join(output_paragraphs)

    @classmethod
    def _is_all_caps_heading(cls, line: str) -> bool:
        """Check if a line looks like an ALL CAPS title or section heading."""
        if not (4 <= len(line) <= 65):
            return False
        has_upper = any(c.isupper() for c in line)
        has_lower = any(c.islower() for c in line)
        if has_upper and not has_lower:
            words = line.split()
            return len(words) >= 1 and not line.startswith(('#', '-', '*', '|', '>', '!', '<', '['))
        return False

    @classmethod
    def _ends_sentence(cls, text: str) -> bool:
        """Check if a line ends with a definitive sentence terminator."""
        stripped = text.rstrip(' "\')]}')
        if not stripped:
            return False

        last_char = stripped[-1]
        if last_char in {'.', '!', '?'}:
            words = stripped.split()
            if words:
                last_word = words[-1].lower()
                if last_word in cls.ABBREVIATIONS:
                    return False
            return True

        if last_char in {':', ';'}:
            return True

        return False

    @classmethod
    def _finalize_whitespace(cls, text: str) -> str:
        """Clean up extraneous spaces and excessive blank lines while preserving code blocks, tables, and indentation."""
        out_lines: List[str] = []
        active_fence: Optional[Tuple[str, int]] = None
        blank_run = 0

        for line in text.split('\n'):
            is_fence, next_fence = cls._update_fence_state(line, active_fence)
            in_code = is_fence or (active_fence is not None)
            active_fence = next_fence

            if in_code:
                blank_run = 0
                out_lines.append(line)
                continue

            if not line.strip():
                blank_run += 1
                if blank_run <= 1:
                    out_lines.append('')
                continue

            blank_run = 0
            if cls.RE_TABLE_ROW.match(line):
                out_lines.append(line.rstrip())
            else:
                m = re.match(r'^([ \t]*)(.*)$', line)
                leading, body = m.group(1), m.group(2)
                cleaned_body = re.sub(r'[ \t]{2,}', ' ', body).rstrip()
                out_lines.append(f"{leading}{cleaned_body}")

        # Trim leading/trailing empty lines without stripping leading indentation on content lines
        while out_lines and not out_lines[0].strip():
            out_lines.pop(0)
        while out_lines and not out_lines[-1].strip():
            out_lines.pop()

        if not out_lines:
            return ''
        return '\n'.join(out_lines) + '\n'


def clean_document(text: str, options: Optional[TextCleanerOptions] = None) -> str:
    """Convenience function for cleaning and reflowing text."""
    return TextCleaner.clean(text, options)
