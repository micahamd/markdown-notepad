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
    RE_CODE_FENCE = re.compile(r'^\s*(```|~~~)')
    RE_TABLE_ROW = re.compile(r'^\s*\|.*\|\s*$')
    RE_MD_HEADING = re.compile(r'^\s*#{1,6}\s+')
    RE_LIST_ITEM = re.compile(r'^\s*(?:[-*+]|\d+[.)]|\([0-9a-zA-Z]+\))\s+')
    RE_BLOCKQUOTE = re.compile(r'^\s*>\s*')
    RE_HORIZONTAL_RULE = re.compile(r'^\s*(?:[-*_]\s*){3,}\s*$')
    
    # Page numbers and form feeds
    RE_FORM_FEED = re.compile(r'\x0c')
    RE_STANDALONE_PAGE_NUM = re.compile(
        r'^\s*(?:(?:Page|p\.|pp\.|-)\s*)?(?:\d+|[ivxlcdm]+)(?:\s*-)?\s*$',
        re.IGNORECASE
    )
    RE_BRACKETED_PAGE_NUM = re.compile(r'^\s*\[\s*\d+\s*\]\s*$')

    # De-hyphenation across line breaks: "infor-\nmation" -> "information"
    RE_HYPHEN_LINEBREAK = re.compile(r'(\b[A-Za-z]+)[-\u2010\u00ad]\s*\n\s*([A-Za-z]+\b)')

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
        """Remove form feeds and standalone page number lines."""
        text = cls.RE_FORM_FEED.sub('\n\n', text)

        lines = text.split('\n')
        cleaned_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and (cls.RE_STANDALONE_PAGE_NUM.match(stripped) or cls.RE_BRACKETED_PAGE_NUM.match(stripped)):
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
        for line in lines:
            s = line.strip()
            if 4 <= len(s) <= 75 and not s.startswith(('#', '```', '|', '>', '-', '*', '1.')):
                candidate_counts[s] = candidate_counts.get(s, 0) + 1

        repeating_headers = {
            s for s, count in candidate_counts.items()
            if count >= 3 and count > (len(lines) / 300)
        }

        if not repeating_headers:
            return text

        cleaned_lines = []
        for line in lines:
            if line.strip() in repeating_headers:
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    @classmethod
    def _dehyphenate(cls, text: str) -> str:
        """
        Merge words broken across line breaks with hyphens.
        'infor-\nmation' -> 'information'
        'self-\nevident' -> 'self-evident'
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

        return cls.RE_HYPHEN_LINEBREAK.sub(replace_hyphen, text)

    @classmethod
    def _reflow_paragraphs(cls, text: str, options: TextCleanerOptions) -> str:
        """
        Intelligently reflow soft line breaks into cohesive paragraphs
        while preserving code blocks, tables, lists, and headings.
        """
        raw_lines = text.split('\n')
        if not raw_lines:
            return text

        non_empty_lengths = [len(l.strip()) for l in raw_lines if l.strip() and not l.startswith(('#', '```', '|'))]
        median_length = statistics.median(non_empty_lengths) if non_empty_lengths else 65
        min_continuation_length = max(35, int(median_length * 0.60))

        output_paragraphs: List[str] = []
        current_paragraph: List[str] = []
        in_code_block = False

        def flush_paragraph():
            if current_paragraph:
                combined = ' '.join(p.strip() for p in current_paragraph if p.strip())
                if combined:
                    output_paragraphs.append(combined)
                current_paragraph.clear()

        for line in raw_lines:
            stripped = line.strip()

            if options.preserve_code_blocks and cls.RE_CODE_FENCE.match(line):
                flush_paragraph()
                in_code_block = not in_code_block
                output_paragraphs.append(line)
                continue

            if in_code_block:
                output_paragraphs.append(line)
                continue

            if not stripped:
                flush_paragraph()
                continue

            if options.preserve_tables and cls.RE_TABLE_ROW.match(line):
                flush_paragraph()
                output_paragraphs.append(line)
                continue

            if cls.RE_MD_HEADING.match(line):
                flush_paragraph()
                output_paragraphs.append(line)
                continue

            if options.auto_promote_headings and cls._is_all_caps_heading(stripped):
                flush_paragraph()
                output_paragraphs.append(f"## {stripped.title()}")
                continue

            if cls.RE_HORIZONTAL_RULE.match(line):
                flush_paragraph()
                output_paragraphs.append(line)
                continue

            if options.preserve_lists and cls.RE_LIST_ITEM.match(line):
                flush_paragraph()
                output_paragraphs.append(line)
                continue

            if options.preserve_blockquotes and cls.RE_BLOCKQUOTE.match(line):
                flush_paragraph()
                output_paragraphs.append(line)
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
            return len(words) >= 1 and not line.startswith(('#', '-', '*', '|', '>'))
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
        """Clean up extraneous spaces and excessive blank lines."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [re.sub(r'[ \t]{2,}', ' ', l).rstrip() for l in text.split('\n')]
        return '\n'.join(lines).strip() + '\n'


def clean_document(text: str, options: Optional[TextCleanerOptions] = None) -> str:
    """Convenience function for cleaning and reflowing text."""
    return TextCleaner.clean(text, options)
