"""
Export Manager Module for MarkItDown Notepad

Provides export functionality to PDF and DOCX formats.
Uses 'markdown' for HTML generation, 'weasyprint' for PDF rendering,
and 'python-docx' for Word document creation.
"""

import html as _html
import os
import re
import base64
import tempfile
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Optional, Dict, Tuple, List, Any, Callable, Set, Union

# Try to import markdown
try:
    import markdown
    from markdown.extensions.tables import TableExtension
    from markdown.extensions.fenced_code import FencedCodeExtension
    from markdown.extensions.codehilite import CodeHiliteExtension
    from markdown.extensions.toc import TocExtension
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# Try to import weasyprint for PDF
try:
    import weasyprint
    from weasyprint import HTML as WeasyprintHTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    weasyprint = None
    WEASYPRINT_AVAILABLE = False

# Try to import python-docx for DOCX
try:
    import docx
    from docx import Document as DocxDocument
    from docx.shared import Inches, Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml.ns import qn
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Try to import PIL for image handling in exports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def get_available_export_formats() -> Dict[str, bool]:
    """Return dict of format name -> availability"""
    return {
        'pdf': WEASYPRINT_AVAILABLE and MARKDOWN_AVAILABLE,
        'docx': DOCX_AVAILABLE,
        'html': MARKDOWN_AVAILABLE,
    }


# =============================================================================
# CSS Stylesheet for PDF/HTML Export
# =============================================================================

def _export_latex_to_unicode(expr: str) -> str:
    """Convert common LaTeX commands, superscripts, and subscripts to readable Unicode for exports."""
    if not expr:
        return ""
    s = expr.strip()
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'√(\1)', s)
    symbols = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ', r'\epsilon': 'ε',
        r'\theta': 'θ', r'\lambda': 'λ', r'\mu': 'μ', r'\pi': 'π', r'\sigma': 'σ',
        r'\tau': 'τ', r'\phi': 'φ', r'\omega': 'ω', r'\Gamma': 'Γ', r'\Delta': 'Δ',
        r'\Theta': 'Θ', r'\Lambda': 'Λ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Omega': 'Ω',
        r'\pm': '±', r'\mp': '∓', r'\times': '×', r'\div': '÷', r'\cdot': '·',
        r'\leq': '≤', r'\le': '≤', r'\geq': '≥', r'\ge': '≥', r'\neq': '≠', r'\ne': '≠',
        r'\approx': '≈', r'\equiv': '≡', r'\propto': '∝', r'\infty': '∞', r'\partial': '∂',
        r'\nabla': '∇', r'\sum': '∑', r'\prod': '∏', r'\int': '∫', r'\sqrt': '√',
        r'\in': '∈', r'\notin': '∉', r'\subset': '⊂', r'\subseteq': '⊆',
        r'\cup': '∪', r'\cap': '∩', r'\emptyset': '∅', r'\forall': '∀', r'\exists': '∃',
        r'\neg': '¬', r'\wedge': '∧', r'\vee': '∨', r'\rightarrow': '→', r'\to': '→',
        r'\leftarrow': '←', r'\leftrightarrow': '↔', r'\Rightarrow': '⇒',
        r'\Leftarrow': '⇐', r'\Leftrightarrow': '⇔', r'\iff': '⇔', r'\dots': '…', r'\cdots': '⋯',
    }
    for cmd in sorted(symbols.keys(), key=len, reverse=True):
        s = re.sub(re.escape(cmd) + r'(?![a-zA-Z])', lambda _, rep=symbols[cmd]: rep, s)
    sup_map = str.maketrans("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
    sub_map = str.maketrans("0123456789+-=()aeioruvxjk", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒᵣᵤᵥₓⱼₖ")
    s = re.sub(r'\^\{([^{}]+)\}', lambda m: m.group(1).translate(sup_map), s)
    s = re.sub(r'\^([0-9+\-=ni])', lambda m: m.group(1).translate(sup_map), s)
    s = re.sub(r'_\{([^{}]+)\}', lambda m: m.group(1).translate(sub_map), s)
    s = re.sub(r'_([0-9+\-=aeioruvxjk])', lambda m: m.group(1).translate(sub_map), s)
    return s


def _build_css(theme: Optional[Dict] = None, page_settings: Optional[Dict] = None) -> str:
    """Build a CSS stylesheet for HTML/PDF export based on theme"""
    # Defaults
    bg = theme.get('visual_bg', '#ffffff') if theme else '#ffffff'
    fg = theme.get('visual_fg', '#1a1a1a') if theme else '#1a1a1a'
    font_family = theme.get('visual_font', 'Georgia') if theme else 'Georgia'
    font_size = theme.get('visual_font_size', 11) if theme else 11

    bq_font = (theme.get('blockquote_font') if theme else None) or font_family
    bq_size_val = (theme.get('blockquote_size') if theme else 0) or 0
    bq_size = int(bq_size_val) if int(bq_size_val) > 0 else font_size
    bq_style_raw = (theme.get('blockquote_style', 'italic') if theme else 'italic') or 'italic'
    bq_font_style = 'italic' if 'italic' in bq_style_raw else 'normal'
    bq_font_weight = 'bold' if 'bold' in bq_style_raw else 'normal'
    bq_fg = (theme.get('blockquote_fg') if theme else None) or '#555555'
    bq_bg = (theme.get('blockquote_bg') if theme else None) or '#f9f9f9'
    bq_bar = (theme.get('blockquote_bar_fg') if theme else None) or '#2563eb'
    bq_pad = int((theme.get('blockquote_padding', 16) if theme else 16) or 16)
    bq_space = int((theme.get('blockquote_spacing', 4) if theme else 4) or 4)
    bq_justify = (theme.get('blockquote_justify', 'left') if theme else 'left') or 'left'
    try:
        line_spacing = max(1.0, min(3.0, float((theme or {}).get('line_spacing', 1.6) or 1.6)))
    except (TypeError, ValueError):
        line_spacing = 1.6
    
    page_size = 'A4'
    margin = '2cm'
    if page_settings:
        page_size = page_settings.get('page_size', 'A4')
        margin = page_settings.get('margin', '2cm')
    
    return f"""
    @page {{
        size: {page_size};
        margin: {margin};
    }}
    body {{
        font-family: '{font_family}', Georgia, 'Times New Roman', serif;
        font-size: {font_size}pt;
        line-height: {line_spacing};
        color: {fg};
        background-color: {bg};
        max-width: 100%;
        padding: 0;
    }}
    h1 {{ font-size: 2em; margin: 0.8em 0 0.4em; border-bottom: 2px solid #ccc; padding-bottom: 0.3em; }}
    h2 {{ font-size: 1.6em; margin: 0.7em 0 0.35em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
    h3 {{ font-size: 1.3em; margin: 0.6em 0 0.3em; }}
    h4 {{ font-size: 1.1em; margin: 0.5em 0 0.25em; }}
    h5, h6 {{ font-size: 1em; margin: 0.5em 0 0.25em; }}
    p {{ margin: 0.8em 0; text-align: justify; text-justify: inter-word; hyphens: auto; }}
    div[align="left"], div[align="left"] p, div[align="left"] h1, div[align="left"] h2, div[align="left"] h3, div[align="left"] h4 {{
        text-align: left !important;
    }}
    div[align="center"], div[align="center"] p, div[align="center"] h1, div[align="center"] h2, div[align="center"] h3, div[align="center"] h4 {{
        text-align: center !important;
    }}
    div[align="right"], div[align="right"] p, div[align="right"] h1, div[align="right"] h2, div[align="right"] h3, div[align="right"] h4 {{
        text-align: right !important;
    }}
    blockquote {{
        margin: {bq_space}px 0;
        padding: 0.5em {bq_pad}px;
        border-left: 4px solid {bq_bar};
        font-family: '{bq_font}', Georgia, serif;
        font-size: {bq_size}pt;
        font-style: {bq_font_style};
        font-weight: {bq_font_weight};
        text-align: {bq_justify};
        color: {bq_fg};
        background: {bq_bg};
    }}
    blockquote p {{
        text-align: {bq_justify};
        margin: 0.3em 0;
    }}
    .math-inline {{
        font-family: 'Cambria Math', 'STIX Two Math', Georgia, serif;
        font-style: italic;
        color: #0f766e;
    }}
    .math-block {{
        font-family: 'Cambria Math', 'STIX Two Math', Georgia, serif;
        font-size: 1.15em;
        text-align: center;
        margin: 0.8em 0;
        padding: 0.5em;
        background: #f8fafc;
        color: #0f766e;
    }}
    .footnote-ref {{
        font-size: 0.8em;
        vertical-align: super;
        font-weight: bold;
    }}
    .footnote-def {{
        font-size: 0.9em;
        color: #475569;
        margin: 0.3em 0 0.3em 1em;
    }}
    code {{
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        background: #f4f4f4;
        padding: 0.15em 0.4em;
        border-radius: 3px;
        font-size: 0.9em;
    }}
    mark {{
        background: #fef08a;
        color: #1a1a1a;
        padding: 0.05em 0.2em;
        border-radius: 2px;
    }}
    pre {{
        background: #f4f4f4;
        padding: 1em;
        border-radius: 5px;
        overflow-x: auto;
        border: 1px solid #ddd;
    }}
    pre code {{
        background: none;
        padding: 0;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }}
    th, td {{
        border: 1px solid #ccc;
        padding: 0.5em 0.75em;
        text-align: left;
    }}
    th {{
        background: #f0f0f0;
        font-weight: bold;
    }}
    tr:nth-child(even) {{
        background: #fafafa;
    }}
    img {{
        max-width: 100%;
        height: auto;
    }}
    hr {{
        border: none;
        border-top: 1px solid #ccc;
        margin: 1.5em 0;
    }}
    ul, ol {{
        margin: 0.5em 0;
        padding-left: 2em;
    }}
    li {{
        margin: 0.2em 0;
    }}
    a {{
        color: #0066cc;
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    a.internal-link {{
        color: #2563eb;
        text-decoration: underline;
        font-weight: 500;
    }}
    .anchor-target {{
        visibility: hidden;
        display: inline-block;
        width: 0;
        height: 0;
        overflow: hidden;
    }}
    """


def _clean_bookmark_name(name: str) -> str:
    """Clean an anchor label for safe use as a Word OpenXML / HTML bookmark identifier."""
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', (name or '').strip().lower())
    clean = re.sub(r'_+', '_', clean).strip('_')
    if not clean or clean[0].isdigit():
        clean = 'bm_' + clean
    return clean[:40]


_INLINE_MATH_RE = re.compile(
    r'(?<!\$)(?<!\\)\$(?!\$)(?!\s)([^$\n]+?)(?<!\s)(?<!\\)\$(?!\$)(?!\d)'
)
_COLOR_SPAN_RE = re.compile(
    r'<span\s+style=["\']\s*color\s*:\s*(#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?)\s*;?\s*["\']\s*>(.*?)</span>',
    re.IGNORECASE | re.DOTALL,
)


def _protect_code_for_preprocess(text: str, placeholders: List[str]) -> str:
    """Replace fenced code blocks and inline code spans with null-delimited placeholders."""
    lines = text.splitlines(keepends=True)
    out_parts: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        m_fence = re.match(r'^\s*(`{3,}|~{3,})', lines[i])
        if m_fence:
            fence = m_fence.group(1)
            fence_char, fence_len = fence[0], len(fence)
            close_re = re.compile(rf'^\s*{re.escape(fence_char)}{{{fence_len},}}\s*$')
            block_lines = [lines[i]]
            i += 1
            while i < n:
                block_lines.append(lines[i])
                if close_re.match(lines[i].rstrip('\r\n')):
                    i += 1
                    break
                i += 1
            block_text = ''.join(block_lines)
            trailing_nl = '\n' if block_text.endswith('\n') else ''
            stored = block_text[:-1] if trailing_nl else block_text
            idx = len(placeholders)
            placeholders.append(stored)
            out_parts.append(f"\x00MDPH{idx}\x00{trailing_nl}")
        else:
            out_parts.append(lines[i])
            i += 1

    text = ''.join(out_parts)

    def _sub_inline_code(m: re.Match) -> str:
        idx = len(placeholders)
        placeholders.append(m.group(0))
        return f"\x00MDPH{idx}\x00"

    text = re.sub(r'(`+)(?!`)([^\n]+?)(?<!`)\1(?!`)', _sub_inline_code, text)
    return text


def _render_footnote_body_html(raw_body: str) -> str:
    """Render footnote body inline Markdown/math/anchors safely without double-escaping or XSS."""
    fn_ph: List[str] = []

    def _stash(html_frag: str) -> str:
        idx = len(fn_ph)
        fn_ph.append(html_frag)
        return f"\x00FNPH{idx}\x00"

    def _sub_anchor(m: re.Match) -> str:
        aid = _clean_bookmark_name(m.group(1).strip())
        return _stash(f'<a id="{aid}" class="anchor-target"></a>')

    def _sub_ref(m: re.Match) -> str:
        raw_lbl = m.group(1).strip()
        aid = _clean_bookmark_name(raw_lbl)
        safe_lbl = _html.escape(raw_lbl, quote=True)
        return _stash(f'<a href="#{aid}" class="internal-link">[@{safe_lbl}]</a>')

    def _sub_math(m: re.Match) -> str:
        rendered = _html.escape(_export_latex_to_unicode(m.group(1)), quote=True)
        return _stash(f'<span class="math-inline">{rendered}</span>')

    def _sub_color(m: re.Match) -> str:
        hex_col = m.group(1)
        inner = _render_footnote_body_html(m.group(2))
        return _stash(f'<span style="color:{hex_col}">{inner}</span>')

    def _sub_mark(m: re.Match) -> str:
        inner = _render_footnote_body_html(m.group(1))
        return _stash(f'<mark>{inner}</mark>')

    work = re.sub(r'\[@(?:anchor|def):\s*([^\]\n]+)\]', _sub_anchor, raw_body, flags=re.IGNORECASE)
    work = re.sub(r'\[@(?!(?:anchor|def):)([^\]\n]+)\]', _sub_ref, work, flags=re.IGNORECASE)
    work = _INLINE_MATH_RE.sub(_sub_math, work)
    work = _COLOR_SPAN_RE.sub(_sub_color, work)
    work = re.sub(r'==([^=\n]+)==', _sub_mark, work)

    escaped = _html.escape(work, quote=True)
    escaped = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', escaped)
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
    escaped = re.sub(r'(?<!\w)__(.+?)__(?!\w)', r'<strong>\1</strong>', escaped)
    escaped = re.sub(r'\*([^\*\n]+?)\*', r'<em>\1</em>', escaped)
    escaped = re.sub(r'(?<!\w)_([^_\n]+?)_(?!\w)', r'<em>\1</em>', escaped)
    escaped = re.sub(r'~~(.+?)~~', r'<del>\1</del>', escaped)

    for idx in range(len(fn_ph) - 1, -1, -1):
        escaped = escaped.replace(f"\x00FNPH{idx}\x00", fn_ph[idx])
    return escaped


def _convert_align_divs_for_html(text: str) -> str:
    """Depth-aware rewrite of <div align="left|center|right">...</div> for md_in_html."""
    tag_re = re.compile(r'<(/?)div\b([^>]*)>', re.IGNORECASE)
    stack: List[Tuple[int, int, Optional[str]]] = []
    edits: List[Tuple[int, int, str]] = []

    for m in tag_re.finditer(text):
        is_close = bool(m.group(1))
        attrs = m.group(2) or ''
        if not is_close:
            m_align = re.match(r'^\s*align\s*=\s*["\'](left|center|right)["\']\s*$', attrs, re.IGNORECASE)
            align_val = m_align.group(1).lower() if m_align else None
            stack.append((m.start(), m.end(), align_val))
        else:
            if stack:
                open_start, open_end, align_val = stack.pop()
                if align_val is not None:
                    edits.append((open_start, open_end, f'<div align="{align_val}" markdown="1">\n\n'))
                    edits.append((m.start(), m.end(), '\n\n</div>'))

    for open_start, open_end, align_val in stack:
        if align_val is not None:
            edits.append((open_start, open_end, f'<div align="{align_val}" markdown="1">'))

    if not edits:
        return text

    edits.sort(key=lambda e: e[0])
    parts: List[str] = []
    pos = 0
    for start, end, repl in edits:
        if start >= pos:
            parts.append(text[pos:start])
            parts.append(repl)
            pos = end
    parts.append(text[pos:])
    return ''.join(parts)


def _preprocess_anchors_for_html(text: str) -> str:
    """Convert [@anchor: point], [@point], footnotes [^id], highlights, color spans, and math $...$ into HTML."""
    placeholders: List[str] = []
    text = _protect_code_for_preprocess(text, placeholders)

    # Process footnote definitions BEFORE global math/anchor substitution so generated HTML is not double-escaped
    def sub_fn_def(m: re.Match) -> str:
        raw_lbl = m.group(1).strip()
        fid = _clean_bookmark_name('fn_' + raw_lbl)
        ref_id = _clean_bookmark_name('fnref_' + raw_lbl)
        label = _html.escape(raw_lbl, quote=True)
        body_html = _render_footnote_body_html(m.group(2).strip())
        fn_html = f'<div id="{fid}" class="footnote-def"><strong>[{label}]</strong> {body_html} <a href="#{ref_id}">↩</a></div>'
        idx = len(placeholders)
        placeholders.append(fn_html)
        return f"\x00MDPH{idx}\x00"

    text = re.sub(r'^\[\^([^\]]+)\]:[ \t]*(.*)$', sub_fn_def, text, flags=re.MULTILINE)

    def sub_anchor(m):
        lbl = m.group(1).strip()
        aid = _clean_bookmark_name(lbl)
        return f'<a id="{aid}" class="anchor-target"></a>'
    text = re.sub(r'\[@(?:anchor|def):\s*([^\]\n]+)\]', sub_anchor, text, flags=re.IGNORECASE)

    def sub_ref(m):
        raw_lbl = m.group(1).strip()
        aid = _clean_bookmark_name(raw_lbl)
        safe_lbl = _html.escape(raw_lbl, quote=True)
        return f'<a href="#{aid}" class="internal-link">[@{safe_lbl}]</a>'
    text = re.sub(r'\[@(?!(?:anchor|def):)([^\]\n]+)\]', sub_ref, text, flags=re.IGNORECASE)

    # Display math $$...$$
    text = re.sub(
        r'\$\$([\s\S]+?)\$\$',
        lambda m: f'<div class="math-block">{_html.escape(_export_latex_to_unicode(m.group(1)), quote=True)}</div>',
        text
    )
    # Inline math $...$ (currency-safe)
    text = _INLINE_MATH_RE.sub(
        lambda m: f'<span class="math-inline">{_html.escape(_export_latex_to_unicode(m.group(1)), quote=True)}</span>',
        text
    )
    # Highlight ==...==
    text = re.sub(r'==([^=\n]+)==', r'<mark>\1</mark>', text)
    # Preserve validated <span style="color:#rrggbb">...</span>
    text = _COLOR_SPAN_RE.sub(
        lambda m: f'<span style="color:{m.group(1)}">{m.group(2)}</span>',
        text,
    )

    # Inline footnote refs [^id]
    def sub_fn_ref(m):
        raw_lbl = m.group(1).strip()
        label = _html.escape(raw_lbl, quote=True)
        fid = _clean_bookmark_name('fn_' + raw_lbl)
        ref_id = _clean_bookmark_name('fnref_' + raw_lbl)
        return f'<sup id="{ref_id}" class="footnote-ref"><a href="#{fid}">[{label}]</a></sup>'
    text = re.sub(r'\[\^([^\]\n]+)\](?!:)', sub_fn_ref, text)

    # Depth-aware <div align="left|center|right">...</div> conversion for md_in_html
    text = _convert_align_divs_for_html(text)

    # Restore placeholders in reverse order
    for idx in range(len(placeholders) - 1, -1, -1):
        text = text.replace(f"\x00MDPH{idx}\x00", placeholders[idx])
    return text


def _add_docx_bookmark(paragraph, name: str, bm_id: int):
    """Add a Word OpenXML bookmark to a paragraph."""
    try:
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        bm_start = OxmlElement('w:bookmarkStart')
        bm_start.set(qn('w:id'), str(bm_id))
        bm_start.set(qn('w:name'), name)
        paragraph._p.append(bm_start)
        bm_end = OxmlElement('w:bookmarkEnd')
        bm_end.set(qn('w:id'), str(bm_id))
        paragraph._p.append(bm_end)
    except Exception:
        pass


def _add_docx_internal_hyperlink(paragraph, text: str, anchor_name: str):
    """Add an internal Word hyperlink referencing a bookmark."""
    try:
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        hyperlink = OxmlElement('w:hyperlink')
        hyperlink.set(qn('w:anchor'), anchor_name)
        hyperlink.set(qn('w:history'), '1')
        r = OxmlElement('w:r')
        rPr = OxmlElement('w:rPr')
        c = OxmlElement('w:color')
        c.set(qn('w:val'), '0066CC')
        rPr.append(c)
        u = OxmlElement('w:u')
        u.set(qn('w:val'), 'single')
        rPr.append(u)
        r.append(rPr)
        t = OxmlElement('w:t')
        t.text = text
        r.append(t)
        hyperlink.append(r)
        paragraph._p.append(hyperlink)
    except Exception:
        run = paragraph.add_run(text)
        run.font.color.rgb = RGBColor(0x00, 0x66, 0xCC)
        run.underline = True


# =============================================================================
# Markdown to HTML Conversion
# =============================================================================

def _markdown_to_html(md_content: str, theme: Optional[Dict] = None,
                       page_settings: Optional[Dict] = None,
                       base_path: Optional[str] = None) -> str:
    """Convert markdown content to a full HTML document"""
    if not MARKDOWN_AVAILABLE:
        raise RuntimeError("Python 'markdown' package is required. Install with: pip install markdown")
    
    extensions = [
        'tables',
        'fenced_code',
        'codehilite',
        'toc',
        'nl2br',
        'sane_lists',
        'smarty',
        'md_in_html',
    ]
    
    extension_configs = {
        'codehilite': {
            'css_class': 'highlight',
            'guess_lang': False,
        },
    }
    
    processed_md = _preprocess_anchors_for_html(md_content)
    html_body = markdown.markdown(processed_md, extensions=extensions,
                                   extension_configs=extension_configs)
    
    # Resolve relative image paths to absolute for PDF rendering
    if base_path:
        html_body = _resolve_image_paths(html_body, base_path)
    
    # Handle base64 images (already work as data URIs)
    css = _build_css(theme, page_settings)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>"""
    
    return html


_SAFE_EXPORT_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}


def _is_safe_export_asset_path(candidate_path: str, base_dir: str) -> bool:
    """Verify *candidate_path* resolves inside *base_dir* and points to an existing allowed image file."""
    if not candidate_path or not base_dir:
        return False
    try:
        real_base = os.path.realpath(base_dir)
        real_cand = os.path.realpath(candidate_path)
        if os.path.commonpath([real_base, real_cand]) != real_base:
            return False
        if not os.path.isfile(real_cand):
            return False
        return os.path.splitext(real_cand)[1].lower() in _SAFE_EXPORT_IMAGE_EXTS
    except Exception:
        return False


def _resolve_image_paths(html: str, base_path: str) -> str:
    """Resolve safe relative image paths in HTML to absolute file:// URIs confined to *base_path*."""
    real_base = os.path.realpath(base_path) if base_path else ""

    def replace_src(match):
        quote_char = match.group(1)
        src = match.group(2)
        if src.startswith(('data:', 'http://', 'https://', 'file://')):
            return match.group(0)

        decoded_src = unquote(src)
        if not decoded_src or os.path.isabs(decoded_src):
            # Reject absolute paths outside base_path; allow only if strictly inside base_path
            if decoded_src and _is_safe_export_asset_path(decoded_src, base_path):
                return f'src={quote_char}{Path(os.path.abspath(decoded_src)).as_uri()}{quote_char}'
            return match.group(0)

        abs_path = os.path.abspath(os.path.join(base_path, decoded_src))
        # Block path traversal escaping base_path before any fallback lookup
        try:
            if real_base and os.path.commonpath([real_base, os.path.realpath(abs_path)]) != real_base:
                return match.group(0)
        except Exception:
            return match.group(0)

        if _is_safe_export_asset_path(abs_path, base_path):
            return f'src={quote_char}{Path(abs_path).as_uri()}{quote_char}'

        # Search safe subdirectories strictly inside base_path (*_assets, assets, images, Images, media)
        if os.path.isdir(base_path):
            fname = os.path.basename(decoded_src)
            try:
                for entry in os.scandir(base_path):
                    if entry.is_dir() and (
                        entry.name.endswith('_assets')
                        or entry.name in ('assets', 'images', 'Images', 'media')
                    ):
                        for rel_cand in (decoded_src, fname):
                            if not rel_cand:
                                continue
                            sub_cand = os.path.abspath(os.path.join(entry.path, rel_cand))
                            if _is_safe_export_asset_path(sub_cand, base_path):
                                return f'src={quote_char}{Path(sub_cand).as_uri()}{quote_char}'
            except Exception:
                pass

        return match.group(0)

    return re.sub(r'src=(["\'])([^"\']*)\1', replace_src, html)


def _make_safe_weasyprint_fetcher(base_path: Optional[str]):
    """Build a restricted URL fetcher for WeasyPrint that blocks network SSRF and out-of-bounds file:// reads."""
    allowed_root = base_path or os.getcwd()

    def _safe_url_fetcher(url: str):
        if not url or not isinstance(url, str):
            raise ValueError("Blocked empty URL in PDF export")
        if url.startswith('data:image/'):
            return weasyprint.default_url_fetcher(url)
        if url.startswith('file://'):
            parsed = urlparse(url)
            raw_path = unquote(parsed.path or '')
            if os.name == 'nt' and raw_path.startswith('/') and len(raw_path) > 2 and raw_path[2] == ':':
                raw_path = raw_path[1:]
            if _is_safe_export_asset_path(raw_path, allowed_root):
                return weasyprint.default_url_fetcher(url)
        raise ValueError(f"Blocked unsafe or external URL fetch in PDF export: {url}")

    return _safe_url_fetcher


# =============================================================================
# Export to PDF
# =============================================================================

def export_to_pdf(md_content: str, output_path: str,
                   theme: Optional[Dict] = None,
                   page_settings: Optional[Dict] = None,
                   base_path: Optional[str] = None) -> bool:
    """
    Export markdown content to PDF.
    
    Args:
        md_content: Markdown text
        output_path: Path for the output PDF file
        theme: Theme dict with font/color settings
        page_settings: Dict with 'page_size' and 'margin'
        base_path: Base directory for resolving relative image paths
    
    Returns:
        True on success
    
    Raises:
        RuntimeError: If required libraries are not installed
    """
    if not WEASYPRINT_AVAILABLE:
        raise RuntimeError("WeasyPrint is required for PDF export. Install with: pip install weasyprint")
    if not MARKDOWN_AVAILABLE:
        raise RuntimeError("Python 'markdown' package is required. Install with: pip install markdown")
    
    html = _markdown_to_html(md_content, theme, page_settings, base_path)
    
    WeasyprintHTML(
        string=html,
        base_url=base_path or '.',
        url_fetcher=_make_safe_weasyprint_fetcher(base_path),
    ).write_pdf(output_path)
    return True


# =============================================================================
# Export to HTML
# =============================================================================

def export_to_html(md_content: str, output_path: str,
                    theme: Optional[Dict] = None,
                    base_path: Optional[str] = None) -> bool:
    """
    Export markdown content to standalone HTML.
    
    Args:
        md_content: Markdown text
        output_path: Path for the output HTML file
        theme: Theme dict
        base_path: Base directory for resolving relative image paths
    
    Returns:
        True on success
    """
    if not MARKDOWN_AVAILABLE:
        raise RuntimeError("Python 'markdown' package is required. Install with: pip install markdown")
    
    html = _markdown_to_html(md_content, theme, base_path=base_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return True


# =============================================================================
# Export to DOCX
# =============================================================================

def _parse_hex_rgb(hex_color: Optional[str]):
    """Validate #rgb or #rrggbb hex string and return RGBColor, or None."""
    if not DOCX_AVAILABLE or not hex_color:
        return None
    s = hex_color.strip()
    m = re.fullmatch(r'#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})', s)
    if not m:
        return None
    h = m.group(1)
    if len(h) == 3:
        h = ''.join(ch * 2 for ch in h)
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _parse_html_img_tag(line_stripped: str) -> Optional[Tuple[str, str]]:
    """Parse a standalone HTML <img ...> tag into (src, alt_spec_with_params)."""
    m = re.match(r'^<img\s+([^>]*?)/?>$', line_stripped, re.IGNORECASE)
    if not m:
        return None
    attrs_str = m.group(1)
    attrs: Dict[str, str] = {}
    for am in re.finditer(r'([a-zA-Z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'>]+))', attrs_str):
        k = am.group(1).lower()
        v = am.group(2) if am.group(2) is not None else (am.group(3) if am.group(3) is not None else (am.group(4) or ''))
        attrs[k] = v.strip()
    src = attrs.get('src', '')
    if not src:
        return None
    alt_base = attrs.get('alt', '')
    params: List[str] = []
    if 'width' in attrs:
        params.append(f"width={attrs['width']}")
    if 'height' in attrs:
        params.append(f"height={attrs['height']}")
    if 'align' in attrs:
        params.append(f"align={attrs['align']}")
    cap = attrs.get('caption') or attrs.get('title')
    if cap:
        params.append(f'caption="{cap}"')
    alt_spec = '|'.join([alt_base] + params) if params else alt_base
    return src, alt_spec


def export_to_docx(md_content: str, output_path: str,
                    theme: Optional[Dict] = None,
                    base_path: Optional[str] = None) -> bool:
    """
    Export markdown content to DOCX (Word document).
    
    Parses the markdown into structural elements and builds a python-docx Document
    with heading styles, paragraph formatting, code blocks, lists, tables, and images.
    
    Args:
        md_content: Markdown text
        output_path: Path for the output DOCX file
        theme: Theme dict
        base_path: Base directory for resolving relative image paths
    
    Returns:
        True on success
    """
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is required for DOCX export. Install with: pip install python-docx")
    
    doc = DocxDocument()
    
    # Configure default style
    style = doc.styles['Normal']
    font_name = theme.get('visual_font', 'Calibri') if theme else 'Calibri'
    font_size = theme.get('visual_font_size', 11) if theme else 11
    style.font.name = font_name
    style.font.size = Pt(font_size)
    style.paragraph_format.space_after = Pt(6)
    raw_ls = theme.get('line_spacing') if theme else None
    try:
        style.paragraph_format.line_spacing = max(1.0, min(3.0, float(raw_ls))) if raw_ls is not None else 1.15
    except (TypeError, ValueError):
        style.paragraph_format.line_spacing = 1.15
    
    # Setup code style
    _ensure_code_style(doc)
    
    lines = md_content.split('\n')
    i = 0
    bm_counter = [0]
    active_align: Optional[str] = None

    def _apply_docx_align(paragraph, align: Optional[str]):
        if align == 'center':
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == 'right':
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        elif align == 'left':
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    
    while i < len(lines):
        line = lines[i]
        
        # Fenced code block (``` or ~~~)
        m_fence = re.match(r'^\s*(`{3,}|~{3,})(.*)$', line)
        if m_fence:
            fence = m_fence.group(1)
            fence_char, fence_len = fence[0], len(fence)
            close_re = re.compile(rf'^\s*{re.escape(fence_char)}{{{fence_len},}}\s*$')
            code_lines = []
            lang = m_fence.group(2).strip()
            i += 1
            while i < len(lines) and not close_re.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # skip closing fence
            _add_code_block(doc, '\n'.join(code_lines), lang)
            continue

        # Single-line <div align="left|center|right">...</div> block
        m_single_div = re.match(
            r'^\s*<div\s+align=["\'](left|center|right)["\']\s*>(.*?)</div>\s*$',
            line, re.IGNORECASE
        )
        if m_single_div:
            blk_align = m_single_div.group(1).lower()
            inner = m_single_div.group(2).strip()
            if inner:
                h_m = re.match(r'^(#{1,6})\s+(.*)', inner)
                if h_m:
                    h = doc.add_heading(level=min(len(h_m.group(1)), 9))
                    bm_counter[0] += 1
                    _add_docx_bookmark(h, _clean_bookmark_name(h_m.group(2).strip()), bm_counter[0])
                    _add_inline_formatting(h, h_m.group(2).strip(), bm_counter)
                    _apply_docx_align(h, blk_align)
                else:
                    p = doc.add_paragraph()
                    _add_inline_formatting(p, inner, bm_counter)
                    _apply_docx_align(p, blk_align)
            i += 1
            continue

        # Opening <div align="left|center|right"> with optional trailing inline text (no closing </div> on same line)
        m_open_div = re.match(
            r'^\s*<div\s+align=["\'](left|center|right)["\']\s*>(.*)$',
            line, re.IGNORECASE
        )
        if m_open_div:
            active_align = m_open_div.group(1).lower()
            trailing = m_open_div.group(2).strip()
            if trailing:
                h_m = re.match(r'^(#{1,6})\s+(.*)', trailing)
                if h_m:
                    h = doc.add_heading(level=min(len(h_m.group(1)), 9))
                    bm_counter[0] += 1
                    _add_docx_bookmark(h, _clean_bookmark_name(h_m.group(2).strip()), bm_counter[0])
                    _add_inline_formatting(h, h_m.group(2).strip(), bm_counter)
                    _apply_docx_align(h, active_align)
                else:
                    p = doc.add_paragraph()
                    _add_inline_formatting(p, trailing, bm_counter)
                    _apply_docx_align(p, active_align)
            i += 1
            continue

        # Closing </div> (matched or unmatched)
        if line.strip().lower() == '</div>':
            active_align = None
            i += 1
            continue

        # Display math block $$...$$
        if line.strip().startswith('$$'):
            stripped_m = line.strip()
            if stripped_m.endswith('$$') and len(stripped_m) > 4:
                expr = stripped_m[2:-2].strip()
                i += 1
            else:
                math_lines = [stripped_m[2:].strip()] if stripped_m[2:].strip() else []
                i += 1
                while i < len(lines) and not lines[i].strip().endswith('$$'):
                    math_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    tail = lines[i].strip()[:-2].strip()
                    if tail:
                        math_lines.append(tail)
                    i += 1
                expr = '\n'.join(math_lines).strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(_export_latex_to_unicode(expr))
            r.font.name = 'Cambria Math'
            r.italic = True
            continue

        # Footnote definition [^id]: text (also handles empty [^id]:)
        fn_def_m = re.match(r'^\[\^([^\]]+)\]:\s*(.*)$', line.strip())
        if fn_def_m:
            fn_id = fn_def_m.group(1).strip()
            fn_body = fn_def_m.group(2).strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.8)
            bm_counter[0] += 1
            _add_docx_bookmark(p, _clean_bookmark_name('fn_' + fn_id), bm_counter[0])
            r_badge = p.add_run(f"[{fn_id}] ")
            r_badge.bold = True
            r_badge.font.size = Pt(max(8, font_size - 1))
            if fn_body:
                _add_inline_formatting(p, fn_body, bm_counter)
            i += 1
            continue
        
        # Heading (allows leading indentation; #hashtag without space falls through to paragraph)
        heading_match = re.match(r'^\s*(#{1,6})\s+(.*)', line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            h = doc.add_heading(level=min(level, 9))
            bm_counter[0] += 1
            bm_name = _clean_bookmark_name(text)
            _add_docx_bookmark(h, bm_name, bm_counter[0])
            _add_inline_formatting(h, text, bm_counter)
            _apply_docx_align(h, active_align)
            i += 1
            continue
        
        # Horizontal rule
        if re.match(r'^[\s]*([-*_])\s*\1\s*\1[\s\-\*_]*$', line):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run('─' * 50)
            run.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
            run.font.size = Pt(8)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue
        
        # Blockquote (supports optional leading indentation)
        if line.lstrip().startswith('>'):
            quote_lines = []
            while i < len(lines) and lines[i].lstrip().startswith('>'):
                quote_lines.append(lines[i].lstrip()[1:].strip())
                i += 1
            p = doc.add_paragraph()
            p.style = doc.styles['Normal']
            bq_pad_px = int((theme.get('blockquote_padding', 16) if theme else 16) or 16)
            p.paragraph_format.left_indent = Cm(max(0.5, bq_pad_px / 12.0))
            pf = p.paragraph_format
            bq_space_pt = int((theme.get('blockquote_spacing', 4) if theme else 4) or 4)
            pf.space_before = Pt(bq_space_pt)
            pf.space_after = Pt(bq_space_pt)
            bq_just = (theme.get('blockquote_justify', 'left') if theme else 'left') or 'left'
            if bq_just == 'center':
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif bq_just == 'right':
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            bq_style_str = (theme.get('blockquote_style', 'italic') if theme else 'italic') or 'italic'
            bq_font_fam = (theme.get('blockquote_font') if theme else None) or font_name
            bq_sz_val = int((theme.get('blockquote_size') if theme else 0) or 0)
            bq_fg_hex = (theme.get('blockquote_fg') if theme else None) or '#555555'
            bq_color = _parse_hex_rgb(bq_fg_hex) or RGBColor(0x55, 0x55, 0x55)
            _add_inline_formatting(
                p,
                ' '.join(quote_lines),
                bm_counter,
                bold=('bold' in bq_style_str),
                italic=('italic' in bq_style_str),
                color_rgb=bq_color,
                font_name=bq_font_fam,
                font_size=bq_sz_val if bq_sz_val > 0 else None,
            )
            continue
        
        # Unordered list
        if re.match(r'^[\s]*[-*+]\s+', line):
            while i < len(lines) and re.match(r'^[\s]*[-*+]\s+', lines[i]):
                text = re.sub(r'^[\s]*[-*+]\s+', '', lines[i])
                p = doc.add_paragraph(style='List Bullet')
                _add_inline_formatting(p, text, bm_counter)
                _apply_docx_align(p, active_align)
                i += 1
            continue
        
        # Ordered list
        if re.match(r'^[\s]*\d+\.\s+', line):
            while i < len(lines) and re.match(r'^[\s]*\d+\.\s+', lines[i]):
                text = re.sub(r'^[\s]*\d+\.\s+', '', lines[i])
                p = doc.add_paragraph(style='List Number')
                _add_inline_formatting(p, text, bm_counter)
                _apply_docx_align(p, active_align)
                i += 1
            continue
        
        # Table
        if '|' in line and i + 1 < len(lines) and re.match(r'^[\s]*\|?[\s:]*[-]+', lines[i + 1]):
            table_lines = []
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            _add_table(doc, table_lines, bm_counter)
            continue
        
        # Standalone Markdown or HTML Image
        stripped_line = line.strip()
        img_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)$', stripped_line)
        if img_match:
            alt_text = img_match.group(1)
            img_path = img_match.group(2)
            _add_image(doc, img_path, alt_text, base_path)
            i += 1
            continue

        html_img = _parse_html_img_tag(stripped_line)
        if html_img:
            _add_image(doc, html_img[0], html_img[1], base_path)
            i += 1
            continue
        
        # Blank line
        if not stripped_line:
            i += 1
            continue
        
        # Regular paragraph — collect contiguous non-blank, non-special lines
        para_lines = []
        while i < len(lines) and lines[i].strip() and not _is_special_line(lines[i]):
            para_lines.append(lines[i])
            i += 1

        # Safety net: force-consume unconsumed non-blank line to prevent infinite loops
        if not para_lines and i < len(lines) and lines[i].strip():
            para_lines.append(lines[i])
            i += 1
        
        if para_lines:
            p = doc.add_paragraph()
            _add_inline_formatting(p, ' '.join(para_lines), bm_counter)
            _apply_docx_align(p, active_align)
    
    doc.save(output_path)
    return True


def _is_special_line(line: str) -> bool:
    """Check if a line is a special markdown element"""
    stripped = line.strip()
    if re.match(r'^\s*#{1,6}\s+', line):
        return True
    if stripped.startswith(('```', '~~~')):
        return True
    if stripped.startswith('$$'):
        return True
    if stripped.startswith('>'):
        return True
    if re.match(r'^<div\s+align=["\'](?:left|center|right)["\']\s*>', stripped, re.IGNORECASE):
        return True
    if stripped.lower() == '</div>':
        return True
    if re.match(r'^\[\^([^\]]+)\]:', stripped):
        return True
    if re.match(r'^[\s]*[-*+]\s+', line):
        return True
    if re.match(r'^[\s]*\d+\.\s+', line):
        return True
    if re.match(r'^[\s]*([-*_])\s*\1\s*\1[\s\-\*_]*$', line):
        return True
    if re.match(r'^!\[([^\]]*)\]\(([^)]+)\)$', stripped):
        return True
    if _parse_html_img_tag(stripped) is not None:
        return True
    if '|' in line and re.match(r'^\s*\|', line):
        return True
    return False


def _ensure_code_style(doc):
    """Ensure a 'Code' paragraph style exists in the document"""
    try:
        doc.styles['Code']
    except KeyError:
        style = doc.styles.add_style('Code', WD_STYLE_TYPE.PARAGRAPH)
        style.font.name = 'Consolas'
        style.font.size = Pt(9)
        style.font.color.rgb = RGBColor(0x1a, 0x1a, 0x1a)
        style.paragraph_format.space_before = Pt(4)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.left_indent = Cm(0.5)


def _add_code_block(doc, code: str, language: str = ""):
    """Add a code block to the document"""
    p = doc.add_paragraph(style='Code')
    run = p.add_run(code)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    # Add background shading via XML (light gray)
    from docx.oxml.ns import qn
    shading = p._element.get_or_add_pPr()
    shd = shading.makeelement(qn('w:shd'), {
        qn('w:val'): 'clear',
        qn('w:color'): 'auto',
        qn('w:fill'): 'F4F4F4'
    })
    shading.append(shd)


_DOCX_INLINE_RE = re.compile(
    r'(?P<code2>``[^`\n]+``)'
    r'|(?P<code1>`[^`\n]+`)'
    r'|(?P<anchor>\[@(?:anchor|def):\s*(?P<anchor_lbl>[^\]\n]+)\])'
    r'|(?P<ref>\[@(?!(?:anchor|def):)(?P<ref_lbl>[^\]\n]+)\])'
    r'|(?P<fn>\[\^(?P<fn_lbl>[^\]\n]+)\])'
    r'|(?P<link>\[(?P<link_txt>[^\]]+)\]\((?P<link_url>[^)]+)\))'
    r'|(?P<cspan><span\s+style=["\']\s*color\s*:\s*(?P<cspan_col>#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?)\s*;?\s*["\']\s*>(?P<cspan_body>.*?)</span>)'
    r'|(?P<mark_html><mark>(?P<mark_html_body>.*?)</mark>)'
    r'|(?P<mark_md>==(?P<mark_md_body>[^=\n]+)==)'
    r'|(?P<strike>~~(?P<strike_body>.+?)~~)'
    r'|(?P<bi_star>\*\*\*(?P<bi_star_body>.+?)\*\*\*)'
    r'|(?P<bi_under>(?<!\w)___(?P<bi_under_body>.+?)___(?!\w))'
    r'|(?P<b_star>\*\*(?P<b_star_body>.+?)\*\*)'
    r'|(?P<b_under>(?<!\w)__(?P<b_under_body>.+?)__(?!\w))'
    r'|(?P<i_star>\*(?P<i_star_body>[^\*\n]+?)\*)'
    r'|(?P<i_under>(?<!\w)_(?P<i_under_body>[^_\n]+?)_(?!\w))'
    r'|(?P<math>(?<!\$)(?<!\\)\$(?!\$)(?!\s)(?P<math_body>[^$\n]+?)(?<!\s)(?<!\\)\$(?!\$)(?!\d))',
    re.IGNORECASE,
)


def _apply_run_style(
    run,
    bold: bool = False,
    italic: bool = False,
    strike: bool = False,
    highlight: bool = False,
    color_rgb=None,
    font_name: Optional[str] = None,
    font_size: Optional[int] = None,
):
    """Apply inherited inline run styles to a python-docx Run."""
    if bold:
        run.bold = True
    if italic:
        run.italic = True
    if strike:
        run.font.strike = True
    if highlight:
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    if color_rgb is not None:
        run.font.color.rgb = color_rgb
    if font_name:
        run.font.name = font_name
    if font_size and font_size > 0:
        run.font.size = Pt(font_size)


def _add_inline_formatting(
    paragraph,
    text: str,
    bm_counter: Optional[List[int]] = None,
    bold: bool = False,
    italic: bool = False,
    strike: bool = False,
    highlight: bool = False,
    color_rgb=None,
    font_name: Optional[str] = None,
    font_size: Optional[int] = None,
):
    """Recursively parse inline markdown/HTML formatting and add styled runs to a paragraph."""
    if not text:
        return

    style_kw = dict(
        bold=bold,
        italic=italic,
        strike=strike,
        highlight=highlight,
        color_rgb=color_rgb,
        font_name=font_name,
        font_size=font_size,
    )

    last_end = 0
    for match in _DOCX_INLINE_RE.finditer(text):
        if match.start() > last_end:
            run = paragraph.add_run(text[last_end:match.start()])
            _apply_run_style(run, **style_kw)

        kind = match.lastgroup
        if kind in ('code2', 'code1'):
            raw = match.group(kind)
            inner = raw[2:-2] if kind == 'code2' else raw[1:-1]
            run = paragraph.add_run(inner)
            _apply_run_style(
                run,
                bold=bold,
                italic=italic,
                strike=strike,
                highlight=highlight,
                color_rgb=color_rgb,
                font_name='Consolas',
                font_size=9,
            )
        elif kind == 'anchor':
            anchor_raw = match.group('anchor_lbl').strip()
            bm_name = _clean_bookmark_name(anchor_raw)
            if bm_counter is not None:
                bm_counter[0] += 1
                bm_id = bm_counter[0]
            else:
                bm_id = 1
            _add_docx_bookmark(paragraph, bm_name, bm_id)
        elif kind == 'ref':
            ref_raw = match.group('ref_lbl').strip()
            bm_name = _clean_bookmark_name(ref_raw)
            _add_docx_internal_hyperlink(paragraph, f"[@{ref_raw}]", bm_name)
        elif kind == 'fn':
            fn_id = match.group('fn_lbl').strip()
            bm_name = _clean_bookmark_name('fn_' + fn_id)
            _add_docx_internal_hyperlink(paragraph, f"[{fn_id}]", bm_name)
        elif kind == 'link':
            link_text = match.group('link_txt')
            link_url = match.group('link_url')
            if link_url.startswith(('#', '@')):
                bm_name = _clean_bookmark_name(link_url.lstrip('#@'))
                _add_docx_internal_hyperlink(paragraph, link_text, bm_name)
            else:
                run = paragraph.add_run(f"{link_text} ({link_url})")
                _apply_run_style(
                    run,
                    bold=bold,
                    italic=italic,
                    strike=strike,
                    highlight=highlight,
                    color_rgb=color_rgb or RGBColor(0x00, 0x66, 0xCC),
                    font_name=font_name,
                    font_size=font_size,
                )
                run.underline = True
        elif kind == 'cspan':
            parsed_rgb = _parse_hex_rgb(match.group('cspan_col'))
            _add_inline_formatting(
                paragraph,
                match.group('cspan_body'),
                bm_counter,
                bold=bold,
                italic=italic,
                strike=strike,
                highlight=highlight,
                color_rgb=parsed_rgb if parsed_rgb is not None else color_rgb,
                font_name=font_name,
                font_size=font_size,
            )
        elif kind in ('mark_html', 'mark_md'):
            body_key = 'mark_html_body' if kind == 'mark_html' else 'mark_md_body'
            _add_inline_formatting(
                paragraph,
                match.group(body_key),
                bm_counter,
                bold=bold,
                italic=italic,
                strike=strike,
                highlight=True,
                color_rgb=color_rgb,
                font_name=font_name,
                font_size=font_size,
            )
        elif kind == 'strike':
            _add_inline_formatting(
                paragraph,
                match.group('strike_body'),
                bm_counter,
                bold=bold,
                italic=italic,
                strike=True,
                highlight=highlight,
                color_rgb=color_rgb,
                font_name=font_name,
                font_size=font_size,
            )
        elif kind in ('bi_star', 'bi_under'):
            body_key = f"{kind}_body"
            _add_inline_formatting(
                paragraph,
                match.group(body_key),
                bm_counter,
                bold=True,
                italic=True,
                strike=strike,
                highlight=highlight,
                color_rgb=color_rgb,
                font_name=font_name,
                font_size=font_size,
            )
        elif kind in ('b_star', 'b_under'):
            body_key = f"{kind}_body"
            _add_inline_formatting(
                paragraph,
                match.group(body_key),
                bm_counter,
                bold=True,
                italic=italic,
                strike=strike,
                highlight=highlight,
                color_rgb=color_rgb,
                font_name=font_name,
                font_size=font_size,
            )
        elif kind in ('i_star', 'i_under'):
            body_key = f"{kind}_body"
            _add_inline_formatting(
                paragraph,
                match.group(body_key),
                bm_counter,
                bold=bold,
                italic=True,
                strike=strike,
                highlight=highlight,
                color_rgb=color_rgb,
                font_name=font_name,
                font_size=font_size,
            )
        elif kind == 'math':
            math_raw = match.group('math_body').strip()
            run = paragraph.add_run(_export_latex_to_unicode(math_raw))
            _apply_run_style(
                run,
                bold=bold,
                italic=True,
                strike=strike,
                highlight=highlight,
                color_rgb=color_rgb,
                font_name='Cambria Math',
                font_size=font_size,
            )

        last_end = match.end()

    if last_end < len(text):
        run = paragraph.add_run(text[last_end:])
        _apply_run_style(run, **style_kw)


def _parse_docx_image_params(alt_text: str):
    """Parse extended image parameters from alt text for DOCX formatting"""
    res = {'alt': alt_text, 'width': None, 'height': None, 'align': 'center', 'caption': None}
    if not alt_text or '|' not in alt_text:
        return res
    parts = alt_text.split('|')
    res['alt'] = parts[0].strip()
    param_str = '|'.join(parts[1:]).strip()

    kv_pattern = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,;|]+))')
    kv_matches = list(kv_pattern.finditer(param_str))
    if kv_matches:
        for m in kv_matches:
            k = m.group(1).lower()
            v = (m.group(2) or m.group(3) or m.group(4) or '').strip()
            if k == 'width':
                dim = re.match(r'(\d+)', v)
                if dim:
                    res['width'] = int(dim.group(1))
            elif k == 'height':
                dim = re.match(r'(\d+)', v)
                if dim:
                    res['height'] = int(dim.group(1))
            elif k == 'align':
                if v.lower() in ('left', 'right', 'center'):
                    res['align'] = v.lower()
            elif k == 'caption':
                res['caption'] = v
    else:
        tokens = [t.strip() for t in re.split(r'[,|]', param_str) if t.strip()]
        for token in tokens:
            m_dim = re.match(r'^(\d+)\s*[xX×]\s*(\d+)$', token)
            if m_dim:
                res['width'] = int(m_dim.group(1))
                res['height'] = int(m_dim.group(2))
                continue
            m_w = re.match(r'^(\d+)(?:px)?$', token)
            if m_w:
                res['width'] = int(m_w.group(1))
                continue
            if token.lower() in ('left', 'right', 'center'):
                res['align'] = token.lower()
                continue
            if not res['caption']:
                res['caption'] = token
    return res


def _parse_md_table_row(line: str) -> List[str]:
    """Parse a Markdown table row, preserving empty cells and unescaping escaped pipes."""
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|') and not s.endswith('\\|'):
        s = s[:-1]
    raw_cells = re.split(r'(?<!\\)\|', s)
    return [c.strip().replace(r'\|', '|') for c in raw_cells]


def _add_table(doc, table_lines: list, bm_counter: Optional[List[int]] = None):
    """Add a styled markdown table to the document"""
    if len(table_lines) < 2:
        return
    if bm_counter is None:
        bm_counter = [0]
    
    # Parse header
    headers = _parse_md_table_row(table_lines[0])
    
    # Parse data rows (skip separator line)
    rows = []
    for line in table_lines[2:]:
        if '|' in line:
            rows.append(_parse_md_table_row(line))
    
    if not headers:
        return
    
    num_cols = max([len(headers)] + [len(r) for r in rows])
    if num_cols <= 0:
        return
    headers = headers + [''] * (num_cols - len(headers))
    normalized_rows = [r + [''] * (num_cols - len(r)) for r in rows]

    table = doc.add_table(rows=1 + len(normalized_rows), cols=num_cols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Header row
    hdr_row = table.rows[0]
    hdr_row._tr.get_or_add_trPr().append(
        hdr_row._tr.makeelement(qn('w:tblHeader'), {})
    )
    for j, header in enumerate(headers):
        cell = hdr_row.cells[j]
        p = cell.paragraphs[0]
        p.text = ''
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        _add_inline_formatting(
            p, header, bm_counter,
            bold=True,
            color_rgb=RGBColor(0x0F, 0x17, 0x2A),
        )
        # Add subtle gray shading to header
        shd = cell._element.get_or_add_tcPr().makeelement(
            qn('w:shd'), {
                qn('w:val'): 'clear',
                qn('w:color'): 'auto',
                qn('w:fill'): 'F1F5F9'
            }
        )
        cell._element.get_or_add_tcPr().append(shd)
    
    # Data rows
    for i, row in enumerate(normalized_rows):
        data_row = table.rows[i + 1]
        is_alt = (i % 2 == 1)
        for j, cell_text in enumerate(row):
            cell = data_row.cells[j]
            p = cell.paragraphs[0]
            p.text = ''
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            _add_inline_formatting(p, cell_text, bm_counter)
            if is_alt:
                shd = cell._element.get_or_add_tcPr().makeelement(
                    qn('w:shd'), {
                        qn('w:val'): 'clear',
                        qn('w:color'): 'auto',
                        qn('w:fill'): 'F8FAFC'
                    }
                )
                cell._element.get_or_add_tcPr().append(shd)
    
    # Add spacing after table
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(4)
    p_after.paragraph_format.space_after = Pt(6)


def _embed_docx_image_file(
    doc,
    image_file_path: str,
    clean_alt: str = "",
    caption: Optional[str] = None,
    width_px: Optional[int] = None,
    align: str = 'center',
):
    """Embed an image file into a DOCX paragraph with alignment, width, and optional caption."""
    if width_px:
        target_w = Inches(min(6.2, max(1.0, width_px / 96.0)))
    else:
        target_w = Inches(5.2)

    p = doc.add_paragraph()
    if align == 'left':
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    elif align == 'right':
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(image_file_path, width=target_w)

    display_caption = caption or clean_alt
    if display_caption:
        cp = doc.add_paragraph(display_caption)
        cp.alignment = p.alignment
        cp.paragraph_format.space_before = Pt(2)
        cp.paragraph_format.space_after = Pt(10)
        for crun in cp.runs:
            crun.font.italic = True
            crun.font.size = Pt(9.5)
            crun.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)


def _add_image(doc, img_path: str, alt_text: str, base_path: Optional[str] = None):
    """Add an image to the document with extended parameter support and fallback path resolution"""
    params = _parse_docx_image_params(alt_text)
    clean_alt = params['alt']
    caption = params['caption']
    width_px = params['width']
    align = params['align']

    # Handle base64 data URIs
    if img_path.startswith('data:'):
        _add_base64_image(doc, img_path, alt_text, width_px=width_px, align=align, caption=caption)
        return
    
    # Fallback search for relative image paths
    decoded_path = unquote(img_path)
    resolved_path = None
    if os.path.isabs(decoded_path) and os.path.exists(decoded_path):
        resolved_path = decoded_path
    elif base_path:
        cand1 = os.path.normpath(os.path.join(base_path, decoded_path))
        if os.path.exists(cand1):
            resolved_path = cand1
        else:
            cand2 = os.path.normpath(os.path.join(base_path, os.path.basename(decoded_path)))
            if os.path.exists(cand2):
                resolved_path = cand2
            elif os.path.isdir(base_path):
                fname = os.path.basename(decoded_path)
                try:
                    for entry in os.scandir(base_path):
                        if entry.is_dir() and (entry.name.endswith('_assets') or entry.name in ('assets', 'images', 'Images', 'media')):
                            sub_cand = os.path.join(entry.path, fname)
                            if os.path.exists(sub_cand):
                                resolved_path = sub_cand
                                break
                except Exception:
                    pass

    if resolved_path and os.path.exists(resolved_path):
        try:
            _embed_docx_image_file(doc, resolved_path, clean_alt, caption, width_px, align)
        except Exception as e:
            doc.add_paragraph(f"[Image: {clean_alt or img_path}] (Failed to embed: {e})")
    else:
        doc.add_paragraph(f"[Image: {clean_alt or img_path}] (File not found)")


def _add_base64_image(
    doc,
    data_uri: str,
    alt_text: str = "",
    width_px: Optional[int] = None,
    align: Optional[str] = None,
    caption: Optional[str] = None,
):
    """Add a base64-encoded image to the document with width/align/caption parameter support."""
    params = _parse_docx_image_params(alt_text)
    clean_alt = params['alt']
    eff_width = width_px if width_px is not None else params['width']
    eff_align = align if align is not None else params['align']
    eff_caption = caption if caption is not None else params['caption']
    try:
        # Parse data URI: data:image/png;base64,xxxxx
        header, data = data_uri.split(',', 1)
        img_bytes = base64.b64decode(data)
        
        # Write to temp file
        suffix = '.png'
        if 'jpeg' in header or 'jpg' in header:
            suffix = '.jpg'
        elif 'gif' in header:
            suffix = '.gif'
        elif 'webp' in header:
            suffix = '.webp'
        elif 'bmp' in header:
            suffix = '.bmp'
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        
        try:
            _embed_docx_image_file(doc, tmp_path, clean_alt, eff_caption, eff_width, eff_align)
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        doc.add_paragraph(f"[Image: {clean_alt or alt_text}] (Failed to decode base64: {e})")
