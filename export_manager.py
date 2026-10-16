"""
Export Manager Module for MarkItDown Notepad

Provides export functionality to PDF and DOCX formats.
Uses 'markdown' for HTML generation, 'weasyprint' for PDF rendering,
and 'python-docx' for Word document creation.
"""

import os
import re
import base64
import tempfile
from pathlib import Path
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
    from weasyprint import HTML as WeasyprintHTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

# Try to import python-docx for DOCX
try:
    import docx
    from docx import Document as DocxDocument
    from docx.shared import Inches, Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
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
        s = s.replace(cmd, symbols[cmd])
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
        line-height: 1.6;
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


def _preprocess_anchors_for_html(text: str) -> str:
    """Convert [@anchor: point], [@point], footnotes [^id], and math $...$ into HTML for PDF/HTML export."""
    def sub_anchor(m):
        lbl = m.group(1).strip()
        aid = _clean_bookmark_name(lbl)
        return f'<a id="{aid}" class="anchor-target"></a>'
    text = re.sub(r'\[@(?:anchor|def):\s*([^\]\n]+)\]', sub_anchor, text, flags=re.IGNORECASE)

    def sub_ref(m):
        lbl = m.group(1).strip()
        aid = _clean_bookmark_name(lbl)
        return f'<a href="#{aid}" class="internal-link">[@{lbl}]</a>'
    text = re.sub(r'\[@(?!(?:anchor|def):)([^\]\n]+)\]', sub_ref, text, flags=re.IGNORECASE)

    # Display math $$...$$
    text = re.sub(
        r'\$\$([\s\S]+?)\$\$',
        lambda m: f'<div class="math-block">{_export_latex_to_unicode(m.group(1))}</div>',
        text
    )
    # Inline math $...$
    text = re.sub(
        r'(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)',
        lambda m: f'<span class="math-inline">{_export_latex_to_unicode(m.group(1))}</span>',
        text
    )
    # Footnote definitions [^id]: text
    def sub_fn_def(m):
        fid = _clean_bookmark_name('fn_' + m.group(1).strip())
        ref_id = _clean_bookmark_name('fnref_' + m.group(1).strip())
        label = m.group(1).strip()
        body = m.group(2).strip()
        return f'<div id="{fid}" class="footnote-def"><strong>[{label}]</strong> {body} <a href="#{ref_id}">↩</a></div>'
    text = re.sub(r'^\[\^([^\]]+)\]:\s*(.+)$', sub_fn_def, text, flags=re.MULTILINE)

    # Inline footnote refs [^id]
    def sub_fn_ref(m):
        label = m.group(1).strip()
        fid = _clean_bookmark_name('fn_' + label)
        ref_id = _clean_bookmark_name('fnref_' + label)
        return f'<sup id="{ref_id}" class="footnote-ref"><a href="#{fid}">[{label}]</a></sup>'
    text = re.sub(r'\[\^([^\]\n]+)\](?!:)', sub_fn_ref, text)
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


def _resolve_image_paths(html: str, base_path: str) -> str:
    """Resolve relative image paths in HTML to absolute file:// URIs"""
    import re
    
    def replace_src(match):
        src = match.group(1)
        # Skip data URIs and absolute URLs
        if src.startswith(('data:', 'http://', 'https://', 'file://')):
            return match.group(0)
        # Resolve relative path
        abs_path = os.path.abspath(os.path.join(base_path, src))
        if os.path.exists(abs_path):
            return f'src="file://{abs_path}"'
        return match.group(0)
    
    return re.sub(r'src="([^"]*)"', replace_src, html)


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
    
    WeasyprintHTML(string=html, base_url=base_path or '.').write_pdf(output_path)
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
    style.paragraph_format.line_spacing = 1.15
    
    # Setup code style
    _ensure_code_style(doc)
    
    lines = md_content.split('\n')
    i = 0
    bm_counter = [0]
    
    while i < len(lines):
        line = lines[i]
        
        # Fenced code block
        if line.strip().startswith('```'):
            code_lines = []
            lang = line.strip()[3:].strip()
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            _add_code_block(doc, '\n'.join(code_lines), lang)
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

        # Footnote definition [^id]: text
        fn_def_m = re.match(r'^\[\^([^\]]+)\]:\s*(.+)$', line.strip())
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
            _add_inline_formatting(p, fn_body, bm_counter)
            i += 1
            continue
        
        # Heading
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            h = doc.add_heading(level=min(level, 9))
            bm_counter[0] += 1
            bm_name = _clean_bookmark_name(text)
            _add_docx_bookmark(h, bm_name, bm_counter[0])
            _add_inline_formatting(h, text, bm_counter)
            i += 1
            continue
        
        # Horizontal rule
        if re.match(r'^[\s]*([-*_])\s*\1\s*\1[\s\-\*_]*$', line):
            # Add a thin paragraph as a visual separator
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run('─' * 50)
            run.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
            run.font.size = Pt(8)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue
        
        # Blockquote
        if line.startswith('>'):
            quote_lines = []
            while i < len(lines) and lines[i].startswith('>'):
                quote_lines.append(lines[i].lstrip('>').strip())
                i += 1
            p = doc.add_paragraph(' '.join(quote_lines))
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
            for run in p.runs:
                run.font.name = bq_font_fam
                if bq_sz_val > 0:
                    run.font.size = Pt(bq_sz_val)
                run.font.italic = ('italic' in bq_style_str)
                run.font.bold = ('bold' in bq_style_str)
                run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            continue
        
        # Unordered list
        if re.match(r'^[\s]*[-*+]\s+', line):
            while i < len(lines) and re.match(r'^[\s]*[-*+]\s+', lines[i]):
                text = re.sub(r'^[\s]*[-*+]\s+', '', lines[i])
                p = doc.add_paragraph(style='List Bullet')
                _add_inline_formatting(p, text, bm_counter)
                i += 1
            continue
        
        # Ordered list
        if re.match(r'^[\s]*\d+\.\s+', line):
            while i < len(lines) and re.match(r'^[\s]*\d+\.\s+', lines[i]):
                text = re.sub(r'^[\s]*\d+\.\s+', '', lines[i])
                p = doc.add_paragraph(style='List Number')
                _add_inline_formatting(p, text, bm_counter)
                i += 1
            continue
        
        # Table
        if '|' in line and i + 1 < len(lines) and re.match(r'^[\s]*\|[\s:]*[-]+', lines[i + 1]):
            table_lines = []
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            _add_table(doc, table_lines)
            continue
        
        # Image
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line.strip())
        if img_match:
            alt_text = img_match.group(1)
            img_path = img_match.group(2)
            _add_image(doc, img_path, alt_text, base_path)
            i += 1
            continue
        
        # Blank line
        if not line.strip():
            i += 1
            continue
        
        # Regular paragraph — collect contiguous non-blank, non-special lines
        para_lines = []
        while i < len(lines) and lines[i].strip() and not _is_special_line(lines[i]):
            para_lines.append(lines[i])
            i += 1
        
        if para_lines:
            p = doc.add_paragraph()
            _add_inline_formatting(p, ' '.join(para_lines), bm_counter)
    
    doc.save(output_path)
    return True


def _is_special_line(line: str) -> bool:
    """Check if a line is a special markdown element"""
    stripped = line.strip()
    if stripped.startswith('#'):
        return True
    if stripped.startswith('```'):
        return True
    if stripped.startswith('$$'):
        return True
    if stripped.startswith('>'):
        return True
    if re.match(r'^\[\^([^\]]+)\]:', stripped):
        return True
    if re.match(r'^[\s]*[-*+]\s+', line):
        return True
    if re.match(r'^[\s]*\d+\.\s+', line):
        return True
    if re.match(r'^[\s]*([-*_])\s*\1\s*\1[\s\-\*_]*$', line):
        return True
    if re.match(r'^!\[', stripped):
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


def _add_inline_formatting(paragraph, text: str, bm_counter: Optional[List[int]] = None):
    """Parse inline markdown formatting and add runs to a paragraph"""
    pattern = re.compile(
        r'(\*\*(.+?)\*\*)'       # bold (1, 2)
        r'|(\*(.+?)\*)'            # italic (3, 4)
        r'|(`(.+?)`)'              # inline code (5, 6)
        r'|(\[([^\]]+)\]\(([^)]+)\))'  # link (7, 8, 9)
        r'|(~~(.+?)~~)'            # strikethrough (10, 11)
        r'|(\[@(?:anchor|def):\s*([^\]\n]+)\])'  # anchor definition (12, 13)
        r'|(\[@(?!(?:anchor|def):)([^\]\n]+)\])'  # anchor reference (14, 15)
        r'|((?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$))'  # inline math (16, 17)
        r'|(\[\^([^\]\n]+)\])'     # footnote reference (18, 19)
    )
    
    last_end = 0
    for match in pattern.finditer(text):
        # Add text before match
        if match.start() > last_end:
            paragraph.add_run(text[last_end:match.start()])
        
        if match.group(2):  # bold
            run = paragraph.add_run(match.group(2))
            run.bold = True
        elif match.group(4):  # italic
            run = paragraph.add_run(match.group(4))
            run.italic = True
        elif match.group(6):  # code
            run = paragraph.add_run(match.group(6))
            run.font.name = 'Consolas'
            run.font.size = Pt(9)
        elif match.group(8):  # link
            link_text = match.group(8)
            link_url = match.group(9)
            if link_url.startswith(('#', '@')):
                bm_name = _clean_bookmark_name(link_url.lstrip('#@'))
                _add_docx_internal_hyperlink(paragraph, link_text, bm_name)
            else:
                run = paragraph.add_run(f"{link_text} ({link_url})")
                run.font.color.rgb = RGBColor(0x00, 0x66, 0xCC)
                run.underline = True
        elif match.group(11):  # strikethrough
            run = paragraph.add_run(match.group(11))
            run.font.strike = True
        elif match.group(12):  # [@anchor: point]
            anchor_raw = match.group(13).strip()
            bm_name = _clean_bookmark_name(anchor_raw)
            if bm_counter is not None:
                bm_counter[0] += 1
                bm_id = bm_counter[0]
            else:
                bm_id = 1
            _add_docx_bookmark(paragraph, bm_name, bm_id)
        elif match.group(14):  # [@point]
            ref_raw = match.group(15).strip()
            bm_name = _clean_bookmark_name(ref_raw)
            _add_docx_internal_hyperlink(paragraph, f"[@{ref_raw}]", bm_name)
        elif match.group(16):  # $math$
            math_raw = match.group(17).strip()
            run = paragraph.add_run(_export_latex_to_unicode(math_raw))
            run.font.name = 'Cambria Math'
            run.italic = True
        elif match.group(18):  # [^footnote]
            fn_id = match.group(19).strip()
            bm_name = _clean_bookmark_name('fn_' + fn_id)
            _add_docx_internal_hyperlink(paragraph, f"[{fn_id}]", bm_name)
        
        last_end = match.end()
    
    # Add remaining text
    if last_end < len(text):
        paragraph.add_run(text[last_end:])


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


def _add_table(doc, table_lines: list):
    """Add a styled markdown table to the document"""
    if len(table_lines) < 2:
        return
    
    # Parse header
    headers = [c.strip() for c in table_lines[0].split('|') if c.strip()]
    
    # Parse data rows (skip separator line)
    rows = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if cells:
            rows.append(cells)
    
    if not headers:
        return
    
    num_cols = len(headers)
    table = doc.add_table(rows=1 + len(rows), cols=num_cols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Header row
    hdr_row = table.rows[0]
    hdr_row._tr.get_or_add_trPr().append(
        hdr_row._tr.makeelement(qn('w:tblHeader'), {})
    )
    for j, header in enumerate(headers):
        if j < num_cols:
            cell = hdr_row.cells[j]
            cell.text = header
            # Add subtle gray shading to header
            shd = cell._element.get_or_add_tcPr().makeelement(
                qn('w:shd'), {
                    qn('w:val'): 'clear',
                    qn('w:color'): 'auto',
                    qn('w:fill'): 'F1F5F9'
                }
            )
            cell._element.get_or_add_tcPr().append(shd)
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(4)
                for run in p.runs:
                    run.bold = True
                    run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    # Data rows
    for i, row in enumerate(rows):
        data_row = table.rows[i + 1]
        is_alt = (i % 2 == 1)
        for j, cell_text in enumerate(row):
            if j < num_cols:
                cell = data_row.cells[j]
                cell.text = cell_text
                if is_alt:
                    shd = cell._element.get_or_add_tcPr().makeelement(
                        qn('w:shd'), {
                            qn('w:val'): 'clear',
                            qn('w:color'): 'auto',
                            qn('w:fill'): 'F8FAFC'
                        }
                    )
                    cell._element.get_or_add_tcPr().append(shd)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.space_after = Pt(3)
    
    # Add spacing after table
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(4)
    p_after.paragraph_format.space_after = Pt(6)


def _add_image(doc, img_path: str, alt_text: str, base_path: Optional[str] = None):
    """Add an image to the document with extended parameter support and fallback path resolution"""
    params = _parse_docx_image_params(alt_text)
    clean_alt = params['alt']
    caption = params['caption']
    width_px = params['width']
    align = params['align']

    # Handle base64 data URIs
    if img_path.startswith('data:'):
        _add_base64_image(doc, img_path, clean_alt)
        return
    
    # Fallback search for relative image paths
    resolved_path = None
    if os.path.isabs(img_path) and os.path.exists(img_path):
        resolved_path = img_path
    elif base_path:
        cand1 = os.path.normpath(os.path.join(base_path, img_path))
        if os.path.exists(cand1):
            resolved_path = cand1
        else:
            cand2 = os.path.normpath(os.path.join(base_path, os.path.basename(img_path)))
            if os.path.exists(cand2):
                resolved_path = cand2
            elif os.path.isdir(base_path):
                fname = os.path.basename(img_path)
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
            run.add_picture(resolved_path, width=target_w)

            # Display caption
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
        except Exception as e:
            doc.add_paragraph(f"[Image: {clean_alt or img_path}] (Failed to embed: {e})")
    else:
        doc.add_paragraph(f"[Image: {clean_alt or img_path}] (File not found)")


def _add_base64_image(doc, data_uri: str, alt_text: str):
    """Add a base64-encoded image to the document"""
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
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        
        try:
            doc.add_picture(tmp_path, width=Inches(5.5))
            if alt_text:
                p = doc.add_paragraph(alt_text)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.italic = True
                    run.font.size = Pt(9)
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        doc.add_paragraph(f"[Image: {alt_text}] (Failed to decode base64: {e})")
