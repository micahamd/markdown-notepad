"""
Document Image Extractor & Inline Placement Engine for MarkItDown Notepad

Extracts embedded images from various document formats (PDF, EPUB, DOCX, PPTX, XLSX, HTML)
and accurately embeds them inline within the generated Markdown content.
"""

import os
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Try importing fitz (PyMuPDF)
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# Try importing PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try importing text cleaner
try:
    from text_cleaner import clean_document
    TEXT_CLEANER_AVAILABLE = True
except ImportError:
    TEXT_CLEANER_AVAILABLE = False


class DocumentImageExtractor:
    """
    Extracts embedded images from documents and integrates them inline into Markdown.
    """

    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}

    @classmethod
    def can_extract(cls, filepath: str) -> bool:
        """Check if image extraction is supported for the given file extension."""
        ext = Path(filepath).suffix.lower()
        return ext in {'.pdf', '.epub', '.docx', '.pptx', '.xlsx', '.html', '.htm'}

    @classmethod
    def get_default_assets_dir(cls, filepath: str) -> str:
        """Generate a companion assets folder path next to the source document."""
        parent_dir = os.path.dirname(os.path.abspath(filepath))
        base_name = Path(filepath).stem
        # Clean special chars in folder name
        safe_base = re.sub(r'[^\w\-. ]', '_', base_name).strip()
        assets_dir = os.path.join(parent_dir, f"{safe_base}_assets")
        return assets_dir

    @classmethod
    def extract_from_pdf(cls, pdf_path: str, output_dir: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract text and embedded images from a PDF using PyMuPDF (fitz).
        Interleaves text blocks and image references based on physical vertical coordinates.
        """
        if not FITZ_AVAILABLE:
            return "", {}

        os.makedirs(output_dir, exist_ok=True)
        extracted_images: Dict[str, str] = {}
        markdown_pages: List[str] = []
        img_counter = 0

        doc = fitz.open(pdf_path)

        for p_idx in range(len(doc)):
            page = doc[p_idx]
            page_num = p_idx + 1
            page_elements = []

            # 1. Retrieve text blocks with bounding boxes: (x0, y0, x1, y1, text, block_no, block_type)
            text_blocks = page.get_text("blocks")
            for b in text_blocks:
                x0, y0, x1, y1, text, b_no, b_type = b
                if b_type == 0 and text.strip():  # Text block
                    page_elements.append({
                        'type': 'text',
                        'y0': y0,
                        'x0': x0,
                        'content': text.rstrip()
                    })

            # 2. Retrieve image placements on this page
            image_info_list = page.get_image_info(xrefs=True)
            for info in image_info_list:
                xref = info.get('xref')
                if not xref:
                    continue

                bbox = info.get('bbox', (0, 0, 0, 0))
                y0 = bbox[1]
                x0 = bbox[0]

                try:
                    base_img = doc.extract_image(xref)
                    if base_img:
                        img_bytes = base_img.get('image')
                        img_ext = base_img.get('ext', 'png')
                        img_counter += 1

                        # Filter out tiny icon/artifact images (< 15x15 px or < 200 bytes)
                        width = base_img.get('width', 0)
                        height = base_img.get('height', 0)
                        if (width > 0 and width < 15) or (height > 0 and height < 15) or len(img_bytes) < 200:
                            continue

                        img_filename = f"img_p{page_num}_{img_counter}.{img_ext}"
                        img_path = os.path.join(output_dir, img_filename)

                        with open(img_path, 'wb') as f:
                            f.write(img_bytes)

                        extracted_images[img_filename] = img_path

                        # Insert markdown reference
                        caption = f"Image (Page {page_num})"
                        if width and height:
                            caption = f"{caption}|{width}x{height}"
                        
                        img_md = f"\n![{caption}]({img_filename})\n"
                        page_elements.append({
                            'type': 'image',
                            'y0': y0,
                            'x0': x0,
                            'content': img_md
                        })
                except Exception as e:
                    print(f"Warning: Failed to extract image xref {xref} on page {page_num}: {e}")

            # Sort page elements from top to bottom (y0 coordinate order)
            page_elements.sort(key=lambda item: (item['y0'], item['x0']))

            # Combine page content
            page_md = "\n\n".join(item['content'] for item in page_elements if item['content'].strip())
            if page_md.strip():
                markdown_pages.append(page_md)

        doc.close()
        full_markdown = "\n\n".join(markdown_pages)
        return full_markdown, extracted_images

    @classmethod
    def extract_from_epub(cls, epub_path: str, output_dir: str, raw_markdown: str = "") -> Tuple[str, Dict[str, str]]:
        """
        Extract all image assets from an EPUB archive and rewrite Markdown image tags
        to point to the extracted local files.
        """
        os.makedirs(output_dir, exist_ok=True)
        extracted_images: Dict[str, str] = {}
        path_mapping: Dict[str, str] = {}

        try:
            with zipfile.ZipFile(epub_path, 'r') as zf:
                for member_name in zf.namelist():
                    ext = Path(member_name).suffix.lower()
                    if ext in cls.IMAGE_EXTENSIONS:
                        filename = os.path.basename(member_name)
                        if not filename:
                            continue

                        # Handle duplicate file names safely
                        target_path = os.path.join(output_dir, filename)
                        counter = 1
                        base, ext_part = os.path.splitext(target_path)
                        while os.path.exists(target_path):
                            target_path = f"{base}_{counter}{ext_part}"
                            counter += 1

                        saved_filename = os.path.basename(target_path)

                        # Extract image file
                        with zf.open(member_name) as src, open(target_path, 'wb') as dst:
                            dst.write(src.read())

                        extracted_images[saved_filename] = target_path

                        # Map original paths to extracted filename
                        path_mapping[member_name] = saved_filename
                        path_mapping[filename] = saved_filename
                        path_mapping[f"../Images/{filename}"] = saved_filename
                        path_mapping[f"Images/{filename}"] = saved_filename
                        path_mapping[f"images/{filename}"] = saved_filename
                        path_mapping[f"../images/{filename}"] = saved_filename
                        path_mapping[f"OEBPS/Images/{filename}"] = saved_filename
        except Exception as e:
            print(f"Error extracting EPUB images: {e}")

        # Rewrite image references in raw markdown
        def rewrite_src(match: re.Match) -> str:
            alt = match.group(1)
            src = match.group(2)

            # Look up normalized src in mapping
            norm_src = src.replace('\\', '/')
            base_src = os.path.basename(norm_src)

            resolved = path_mapping.get(norm_src) or path_mapping.get(base_src)
            if resolved:
                return f"![{alt}]({resolved})"
            return match.group(0)

        # Pattern: ![alt](src)
        updated_markdown = re.sub(
            r'!\[([^\]]*)\]\(([^)\s]+)\)',
            rewrite_src,
            raw_markdown
        )

        return updated_markdown, extracted_images

    @classmethod
    def extract_from_office(cls, office_path: str, output_dir: str, raw_markdown: str = "") -> Tuple[str, Dict[str, str]]:
        """
        Extract embedded media from DOCX, PPTX, or XLSX archives and link them in Markdown.
        """
        os.makedirs(output_dir, exist_ok=True)
        extracted_images: Dict[str, str] = {}
        path_mapping: Dict[str, str] = {}

        try:
            with zipfile.ZipFile(office_path, 'r') as zf:
                for member_name in zf.namelist():
                    # Images in Office XML are stored under word/media/, ppt/media/, xl/media/
                    if '/media/' in member_name.lower():
                        ext = Path(member_name).suffix.lower()
                        if ext in cls.IMAGE_EXTENSIONS:
                            filename = os.path.basename(member_name)
                            target_path = os.path.join(output_dir, filename)

                            counter = 1
                            base, ext_part = os.path.splitext(target_path)
                            while os.path.exists(target_path):
                                target_path = f"{base}_{counter}{ext_part}"
                                counter += 1

                            saved_filename = os.path.basename(target_path)

                            with zf.open(member_name) as src, open(target_path, 'wb') as dst:
                                dst.write(src.read())

                            extracted_images[saved_filename] = target_path
                            path_mapping[member_name] = saved_filename
                            path_mapping[filename] = saved_filename
                            path_mapping[f"word/media/{filename}"] = saved_filename
                            path_mapping[f"media/{filename}"] = saved_filename
        except Exception as e:
            print(f"Error extracting Office images: {e}")

        # If raw markdown already has image references, update them
        if raw_markdown and re.search(r'!\[.*?\]\(.*?\)', raw_markdown):
            def rewrite_office_src(match: re.Match) -> str:
                alt = match.group(1)
                src = match.group(2)
                base_src = os.path.basename(src.replace('\\', '/'))
                resolved = path_mapping.get(base_src) or path_mapping.get(src)
                if resolved:
                    return f"![{alt}]({resolved})"
                return match.group(0)

            updated_markdown = re.sub(
                r'!\[([^\]]*)\]\(([^)\s]+)\)',
                rewrite_office_src,
                raw_markdown
            )
        else:
            # If markdown has no inline references, append extracted images
            updated_markdown = raw_markdown
            if extracted_images:
                img_section = "\n\n---\n\n## Extracted Images\n\n"
                for img_name in sorted(extracted_images.keys()):
                    img_section += f"![{img_name}]({img_name})\n\n"
                updated_markdown += img_section

        return updated_markdown, extracted_images

    @classmethod
    def extract_and_integrate(
        cls,
        filepath: str,
        output_dir: Optional[str] = None,
        raw_markdown: str = "",
        clean_text: bool = True
    ) -> Tuple[str, Dict[str, str], str]:
        """
        Main extraction entry point. Dispatches by document type and returns
        (final_markdown, extracted_images_dict, assets_directory).
        """
        if not output_dir:
            output_dir = cls.get_default_assets_dir(filepath)

        ext = Path(filepath).suffix.lower()
        extracted_images: Dict[str, str] = {}
        processed_md = raw_markdown

        if ext == '.pdf':
            # PyMuPDF spatial extraction
            pdf_md, images = cls.extract_from_pdf(filepath, output_dir)
            if pdf_md and len(pdf_md.strip()) > 50:
                processed_md = pdf_md
                extracted_images = images
            else:
                # Fallback if fitz produced empty output
                processed_md = raw_markdown
        elif ext == '.epub':
            processed_md, extracted_images = cls.extract_from_epub(filepath, output_dir, raw_markdown)
        elif ext in {'.docx', '.pptx', '.xlsx'}:
            processed_md, extracted_images = cls.extract_from_office(filepath, output_dir, raw_markdown)
        else:
            processed_md = raw_markdown

        # Apply algorithmic cleanup while preserving image markdown tags
        if clean_text and TEXT_CLEANER_AVAILABLE and processed_md.strip():
            processed_md = clean_document(processed_md)

        return processed_md, extracted_images, output_dir


def extract_document_images(filepath: str, output_dir: Optional[str] = None, raw_markdown: str = "") -> Tuple[str, Dict[str, str], str]:
    """Convenience function for document image extraction."""
    return DocumentImageExtractor.extract_and_integrate(filepath, output_dir, raw_markdown)
