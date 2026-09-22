import io

import pytest
from PIL import Image

fitz = pytest.importorskip("fitz")

from utils.pdf import render_pdf_page_to_pil


def make_pdf(tmp_path, kind="scan", rotation=0):
    source = Image.new("RGB", (600, 800), "white")
    source.paste((10, 30, 90), (20, 30, 201, 302))
    stream = io.BytesIO()
    source.save(stream, format="PNG")
    path = tmp_path / "page.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=150, height=200)
        rect = page.rect if kind != "margin" else fitz.Rect(10, 10, 140, 190)
        if kind != "vector":
            xref = page.insert_image(rect, stream=stream.getvalue())
            # 使用标准 RGB 扫描图；新版 PyMuPDF 插图默认创建 ICC 色彩空间。
            doc.xref_set_key(xref, "ColorSpace", "/DeviceRGB")
        if kind == "text":
            page.insert_text((20, 20), "Visible title")
        if kind == "hidden_text":
            page.insert_text((20, 20), "OCR text", render_mode=3)
        if kind == "vector":
            page.draw_rect(fitz.Rect(10, 10, 80, 90), color=(1, 0, 0))
        if kind == "multiple":
            page.insert_image(fitz.Rect(10, 10, 40, 50), stream=stream.getvalue())
        if kind == "annotation":
            page.add_rect_annot(fitz.Rect(10, 10, 80, 90))
        if kind == "clip":
            xref = page.get_contents()[0]
            doc.update_stream(xref, b"q 0 0 70 100 re W n\n" + doc.xref_stream(xref) + b"\nQ")
        if kind == "mask":
            xref = page.get_images()[0][0]
            doc.xref_set_key(xref, "Mask", "[0 0 0 0 0 0]")
        if kind == "crop":
            page.set_cropbox(fitz.Rect(10, 10, 140, 190))
        page.set_rotation(rotation)
        doc.save(path)
    return path, source


@pytest.mark.parametrize("dpi", [72, 300])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_scan_preserves_original_pixels(tmp_path, dpi, rotation):
    path, source = make_pdf(tmp_path, rotation=rotation)
    actual = render_pdf_page_to_pil(path, 0, dpi=dpi)
    expected = source.rotate(-rotation, expand=True)
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind", ["text", "hidden_text", "vector", "margin", "multiple", "annotation", "clip", "mask", "crop"])
def test_complex_page_matches_dpi_render(tmp_path, kind):
    path, _ = make_pdf(tmp_path, kind)
    actual = render_pdf_page_to_pil(path, 0, dpi=144)
    with fitz.open(path) as doc:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        expected = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()


def test_extraction_failure_still_renders_with_pymupdf(tmp_path, monkeypatch):
    path, _ = make_pdf(tmp_path)
    def fail(*args):
        raise ValueError("unsupported image")
    monkeypatch.setattr("utils.pdf._extract_full_page_scan", fail)
    assert render_pdf_page_to_pil(path, 0, dpi=72).size == (150, 200)
