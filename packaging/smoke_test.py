"""Smoke tests executed from the actual frozen runtime (no source checkout needed)."""
import json
import platform
import tempfile
from pathlib import Path


def smoke_test():
    import sys
    assert getattr(sys, "frozen", False), "Run smoke tests from the packaged application"
    from PySide6.QtWidgets import QApplication
    from app import MainWindow
    from PIL import Image
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    window.hide()
    from concurrent.futures import ThreadPoolExecutor
    def check_io_and_ocr():
        import fitz
        from cutbatch import OcrHandle
        from pdf_utils import render_pdf_page_to_pil
        import xlsxwriter
        with tempfile.TemporaryDirectory(prefix='标注打包验证-') as temp:
            pdf = Path(temp) / '测试.pdf'
            doc = fitz.open()
            doc.new_page(width=128, height=128)
            doc.save(pdf)
            doc.close()
            page = render_pdf_page_to_pil(str(pdf), 0, 72)
            assert page.size == (128, 128)
            ocr = OcrHandle()
            ocr.text_predict(Image.new('RGB', (128,128), 'white'), 128)
            workbook = xlsxwriter.Workbook(str(Path(temp) / '测试.xlsx'))
            workbook.add_worksheet('测试').write(0, 0, '字')
            workbook.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(check_io_and_ocr).result(timeout=120)
    window.close()
    print(json.dumps({'smoke_test':'passed', 'arch':platform.machine(),
                      'checks':['Qt window','bundled OCR models','PDF rendering','Excel']}, ensure_ascii=False))
