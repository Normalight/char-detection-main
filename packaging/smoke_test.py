"""Smoke tests executed from the actual frozen runtime (no source checkout needed)."""
import json
import os
import platform
import tempfile
from pathlib import Path


def smoke_test():
    import sys
    assert getattr(sys, "frozen", False), "Run smoke tests from the packaged application"
    from PySide6.QtWidgets import QApplication
    from ui.window import MainWindow
    from hanzi_auth import require_login, protect_window
    assert callable(require_login) and callable(protect_window)
    from PIL import Image
    app = QApplication.instance() or QApplication([])
    from hanzi_auth.desktop import LoginDialog
    from hanzi_auth.updates import build_info
    from hanzi_auth.update_ui import UpdateController
    product = 'char-detection-main'
    assert build_info(product)["product"] == product
    from hanzi_auth.single_instance import SingleInstance
    with tempfile.TemporaryDirectory(prefix='single-instance-smoke-') as lock_dir:
        owner = SingleInstance(product, lock_dir)
        duplicate = SingleInstance(product, lock_dir)
        assert owner.acquire()
        assert not duplicate.acquire()
        owner.close()
        assert duplicate.acquire()
        duplicate.close()
    from hanzi_auth.credentials import CredentialStore
    with tempfile.TemporaryDirectory(prefix='credentials-smoke-') as credential_dir:
        vault = CredentialStore('https://smoke.invalid', credential_dir)
        vault.set('smoke-test', 'temporary-smoke-value')
        assert vault.get('smoke-test') == 'temporary-smoke-value'
        vault.delete('smoke-test')
        assert vault.get('smoke-test') == ''
    login = LoginDialog(product)
    assert login.server_url.startswith(("https://", "http://"))
    login.reject()
    window = MainWindow()
    updates = UpdateController(window, product, is_locked=lambda: True)
    updates.startup_timer.stop()
    assert updates.action.text() == "检查更新…"
    window.show()
    app.processEvents()
    window.hide()
    # Exercise the retained Qt image plugins and Unicode text widgets.
    from PySide6.QtGui import QImageReader
    from PySide6.QtWidgets import QLineEdit
    text_input = QLineEdit()
    text_input.setText('中文输入测试')
    assert text_input.text() == '中文输入测试'
    with tempfile.TemporaryDirectory(prefix='image-codec-smoke-') as temp:
        for extension in ('png', 'jpg', 'bmp', 'tiff', 'webp'):
            image_path = Path(temp) / ('图片.' + extension)
            Image.new('RGB', (32, 24), 'white').save(image_path)
            decoded = QImageReader(str(image_path)).read()
            assert not decoded.isNull(), f'Qt cannot read {extension}'
            assert (decoded.width(), decoded.height()) == (32, 24)
    from concurrent.futures import ThreadPoolExecutor
    def check_io_and_ocr():
        import fitz
        from ocr.pipeline import OcrHandle
        from utils.pdf import render_pdf_page_to_pil
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
            fixture = os.environ.get('HANZI_SMOKE_IMAGE')
            image = Image.open(fixture).convert('RGB') if fixture else Image.new('RGB', (128,128), 'white')
            image.thumbnail((1280, 1280))
            result = ocr.text_predict(image, 640 if fixture else 128)
            if fixture:
                assert result, 'Text fixture produced no OCR results'
            workbook = xlsxwriter.Workbook(str(Path(temp) / '测试.xlsx'))
            workbook.add_worksheet('测试').write(0, 0, '字')
            workbook.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(check_io_and_ocr).result(timeout=120)
    window.close()
    print(json.dumps({'smoke_test':'passed', 'arch':platform.machine(),
                      'checks':['login and update UI','Qt window','image codecs','Unicode widgets','bundled OCR models','PDF rendering','Excel']}, ensure_ascii=False))
