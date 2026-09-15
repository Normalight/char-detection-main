import numpy as np

import cutbatch


def test_recognition_failure_is_isolated_to_one_text_row(monkeypatch):
    class PartiallyFailingRecognizer:
        def predict_rbg_batch(self, _items):
            raise ValueError("one row has degenerate geometry")

        def predict_rbg(self, _crop, bbox):
            if float(np.asarray(bbox)[0, 0]) == 20.0:
                raise ValueError("bad row")
            return {
                "line_pred": "甲",
                "char_list": [{"char": "甲", "bbox": bbox}],
                "words": [],
            }

    handle = cutbatch.OcrHandle.__new__(cutbatch.OcrHandle)
    handle.crnn_handle = PartiallyFailingRecognizer()
    monkeypatch.setattr(
        cutbatch,
        "get_rotate_crop_image",
        lambda _image, _box: np.full((12, 20, 3), 255, dtype=np.uint8),
    )
    boxes = np.asarray(
        [
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[20, 0], [30, 0], [30, 10], [20, 10]],
        ],
        dtype=np.float32,
    )

    results = handle.crnnRecWithBox(
        np.full((40, 40, 3), 255, dtype=np.uint8),
        boxes,
        np.asarray([0.9, 0.8], dtype=np.float32),
    )

    assert len(results) == 2
    assert results[0]["result"]["line_pred"] == "甲"
    assert results[1]["result"]["line_pred"] == ""
