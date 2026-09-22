"""Shared bounded OCR images and exact original-coordinate mapping."""
from dataclasses import dataclass
import math
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class OcrImage:
    image: Image.Image
    source_size: tuple

    @property
    def target_size(self):
        return self.image.size

    @property
    def scale_back(self):
        return np.asarray(self.source_size,dtype=float)/self.target_size


def target_size(source_size, scale=1.0, max_side=2560):
    scale=float(scale)
    if not math.isfinite(scale) or scale<=0:
        raise ValueError('OCR scale must be finite and positive')
    if min(source_size)<=0:
        raise ValueError('image dimensions must be positive')
    limit=int(max_side)
    # A disabled/invalid legacy setting must not expose the GUI to unbounded
    # full-resolution OCR. The source pixels used for export are unaffected.
    if limit<=0:
        limit=2560
    ratio=min(1.0,scale,limit/max(source_size))
    return tuple(max(1,round(d*ratio)) for d in source_size)


def prepare_ocr_image(image, scale=1.0, max_side=2560):
    source=image.size
    target=target_size(source,scale,max_side)
    if target!=source:
        image=image.resize(target,Image.Resampling.LANCZOS,reducing_gap=3.0)
    if image.mode!='RGB':
        image=image.convert('RGB')
    return OcrImage(image,source)


def load_ocr_image(path, scale=1.0, max_side=2560):
    with Image.open(path) as image:
        source=image.size
        target=target_size(source,scale,max_side)
        # JPEG scaled decoding saves full-page allocation in background OCR.
        # Other decoders safely ignore draft and use the same final resize.
        image.draft('RGB',target)
        if image.size!=target:
            image=image.resize(target,Image.Resampling.LANCZOS,reducing_gap=3.0)
        image=image.convert('RGB')
        image.load()
        return OcrImage(image,source)


def map_results_to_source(rows, prepared):
    factors=prepared.scale_back
    geometry={'source_size':list(prepared.source_size),'ocr_size':list(prepared.target_size)}
    for row in rows:
        row['ocr_geometry']=geometry.copy()
        if prepared.source_size==prepared.target_size:
            continue
        row['bbox']=np.asarray(row['bbox'],dtype=np.float32)*factors
        result=row.get('result',{})
        for item in result.get('char_list',[])+result.get('words',[]):
            for key in ('bbox','line'):
                if key in item:
                    item[key]=np.asarray(item[key],dtype=np.float32)*factors
    return rows
