"""图片/PDF 标注导出的目录、前缀及兼容 JSON 路径。"""
import json
import ntpath
import os
from typing import List, Tuple


def assign_image_output_names(items: List[dict]) -> None:
    """Keep ordinary stems; disambiguate duplicate image stems for this session."""
    groups = {}
    reserved = {os.path.splitext(os.path.basename(it['source_path']))[0].casefold()
                for it in items}
    for item in items:
        if item.get('type') == 'image':
            stem = os.path.splitext(os.path.basename(item['source_path']))[0]
            groups.setdefault(stem.casefold(), []).append(item)
    for group in groups.values():
        if len(group) < 2:
            continue
        for item in sorted(group, key=lambda it: it['source_path']):
            stem, ext = os.path.splitext(os.path.basename(item['source_path']))
            base = f'{stem}_{ext[1:].lower()}'
            name, number = base, 2
            while name.casefold() in reserved:
                name = f'{base}_{number}'
                number += 1
            reserved.add(name.casefold())
            item['output_stem'] = name


def get_output_dir_for_item(output_root_dir: str, item: dict) -> Tuple[str, str]:
    """
    返回当前 item 的输出文件夹与文件名前缀。

    image:      <output>/<output_stem or image_stem>/, prefix matches folder
    pdf page:   <output>/<pdf_stem>/page_001/, prefix=<pdf_stem>_p001
    """
    if item.get("type") == "image":
        stem = item.get('output_stem') or os.path.splitext(os.path.basename(item.get("source_path", "image")))[0]
        return os.path.join(output_root_dir, stem), stem

    pdf_stem = os.path.splitext(os.path.basename(item.get("source_path", "pdf")))[0]
    page_index = int(item.get("page_index") or 0)
    page_folder = f"page_{page_index + 1:03d}"
    prefix = f"{pdf_stem}_p{page_index + 1:03d}"
    return os.path.join(output_root_dir, pdf_stem, page_folder), prefix



def get_json_candidates(output_root_dir: str, item: dict) -> List[str]:
    """兼容旧版本可能生成的 page_001.json 等路径。"""
    folder, prefix = get_output_dir_for_item(output_root_dir, item)
    candidates = [os.path.join(folder, f"{prefix}.json"), os.path.join(folder, f"{os.path.basename(folder)}.json")]
    if item.get("type") == "image" and item.get('output_stem'):
        legacy_stem = os.path.splitext(os.path.basename(item['source_path']))[0]
        legacy_path = os.path.join(output_root_dir, legacy_stem, f"{legacy_stem}.json")
        if _legacy_image_matches(legacy_path, item):
            candidates.append(legacy_path)
    out, seen = [], set()
    for p in candidates:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _legacy_image_matches(path: str, item: dict) -> bool:
    """Only reuse a shared legacy folder when its annotation belongs to this image."""
    try:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        source = item["source_path"]
        filename = os.path.basename(source)
        saved_source = data.get("source_path")
        if saved_source:
            # Accept moved input folders and annotations created on Windows.
            return ntpath.basename(saved_source) == filename
        # Very old annotations may omit the source. Require an unambiguous
        # existing input; never give the same legacy boxes to two image files.
        stem = os.path.splitext(filename)[0].casefold()
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        with os.scandir(os.path.dirname(source) or ".") as entries:
            matches = [entry.name for entry in entries if entry.is_file()
                       and os.path.splitext(entry.name)[0].casefold() == stem
                       and os.path.splitext(entry.name)[1].lower() in extensions]
        return matches == [filename]
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return False
