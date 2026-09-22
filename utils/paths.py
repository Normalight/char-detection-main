"""图片/PDF 标注导出的目录、前缀及兼容 JSON 路径。"""
import os
from typing import List, Tuple


def get_output_dir_for_item(output_root_dir: str, item: dict) -> Tuple[str, str]:
    """
    返回当前 item 的输出文件夹与文件名前缀。

    image:      <output>/<image_stem>/, prefix=<image_stem>
    pdf page:   <output>/<pdf_stem>/page_001/, prefix=<pdf_stem>_p001
    """
    if item.get("type") == "image":
        stem = os.path.splitext(os.path.basename(item.get("source_path", "image")))[0]
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
    out, seen = [], set()
    for p in candidates:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

