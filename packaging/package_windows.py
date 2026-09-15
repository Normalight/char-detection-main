"""Smoke-test the frozen Windows app, then archive the complete onedir bundle."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAME = 'interactive_annotator'


def main():
    if sys.platform != "win32":
        raise RuntimeError("Run Windows packaging on Windows")
    bundle = ROOT / "dist" / NAME
    executable = bundle / (NAME + ".exe")
    subprocess.run([str(executable), "--smoke-test"], check=True, timeout=180, cwd=bundle)
    (bundle / "使用说明.txt").write_text(
        "解压后打开 " + NAME + ".exe。请保留整个文件夹及 _internal 目录。\n"
        "无需另装 Python 或模型；请选择自己的可写文件夹保存结果。\n", encoding="utf-8-sig")
    shutil.make_archive(str(ROOT / "dist" / (NAME + "-windows-x64")),
                                      "zip", root_dir=bundle.parent, base_dir=bundle.name)



if __name__ == "__main__":
    main()
