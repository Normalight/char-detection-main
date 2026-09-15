"""Download the default release models at fixed upstream revisions."""
from pathlib import Path
import logging
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS = [('PaddlePaddle/PP-OCRv6_small_det_onnx', '28fe5895c24fd108c19eb3e8479f4ab385fbfc62', 'models/ppocrv6/small'), ('PaddlePaddle/PP-OCRv6_small_rec_onnx', 'b8f84f0b80c529de40b4fbb3544b84fa7233a513', 'models/ppocrv6_rec/small')]


def main():
    for repository, revision, target in MODELS:
        folder = ROOT / target
        folder.mkdir(parents=True, exist_ok=True)
        for filename in ("inference.onnx", "inference.yml"):
            url = f"https://huggingface.co/{repository}/resolve/{revision}/{filename}"
            destination = folder / filename
            temporary = destination.with_suffix(destination.suffix + ".part")
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "character-desktop-build"})
                    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                    if temporary.stat().st_size < 100:
                        raise ValueError(f"Incomplete model resource: {url}")
                    temporary.replace(destination)
                    break
                except Exception:
                    temporary.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
                    logging.warning("Download failed, retrying %s", url)
                    time.sleep(2 ** attempt)
            logging.info("Downloaded %s", destination.relative_to(ROOT))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
