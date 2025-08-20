import hashlib
import os

import requests
import torch
from tqdm import tqdm

_links = [
    (
        "https://github.com/hkchengrex/Cutie/releases/download/v1.0/coco_lvis_h18_itermask.pth",
        "6fb97de7ea32f4856f2e63d146a09f31",
    ),
    (
        "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth",
        "a6071de6136982e396851903ab4c083a",
    ),
]


def download_models_if_needed() -> str:
    """Download models if needed, handling both package and development modes"""

    # Try package-relative path first (development mode)
    try:
        weight_dir = os.path.join(os.path.dirname(__file__), "..", "..", "weights")
        if os.path.exists(weight_dir):
            # Development mode - use relative path
            pass
        else:
            raise FileNotFoundError("Development weights directory not found")
    except:
        # Package mode or fallback - use user directory
        weight_dir = os.path.expanduser("~/.cutie/weights")

    os.makedirs(weight_dir, exist_ok=True)

    for link, md5 in _links:
        # download file if not exists with a progressbar
        filename = link.split("/")[-1]
        filepath = os.path.join(weight_dir, filename)

        if (
            not os.path.exists(filepath)
            or hashlib.md5(open(filepath, "rb").read()).hexdigest() != md5
        ):
            print(f"Downloading {filename} to {weight_dir}...")
            r = requests.get(link, stream=True)
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            t = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(filepath, "wb") as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                raise RuntimeError("Error while downloading %s" % filename)
    return weight_dir


if __name__ == "__main__":
    download_models_if_needed()
