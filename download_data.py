import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_avss_dataset(data_dir="data/datasets/avss", public_url=None):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "dla_dataset.zip"

    print(public_url)
    api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={public_url}"

    download_url = requests.get(api_url).json()["href"]

    print("Dowload dataset.")
    response = requests.get(download_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(
        desc="Progress",
        total=total_size,
        unit="B",
        unit_scale=True,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print("Unzip data")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print("Dataset is ready")
    print(f"Location: {data_dir / 'dla_dataset'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AVSS dataset")
    parser.add_argument(
        "--public-url", type=str, help="Public URL for Yandex.Disk download"
    )
    args = parser.parse_args()

    download_avss_dataset(public_url=args.public_url)
