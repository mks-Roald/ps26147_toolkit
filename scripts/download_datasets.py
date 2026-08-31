import os
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

RADIO_ML_URL = "https://www.deepsig.org/datasets/RadioML2016.10a.zip"

def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_radio_ml():
    ensure_data_dir()
    zip_path = DATA_DIR / "RadioML2016.10a.zip"
    if zip_path.exists():
        print(f"{zip_path} already exists, skipping download.")
        return
    print(f"Downloading RadioML2016.10a from {RADIO_ML_URL} ...")
    urllib.request.urlretrieve(RADIO_ML_URL, zip_path)
    print("Download complete. Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction finished.")

if __name__ == "__main__":
    download_radio_ml()
