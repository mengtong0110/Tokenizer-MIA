import os
import tqdm
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
from datasets import load_dataset

base_dir = Path(__file__).parent

custom_cache_dir = base_dir / "downloaded_data"
os.environ["HF_HOME"] = str(custom_cache_dir)
os.environ["HF_DATASETS_CACHE"] = str(custom_cache_dir)
os.environ["HF_DATASETS_DOWNLOADS"] = str(custom_cache_dir)


def extract_website(url):
    netloc = urlparse(url).netloc
    return netloc

dataset = load_dataset(
    "allenai/c4",
    name="en",
    split="train",
    streaming=True,
    cache_dir=str(custom_cache_dir),
)
N = 5000000

for i, sample in enumerate(tqdm.tqdm(dataset, total=N)):
    try:
        url = sample.get("url", None)
        text = sample.get("text", "")
        website = extract_website(url)
        filename = f"downloaded_data/{website}.jsonl"
        file_path = base_dir / filename
        if not file_path.exists():
            with open(file_path, "w") as f:
                json.dump([text],f)
        else:
            with open(file_path, "r") as f:
                web_data = json.load(f)
            web_data.append(text)
            with open(file_path, "w") as f:
                json.dump(web_data,f)
        if i + 1 >= N:
            break
    except:
        pass
        continue

directory = Path(base_dir /'downloaded_data')
os.makedirs(base_dir /'website_data/', exist_ok=True)
files = [f for f in directory.iterdir() if f.is_file()]

for file in tqdm.tqdm(files):
    try:
        with open(file , 'r') as f:
            data = json.load(f)
        if len(data) >= 200:
            target_path = base_dir / 'website_data' / file.name
            shutil.copyfile(file, target_path)
    except:
        pass
        continue


