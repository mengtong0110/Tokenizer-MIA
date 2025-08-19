import os
import json
import tqdm
import random
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

base_dir = Path(__file__).parent
directory = base_dir / 'website_data'
files = [f.name[:-5] for f in directory.iterdir() if f.is_file()]
random.shuffle(files)

vocab_sizes = [200000, 170000, 140000, 110000, 80000]
special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

def get_training_data(files):
    training_data = []
    member_urls = random.sample(files, int(0.5 * len(files)))
    for url in member_urls:
        with open(base_dir / f'website_data/{url}.json', 'r') as f:
            url_data = json.load(f)
            training_data.extend(url_data)
    return {
        'training_data': training_data,
        'member_datasets': member_urls,
    }

def train_and_export_tokenizers(training_info, tokenizer_base_name, vocab_sizes, iteration):
    max_vocab_size = max(vocab_sizes)
    training_data = training_info['training_data']
    member_datasets = training_info['member_datasets']
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=max_vocab_size,
        show_progress=False,
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(training_data, trainer=trainer)

    temp_dir = base_dir / f"shadow_tokenizer/tmp_{tokenizer_base_name}_{iteration}"
    temp_dir.mkdir(exist_ok=True, parents=True)
    tokenizer.model.save(str(temp_dir))

    with open(temp_dir / "vocab.json", "r", encoding="utf-8") as f:
        full_vocab = json.load(f)
    with open(temp_dir / "merges.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    merges = [line for line in lines[1:] if line.strip() and not line.startswith("#")]
    initial_vocab_size = len(full_vocab) - len(merges)  

    for vocab_size in tqdm.tqdm(vocab_sizes,desc='[steps]'):
        n_merges = vocab_size - initial_vocab_size
        if n_merges < 0:
            raise ValueError(f"Target vocab_size {vocab_size} too small for the initial vocab ({initial_vocab_size})")
        truncated_merges = merges[:n_merges]
        merges_list = [tuple(line.strip().split()) for line in truncated_merges if line.strip()]

        truncated_vocab = {}
        idx = 0
        for token in special_tokens:
            if token in full_vocab and token not in truncated_vocab:
                truncated_vocab[token] = idx
                idx += 1
        for token, _ in full_vocab.items():
            if token not in truncated_vocab and len(truncated_vocab) < vocab_size:
                truncated_vocab[token] = idx
                idx += 1

        new_tokenizer = Tokenizer(BPE(
            vocab=truncated_vocab,
            merges=merges_list,
            unk_token="[UNK]"
        ))
        new_tokenizer.pre_tokenizer = Whitespace()
        save_path = base_dir / f"shadow_tokenizer/{tokenizer_base_name}-{vocab_size}_{iteration}.json"
        new_tokenizer.save(str(save_path))

        info_path = base_dir / f"tokenizer_info/{tokenizer_base_name}-{vocab_size}_{iteration}.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, 'w') as f:
            json.dump({'member_datasets': member_datasets}, f)


(base_dir / 'shadow_tokenizer').mkdir(exist_ok=True)
(base_dir / 'tokenizer_info').mkdir(exist_ok=True)

for iteration in tqdm.trange(128, desc="Iterations"):
    train_data_path = base_dir / f"shadow_tokenizer/training_data_{iteration}.json"

    if not train_data_path.exists():
        training_info = get_training_data(files)
        with open(train_data_path, 'w') as f:
            json.dump(training_info, f)
    else:
        with open(train_data_path, 'r') as f:
            training_info = json.load(f)

    done = all((base_dir / f"shadow_tokenizer/shadow_tokenizer-{vocab_size}_{iteration}.json").exists() for vocab_size in vocab_sizes)
    if done:
        continue

    train_and_export_tokenizers(
        training_info,
        tokenizer_base_name='shadow_tokenizer',
        vocab_sizes=vocab_sizes,
        iteration=iteration
    )
