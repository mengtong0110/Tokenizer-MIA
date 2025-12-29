import json
import tqdm
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from collections import defaultdict
from tokenizers.pre_tokenizers import Whitespace

base_dir = Path(__file__).parent
directory = base_dir / '../website_data'
files = [f.name[:-5] for f in directory.iterdir() if f.is_file()]

vocab_sizes = [200000, 170000, 140000, 110000, 80000]

n_min = 48


def filter_tokenizer_by_frequency(tokenizer_path, token2cnt, output_path, vocab_size):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    vocab = tokenizer.get_vocab()
    with open(str(tokenizer_path)) as f:
        token_files = json.load(f)
    
    merges = token_files["model"]["merges"]

    special_tokens = ["[UNK]"]
    new_vocab = {}
    idx = 0

    for token in special_tokens:
        if token in vocab:
            new_vocab[token] = idx
            idx += 1

    for token, old_idx in vocab.items():
        if token in new_vocab:
            continue
        if token2cnt.get(token, 0) < n_min:
            continue
        if len(new_vocab) >= vocab_size:
            break
        new_vocab[token] = idx
        idx += 1

    new_merges = [tuple(merge) for merge in merges 
              if merge[0] in new_vocab and merge[1] in new_vocab and merge[0]+merge[1] in new_vocab]

    new_tokenizer = Tokenizer(BPE(vocab=new_vocab, merges=new_merges, unk_token="[UNK]"))
    new_tokenizer.pre_tokenizer = Whitespace()
    new_tokenizer.save(str(output_path))



for vocab_size in tqdm.tqdm(vocab_sizes, desc='[Filtering Tokenizers]'):
    tokenizer_name = f"target_tokenizer-{vocab_size}"
    tokenizer_path = base_dir / f"../trained_tokenizer/{tokenizer_name}.json"
    dataset2cnt_path = base_dir / f"../trained_tokenizer/{vocab_size}-dataset2token_cnt.json"

    with open(f'../tokenizer_info/{tokenizer_name}.json', 'r') as f:
        training_info = json.load(f)
    member_datasets = training_info["member_datasets"]

    with open(dataset2cnt_path, 'r') as f:
        dataset2token_cnt = json.load(f)
    token2cnt = defaultdict(int)
    for dataset in tqdm.tqdm(member_datasets, desc=f"[token count for {vocab_size}]"):
        for token in dataset2token_cnt[dataset]:
            token2cnt[token] += dataset2token_cnt[dataset][token]

    output_path = base_dir / f"./{tokenizer_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_tokenizer_by_frequency(tokenizer_path, token2cnt, output_path, vocab_size)
