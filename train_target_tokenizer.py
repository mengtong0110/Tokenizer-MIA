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
v_size = [200000,170000,140000,110000,80000]


os.makedirs(base_dir / 'trained_tokenizer', exist_ok=True)
os.makedirs(base_dir / 'tokenizer_info', exist_ok=True) 
os.makedirs(base_dir / 'shadow_tokenizer', exist_ok=True) 
os.makedirs(base_dir / 'infer_results', exist_ok=True)

def training_data(files):
    training_path = base_dir / 'trained_tokenizer/training_data.json'
    if not training_path.exists():
        training_data = []
        random.shuffle(files)
        member_urls = set(f[:-5] for f in files[:len(files)//2])
        non_member_urls = set(f[:-5] for f in files[len(files)//2:])
        for url in member_urls:
            with open(base_dir / f'website_data/{url}.json', 'r') as f:
                url_data = json.load(f)
                training_data.extend(url_data)
        data = {
            'training_data': training_data,
            'member_datasets': list(member_urls),
            'non_member_datasets': list(non_member_urls)
        }
        with open(training_path, 'w') as f:
            json.dump(data, f)
    else:
        with open(training_path, 'r') as f:
            data = json.load(f)
    return data

def train_tokenizer(training_data, tokenizer_name='tokenizer_base', vocab_size=160000):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        vocab_size=vocab_size,
        show_progress=False,
        #min_frequency=100 
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(training_data, trainer=trainer)
    tokenizer.save(f"{base_dir}/trained_tokenizer/{tokenizer_name}.json")


training_info = None 
for vocab_size in tqdm.tqdm(v_size):
    #======================================================
    # Train Target Tokenizer
    #======================================================
    tokenizer_name = f'target_tokenizer-{vocab_size}'
    if not os.path.exists(f'{base_dir}/trained_tokenizer/' + tokenizer_name + '.json'):
        directory = base_dir / 'website_data'
        files = [f'{f.name}' for f in directory.iterdir() if f.is_file()]
        if training_info == None:
            training_info = training_data(files)
        train_tokenizer(training_info['training_data'], tokenizer_name=tokenizer_name, vocab_size=vocab_size)
        with open(f'{base_dir}/tokenizer_info/target_tokenizer-{vocab_size}.json', 'w') as f:
            json.dump({'member_datasets': training_info['member_datasets'], 'non_member_datasets': training_info['non_member_datasets']}, f)
