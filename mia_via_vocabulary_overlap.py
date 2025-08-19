import json
import tqdm
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from sklearn.metrics import roc_curve, auc
from concurrent.futures import ProcessPoolExecutor, as_completed

_shadow_vocabs = None
_shadow_datasets_list = None
_target_vocab_tokens = None

def init_worker(vocabs, datasets_list, target_tokens):
    global _shadow_vocabs, _shadow_datasets_list, _target_vocab_tokens
    _shadow_vocabs = vocabs
    _shadow_datasets_list = datasets_list
    _target_vocab_tokens = target_tokens


def compute_dataset_score(dataset):
    tokens_in = set()
    tokens_out = set()
    v_in_sets = []
    v_out_sets = []

    for vocab, datasets in zip(_shadow_vocabs, _shadow_datasets_list):
        vocab_tokens = set(vocab.keys())
        if dataset in datasets:
            v_in_sets.append(vocab_tokens)
            tokens_in.update(vocab_tokens)
        else:
            v_out_sets.append(vocab_tokens)
            tokens_out.update(vocab_tokens)

    excluded_tokens = tokens_in & tokens_out
    filtered_target_tokens = _target_vocab_tokens - excluded_tokens

    def jaccard_score(vocab_tokens):
        filtered_vocab = vocab_tokens - excluded_tokens
        union = filtered_vocab | filtered_target_tokens
        if not union:
            return 0.0
        intersection = filtered_vocab & filtered_target_tokens
        return len(intersection) / len(union)

    in_scores = [jaccard_score(v) for v in v_in_sets]
    out_scores = [jaccard_score(v) for v in v_out_sets]

    y_pred = (np.mean(in_scores or [0]) - np.mean(out_scores or [0]) + 1) / 2
    return (dataset, float(y_pred))

if __name__ == '__main__':
    base_dir = Path(__file__).parent
    directory = base_dir / 'website_data'
    datasets = [f.name[:-5] for f in directory.iterdir() if f.is_file() and f.name.endswith('.json')]
    v_size = [200000, 170000, 140000, 110000, 80000]
    shadow_num = 96

    for vocab_size in v_size:
        target_tokenizer = Tokenizer.from_file(f"{base_dir}/trained_tokenizer/target_tokenizer-{vocab_size}.json")
        target_vocab = target_tokenizer.get_vocab()
        target_vocab_tokens = set(target_vocab.keys())

        shadow_vocabs = []
        shadow_datasets_list = []

        for iteration in range(shadow_num):
            shadow_tokenizer = Tokenizer.from_file(
                f"{base_dir}/shadow_tokenizer/shadow_tokenizer-{vocab_size}_{iteration}.json")
            shadow_vocabs.append(shadow_tokenizer.get_vocab())
            with open(f"{base_dir}/tokenizer_info/shadow_tokenizer-{vocab_size}_{iteration}.json", "r") as f:
                training_info = json.load(f)
            member_datasets = set(training_info["member_datasets"])
            shadow_datasets_list.append(member_datasets)

        dataset2pred = dict()
        with ProcessPoolExecutor(
            max_workers=7,
            initializer=init_worker,
            initargs=(shadow_vocabs, shadow_datasets_list, target_vocab_tokens)
        ) as executor:
            futures = {
                executor.submit(compute_dataset_score, dataset): dataset
                for dataset in datasets
            }
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Computing scores"):
                dataset, pred = future.result()
                dataset2pred[dataset] = pred

        with open(f'{base_dir}/tokenizer_info/target_tokenizer-{vocab_size}.json', 'r') as f:
            training_info = json.load(f)

        y_pred, y_true = [], []
        details = {'members': [], 'non-members': []}

        for dataset in tqdm.tqdm(training_info['member_datasets'], desc='[2/4]'):
            if dataset in datasets:
                pred_score = dataset2pred.get(dataset, 0)
                y_pred.append(pred_score)
                y_true.append(1)
                details['members'].append((dataset, str(pred_score)))

        for dataset in tqdm.tqdm(training_info['non_member_datasets'], desc='[3/4]'):
            if dataset in datasets:
                pred_score = dataset2pred.get(dataset, 0)
                y_pred.append(pred_score)
                y_true.append(0)
                details['non-members'].append((dataset, str(pred_score)))

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        balanced_accuracy = np.max(1 - (fpr + (1 - tpr)) / 2)
        tpr_at_low_fpr = tpr[np.where(np.array(fpr) < 0.01)[0][-1]] if np.any(np.array(fpr) < 0.01) else 0.0

        result = {
            'roc_auc': roc_auc,
            'balanced_accuracy': balanced_accuracy,
            'tpr_at_low_fpr': tpr_at_low_fpr,
            'fpr': fpr.tolist(), 
            'tpr': tpr.tolist(),  
            'details': details
        }

        print(f'roc_auc : {roc_auc:.4f}, balanced_accuracy : {balanced_accuracy:.4f}, tpr@fpr<1% : {tpr_at_low_fpr:.4f}')
        with open(f'{base_dir}/infer_results/MIA via Vocabulary Overlap - v_size_{vocab_size}.json', 'w') as f:
            json.dump(result, f, indent=4)
