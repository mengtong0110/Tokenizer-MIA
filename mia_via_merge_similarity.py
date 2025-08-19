import json
import tqdm
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from sklearn.metrics import roc_curve, auc
from joblib import Parallel, delayed

shared_shadow_vocabs = []
shared_shadow_datasets_list = []
shared_target_vocab = {}
dataset2in_out_index = {}

def compute_dataset_score(dataset, in_indices, out_indices):
    in_score = []
    out_score = []

    for i in in_indices:
        v = shared_shadow_vocabs[i]
        shared_tokens = set(v.keys()) & set(shared_target_vocab.keys())
        X = [v[token] + 1 for token in shared_tokens]
        Y = [shared_target_vocab[token] + 1 for token in shared_tokens]
        if len(X) > 1:
            corr = np.corrcoef(X, Y)[0, 1]
            in_score.append(corr)

    for i in out_indices:
        v = shared_shadow_vocabs[i]
        shared_tokens = set(v.keys()) & set(shared_target_vocab.keys())
        X = [v[token] + 1 for token in shared_tokens]
        Y = [shared_target_vocab[token] + 1 for token in shared_tokens]
        if len(X) > 1:
            corr = np.corrcoef(X, Y)[0, 1]
            out_score.append(corr)

    if len(in_score) == 0 or len(out_score) == 0:
        return (dataset, 0.0)

    y_pred = (np.mean(in_score) - np.mean(out_score) + 2) / 4
    return (dataset, float(y_pred))


def batch_compute_dataset_scores(batch):
    return [
        compute_dataset_score(dataset,
                              dataset2in_out_index[dataset]["in"],
                              dataset2in_out_index[dataset]["out"])
        for dataset in batch
    ]


if __name__ == '__main__':
    base_dir = Path(__file__).parent
    directory = base_dir / 'website_data'
    datasets = [f.name[:-5] for f in directory.iterdir() if f.is_file() and f.name.endswith('.json')]
    v_size = [200000, 170000, 140000, 110000, 80000]

    for vocab_size in v_size:
        target_tokenizer = Tokenizer.from_file(f"{base_dir}/trained_tokenizer/target_tokenizer-{vocab_size}.json")
        shared_target_vocab = target_tokenizer.get_vocab()

        shared_shadow_vocabs.clear()
        shared_shadow_datasets_list.clear()
        for iteration in range(96):
            shadow_tokenizer = Tokenizer.from_file(f"{base_dir}/shadow_tokenizer/shadow_tokenizer-{vocab_size}_{iteration}.json")
            shared_shadow_vocabs.append(shadow_tokenizer.get_vocab())
            with open(f"{base_dir}/tokenizer_info/shadow_tokenizer-{vocab_size}_{iteration}.json", "r") as f:
                training_info = json.load(f)
            shared_shadow_datasets_list.append(set(training_info["member_datasets"]))

        dataset2in_out_index = {
            dataset: {
                "in": [i for i in range(96) if dataset in shared_shadow_datasets_list[i]],
                "out": [i for i in range(96) if dataset not in shared_shadow_datasets_list[i]],
            }
            for dataset in datasets
        }

        batch_size = 8
        dataset_batches = [datasets[i:i + batch_size] for i in range(0, len(datasets), batch_size)]

        results = Parallel(n_jobs=10, backend="loky")(
            delayed(batch_compute_dataset_scores)(batch) for batch in tqdm.tqdm(dataset_batches, desc="Computing scores")
        )

        dataset2pred = dict(item for batch in results for item in batch)

        with open(f'{base_dir}/tokenizer_info/target_tokenizer-{vocab_size}.json', 'r') as f:
            training_info = json.load(f)

        y_pred, y_true = [], []
        details = {'members': [], 'non-members': []}
        for dataset in tqdm.tqdm(training_info['member_datasets'], desc='[2/4]'):
            pred_score = dataset2pred.get(dataset, 0)
            y_pred.append(pred_score)
            y_true.append(1)
            details['members'].append((dataset, str(pred_score)))
        for dataset in tqdm.tqdm(training_info['non_member_datasets'], desc='[3/4]'):
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
        with open(f'{base_dir}/infer_results/MIA via Merge Similarity - v_size_{vocab_size}.json', 'w') as f:
            json.dump(result, f, indent=4)
