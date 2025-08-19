import tqdm
import json
import numpy as np
from mpmath import exp
from pathlib import Path
from tokenizers import Tokenizer
from sklearn.metrics import roc_curve, auc 

def sigmoid(x):
    return 1 / (1 + exp(-x))

fpr_target = 0.01
base_dir = Path(__file__).parent
directory = base_dir / 'website_data'
datasets = [f.name[:-5] for f in directory.iterdir() if f.is_file() and f.name.endswith('.json')]

v_size = [200000,170000,140000,110000,80000]

for vocab_size in v_size:
    tokenizer_name = f'tokenizer_{vocab_size}'  
    tokenizer = Tokenizer.from_file(f"{base_dir}/trained_tokenizer/target_tokenizer-{vocab_size}.json")

    dataset2score = dict()
    for dataset in tqdm.tqdm(datasets, desc='[0/3]'):
        total_bytes = 0
        total_tokens = 0
        with open(base_dir / f'website_data/{dataset}.json', 'r') as f:
            url_data = json.load(f)
        for text in url_data:
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(tokenizer.encode(text).ids)
        average_bpt = total_bytes / total_tokens
        dataset2score[dataset] = average_bpt

    with open(f'{base_dir}/tokenizer_info/target_tokenizer-{vocab_size}.json', 'r') as f:
        training_info = json.load(f)

    details = {'members':[],'non-members':[]}
    y_pred, y_true = [], []
    for dataset in tqdm.tqdm(training_info['member_datasets'], desc='[1/3]'):
        pred_score = sigmoid(dataset2score[dataset])
        y_pred.append(pred_score)
        y_true.append(1)
        details['members'].append([dataset, float(pred_score)])
    for dataset in tqdm.tqdm(training_info['non_member_datasets'], desc='[2/3]'):
        pred_score = sigmoid(dataset2score[dataset])
        y_pred.append(pred_score)
        y_true.append(0)
        details['non-members'].append([dataset, float(pred_score)])

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    tpr_at_low_fpr = tpr[np.where(np.array(fpr) < 0.01)[0][-1]] if np.any(np.array(fpr) < 0.01) else 0.0
    balanced_accuracy = np.max(1 - (fpr + (1 - tpr)) / 2)

    result = {
        'roc_auc': roc_auc,
        'balanced_accuracy': balanced_accuracy,
        'tpr_at_low_fpr': tpr_at_low_fpr,
        'fpr': fpr.tolist(), 
        'tpr': tpr.tolist(),  
        'details': details
    }

    with open(f'{base_dir}/infer_results/MIA via Compression Rate - v_size_{vocab_size}.json', 'w') as f:
        json.dump(result, f, indent=4)




        
            