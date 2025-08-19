import re
import os
import json
import tqdm
import heapq
import random
import powerlaw
import numpy as np 
from pathlib import Path
from mpmath import mp, exp, mpf
from functools import partial
from tokenizers import Tokenizer
from sklearn.metrics import roc_curve, auc
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


# ===================== CONFIG ======================
fpr_target = 0.01
mp.dps = 100
v_size = [200000, 170000, 140000, 110000, 80000]

sampling_iteration = 10

base_dir = Path(__file__).parent
directory = base_dir / 'website_data'
datasets = [f.name[:-5] for f in directory.iterdir() if f.is_file() and f.name.endswith('.json')]

mp.dps = 30 

# ===================================================
# Global shared variable holder for multiprocessing
# ===================================================
_shared_data = {}

def sigmoid(x):
    return 1 / (1 + exp(-x))

def check_token(token, substring_to_words):
    return substring_to_words.get(token, set())

def init_worker(word_id2tokens, dataset2id2cnt, vocab):
    global _shared_data
    _shared_data['dataset2id2cnt'] = dataset2id2cnt
    _shared_data['word_id2tokens'] = word_id2tokens
    _shared_data['vocab'] = vocab

def compute_token_cnt_for_dataset(dataset):
    dataset2id2cnt = _shared_data['dataset2id2cnt']
    word_id2tokens = _shared_data['word_id2tokens']
    vocab = _shared_data['vocab']

    word2cnt = dataset2id2cnt[dataset]
    token_cnt_raw = defaultdict(int)
    for word_id, cnt in word2cnt.items():
        for token in word_id2tokens.get(str(word_id), []):
            token_cnt_raw[token] += cnt
    token_cnt_dict = {token: cnt for token, cnt in token_cnt_raw.items() if token in vocab}
    return dataset, token_cnt_dict

def token_to_word_ids_worker(args):
    token, substring_to_words, word2id = args
    matched = check_token(token, substring_to_words)
    matched_words = set(matched)
    if token in word2id:
        matched_words.add(word2id[token])
    return token, list(matched_words)

def threaded_find_xmin(self, xmin_distance=None):
    from numpy import unique, asarray, argmin, nan
    from numpy.ma import masked_array

    if not self.given_xmin:
        possible_xmins = self.data
    else:
        possible_ind = min(self.given_xmin) <= self.data
        possible_ind *= self.data <= max(self.given_xmin)
        possible_xmins = self.data[possible_ind]
    xmins, xmin_indices = unique(possible_xmins, return_index=True)
    xmins = xmins[:-1]
    xmin_indices = xmin_indices[:-1]

    if xmin_distance is None:
        xmin_distance = self.xmin_distance

    if len(xmins) <= 0:
        from numpy import array
        self.xmin = nan
        self.D = nan
        self.V = nan
        self.Asquare = nan
        self.Kappa = nan
        self.alpha = nan
        self.sigma = nan
        self.n_tail = nan
        setattr(self, xmin_distance + 's', array([nan]))
        self.alphas = array([nan])
        self.sigmas = array([nan])
        self.in_ranges = array([nan])
        self.xmins = array([nan])
        self.noise_flag = True
        return self.xmin

    def fit_function(xmin, idx, num_xmins):
        print(f'xmin progress: {int(idx / num_xmins * 100):02d}%', end='\r')
        pl = self.xmin_distribution(
            xmin=xmin,
            xmax=self.xmax,
            discrete=self.discrete,
            estimate_discrete=self.estimate_discrete,
            fit_method=self.fit_method,
            data=self.data,
            parameter_range=self.parameter_range,
            parent_Fit=self
        )
        if not hasattr(pl, 'sigma'):
            pl.sigma = nan
        if not hasattr(pl, 'alpha'):
            pl.alpha = nan
        return getattr(pl, xmin_distance), pl.alpha, pl.sigma, pl.in_range()

    num_xmins = len(xmins)
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fit_function, xmin, idx, num_xmins): idx for idx, xmin in enumerate(xmins)}
        fits = [None] * num_xmins
        for future in as_completed(futures):
            idx = futures[future]
            fits[idx] = future.result()
    fits = asarray(fits)

    setattr(self, xmin_distance + 's', fits[:, 0])
    self.alphas = fits[:, 1]
    self.sigmas = fits[:, 2]
    self.in_ranges = fits[:, 3].astype(bool)
    self.xmins = xmins

    good_values = self.in_ranges
    if self.sigma_threshold:
        good_values = good_values * (self.sigmas < self.sigma_threshold)

    if good_values.any():
        valid_xmins = self.xmins[good_values]
        min_D_index = np.where(self.xmins == valid_xmins.min())[0][0]
        self.noise_flag = False
    else:
        min_D_index = argmin(getattr(self, xmin_distance + 's'))  
        self.noise_flag = True

    if self.noise_flag:
        print("No valid fits found.")

    self.xmin = xmins[min_D_index]
    setattr(self, xmin_distance, getattr(self, xmin_distance + 's')[min_D_index])
    self.alpha = self.alphas[min_D_index]
    self.sigma = self.sigmas[min_D_index]
    self.fitting_cdf_bins, self.fitting_cdf = self.cdf()
    return self.xmin

import numpy as np

def compute_tpr_at_fpr_threshold(y_true, y_pred, target_fpr=0.01):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    desc_sort_idx = np.argsort(-y_pred)
    y_true_sorted = y_true[desc_sort_idx]
    y_pred_sorted = y_pred[desc_sort_idx]

    cum_tp = np.cumsum(y_true_sorted == 1)
    cum_fp = np.cumsum(y_true_sorted == 0)

    total_neg = np.sum(y_true == 0)
    total_pos = np.sum(y_true == 1)

    fpr = cum_fp / total_neg
    tpr = cum_tp / total_pos

    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0, None  
    best_idx = valid[-1]
    best_threshold = y_pred_sorted[best_idx]
    best_tpr = tpr[best_idx]
    return best_tpr


powerlaw.Fit.find_xmin = threaded_find_xmin

if __name__ == "__main__":
    for vocab_size in v_size:
        #=======================================================
        # Prepare for Implement Experiments
        #=======================================================
        dataset2id2cnt_path = base_dir / 'trained_tokenizer' /'dataset2id2cnt.json'
        token2word_ids_path = base_dir / f"trained_tokenizer/{vocab_size}-token2word_ids.json"
        word_id2tokens_path = base_dir / f"trained_tokenizer/{vocab_size}-word_id2tokens.json"
        dataset2cnt_path = base_dir / f"trained_tokenizer/{vocab_size}-dataset2token_cnt.json"
        token2datasets_path = base_dir / f"trained_tokenizer/{vocab_size}-token2datasets.json"

        if not dataset2id2cnt_path.exists():
            word2id = dict()
            id2word = dict()
            next_id = 0
            dataset2id2cnt = dict()  
            for dataset in tqdm.tqdm(datasets):
                counter = Counter()
                with open(directory / f"{dataset}.json", 'r', encoding='utf-8') as f:
                    url_data = json.load(f)
                for data in url_data:
                    words = re.findall(r'\w+|[^\w\s]+', data)
                    for word in words:
                        if word not in word2id:
                            word2id[word] = next_id
                            id2word[next_id] = word
                            next_id += 1
                        counter[word2id[word]] += 1
                dataset2id2cnt[dataset] = dict(counter)
            with open(base_dir /'trained_tokenizer' /'word2id.json', 'w', encoding='utf-8') as f:
                json.dump(word2id, f, ensure_ascii=False, indent=2)
            with open(base_dir / 'trained_tokenizer' /'id2word.json', 'w', encoding='utf-8') as f:
                json.dump(id2word, f, ensure_ascii=False, indent=2)
            with open(base_dir / 'trained_tokenizer' /'dataset2id2cnt.json', 'w', encoding='utf-8') as f:
                json.dump(dataset2id2cnt, f, ensure_ascii=False, indent=2)

        if not token2word_ids_path.exists():
            with open(base_dir / 'trained_tokenizer/word2id.json', 'r') as f:
                word2id = json.load(f)
            words = set(word2id.keys())
            tokenizer = Tokenizer.from_file(str(base_dir / f"trained_tokenizer/target_tokenizer-{vocab_size}.json"))
            target_vocab = set(tokenizer.get_vocab().keys())
            max_word_length = max(len(token) for token in target_vocab)
            token2word_ids = {}
            word_id2tokens = {}
            for substring_len in tqdm.trange(1, max_word_length + 1, desc='[0/5] Pre-process for Experiments'):
                substring_to_words = defaultdict(set)
                vocab_l = [token for token in target_vocab if len(token) == substring_len]
                for word in words:
                    if len(word) >= substring_len:
                        for start in range(len(word) - substring_len + 1):
                            substring = word[start:start + substring_len]
                            substring_to_words[substring].add(word2id[word])
                args_list = [(token, substring_to_words, word2id) for token in vocab_l]
                with ThreadPoolExecutor() as executor:
                    for token_v, matched_ids in tqdm.tqdm(executor.map(token_to_word_ids_worker, args_list),
                                                          total=len(args_list),
                                                          desc=f"Process substrings of length {substring_len}"):
                        token2word_ids[token_v] = matched_ids
                        for word_id in matched_ids:
                            word_id2tokens.setdefault(str(word_id), [])
                            word_id2tokens[str(word_id)].append(token_v)
            with open(token2word_ids_path, 'w') as f:
                json.dump(token2word_ids, f)
            with open(word_id2tokens_path, 'w') as f:
                json.dump(word_id2tokens, f)

        if not dataset2cnt_path.exists():
            tokenizer = Tokenizer.from_file(str(base_dir / f"trained_tokenizer/target_tokenizer-{vocab_size}.json"))
            vocab = tokenizer.get_vocab()
            with open(token2word_ids_path, 'r') as f:
                token2word_ids = json.load(f)
            with open(word_id2tokens_path, 'r') as f:
                word_id2tokens = json.load(f)
            with open(base_dir / 'trained_tokenizer/dataset2id2cnt.json', 'r') as f:
                dataset2id2cnt = json.load(f)
            dataset2token_cnt = {}
            with ThreadPoolExecutor(initializer=init_worker, initargs=(word_id2tokens, dataset2id2cnt, vocab)) as executor:
                for ds, token_cnt_dict in tqdm.tqdm(executor.map(compute_token_cnt_for_dataset, dataset2id2cnt.keys()),
                                                    total=len(dataset2id2cnt),
                                                    desc="[Parallel] Compute dataset2token_cnt"):
                    dataset2token_cnt[ds] = token_cnt_dict
            with open(dataset2cnt_path, 'w') as f:
                json.dump(dataset2token_cnt, f)

        if not token2datasets_path.exists():
            tokenizer = Tokenizer.from_file(str(base_dir / f"trained_tokenizer/target_tokenizer-{vocab_size}.json"))
            vocab = tokenizer.get_vocab()
            with open(dataset2cnt_path, 'r') as f:
                dataset2token_cnt = json.load(f)
            token2datasets = defaultdict(list)
            for token in vocab.keys():
                token2datasets[token] = []
            for dataset in tqdm.tqdm(dataset2token_cnt):
                for token in dataset2token_cnt[dataset]:
                    token2datasets[token].append(dataset)
            with open(token2datasets_path, 'w') as f:
                json.dump(token2datasets, f)


        #=======================================================
        # Membership Inference for Datasets
        #=======================================================
        target_tokenizer = Tokenizer.from_file(f"{base_dir}/trained_tokenizer/target_tokenizer-{vocab_size}.json")
        target_vocab = target_tokenizer.get_vocab()
        involved_datasets = set()
        for iteration in range(sampling_iteration):
            with open(f'{base_dir}/tokenizer_info/shadow_tokenizer-{vocab_size}_{iteration}.json', 'r') as f:
                sampled_datasets = json.load(f)
            sampled_datasets = set(sampled_datasets['member_datasets'])
            involved_datasets |= sampled_datasets
        with open(dataset2cnt_path, 'r') as f:
            dataset2token_cnt = json.load(f)
        estimated_token2cnt = defaultdict(int)
        for dataset in tqdm.tqdm(list(involved_datasets),desc='[0/4]'):
            for token in dataset2token_cnt[dataset]:
                estimated_token2cnt[token] += dataset2token_cnt[dataset][token]

        with open(f'{base_dir}/tokenizer_info/target_tokenizer-{vocab_size}.json', 'r') as f:
            training_info = json.load(f)

        for topk in [60000, 40000, 20000]:
            sorted_tokens = [k for k, v in sorted(target_vocab.items(), key=lambda item: item[1], reverse=True)]
            token_set = sorted_tokens[:topk]
            dataset2score = dict()
            for dataset in tqdm.tqdm(datasets):
                score = 1
                if dataset not in involved_datasets:
                    for token in token_set:
                        denom = dataset2token_cnt[dataset].get(token,0)+estimated_token2cnt[token]
                        if denom != 0:
                            score *= (1-dataset2token_cnt[dataset].get(token,0)/denom)
                else:
                    for token in token_set:
                        denom = estimated_token2cnt[token]
                        if denom != 0:
                            score *= (1-dataset2token_cnt[dataset].get(token,0)/denom)
                dataset2score[dataset] = mpf(1)  -  score

            details = {'members':[],'non-members':[]}
            y_pred, y_true = [], []
            for dataset in tqdm.tqdm(training_info['member_datasets'], desc='[3/4]'):
                pred_score = (dataset2score[dataset])
                y_pred.append(pred_score)
                y_true.append(1)
                details['members'].append([dataset, float(pred_score)])
            for dataset in tqdm.tqdm(training_info['non_member_datasets'], desc='[4/4]'):
                pred_score = (dataset2score[dataset])
                y_pred.append(pred_score)
                y_true.append(0)
                details['non-members'].append([dataset, float(pred_score)])

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            tpr_at_low_fpr = compute_tpr_at_fpr_threshold(y_true, y_pred, target_fpr=0.01)
            balanced_accuracy = np.max(1 - (fpr + (1 - tpr)) / 2)

            result = {
                'roc_auc': roc_auc,
                'balanced_accuracy': balanced_accuracy,
                'tpr_at_low_fpr': tpr_at_low_fpr,
                'fpr': fpr.tolist(), 
                'tpr': tpr.tolist(),  
                'details': details
            }
            print(f'roc_auc : {roc_auc:.4f}, balanced_accuracy : {balanced_accuracy:.4f}, tpr@fpr<1% : {tpr_at_low_fpr:.4f}')
            with open(f'{base_dir}/infer_results/MIA via Naive Bayes - K_{topk} - v_size_{vocab_size}.json', 'w') as f:
                json.dump(result, f, indent=4)
