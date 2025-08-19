# Membership Inference Attacks on Tokenizers of Large Language Models

Code for the Security'26 submission "Membership Inference Attacks on Tokenizers of Large Language Models"

Note that this repo is anonymous and only intended for review purpose only.

## Implement MIAs against tokenizers

### Step 0. Install required packages

Prepare the evaluation environment:

```shell
conda create -n MIA python=3.10
conda activate MIA
pip install -r requirements.txt
```

### Step 1. Download datasets for evaluations

Download datasets collected by Google: 

```shell
python download_datasets.py
```

### Step 2. Train target tokenizers

Train the target tokenizers of LLMs:

```shell
python train_target_tokenizer.py
```

### Step 3. Train shadow tokenizers

Train shadow tokenizers for MIAs: 

```shell
python train_shadow_tokenizer.py
```

### Step 4. MIA via vocabulary overlap

Run the following code to conduct membership inference on tokenizers

```shell
python mia_via_compression_rate.py
python mia_via_vocabulary_overlap.py
python mia_via_frequency_estimation.py
python mia_via_merge_similarity.py
python mia_via_naive_bayes.py
python mia_via_compression_rate.py


```

All the experimental results will be shown in the folder "infer_results".
