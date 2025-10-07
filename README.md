# Membership Inference Attacks on Tokenizers of Large Language Models

Code for the submission "Membership Inference Attacks on Tokenizers of Large Language Models"

Note that this repo is anonymous and only intended for review purpose only.

## Implementation Steps

### Step 0. Install Required Packages

First, set up the Python environment and install all required dependencies.

```shell
conda create -n MIA python=3.10
conda activate MIA
pip install -r requirements.txt
```

### Step 1. Download Datasets for Evaluation

Next, download the datasets used in our evaluations. These datasets have been collected by Google

```shell
python download_datasets.py
```

### Step 2. Train Target Tokenizers

In this step, train the target tokenizers, which serve as the attack targets in MIA experiments.

```shell
python train_target_tokenizer.py
```

### Step 3. Train Shadow Tokenizers

Shadow tokenizers are trained to mimic the behavior of the target tokenizer. These are used in the attack phase to help infer membership.

```shell
python train_shadow_tokenizer.py
```

### Step 4. Perform Membership Inference Attacks

Now, conduct membership inference attacks using various methods. Each script below implements a different attack method.

```shell
python mia_via_compression_rate.py
python mia_via_vocabulary_overlap.py
python mia_via_frequency_estimation.py
python mia_via_merge_similarity.py
python mia_via_naive_bayes.py


```

All experimental results will be saved in the **infer_results** folder for further analysis.
