# Membership Inference Attacks on Tokenizers of Large Language Models

Code for the Security'26 submission "Membership Inference Attacks on Tokenizers of Large Language Models"

Note that this repo is anonymous and only intended for review purpose only.

## Implementation Steps

### Step 0. Install Required Packages

First, set up the Python environment and install all required dependencies.

```shell
conda create -n MIA python=3.12
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

### Step 5. Min Count Mechanism against MIAs

The code for the min count defense is provided in the 'min_defense' folder. It can be deployed using the following code:
```shell
 python min_defense.py
```

### Step 6. Differentially Private Mechanism against MIAs

We implement the tokenizer training with DP via the modification of Hugging Face's Rust code. It requires a new conda environment. Specifically, the codes can be found in 'dp_defense' folder. We modified code in lines 486-505 of ‘dp_defense\source_code\tokenizers\src\models\bpe\trainer.rs’. The DP training with epsilon=30.0 is as follows:

```shell
conda create -n MIA_dp python=3.12
conda activate MIA_dp
cd dp_defense/source_code/bindings/python/
pip install .
pip install datasets
pip install joblib
pip install mpmath
pip install numpy
pip install powerlaw
pip install scikit_learn
pip install tqdm
cd ../../../../
python train_target_tokenizer.py #Before running, delete the previously trained tokenizers in step 2.
```
