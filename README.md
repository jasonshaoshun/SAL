# Spectral Removal of Guarded Attribute Information

This repository contains the code for the experiments and algorithm from the paper [Spectral Removal of Guarded Attribute Information](https://arxiv.org/abs/2203.07893) (appears at EACL 2023).

# Introduction

We propose to erase information from neural representations by truncating a singular value decomposition of a covariance matrix between the neural representations and the examples representing the information to be removed or protected attributes. The truncation is done by taking the small singular value principal directions (indicating directions that covary less with the protected attribute).

In addition, we also describe a kernel method to solve the same problem. Rather than performing SVD on the covariance matrix, the kernel method performs a series of spectral operations on the kernel  matrices of the input neural representations and the protected attributes.

# Experimental Setting and Datasets

We use the experimental settings from the paper [&#34;Null it out: guarding protected attributes by iterative nullspsace projection&#34;](https://www.aclweb.org/anthology/2020.acl-main.647/), as use the algorithm in that paper as a benchmark.

# Algorithm

The implementation is available for Python [ksal.py](src/ksal) and [Matlab](src/matlab_version).

Given an example representation of X in the shape of (number of samples, number of dimensions) with a label of biases Z and an optional label for main purpose Y, ksal.py designed to remove the information of Z and we found it is good at keeping the information about Y. We evaluate the biases before and after debiasing by using different classifiers on the pair of (X, Z), tpr-gap between different populations (p(Y=Y'|X,Z)) and some other popular metrics like WEAT.

# Experiments

Start a new virtual environment:

```sh
conda create -n SAL python=3.7 anaconda
conda activate SAL
```

Install jsonnet from conda-forge and other dependencies from requirement.txt

```sh
conda install -c conda-forge jsonnet
pip install -r requirements.txt

```

## Setup

Use the following script to download the datasets used in this repository:

```sh
./download_data.sh
```

Download EN library from spaCy

```sh

```

## Word Embedding Experiments (Section 6.1 in the paper)

```py

python src/data/to_word2vec_format.py data/embeddings/glove.42B.300d.txt

python src/data/filter_vecs.py \
--input-path data/embeddings/glove.42B.300d.txt \
--output-dir data/embeddings/ \
--top-k 150000  \
--keep-inherently-gendered  \
--keep-names 
```

And run the notebook [notebook](notebooks/notebook_word-embedding.ipynb)

To run the Word similarity Experiments (table 1)

Please check the notebook [notebook](notebooks/simlex_SAL.ipynb) for our method, and [notebook](notebooks/simlex_INLP.ipynb) for INLP

## Controlled Demographic experiments (Section 6.2.1 in the paper)

export PYTHONPATH=/path_to/nullspace_projection

```sh
./run_deepmoji_debiasing.sh
```

In order to recreate the evaluation used in the paper, check out the following [sal notebook](notebooks/notebook_fair-sentiment.ipynb) 
<!-- and [ksal notebook](notebooks/notebook_FairClassification_ksal.ipynb) -->

## Bias Bios experiments (Section 6.2.2 in the paper)

Assumes the bias-in-bios dataset from [De-Arteaga, Maria, et al. 2019](https://arxiv.org/abs/1901.09451) saved at `data/biasbios/BIOS.pkl`.

```py
python src/data/create_dataset_biasbios.py \
        --input-path data/biasbios/BIOS.pkl \
        --output-dir data/biasbios/ \
        --vocab-size 250000
```

```sh
./run_bias_bios.sh
```

Run the BERT experiments in [BERT sal notebook](notebooks/notebook_fair-profession_bert.ipynb) 
<!-- and [BERT ksal notebook](notebooks/biasbios_bert_ksal.ipynb) -->

Run the FastText experiments in [FastText sal notebook](notebooks/biasbios_fair-profession_fasttext.ipynb) 
<!-- and [FastText ksal notebook](notebooks/biasbios_fasttext_ksal.ipynb) -->

# Trained Models

We will release the model later
