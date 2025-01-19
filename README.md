# Parametric RAG

## Overview

**Welcome to the Official Repository of Parametric Retrieval-Augmented Generation (Parametric RAG)!**

This repository contains the code, datasets models used in our paper:
 **"Parametric Retrieval-Augmented Generation"**.

------

#### What is Parametric RAG?

Parametric RAG introduces a new paradigm for retrieval-augmented generation by embedding external knowledge directly into the parametric space of Large Language Models (LLMs). This approach overcomes the limitations of traditional in-context RAG methods by:

- Reducing computational overhead by avoiding large context windows.

- Deeply integrating external knowledge into the Feed-Forward Networks (FFN) of LLMs for improved reasoning and synthesis.

------

#### What’s Included?

- End-to-end implementation of the Parametric RAG pipeline.
- Preprocessed benchmark datasets for experiments and scripts for customizing and adding new datasets.

## Reproduce Paper Results

In the following GitHub repository, we demonstrate how to test the performance of Parametric RAG on various QA datasets. Specifically, follow these steps to run Parametric RAG:

- **Run the Data Augmentation Module**: This step corresponds to Section 3.2.1 *Self-Augmentation* in the original paper, where documents are transformed into a data-augmented dataset.
- **Generate Parametric Representations of Documents**: This step corresponds to Section 3.2.2 *Additional Parameter Training* in the original paper, where additional LoRA parameters are trained.
- **Inference**: Merge the parametric representations of relevant documents, insert them into the LLM, and use the updated LLM for inference.

### Install Environment

```
conda create -n prag python=3.10.4
conda activate prag
pip install torch==1.13.1
pip install -r requirements.txt
```

Please change the `ROOT_DIR` variable in `src/root_dir_path.py` to the folder address where you store PRAG.

### Self-Augmentation

You can directly use the pre-augmented data file `data_aug.tar.gz`. To extract it, run the command `tar -xzvf data_aug.tar.gz` in your terminal.

If you want to perform data augmentation yourself, please process it as follows.

Prepare BM25 for retrieval

1. Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

2. Use Elasticsearch to index the Wikipedia dump

```bash
cd data
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-8.15.0.tar.gz
rm elasticsearch-8.15.0.tar.gz 
cd elasticsearch-8.15.0
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
```

[TODO] Dowload dataset, and save to path `data/{dataset_name}`.

Data Augmentation:

```bash
python3 src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300  \
    --topk 3
```

| `parameter` | `example/options` |
| --- | --- | 
| `model_name` | `llama3.2-1b-instruct`, `qwen2.5-1.5b-instruct`, `llama3-8b-instruct` |
| `dataset` | `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions` |
| `data_path` | `folder to the saved data` |
| `sample` | only augment the first {sample} questions |
| `topk` | retrieval number |

The results of data augmentation will be stored in the file `data_aug/{dataset}/{data_type}.json`.

[TODO] Own dataset
