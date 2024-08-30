# Hoaxpedia: A Unified Wikipedia Hoax Articles Dataset
_TL:DR: Hoaxpedia is a Dataset containing Hoax articles collected from Wikipedia and semantically similar Legitimate article in 2 settings: Fulltext and Definition and in 3 splits based on Hoax:Legit ratio (1:2,1:10,1:100)._

We introduce HOAXPEDIA, a collection of 311 hoax articles (from existing literature and official Wikipedia lists), together with semantically similar legitimate articles, which together form a binary text classification dataset aimed at fostering research in automated hoax detection.


## Dataset Description


### Calling the dataset

```python
from datasets import load_dataset
dataset = load_dataset('hsuvaskakoty/hoaxpedia','datasetSetting_datasetSplit')
```

## Functin to extract the Legitimate articles from the dataset

We extract the Legitimate articles from the dataset by calling the following function:

```python
python collect_real.py  --data_path ../data/hoax_unified_v4.csv 
                        --output_path ../data
```

Where the hoax articles are already given in the data folder. The function will extract the Legitimate articles from the dataset and save them in the data folder.

## Classification of Language Model

The following function will call the Language Models in training to classify the articles into Hoax and Legitimate articles.

```python
python classifier.py --input_train_path ../data/INPUT_TRAIN_PATH.csv 
                     --input_test_path ../data/INPUT_TEST.csv 
                     --output_path ../data 
                     --task definition 
                     --huggingface_key <key> 
                     --repo_id <repo_id>

```



## Citation

If you use our dataset, please cite our paper:

```bibtex
@article{borkakoty2024hoaxpedia,
  title={Hoaxpedia: A Unified Wikipedia Hoax Articles Dataset},
  author={Borkakoty, Hsuvas and Espinosa-Anke, Luis},
  journal={arXiv preprint arXiv:2405.02175},
  year={2024}
}
```