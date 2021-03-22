# InFoRM: Individual Fairness on Graph Mining

This is a Python implementation of InFoRM: Individual Fairness on Graph Mining for the task of PageRank, spectral clustering and LINE, as described in our paper:
 
Jian Kang, Jingrui He, Ross Maciejewski, Hanghang Tong. [InFoRM: Individual Fairness on Graph Mining](http://jiank2.web.illinois.edu/files/kdd20/kang20inform.pdf). In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 379-389. 2020 (KDD 2020).

## Requirements
* python 3 (>3.7)
* numpy
* scipy
* sklearn
* networkx

## Data

We provide data used in the paper in `data` folder. Have a look at the `load_graph.py` for your reference.

In the demos, we load PPI dataset.

## Models

We provide three mutually exclusive debiasing method in `method` folder: 
* `debias_graph.py`: Debiasing the input graph. Feel free to override `__init__()` and `fit()` functions to debias your own method.
* `debias_model.py`: Debiasing the mining model. Feel free to override `__init__()` and `fit()` functions to debias your own method.
* `debias_result.py`: Debiasing the mining results.

## Demos

Please check our demos in `demo_{#1}.ipynb` where `{#1}` can be PageRank, spectral_clustering or LINE.

## Reference

Please cite our paper if you use this code in your own work:

```
@inproceedings{kang2020inform,
  title={InFoRM: Individual Fairness on Graph Mining},
  author={Kang, Jian and He, Jingrui and Maciejewski, Ross and Tong, Hanghang},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={379–389},
  year={2020},
  organization={ACM}
}
```