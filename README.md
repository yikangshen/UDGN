# StructFormer

This repository contains the code used for masked language model and unsupervised parsing experiments in 
[Unsupervised Dependency Graph Network](https://openreview.net/forum?id=yYJhaF4-dZ9) paper.
If you use this code or our results in your research, we'd appreciate if you cite our paper as following:

```
@misc{shen2020unsupervised,
      title={Unsupervised Dependency Graph Network}, 
      author={Yikang Shen, Shawn Tan, Alessandro Sordoni, Peng Li, Jie Zhou, Aaron Courville},
      booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
      year={2022},
}
```

## Software Requirements
Python 3, NLTK, ufal.chu_liu_edmonds and PyTorch are required for the current codebase.

## Steps

1. Install PyTorch and NLTK

2. Download [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42) and [BLLIP](https://catalog.ldc.upenn.edu/LDC2000T43). Put [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42) into NLTK's corpus folder. Put [BLLIP](https://catalog.ldc.upenn.edu/LDC2000T43) into `./data` folder

3. Scripts and commands, from `google-research/`:

  	+  Train Language Modeling
  	```python main.py --cuda --save /path/to/your/model```

  	+ Test Unsupervised Parsing
      ```python test_phrase_grammar.py --cuda --checkpoint /path/to/your/model --print```
    
    The default setting in `main.py` achieves a perplexity of approximately `59.3` on PTB test set, Directed Dependency Accuracy of approximately `49.9` on WSJ test set, and Undirected Dependency Accuracy of approximately `61.8`.
    
## Acknowledgements
Much of our preprocessing and evaluation code is based on the following repository:  
- [Structformer](https://github.com/google-research/google-research/tree/master/structformer)  
