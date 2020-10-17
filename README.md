Code for paper "[Federated Accelerated Stochastic Gradient Descent](https://arxiv.org/abs/2006.08950)", NeurIPS 2020 (best paper in International Workshop on Federated Learning for User Privacy and Data Confidentiality
in Conjunction with ICML 2020 ([FL-ICML'20](http://federated-learning.org/fl-icml-2020/))).

bibtex:
```
@article{yuan2020federated, 
 title={Federated Accelerated Stochastic Gradient Descent}, 
 author={Yuan, Honglin and Ma, Tengyu}, 
 journal={arXiv preprint arXiv:2006.08950}, 
 year={2020} 
 }
```

---

Dependencies:

- python 3.7 with the following packages: numpy, matplotlib, scipy, pandas, sklearn
- Datasets `a9a` and `epsilon` can be downloaded from `https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/`. 
  - please place datasets to `libsvm_datasets` directory. Unzip `epsilon`.

Scripts:

- `logistic.py`: main functions
- `analysis_utils.py`: auxiliary functions for analysis

To reproduce the results in paper: (please be aware that the following scripts can take long time to run.)
- Figure 1, 2: a9a with l2 reg = 1e-3: run `python a9a_1e-03.py`;
- Figure 3, 4: a9a with l2 reg = 1e-2: run `python a9a_1e-02.py`;
- Figure 5, 6: epsilon with l2 reg = 1e-4: run `python epsilon_1e-04.py`;
- Figure 7: a9a with l2 reg = 1e-4: run `python a9a_1e-04.py`.
These commands will generate results in `out` directory.

Alternatively, you can plot the figures directly based on our results in Jupyter notebook `browse_figures.ipynb`.

