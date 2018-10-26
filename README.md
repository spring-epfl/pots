# Protective optimization techonologies: a credit scoring case study

This is the accompanying code to the papers "[POTS: Protective Optimization Technologies](https://arxiv.org/abs/1806.02711)" and "Protective Adversarial Machine Learning".

Cite as follows:
```
@article{pots,
  title={POTs: Protective Optimization Technologies},
  author={Overdorf, Rebekah and Kulynych, Bogdan and Balsa, Ero and Troncoso, Carmela and G{\"u}rses, Seda},
  journal={arXiv preprint arXiv:1806.02711},
  year={2018}
}
```


## Installation

### System packages
On a Debian-based system these packages should have you covered:
```
apt install python3 python3-matplotlib python3-numpy python3-scipy python3-sklearn
```

### Python
You need to have Python 3.5 or later. To install the packages, run:
```
pip install -r requirements.txt
```

## Structure

* `src` --- common utilities for credit scoring.
* `scripts` --- a script for running the _poisoning_ experiments
* `notebooks` --- Jupyter notebooks for both evasion and poisoning
* `images` --- upon running, the notebooks save plots here
* `out` --- the scripts saves simulation data here
* `data` --- German credit risk dataset

## Running the poisoning experiments

```
PYTHONPATH=. python scripts/credit_poisoning.py
```

The experiments for evasion run fast, hence they are directly in the corresponding notebook.

## Running notebooks

```
jupyter notebook
```
And choose the notebooks in the `notebook` folder
