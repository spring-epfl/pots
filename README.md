# Protective optimization techonologies: case studies

This is the accompanying code to the paper "[POTS: Protective Optimization Technologies](https://arxiv.org/abs/1806.02711)".

Cite as follows:
```
@article{pots,
  author    = {Bogdan Kulynych and
               Rebekah Overdorf and
               Carmela Troncoso and
               Seda G\"urses},
  title     = {{POTs: Protective Optimization Technologies}},
  journal   = {CoRR},
  volume    = {abs/1806.02711},
  year      = {2018},
  url       = {http://arxiv.org/abs/1806.02711},
  archivePrefix = {arXiv},
  eprint    = {1806.02711},
}
```


## Installation

### System packages
On a Debian-based system these packages should have you covered:
```
apt install python3 python3-matplotlib python3-numpy python3-scipy python3-sklearn
```

### Python
You need to have Python 3.6 or later. To install the packages, run:
```
pip install -r requirements.txt
```

## Structure

* `src` --- common utilities for credit scoring.
* `scripts` --- a script for running the _poisoning_ experiments
* `notebooks` --- Jupyter notebooks for both evasion and poisoning
* `images` --- after running, the notebooks and scripts save plots here
* `out` --- the scripts saves simulation data here
* `data` --- German credit risk dataset

## Running the poisoning experiments

```
PYTHONPATH=. python scripts/credit_poisoning.py
```

The experiments for evasion run fast, hence they are directly in the corresponding notebook.

## Running the traffic routing experiments

Check the instructions here:
```
PYTHONPATH=. python scripts/anti_waze.py --help
```

Example run:
```
PYTHONPATH=. python scripts/anti_waze.py --town leonia run-one-experiment --target_time 5
```

## Running notebooks

```
jupyter notebook
```
And choose the notebooks in the `notebook` folder
