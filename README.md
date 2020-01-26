# Protective optimization techonologies: case studies

This is the accompanying code to the paper "[POTS: Protective Optimization Technologies](https://arxiv.org/abs/1806.02711)".

It contains code for the two case studies in the latest version of the paper:

* Poisoning a credit-scoring model.
* Changine speed limits in a town to avoid Waze routing through it.

One POT in this repo that did not make into the paper:

* Evasion/"gaming" of the credit-scoring model.

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

* `scripts` --- a script for running the _poisoning_ credit-scoring experiments, and the traffic
    routing POT.
* `notebooks` --- Jupyter notebooks for evasion and poisoning credit scoring experiments
* `src` --- common utilities and tools.
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

## Citing

```
@inproceedings{KulynychOTG20,
  author    = {Bogdan Kulynych and
               Rebekah Overdorf and
               Carmela Troncoso and
               Seda F. G{\"{u}}rses},
  title     = {POTs: Protective Optimization Technologies},
  booktitle = {FAT* '20: Conference on Fairness, Accountability, and Transparency,
               Barcelona, Spain, January 27-30, 2020},
  pages     = {177--188},
  year      = {2020},
  url       = {https://doi.org/10.1145/3351095.3372853},
  doi       = {10.1145/3351095.3372853},
}
```
