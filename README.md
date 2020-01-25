# Protective optimization techonologies: case studies

This is the accompanying code to the paper "[POTS: Protective Optimization Technologies](https://arxiv.org/abs/1806.02711)".

Cite as follows:
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
