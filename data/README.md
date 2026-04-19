# Datasets

The synthetic experiments (Exp 1, 2, 3, 5) need no downloads.

Experiment 4 uses real data. Download it here before running `make exp4`:

## MovieLens 1M (primary)

```bash
curl -L https://files.grouplens.org/datasets/movielens/ml-1m.zip -o ml-1m.zip
unzip ml-1m.zip && rm ml-1m.zip
# Creates: data/ml-1m/ratings.dat
```

## MovieLens 100K (fallback if 1M too large)

```bash
curl -L https://files.grouplens.org/datasets/movielens/ml-100k.zip -o ml-100k.zip
unzip ml-100k.zip && rm ml-100k.zip
# Creates: data/ml-100k/u.data
```

## 20 Newsgroups

Fetched automatically by scikit-learn on first use; cached in `~/scikit_learn_data/`.
