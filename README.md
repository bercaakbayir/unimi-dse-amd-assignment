# Frequent Itemset Mining Algorithms: Implementation and Review - Case of IMDB Movie Stars

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bercaakbayir/unimi-dse-amd-assignment/blob/main/main.ipynb)

## Overview

This project is an implementation and comparison of frequent itemset mining algorithms applied to a real-world dataset. The dataset used is the [IMDB Top 1000 Movies](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows) from Kaggle, where each movie is treated as a **basket** containing its four credited actors. The goal is to discover which actors frequently appear together and to generate association rules from these co-occurrences.

## Dataset

- 1,000 movies, each with 4 actors → 1,000 baskets
- 2,709 unique actors
- Support threshold used: **5** (an actor pair must co-occur in at least 5 movies)

## Algorithms Implemented

### A-Priori
The classical approach. It scans the dataset multiple times — once per itemset size — and prunes candidates whose subsets are not frequent. Simple and exact, but gets slow as data grows since every pass reads the full dataset.

### Multihash (PCY variant)
An improvement over A-Priori. In the first pass, it hashes all pairs into multiple hash tables simultaneously. Only pairs that pass all hash table bitmaps are counted in the second pass. This reduces the number of candidates significantly and makes the second pass much faster.

### SON Algorithm
Splits the data into chunks and runs Multihash locally on each chunk with a proportionally scaled support threshold. Any globally frequent itemset must be frequent in at least one chunk — so no frequent itemset is ever missed. A final global pass verifies all candidates and removes false positives.

### SON with MapReduce
The same logic as SON, but both phases run in parallel using Python's `multiprocessing`. Each chunk is processed by a separate worker process. At small scales this is actually slower due to process overhead, but it becomes the most scalable option as data grows.

## Association Rule Extraction Metrics

Each discovered itemset is evaluated with three measures:

- **Support** — how many baskets contain the itemset
- **Confidence** — how likely is actor B to appear given actor A is present
- **Interest** — confidence minus the base probability of actor B (measures how much actor A actually influences actor B's presence)

## Project Structure

```
├── main.ipynb          # Main notebook with all analysis and results
├── src/
│   ├── algorithms.py   # Algorithm implementations
│   └── helper.py       # Data loading, metrics, association rule generation
├── data/
│   └── imdb_top_1000.csv
└── requirements.txt
```

## Requirements

```
pip install -r requirements.txt
```
