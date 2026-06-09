import pandas as pd
import numpy as np
from itertools import combinations
import math
import random
from collections import defaultdict
from typing import Iterable
import itertools


def triangular_matrix_method(baskets, item_filter=None):
    all_items = sorted({
        item for basket in baskets for item in basket
        if item_filter is None or item in item_filter
    })
    item_to_idx = {item: i+1 for i, item in enumerate(all_items)}
    n = len(all_items)

    size = n * (n - 1) // 2
    a = [0] * (size + 1)

    def get_index(i, j):
        if i > j:
            i, j = j, i
        # Book formula: k = (i-1)(n - i/2) + (j-i)
        # i*(i-1) is always even (consecutive integers), so integer division is exact.
        return (i - 1) * n - i * (i - 1) // 2 + (j - i)

    for basket in baskets:
        indices = sorted([
            item_to_idx[item] for item in basket
            if item_filter is None or item in item_filter
        ])
        for i, j in combinations(indices, 2):
            k = get_index(i, j)
            a[k] += 1

    result = {}
    for item_i, item_j in combinations(all_items, 2):
        i, j = item_to_idx[item_i], item_to_idx[item_j]
        k = get_index(i, j)
        if a[k] > 0:
            result[frozenset({item_i, item_j})] = a[k]

    return result


def triples_method(baskets, item_filter=None, pair_filter=None):
    triples = {}

    for basket in baskets:
        items = [item for item in basket if item_filter is None or item in item_filter]
        for item_i, item_j in combinations(sorted(items), 2):
            key = frozenset({item_i, item_j})
            if pair_filter is not None and not pair_filter(key):
                continue
            triples[key] = triples.get(key, 0) + 1

    triples = dict(sorted(triples.items(), key=lambda x: x[1], reverse=True))
    return triples


def apriori(baskets, support_threshold, verbose=False):

    if verbose: print("Pass 1 : Calculation occureness of items")
    singleton_counts = {}
    for basket in baskets:
        for item in basket:
            singleton_counts[item] = singleton_counts.get(item, 0) + 1

    if verbose: print('Occureness of Items:', singleton_counts)

    L1 = {frozenset([item]) for item, count in singleton_counts.items() if count >= support_threshold}

    if verbose: print(f"\n=== BETWEEN PASSES: L1 (frequent singletons, support >= {support_threshold}) ===")
    if verbose: print(f"  {[set(s) for s in L1]}\n")

    if not L1:
        return {}

    frequent_itemsets = {1: L1}
    support_counts = {frozenset([item]): count for item, count in singleton_counts.items() if count >= support_threshold}

    k = 2

    while True:

        Lk_prev = frequent_itemsets[k - 1]
        Ck_counts = {}

        if k == 2:
            frequent_items_set = {item for fs in L1 for item in fs}
            Ck_counts = triples_method(baskets, item_filter=frequent_items_set)
        else:
            for basket in baskets:
                # Start with items in this basket that are frequent singletons
                frequent_in_basket = [item for item in basket if frozenset([item]) in L1]
                basket_set = set(frequent_in_basket)

                # Book (Example 6.8): eliminate any item that does not appear in
                # at least two frequent (k-1)-itemsets that are subsets of this basket.
                # Only items surviving this pruning can contribute to a frequent k-set.
                eligible = [
                    item for item in frequent_in_basket
                    if sum(
                        1 for s in Lk_prev
                        if item in s and s <= basket_set
                    ) >= 2
                ]

                for candidate in combinations(sorted(eligible), k):
                    subsets = [frozenset(candidate) - {item} for item in candidate]
                    if all(s in Lk_prev for s in subsets):
                        key = frozenset(candidate)
                        Ck_counts[key] = Ck_counts.get(key, 0) + 1

        if verbose:
            print(f"=== PASS {k}: Candidate itemsets C{k} counts ===")
            for itemset, count in sorted(Ck_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {set(itemset)}: {count}")

        Lk = {itemset for itemset, count in Ck_counts.items() if count >= support_threshold}

        if verbose: print(f"\n=== BETWEEN PASSES: L{k} (frequent {k}-itemsets, support >= {support_threshold}) ===")
        if verbose: print(f"  {[set(s) for s in Lk]}\n")

        if not Lk:
            if verbose: print(f"  No frequent {k}-itemsets found. Stopping.")
            break

        for itemset in Lk:
            support_counts[itemset] = Ck_counts[itemset]

        frequent_itemsets[k] = Lk
        k += 1

    return support_counts



def _default_hash_functions(num_tables: int, num_buckets: int, seed: int = 42):
    """
    hash: h(x) = (a*x + b) mod p mod B
    """

    p = (1 << 31) - 1  # 2^31 - 1

    rng = random.Random(seed)

    def make_fn(table_idx: int):
        a = rng.randint(1, p - 1)
        b = rng.randint(0, p - 1)

        def _hash(pair: frozenset) -> int:
            x, y = sorted(str(item) for item in pair)
            combined = hash((x, y))
            return ((a * combined + b) % p) % num_buckets
        return _hash

    return [make_fn(t) for t in range(num_tables)]



def multihash_algorithm(
    baskets:         list[set],
    support:         int,
    num_hash_tables: int = 2,
    num_buckets:     int | None = None,
) -> dict[frozenset, int]:

    if not baskets:
        return {}

    if num_buckets is None:
        total_pairs = sum(math.comb(len(b), 2) for b in baskets)
        num_buckets = max(11, total_pairs // max(1, support) * 2 + 1)
        if num_buckets % 2 == 0:
            num_buckets += 1

    hash_fns = _default_hash_functions(num_hash_tables, num_buckets)

    # Pass 1
    item_counts   = defaultdict(int)
    bucket_counts = [[0] * num_buckets for _ in range(num_hash_tables)]

    for basket in baskets:
        for item in basket:
            item_counts[item] += 1
        items = list(basket)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair = frozenset({items[i], items[j]})
                for t, hfn in enumerate(hash_fns):
                    bucket_counts[t][hfn(pair)] += 1

    # Between passes : 1st pass <-> 2nd pass
    frequent_items   = {item for item, cnt in item_counts.items()
                        if cnt >= support}
    frequent_buckets = [
        [cnt >= support for cnt in bucket_counts[t]]
        for t in range(num_hash_tables)
    ]

    # Pass 2: count candidate pairs via triples_method
    pair_filter = lambda pair: all(
        frequent_buckets[t][hash_fns[t](pair)] for t in range(num_hash_tables)
    )
    pair_counts = triples_method(baskets, item_filter=frequent_items, pair_filter=pair_filter)

    frequent_pairs = {pair: cnt for pair, cnt in pair_counts.items()
                      if cnt >= support}

    all_frequent: dict[frozenset, int] = {}

    for item in frequent_items:
        all_frequent[frozenset({item})] = item_counts[item]

    all_frequent.update(frequent_pairs)

    # k>=3
    def generate_candidates(prev_frequent: dict[frozenset, int], k: int) -> set[frozenset]:
        candidates = set()
        prev_list  = list(prev_frequent.keys())

        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                union = prev_list[i] | prev_list[j]
                if len(union) == k:
                    if not union <= frequent_items:
                        continue
                    if all(frozenset(sub) in prev_frequent
                           for sub in itertools.combinations(union, k - 1)):
                        candidates.add(frozenset(union))
        return candidates

    prev_frequent_dict = frequent_pairs
    k = 3

    while prev_frequent_dict:
        candidates = generate_candidates(prev_frequent_dict, k)
        if not candidates:
            break

        candidate_counts = defaultdict(int)
        for basket in baskets:
            basket_fs = frozenset(basket)
            for candidate in candidates:
                if candidate <= basket_fs:
                    candidate_counts[candidate] += 1

        new_frequent = {c: cnt for c, cnt in candidate_counts.items()
                        if cnt >= support}
        all_frequent.update(new_frequent)
        prev_frequent_dict = new_frequent
        k += 1

    return all_frequent


def son_algorithm(
    baskets:           list[set],
    support_threshold: int,
    num_chunks:        int = 5,
    num_hash_tables:   int = 2,
    num_buckets:       int | None = None,
) -> dict:

    n          = len(baskets)
    chunk_size = max(1, n // num_chunks)
    p          = chunk_size / n

    local_support = max(1, int(math.floor(p * support_threshold)))

    all_candidates: set[frozenset] = set()

    chunks = [
        baskets[i : i + chunk_size]
        for i in range(0, n, chunk_size)
    ]

    for chunk in chunks:
        if not chunk:
            continue
        local_frequent = multihash_algorithm(
            chunk,
            support=local_support,
            num_hash_tables=num_hash_tables,
            num_buckets=num_buckets,
        )
        all_candidates |= local_frequent.keys()

    # PASS 2: count every candidate over the FULL dataset
    global_counts: dict[frozenset, int] = defaultdict(int)

    for basket in baskets:
        basket_set = frozenset(basket)
        for candidate in all_candidates:
            if candidate <= basket_set:
                global_counts[candidate] += 1

    frequent_itemsets = {
        c for c, cnt in global_counts.items()
        if cnt >= support_threshold
    }

    false_positives = all_candidates - frequent_itemsets

    return {
        "frequent_itemsets" : frequent_itemsets,
        "candidates"        : all_candidates,
        "global_counts"     : dict(global_counts),
        "false_positives"   : false_positives,
        "num_chunks"        : len(chunks),
        "local_support"     : local_support,
    }


def son_mapreduce(
    baskets:           list[set],
    support_threshold: int,
    num_chunks:        int = 5,
    num_hash_tables:   int = 2,
    num_buckets:       int | None = None,
) -> dict:

    n          = len(baskets)
    chunk_size = max(1, n // num_chunks)
    p          = chunk_size / n

    local_support = max(1, int(math.floor(p * support_threshold)))

    chunks = [
        baskets[i : i + chunk_size]
        for i in range(0, n, chunk_size)
    ]

    # MapReduce Job 1

    # PHASE 1 - MAP
    phase1_mapped: list[list[tuple[frozenset, int]]] = []

    for chunk in chunks:
        if not chunk:
            continue
        local_frequent = multihash_algorithm(
            chunk,
            support=local_support,
            num_hash_tables=num_hash_tables,
            num_buckets=num_buckets,
        )
        phase1_mapped.append([(itemset, 1) for itemset in local_frequent])

    # PHASE 1 - REDUCE
    all_candidates: set[frozenset] = set()

    for kv_pairs in phase1_mapped:
        for itemset, _ in kv_pairs:
            all_candidates.add(itemset)

    # MapReduce Job 2

    # PHASE 2 - MAP
    phase2_mapped: list[list[tuple[frozenset, int]]] = []

    for chunk in chunks:
        local_counts: dict[frozenset, int] = defaultdict(int)
        for basket in chunk:
            basket_set = frozenset(basket)
            for candidate in all_candidates:
                if candidate <= basket_set:
                    local_counts[candidate] += 1
        phase2_mapped.append(
            [(candidate, count) for candidate, count in local_counts.items() if count > 0]
        )

    # PHASE 2 - REDUCE
    global_counts: dict[frozenset, int] = defaultdict(int)

    for kv_pairs in phase2_mapped:
        for candidate, count in kv_pairs:
            global_counts[candidate] += count

    frequent_itemsets = {
        c for c, cnt in global_counts.items()
        if cnt >= support_threshold
    }

    false_positives = all_candidates - frequent_itemsets

    return {
        "frequent_itemsets" : frequent_itemsets,
        "candidates"        : all_candidates,
        "global_counts"     : dict(global_counts),
        "false_positives"   : false_positives,
        "num_chunks"        : len(chunks),
        "local_support"     : local_support,
    }