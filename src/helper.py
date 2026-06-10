import re
import pandas as pd
from pathlib import Path
import zipfile
from collections import defaultdict
from itertools import combinations


def kaggle_data_handler(kaggle_url, csv_index=0):
    match = re.search(r"kaggle\.com/datasets/([^/]+)/([^/?#]+)", kaggle_url)
    if not match:
        raise ValueError(f"Could not parse Kaggle dataset URL: {kaggle_url}")

    owner, slug = match.group(1), match.group(2)

    try:
        download_dir = Path(__file__).parent.parent / "data"
    except NameError:
        download_dir = Path.cwd() / "data"

    existing_csvs = sorted(download_dir.glob("*.csv")) if download_dir.exists() else []

    if not existing_csvs:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        download_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(f"{owner}/{slug}", path=str(download_dir), unzip=False, quiet=False)

        for z in download_dir.glob("*.zip"):
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(download_dir)
            z.unlink()

        existing_csvs = sorted(download_dir.glob("*.csv"))

    return pd.read_csv(existing_csvs[csv_index])


def print_results(result: dict, top_n: int = 20) -> None:
    fi      = result["frequent_itemsets"]
    cands   = result["candidates"]
    fp      = result["false_positives"]
    counts  = result["global_counts"]

    print("=" * 60)
    print("Results")
    print("=" * 60)
    effective = result.get("effective_chunks", result["num_chunks"])
    if effective < result["num_chunks"]:
        print(f"  Chunks used : {effective} (capped from {result['num_chunks']} — support too low)")
    else:
        print(f"  Chunks used : {result['num_chunks']}")
    print(f"  Local support : {result['local_support']}")
    print(f"  Candidate itemsets : {len(cands)}")
    print(f"  False positives : {len(fp)}  (eliminated in pass 2)")
    print(f"  Truly frequent : {len(fi)}")
    print()

    sorted_fi = sorted(fi, key=lambda s: (len(s), -counts.get(s, 0)))
    by_size: dict = defaultdict(list)
    for s in sorted_fi:
        by_size[len(s)].append(s)

    for size in sorted(by_size):
        label = {1: "Singletons", 2: "Pairs", 3: "Triples"}.get(
            size, f"Size-{size} sets"
        )
        items_at_size = sorted(by_size[size], key=lambda s: -counts.get(s, 0))
        print(f"  {label} ({len(items_at_size)} frequent):")
        for itemset in items_at_size[:top_n]:
            cnt = counts.get(itemset, 0)
            items_str = ", ".join(str(x) for x in sorted(itemset))
            print(f"    {{{items_str}}}  [support={cnt}]")
        if len(items_at_size) > top_n:
            print(f"    ... and {len(items_at_size) - top_n} more")
        print()
        
        
        


def support(itemset, baskets) -> float:
    """Fraction of baskets that contain every item in itemset."""
    itemset = frozenset(itemset)
    return sum(1 for b in baskets if itemset.issubset(b)) / len(baskets)



def confidence(lhs, rhs, baskets, verbose=False) -> float | None:
    """P(lhs ∪ rhs) / P(lhs)."""
    lhs, rhs = frozenset(lhs), frozenset(rhs)
    sup_lhs   = sum(1 for b in baskets if lhs.issubset(b))
    sup_union = sum(1 for b in baskets if (lhs | rhs).issubset(b))

    if sup_lhs == 0:
        return None

    conf = sup_union / sup_lhs

    if verbose:
        print(f"  support({set(lhs)})          = {sup_lhs}")
        print(f"  support({set(lhs | rhs)})    = {sup_union}")
        print(f"  confidence({set(lhs)} → {set(rhs)}) "
              f"= {sup_union}/{sup_lhs} = {conf:.4f}")
    return conf



def interest(lhs, rhs, baskets, verbose=False) -> float | None:
    """confidence(lhs → rhs) − P(rhs)."""
    rhs  = frozenset(rhs)
    conf = confidence(lhs, rhs, baskets, verbose=verbose)

    if conf is None:
        return None

    sup_rhs  = sum(1 for b in baskets if rhs.issubset(b))
    prob_rhs = sup_rhs / len(baskets)
    score    = conf - prob_rhs

    if verbose:
        print(f"  P({set(rhs)})"
              f"= {sup_rhs}/{len(baskets)} = {prob_rhs:.4f}")
        print(f"  interest({set(lhs)} → {set(rhs)}) "
              f"= {conf:.4f} - {prob_rhs:.4f} = {score:.4f}")
    return score



def generate_association_rules(
    frequent_itemsets,          
    baskets,                    
    global_counts=None,         
    min_confidence: float = 0.0,
    min_interest:   float | None = None) -> pd.DataFrame:
    
    if isinstance(frequent_itemsets, dict):
        itemsets = set(frequent_itemsets.keys())
    else:
        if global_counts is None:
            raise ValueError(
                "global_counts must be provided when frequent_itemsets is a set"
            )
        itemsets = set(frequent_itemsets)

    n = len(baskets)
    rows = []

    for itemset in itemsets:
        if len(itemset) < 2:
            continue

        # Every single-item consequent derived from this itemset
        for item in itemset:
            rhs = frozenset([item])
            lhs = itemset - rhs

            conf = confidence(lhs, rhs, baskets)
            if conf is None or conf < min_confidence:
                continue

            sup   = support(itemset, baskets)
            score = interest(lhs, rhs, baskets)

            if min_interest is not None and (score is None or score < min_interest):
                continue

            rows.append({
                "antecedents": set(lhs),
                "consequents": set(rhs),
                "support":     round(sup * n),
                "confidence":  round(conf * 100, 2), 
                "interest":    round(score, 4) if score is not None else None,
            })

    df = (
        pd.DataFrame(rows, columns=["antecedents", "consequents",
                                    "support", "confidence", "interest"])
        .sort_values("confidence", ascending=False)
        .reset_index(drop=True)
    )
    return df