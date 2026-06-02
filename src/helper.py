import re
import pandas as pd
from pathlib import Path
import zipfile
from collections import defaultdict

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
        
        
        
def generate_association_rules(
    frequent_itemsets,          
    global_counts=None,         
) -> pd.DataFrame:
    
    if isinstance(frequent_itemsets, dict):
        counts = frequent_itemsets
        itemsets = set(frequent_itemsets.keys())
    else:
        if global_counts is None:
            raise ValueError("global_counts must be provided when frequent_itemsets is a set")
        counts = global_counts
        itemsets = set(frequent_itemsets)

    rules = []
    for itemset in itemsets:
        if len(itemset) < 2:
            continue
        for item in itemset:
            rhs = frozenset([item])
            lhs = itemset - rhs
            if lhs not in counts:
                continue
            support = counts[itemset]
            confidence = support / counts[lhs]
            rules.append({
                "antecedents": set(lhs),
                "consequents": set(rhs),
                "support":     support,
                "confidence":  round(confidence * 100, 2),
            })

    df = pd.DataFrame(rules, columns=["antecedents", "consequents", "support", "confidence"])
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    return df