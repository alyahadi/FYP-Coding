import glob
from evaluate import load

# pip install evaluate
# pip install bert-score

def compute_bertscore(ref_pattern, pred_pattern, lang="en"):
    # 1) collect & sort paths
    ref_paths  = sorted(glob.glob(ref_pattern))
    pred_paths = sorted(glob.glob(pred_pattern))

    # 2) read them in
    references  = [open(fp,  encoding="utf-8").read().strip() for fp in ref_paths]
    predictions = [open(fp,  encoding="utf-8").read().strip() for fp in pred_paths]

    # 3) load and compute
    bs = load("bertscore")
    results = bs.compute(predictions=predictions, references=references, lang=lang)

    # 4) average each list
    avg = lambda key: sum(results[key]) / len(results[key])
    return avg("precision"), avg("recall"), avg("f1")


# map each method to its summaries folder
configs = {
    "LexRank"  : ("Gold Summary/*.txt",       "Lexrank Summary/*.txt"),
    "BART"     : ("Gold Summary/*.txt",       "BART Keywords/*.txt"),
    "Pegasus"  : ("Gold Summary/*.txt",       "Pegasus Keywords/*.txt"),
    "BERT"  : ("Gold Summary/*.txt",       "BERT Summaries/*.txt"),
    "Naive"    : ("Gold Summary/*.txt",       "Naive Summary/*.txt"),
}

print("→ BERTScore ←")
for name, (ref_pat, pred_pat) in configs.items():
    p, r, f1 = compute_bertscore(ref_pat, pred_pat)
    print(f"{name:>9} →  P: {p:.3f}   R: {r:.3f}   F1: {f1:.3f}")



#      BART →  P: 0.841   R: 0.836   F1: 0.839
#      BERT →  P: 0.826   R: 0.830   F1: 0.828
#   Pegasus →  P: 0.827   R: 0.831   F1: 0.829
# Pegasus_ZeroShot →  P: 0.823   R: 0.833   F1: 0.828
#   LexRank →  P: 0.822   R: 0.822   F1: 0.822