import glob
from evaluate import load

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

# def compute_rouge(ref_pattern, pred_pattern, use_stemmer=True):
#     # 1) collect & sort paths
#     ref_paths  = sorted(glob.glob(ref_pattern))
#     pred_paths = sorted(glob.glob(pred_pattern))

#     # 2) read them in
#     references  = [open(fp, encoding="utf-8").read().strip() for fp in ref_paths]
#     predictions = [open(fp, encoding="utf-8").read().strip() for fp in pred_paths]

#     # 3) load rouge and compute
#     rouge = load("rouge")
#     results = rouge.compute(
#         predictions=predictions,
#         references=references,
#         use_stemmer=use_stemmer
#     )
#     # results is a dict with keys 'rouge1','rouge2','rougeL','rougeLsum'
#     return results

# map each method to its summaries folder
configs = {
    "LexRank"  : ("Gold Summary/*.txt",       "Lexrank Summary/*.txt"),
    "HetFormer": ("Gold Summary/*.txt",       "HetFormer Summaries/*.txt"),
    "BART"     : ("Gold Summary/*.txt",       "BART Keywords/*.txt"),
    "Pegasus"  : ("Gold Summary/*.txt",       "Pegasus Keywords/*.txt"),
    "BERT"  : ("Gold Summary/*.txt",       "BERT Summaries/*.txt"),
    "Pegasus_ZeroShot": ("Gold Summary/*.txt", "Pegasus_ZeroShot/*.txt"),
    "HetFormer 2.0": ("Gold Summary/*.txt",       "HetFormer_Summaries/*.txt"),
}

print("→ BERTScore ←")
for name, (ref_pat, pred_pat) in configs.items():
    p, r, f1 = compute_bertscore(ref_pat, pred_pat)
    print(f"{name:>9} →  P: {p:.3f}   R: {r:.3f}   F1: {f1:.3f}")

# print("→ ROUGE Scores (F1) ←")
# for name, (ref_pat, pred_pat) in configs.items():
#     scores = compute_rouge(ref_pat, pred_pat)
#     print(f"{name:>9} " +
#           f" R1: {scores['rouge1']:.3f} " +
#           f" R2: {scores['rouge2']:.3f} " +
#           f" RL: {scores['rougeL']:.3f}")


#      BART →  P: 0.841   R: 0.836   F1: 0.839
#      BERT →  P: 0.826   R: 0.830   F1: 0.828
#   Pegasus →  P: 0.827   R: 0.831   F1: 0.829
# Pegasus_ZeroShot →  P: 0.823   R: 0.833   F1: 0.828
#   LexRank →  P: 0.822   R: 0.822   F1: 0.822