# import glob
# from evaluate import load

# def compute_rouge(ref_pattern, pred_pattern, use_stemmer=True):
#     ref_paths  = sorted(glob.glob(ref_pattern))
#     pred_paths = sorted(glob.glob(pred_pattern))

#     references  = [open(fp, encoding="utf-8").read().strip() for fp in ref_paths]
#     predictions = [open(fp, encoding="utf-8").read().strip() for fp in pred_paths]

#     rouge = load("rouge")
#     results = rouge.compute(
#         predictions=predictions,
#         references=references,
#         use_stemmer=use_stemmer
#     )
#     return results  # {'rouge1': F1, 'rouge2': F1, 'rougeL': F1}

# configs = {
#     "LexRank"  : ("Gold Summary/*.txt",       "Lexrank Summary/*.txt"),
#     "BART"     : ("Gold Summary/*.txt",       "BART Keywords/*.txt"),
#     "Pegasus"  : ("Gold Summary/*.txt",       "Pegasus Keywords/*.txt"),
#     "BERT"  : ("Gold Summary/*.txt",       "BERT Summaries/*.txt"),
#     "Naive"    : ("Gold Summary/*.txt",       "Naive Summary/*.txt"),
# }

# print("→ ROUGE Scores (F1) ←")
# for name, (ref_pat, pred_pat) in configs.items():
#     scores = compute_rouge(ref_pat, pred_pat)
#     print(f"{name:>9} " +
#           f" R1: {scores['rouge1']:.3f} " +
#           f" R2: {scores['rouge2']:.3f} " +
#           f" RL: {scores['rougeL']:.3f}")

from rouge_score import rouge_scorer
import glob

# pip install rouge-score
# pip install absl-py
# pip install evaluate

def compute_rouge_metrics(ref_pattern, pred_pattern):
    ref_paths  = sorted(glob.glob(ref_pattern))
    pred_paths = sorted(glob.glob(pred_pattern))

    references  = [open(fp, encoding="utf-8").read().strip() for fp in ref_paths]
    predictions = [open(fp, encoding="utf-8").read().strip() for fp in pred_paths]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = {
        'rouge1': {'precision': [], 'recall': [], 'f1': []},
        'rouge2': {'precision': [], 'recall': [], 'f1': []},
        'rougeL': {'precision': [], 'recall': [], 'f1': []},
    }

    for ref, pred in zip(references, predictions):
        result = scorer.score(ref, pred)
        for metric in scores:
            scores[metric]['precision'].append(result[metric].precision)
            scores[metric]['recall'].append(result[metric].recall)
            scores[metric]['f1'].append(result[metric].fmeasure)

    # Average
    for metric in scores:
        for key in scores[metric]:
            scores[metric][key] = sum(scores[metric][key]) / len(scores[metric][key])

    return scores

configs = {
    "LexRank"  : ("Gold Summary/*.txt",       "Lexrank Summary/*.txt"),
    "BERT"  : ("Gold Summary/*.txt",       "BERT Summaries/*.txt"),
    "BART"     : ("Gold Summary/*.txt",       "BART Keywords/*.txt"),
    "Pegasus"  : ("Gold Summary/*.txt",       "Pegasus Keywords/*.txt"),
    "Naive"    : ("Gold Summary/*.txt",       "Naive Summary/*.txt"),
}

for name, (ref_pat, pred_pat) in configs.items():
    scores = compute_rouge_metrics(ref_pat, pred_pat)
    print(f"{name:>9} | "
          f"{scores['rouge1']['precision']:.3f} {scores['rouge1']['recall']:.3f} {scores['rouge1']['f1']:.3f} | "
          f"{scores['rouge2']['precision']:.3f} {scores['rouge2']['recall']:.3f} {scores['rouge2']['f1']:.3f} | "
          f"{scores['rougeL']['precision']:.3f} {scores['rougeL']['recall']:.3f} {scores['rougeL']['f1']:.3f}")

# ROUGE-1	ROUGE-2	ROUGE-L
# 0.355	0.111	0.163 (LR)
# 0.447	0.121	0.189 (BERT)
# 0.453	0.129	0.203 (BART)
# 0.441	0.121	0.189 (PEGASUS)
# 0.438	0.115	0.186 (Naive)
