import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

# pip install transformers sentence-transformers scikit-learn

# Load the BERT model (same as summarizer used internally)
model = SentenceTransformer('bert-base-uncased')

def extract_custom_topics(text: str) -> dict:
    """Extracts blocks under --- Topic --- headings into a {topic:content} dict."""
    pattern = r'---\s*(.*?)\s*---\n(.*?)(?=(\n---)|\Z)'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return {topic.strip(): content.strip() for topic, content, _ in matches}

def bert_extractive_summary(text: str, num_sentences: int = 5) -> str:
    """Extract top-N representative sentences using BERT embeddings + cosine similarity."""
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    embeddings = model.encode(sentences)
    sim_matrix = cosine_similarity([np.mean(embeddings, axis=0)], embeddings)[0]
    top_indices = sim_matrix.argsort()[-num_sentences:][::-1]
    top_sentences = [sentences[i] for i in sorted(top_indices)]
    return " ".join(top_sentences)

input_dir = "output"
output_dir = "BERT Summaries"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".txt"):
        continue

    with open(os.path.join(input_dir, filename), encoding="utf-8") as f:
        full_text = f.read()

    topics = extract_custom_topics(full_text)
    summary_output = ""

    for topic, content in topics.items():
        summary_output += f"--- {topic} Summary ---\n"
        if content.strip():
            summary = bert_extractive_summary(content, num_sentences=3)
            summary_output += summary + "\n\n"
        else:
            summary_output += "No content for this topic.\n\n"

    base = filename.replace("_output.txt", "")
    out_path = os.path.join(output_dir, f"{base}_BERT.txt")
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write(summary_output)

    print(f"BERT Summary written to {out_path}")
