from nltk.tokenize import sent_tokenize
import os
import re

def naive_first_n_sentences(text, n=5):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:n])

def extract_custom_topics(text):
    # This regex pattern assumes that each topic block starts with a line like:
    # --- Topic Name ---
    # and continues until the next topic marker (or end of text)
    pattern = r'---\s*(.*?)\s*---\n(.*?)(?=(\n---)|\Z)'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    topics = {}
    for topic, content, _ in matches:
        topics[topic.strip()] = content.strip()
    return topics

input_dir = "output"
output_dir = "Naive Summary"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            text = f.read()

        topics = extract_custom_topics(text)  # same function you use for other models
        summary_text = ""

        for topic, content in topics.items():
            summary_text += f"--- {topic} Summary ---\n"
            if content.strip():
                summary = naive_first_n_sentences(content, n=3)
                summary_text += summary + "\n\n"
            else:
                summary_text += "No content available.\n\n"

        out_path = os.path.join(output_dir, file.replace("_output.txt", "_naive.txt"))
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(summary_text)

        print(f"Naive summary written to {out_path}")

