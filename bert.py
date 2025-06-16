# pip install bert-extractive-summarizer

import os, re
from summarizer import Summarizer

# 1) Initialise the Summarizer (uses bert-base-uncased by default)
model = Summarizer()

def extract_custom_topics(text: str) -> dict:
    """Extracts blocks under --- Topic --- headings into a {topic:content} dict."""
    pattern = r'---\s*(.*?)\s*---\n(.*?)(?=(\n---)|\Z)'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return {topic.strip(): content.strip() for topic, content, _ in matches}

def bert_extractive_summary(text: str, num_sentences: int = 5) -> str:
    """Returns the top-N sentences from `text` as a single string."""
    return model(text, num_sentences=num_sentences)

# 3) Process all files
input_dir  = "output"
output_dir = "BERT Summaries"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".txt"):
        continue

    # Read the raw, topic-segmented file
    with open(os.path.join(input_dir, filename), encoding="utf-8") as f:
        full_text = f.read()

    topics = extract_custom_topics(full_text)
    summary_output = ""

    # Summarize each topic block separately
    for topic, content in topics.items():
        summary_output += f"--- {topic} Summary ---\n"
        if content:
            summary = bert_extractive_summary(content, num_sentences=3)
            summary_output += summary + "\n\n"
        else:
            summary_output += "No content for this topic.\n\n"

    # Write out
    stem = os.path.splitext(filename)[0]                # e.g. "Abu Ubaidah Al-Jarrah_output"
    # strip off the "_output" suffix if present
    base = stem[:-len("_output")] if stem.endswith("_output") else stem  
    # add your new suffix
    output_file = os.path.join(output_dir, f"{base}_BERT.txt")
        
    with open(output_file, "w", encoding="utf-8") as out:
            out.write(summary_output)

    print(f"BERT Summary written to {output_file}")
