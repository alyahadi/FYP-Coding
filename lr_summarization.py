import os
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# pip install sumy


# Function to apply LexRank summarization to a block of text.
def lexrank_summarize(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

# Function to extract custom topic blocks using regex.
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

# Define input and output directories.
input_dir = "output"
output_dir = "Lexrank Summary"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each .txt file in the input directory.
for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        input_path = os.path.join(input_dir, file)
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Extract custom topic blocks from the file.
        topics = extract_custom_topics(text)
        
        # Prepare a summary text by applying LexRank to each topic block.
        summary_text = ""
        for topic, content in topics.items():
            summary = lexrank_summarize(content, sentence_count=7)
            summary_text += f"--- {topic} Summary ---\n{summary}\n\n"
        
        # Create a unique output filename for this input file.
        # strip off “.txt”
        stem = os.path.splitext(file)[0]                # e.g. "Abu Ubaidah Al-Jarrah_output"
        # strip off the "_output" suffix if present
        base = stem[:-len("_output")] if stem.endswith("_output") else stem  
        # add your new suffix
        output_file = os.path.join(output_dir, f"{base}_lexrank.txt")
        
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(summary_text)
        
        print(f"LexRank Summary written to {output_file}")
