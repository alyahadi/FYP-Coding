from transformers import BartTokenizer, BartForConditionalGeneration
import os
import re

# Initialize BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def bart_summarize_with_keywords(
    text, topic, keywords,
    max_length=250, min_length=50,
    num_beams=5, length_penalty=2.0,
    num_sentences=5         
):
    """Summarizes text, emphasizing keywords, then truncates to N sentences."""
    keyword_str  = ", ".join(keywords)
    prompt       = f"Summarize the key aspects of {topic}, focusing on these keywords: {keyword_str}."
    combined     = prompt + " " + text
    inputs       = tokenizer([combined],
                              max_length=1024,
                              truncation=True,
                              return_tensors="pt")
    summary_ids  = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        num_beams=num_beams,
        early_stopping=True
    )
    raw_summary  = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    raw_summary  = re.sub(r"Read more at CNN\.com.*", "", raw_summary)

    # ---- new: split into sentences and take only the first num_sentences ----
    # A simple regex-based splitter:
    sentences = re.split(r'(?<=[\.\!\?])\s+', raw_summary.strip())
    limited   = sentences[:num_sentences]
    summary   = " ".join(limited).strip()

    return summary

def extract_custom_topics(text):
    pattern = r'---\s*(.*?)\s*---\n(.*?)(?=(\n---)|\Z)'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    topics = {}
    for topic, content, _ in matches:
        topics[topic.strip()] = content.strip()
    return topics

input_dir = "output"
output_dir = "BART Keywords"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        input_path = os.path.join(input_dir, file)
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        topics = extract_custom_topics(text)

        summary_text = ""
        for topic, content in topics.items():
            if content.strip():
                # Define keywords here, based on the topic.
                if topic == "Family and Lineage Overview":
                    keywords = ["called", "name", "lineage", "tribe", "family", "cousin", "uncle", "sister", "mother", "father", "born", "Banu"]
                elif topic == "Conversion and Early Islam":
                    keywords = ["conversion", "embrace", "accept"]
                elif topic == "Persecution":
                    keywords = ["persecution", "torture", "beat"]
                elif topic == "Hijra":
                    keywords = ["migration", "Ansar", "Muhajireen", "Medina", "hijrah", "Abyssinia"]
                elif topic == "Battle":
                    keywords = ["battle", "fight", "mission", "war", "sword", "horse", "Badr", "Uhud", "Yarmouk", "Tabuk", "Mu'tah", "Khaybar", "Khandaq"]
                elif topic == "Quality or Trait":
                    keywords = ["brave", "courage", "generous", "honor", "virtue", "the first"]
                elif topic == "Death":
                    keywords = ["death", "martyr", "shaheed", "killed", "dies", "passed away", "Amwas", "plague"]
                else:
                    keywords = [] #if no keywords are defined.

                summary = bart_summarize_with_keywords(content, topic, keywords)
                summary_text += f"--- {topic} Summary ---\n{summary}\n\n"
            else:
                summary_text += f"--- {topic} Summary ---\nNo summary available due to lack of information.\n\n"

        # Write out
        stem = os.path.splitext(file)[0]                # e.g. "Abu Ubaidah Al-Jarrah_output"
        # strip off the "_output" suffix if present
        base = stem[:-len("_output")] if stem.endswith("_output") else stem  
        # add your new suffix
        output_file = os.path.join(output_dir, f"{base}_bart.txt")
        
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(summary_text)

        print(f"BART Summary written to {output_file}")