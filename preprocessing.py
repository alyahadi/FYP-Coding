import os
import re
import glob
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# pip install nltk
# pip install spacy
# python -m spacy download en_core_web_sm


# Download required NLTK resources (first-time only)
nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Step 0: Read .txt files from "FYP Dataset"
file_path_pattern = "FYP Dataset/*.txt"
text_files = glob.glob(file_path_pattern)

if not text_files:
    print(f"No files found using the pattern: {file_path_pattern}")
else:
    print(f"Found {len(text_files)} file(s).")

for file_path in text_files:
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"Processing file: {file_path}")

    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # If empty, note it and skip processing but still write out
    if not text:
        output_lines.append("Warning: file is empty. Skipping further processing.")
    else:
        # Debug: Preview
        output_lines.append("Text preview: " + text[:200])
        output_lines.append("Total text length: " + str(len(text)))

        # Step 1: Remove filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'i mean', 'actually',
                        'basically', 'so', 'okay', 'right']
        for filler in filler_words:
            text = re.sub(r'\b' + re.escape(filler) + r'\b', '', text, flags=re.IGNORECASE)

        # Step 2: Remove redundant phrases & Arabic translit.
        redundant_phrases = [
            "(SAW)", "(subhanAllah)", "(Subhanallah)", "(insha Allah ta'ala)",
            "(inshaAllah ta'ala)", "(insha Allah)", "(SWT)", "(RA)", "(alhamdulillah)",
            "(Allahumma ameen)", "(a.s)", "(As-salamu alaikum wa rahmatullahi wa barakatuhu)",
            "(subhanahu wa ta'ala)", "(said)", "(say)"
        ]
        for phrase in redundant_phrases:
            cleaned = phrase.strip("()")
            pattern = r'\(?\b' + re.escape(cleaned) + r'\)?\b'
            text = re.sub(pattern, '', text)
        text = re.sub(r'\([^)]*[\u0600-\u06FF][^)]*\)', '', text)  # remove Arabic-script in parens

        # Step 3: Sentence segmentation
        sentences = sent_tokenize(text)
        output_lines.append("Number of sentences: " + str(len(sentences)))

        # Step 4: Tokenize each sentence
        tokenized = [word_tokenize(s) for s in sentences]

        # Step 5: Remove stopwords & punctuation
        stop_words = set(stopwords.words('english'))
        punct = set(string.punctuation)
        filtered_tokens = []
        for sent in tokenized:
            filtered = [w for w in sent if w.lower() not in stop_words and w not in punct]
            filtered_tokens.append(filtered)

        # Step 6: Lemmatization using spaCy
        lemmatized_tokens = []
        for tokens in filtered_tokens:
            doc = nlp(" ".join(tokens))
            lemmas = [token.lemma_ for token in doc]
            lemmatized_tokens.append(lemmas)
        filtered_tokens = lemmatized_tokens

        # Custom topic extraction on original sentences
        custom_topics = {
            "Lineage Overview": ["called", "name", "lineage", "tribe", "family", "cousin", "uncle", "sister", "mother", "father", "born", "Banu", "brother"],
            "Conversion and Early Islam": ["conversion", "embrace", "accept"],
            "Persecution": ["persecution", "torture", "beat"],
            "Hijra": ["migration", "Ansar", "Muhajireen", "Medina", "hijrah", "Abyssinia"],
            "Battle": ["battle", "fight", "mission", "war", "sword", "horse", "Badr", "Uhud", "Yarmouk", "Tabuk", "Mu'tah", "Khaybar", "Khandaq"],
            "Virtue(s)": ["brave", "courage", "generous", "honor", "virtue", "the first"],
            "Death": ["death", "martyr", "shaheed", "killed", "dies", "passed away", "Amwas", "plague"]
        }
        topic_sentences = {t: [] for t in custom_topics}
        for sent in sentences:
            low = sent.lower()
            for topic, keywords in custom_topics.items():
                if any(kw in low for kw in keywords):
                    topic_sentences[topic].append(sent)

        # Append topic extraction results
        output_lines.append("\n=== Custom Topic Extraction ===")
        for topic, sents in topic_sentences.items():
            output_lines.append(f"\n--- {topic} ---")
            if sents:
                output_lines.extend(sents)
            else:
                output_lines.append("No sentences matched this topic.")

    # Unified write-out (handles both empty and non-empty cases)
    stem = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(output_dir, f"{stem}_output.txt")
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(output_lines))

    print(f"Output written to {out_path}")
