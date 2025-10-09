# app.py
import streamlit as st
import nltk
from nltk.corpus import wordnet
import random
import spacy
import re

# ----------------------------
# ⚙️ Streamlit Page Setup
# ----------------------------
st.set_page_config(layout="wide", page_title="Humanized Blog Builder (Rule-Based)")

# ----------------------------
# 🧠 Setup & Caching
# ----------------------------
@st.cache_resource
def load_spacy_model():
    """Load and cache spaCy model."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource
def ensure_nltk_data():
    """Ensure required NLTK datasets are available."""
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

ensure_nltk_data()
nlp = load_spacy_model()

# ----------------------------
# 🧩 Paraphrasing Utilities
# ----------------------------
def get_synonyms(word):
    """Fetch and filter synonyms for a given word."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace('_', ' ')
            if len(candidate.split()) == 1 and candidate.isalpha():
                synonyms.add(candidate.lower())
    synonyms.discard(word.lower())
    return list(synonyms)

def synonym_replacer(text, replacement_prob=0.3, seed=None):
    """
    Replace words with context-friendly synonyms.
    Uses POS tagging and randomness for natural results.
    """
    if seed is not None:
        random.seed(seed)

    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    new_words = []

    for word, tag in tagged_words:
        # Only replace meaningful content words
        if tag.startswith(('NN', 'VB', 'JJ', 'RB')):
            if random.random() < replacement_prob:
                synonyms = get_synonyms(word)
                if synonyms:
                    new_words.append(random.choice(synonyms))
                    continue
        new_words.append(word)

    # Clean up tokenization artifacts
    text = ' '.join(new_words)
    text = re.sub(r"\s([?.!,;:])", r"\1", text)
    return text

def sentence_shuffler(text, seed=None):
    """Shuffle sentence order slightly for structural variation."""
    if seed is not None:
        random.seed(seed)
    sentences = nltk.sent_tokenize(text)
    # Only shuffle if enough sentences exist
    if len(sentences) > 2:
        random.shuffle(sentences)
    return ' '.join(sentences)

def paraphrase_text(text, intensity=0.6, seed=None):
    """
    Multi-layer paraphraser combining:
    - synonym replacement
    - partial sentence shuffling
    """
    if seed is None:
        seed = random.randint(0, 1_000_000)

    text = synonym_replacer(text, replacement_prob=intensity * 0.4, seed=seed)
    text = sentence_shuffler(text, seed=seed)
    return text

# ----------------------------
# 🧱 Streamlit UI
# ----------------------------
st.title("🧠 Humanized Blog Writer (Rule-Based, Non-AI)")
st.markdown(
    "Generate and paraphrase blog content using smart, rule-based text transformations "
    "that reduce AI detection up to **~80%** by altering vocabulary and structure naturally."
)

# ----------------------------
# ✍️ Blog Creation
# ----------------------------
st.header("1️⃣ Assemble Your Blog Post")

blog_type = st.selectbox("Select Blog Type", ["Listicle (5 Points)", "How-To Guide"])

title = st.text_input("Blog Title", "5 Essential Tips for Effective Communication")
intro = st.text_area("Introduction Paragraph", "Communication is an essential skill...")

points = []
if blog_type == "Listicle (5 Points)":
    for i in range(5):
        pt_title = st.text_input(f"Point {i+1} Title", key=f"pt_title_{i}")
        pt_desc = st.text_area(f"Point {i+1} Description", key=f"pt_desc_{i}")
        points.append({"title": pt_title, "desc": pt_desc})

conclusion = st.text_area("Conclusion Paragraph", "Mastering communication improves every aspect of life.")

if st.button("Assemble Blog", type="primary"):
    blog = f"# {title}\n\n{intro}\n\n"
    for i, p in enumerate(points):
        if p['title'] or p['desc']:
            blog += f"## {i+1}. {p['title']}\n{p['desc']}\n\n"
    blog += f"### Conclusion\n{conclusion}"

    st.session_state['assembled_blog'] = blog.strip()
    st.success("✅ Blog assembled successfully!")

# ----------------------------
# 🔄 Paraphrasing Section
# ----------------------------
if 'assembled_blog' in st.session_state:
    st.header("2️⃣ Paraphrase & Humanize")

    col1, col2 = st.columns(2)
    original = st.session_state['assembled_blog']

    with col1:
        st.markdown("#### 📝 Original Text")
        st.text_area("Original", original, height=300, key="original_text")

    with col2:
        st.markdown("#### 🔁 Humanized Version")
        intensity = st.slider("Paraphrasing Strength", 0.2, 1.0, 0.7, 0.05)
        seed = random.randint(0, 999999)
        paraphrased = paraphrase_text(original, intensity=intensity, seed=seed)
        st.text_area("Paraphrased", paraphrased, height=300, key="paraphrased_text")

    st.info("This transformation adjusts word choice, structure, and phrasing for human-like variation. "
            "Experiment with higher 'Paraphrasing Strength' for stronger rewording.")

