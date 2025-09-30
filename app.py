# app.py
import streamlit as st
import nltk
from nltk.corpus import wordnet
import random
import spacy

# --- Streamlit page config ---
st.set_page_config(layout="wide", page_title="Non-AI Blog Assistant")

# --- Setup & Caching ---
@st.cache_resource
def load_spacy_model():
    """Load spaCy model once and cache it."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource
def ensure_nltk_data():
    """Ensure NLTK resources are available."""
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

# Load resources once
ensure_nltk_data()
nlp = load_spacy_model()

# --- Paraphrasing Functions ---
def get_synonyms(word):
    """Find synonyms for a word using NLTK WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    synonyms.discard(word)  # safer removal
    return list(synonyms)

def synonym_replacer(text, replacement_prob=0.2, seed=None):
    """Replaces words with synonyms, controlled randomness."""
    if seed is not None:
        random.seed(seed)  # reproducibility
    
    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    new_words = []

    for word, tag in tagged_words:
        if tag.startswith(('NN', 'VB', 'JJ', 'RB')):
            if random.random() < replacement_prob:
                synonyms = get_synonyms(word)
                if synonyms:
                    new_words.append(random.choice(synonyms))
                    continue
        new_words.append(word)
        
    return ' '.join(new_words).replace(" .", ".").replace(" ,", ",")

def shuffle_sentences(text, seed=None):
    """Shuffles the order of sentences in a paragraph."""
    if seed is not None:
        random.seed(seed)
    sentences = nltk.sent_tokenize(text)
    random.shuffle(sentences)
    return ' '.join(sentences)

# --- UI ---
st.title("🛠️ Rule-Based Blog Assembler & Paraphraser")
st.markdown("This tool uses templates and rule-based transformations (like synonym replacement) to create and modify blog content without generative AI.")

# --- Blog Generator ---
st.header("1. Assemble Your Blog Post")

blog_type = st.selectbox("Choose a Blog Type", ["Listicle (5 Points)", "How-To Guide"])

if blog_type == "Listicle (5 Points)":
    st.subheader("Fill in the details for your Listicle:")
    title = st.text_input("Blog Title", "5 Essential Tips for Great Public Speaking")
    intro = st.text_area("Introduction Paragraph", "Public speaking can be daunting...")

    points = []
    for i in range(5):
        point_title = st.text_input(f"Title for Point {i+1}", key=f"p_title_{i}")
        point_desc = st.text_area(f"Description for Point {i+1}", key=f"p_desc_{i}")
        points.append({"title": point_title, "desc": point_desc})

    conclusion = st.text_area("Conclusion Paragraph", "By keeping these tips in mind...")

    if st.button("Assemble Blog Post", type="primary"):
        assembled_blog = f"# {title}\n\n{intro}\n\n"
        for i, p in enumerate(points):
            assembled_blog += f"## {i+1}. {p['title']}\n{p['desc']}\n\n"
        assembled_blog += f"### Conclusion\n{conclusion}"
        st.session_state['assembled_blog'] = assembled_blog
        st.success("Blog assembled successfully!")

# --- Display & Paraphrase ---
if 'assembled_blog' in st.session_state:
    st.subheader("Assembled Blog Content")
    st.text_area("Generated Content", st.session_state['assembled_blog'], height=300)

    st.header("2. Paraphrase and 'Humanize' the Text")

    text_to_paraphrase = st.session_state['assembled_blog']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Original Text")
        st.text_area("Original", text_to_paraphrase, height=300, key="original_text_display")

    with col2:
        st.markdown("#### Transformed Text")
        # Use session state seed so result doesn’t change on every widget interaction
        if "paraphrase_seed" not in st.session_state:
            st.session_state["paraphrase_seed"] = random.randint(0, 1_000_000)

        paraphrased_text = synonym_replacer(text_to_paraphrase, replacement_prob=0.15, seed=st.session_state["paraphrase_seed"])
        
        st.text_area("Paraphrased", paraphrased_text, height=300, key="paraphrased_text_display")
        
    st.info("The transformed text uses synonym replacement. You can extend this with more rules like sentence shuffling or voice changes.")
