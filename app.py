# app.py

import streamlit as st
import nltk
from nltk.corpus import wordnet
import random
import spacy

# --- First-time setup for NLTK and spaCy ---
# You need to run this setup once locally, or include it in your deployment setup.
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    st.info("Downloading necessary NLTK data... This will run only once.")
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.info("Downloading necessary spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# --- Paraphrasing and "Humanizing" Functions (Non-AI) ---

def get_synonyms(word):
    """Finds synonyms for a word using NLTK WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacer(text, replacement_prob=0.2):
    """Replaces words in the text with synonyms."""
    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    new_words = []
    
    for word, tag in tagged_words:
        # Only replace nouns, verbs, adverbs, and adjectives to maintain sentence structure
        if tag.startswith(('NN', 'VB', 'JJ', 'RB')):
            if random.random() < replacement_prob:
                synonyms = get_synonyms(word)
                if synonyms:
                    new_word = random.choice(synonyms)
                    new_words.append(new_word)
                    continue
        new_words.append(word)
        
    # Reconstruct the sentence (this is a simplified approach)
    return ' '.join(new_words).replace(" .", ".").replace(" ,", ",")

def shuffle_sentences(text):
    """Shuffles the order of sentences in a paragraph."""
    sentences = nltk.sent_tokenize(text)
    random.shuffle(sentences)
    return ' '.join(sentences)

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Non-AI Blog Assistant")

st.title("🛠️ Rule-Based Blog Assembler & Paraphraser")
st.markdown("This tool uses templates and rule-based transformations (like synonym replacement) to create and modify blog content without generative AI.")

# --- Blog Generation Section ---
st.header("1. Assemble Your Blog Post")

blog_type = st.selectbox("Choose a Blog Type", ["Listicle (5 Points)", "How-To Guide"])

# Template-based input fields
if blog_type == "Listicle (5 Points)":
    st.subheader("Fill in the details for your Listicle:")
    title = st.text_input("Blog Title", "5 Essential Tips for Great Public Speaking")
    intro = st.text_area("Introduction Paragraph", "Public speaking can be daunting, but with a few key strategies, anyone can become a confident and effective speaker. Here are our top five tips to help you master the art.")
    
    points = []
    for i in range(5):
        point_title = st.text_input(f"Title for Point {i+1}", key=f"p_title_{i}")
        point_desc = st.text_area(f"Description for Point {i+1}", key=f"p_desc_{i}")
        points.append({"title": point_title, "desc": point_desc})
        
    conclusion = st.text_area("Conclusion Paragraph", "By keeping these tips in mind, you'll be well on your way to delivering powerful and memorable presentations. Practice is key, so take every opportunity to speak and refine your skills.")

    if st.button("Assemble Blog Post", type="primary"):
        # --- Assemble the blog from the template ---
        assembled_blog = f"# {title}\n\n"
        assembled_blog += f"{intro}\n\n"
        for i, p in enumerate(points):
            assembled_blog += f"## {i+1}. {p['title']}\n"
            assembled_blog += f"{p['desc']}\n\n"
        assembled_blog += f"### Conclusion\n{conclusion}"
        
        st.session_state['assembled_blog'] = assembled_blog
        st.success("Blog assembled successfully! Now you can paraphrase it below.")

# Display assembled blog if it exists
if 'assembled_blog' in st.session_state:
    st.subheader("Assembled Blog Content")
    st.text_area("Generated Content", st.session_state['assembled_blog'], height=300)

    # --- Paraphrasing Section ---
    st.header("2. Paraphrase and 'Humanize' the Text")
    
    text_to_paraphrase = st.session_state.get('assembled_blog', '')
    
    if text_to_paraphrase:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Text")
            st.text_area("Original", text_to_paraphrase, height=300, key="original_text_display")

        with col2:
            st.markdown("#### Transformed Text")
            # Apply transformations
            paraphrased_text = synonym_replacer(text_to_paraphrase, replacement_prob=0.15)
            # You could chain more transformations here, e.g., sentence shuffling
            
            st.text_area("Paraphrased", paraphrased_text, height=300, key="paraphrased_text_display")
            
        st.info("The transformed text uses synonym replacement to alter the content. More complex rules like voice change could be added for better results.")
