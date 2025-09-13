import streamlit as st
import re
from transformers import pipeline
import nltk

# --- NLTK SETUP ---
# Download the sentence tokenizer model (only needs to be done once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    st.info("First-time setup: Downloading NLTK sentence tokenizer...")
    nltk.download('punkt')
    st.success("Setup complete!")


# --- NLP MODEL ---
# Using st.cache_resource to load the model only once
@st.cache_resource
def load_paraphraser():
    """Loads the paraphrasing model and returns the pipeline."""
    return pipeline(
        "text2text-generation",
        model="Vamsi/T5_Paraphrase_Paws",
        device=-1  # ensures CPU is used
    )

paraphraser = load_paraphraser()


def clean_text(text: str) -> str:
    """
    Remove unwanted unicode characters and normalize whitespace.
    """
    text = text.encode("ascii", "ignore").decode()  # remove unicode
    text = re.sub(r'\s+', ' ', text).strip()        # normalize whitespace
    return text


def humanize_long_text(text: str, batch_size: int = 5) -> str:
    """
    Paraphrases long AI-generated text by processing it in sentence batches
    to preserve content length and structure.

    Args:
        text (str): The AI-generated text to humanize.
        batch_size (int): The number of sentences to process in each batch.

    Returns:
        str: The humanized text.
    """
    # 1. Clean and split the text into sentences
    cleaned_text = clean_text(text)
    sentences = nltk.sent_tokenize(cleaned_text)

    if not sentences:
        return ""

    # 2. Process sentences in batches for efficiency
    humanized_sentences = []
    progress_bar = st.progress(0, text="Humanizing text...")
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        # The model expects this specific prefix
        prefixed_batch = [f"paraphrase: {sentence}" for sentence in batch]

        # 3. Paraphrase the batch
        try:
            results = paraphraser(
                prefixed_batch,
                max_new_tokens=128,  # Max tokens *per sentence* in the batch
                num_return_sequences=1,
                do_sample=True,
                top_k=100,
                top_p=0.95,
            )
            # Extract the generated text from the results
            paraphrased_batch = [result['generated_text'] for result in results]
            humanized_sentences.extend(paraphrased_batch)
        except Exception as e:
            st.error(f"An error occurred during paraphrasing: {e}")
            # In case of an error, you might want to append the original batch
            # to avoid losing content.
            humanized_sentences.extend(batch)

        # 4. Update progress bar
        progress_percentage = min((i + batch_size) / len(sentences), 1.0)
        progress_bar.progress(progress_percentage, text=f"Humanizing... Batch {i//batch_size + 1}/{total_batches} complete.")

    progress_bar.empty() # Remove the progress bar after completion
    
    # 5. Join the humanized sentences back into a single text
    return " ".join(humanized_sentences)


# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Text Humanizer", page_icon="📝", layout="wide")

st.title("📝 AI Text Humanizer")
st.write(
    "Paste AI-generated content below (even long articles!), and I'll clean and "
    "humanize it without losing the original meaning or length."
)

# Input text
input_text = st.text_area("Enter AI-generated text (up to 2000+ words):", height=250)

if st.button("Humanize", type="primary"):
    if input_text.strip():
        # The spinner is good for the initial setup, but the progress bar
        # will handle the feedback during the main processing.
        with st.spinner("Preparing to process..."):
            humanized = humanize_long_text(input_text)

        st.subheader("✅ Humanized Output")
        st.write(humanized)

        st.download_button(
            label="Download Result",
            data=humanized,
            file_name="humanized_text.txt",
            mime="text/plain"
        )
    else:
        st.warning("⚠️ Please enter some text first!")
