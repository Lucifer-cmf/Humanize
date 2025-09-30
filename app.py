import streamlit as st
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import os

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
def load_paraphraser_model():
    """Loads a more capable paraphrasing model and returns the pipeline."""
    
    # Using t5-large for better "humanization" and stylistic control
    # You can change this to other suitable models on Hugging Face if needed,
    # e.g., "humarif/chatgpt-paraphraser-long" or "tuner007/pegasus_paraphrase"
    model_name = "t5-large" 
    
    # Check if a specific model name is set in environment variables (for deployment flexibility)
    if os.getenv("PARAPHRASER_MODEL_NAME"):
        model_name = os.getenv("PARAPHRASER_MODEL_NAME")

    st.write(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # -1 for CPU, 0 for GPU (if available)
        max_new_tokens=256, # Increased default max_new_tokens for potentially longer, more varied outputs per sentence.
    )

paraphraser_pipeline = load_paraphraser_model()


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
    to preserve content length and structure, aiming to reduce AI detection score
    and adopt a blog-like style.

    Args:
        text (str): The AI-generated text to humanize.
        batch_size (int): The number of sentences to process in each batch.

    Returns:
        str: The humanized text in a more blog-like style.
    """
    # 1. Clean and split the text into sentences
    cleaned_text = clean_text(text)
    sentences = nltk.sent_tokenize(cleaned_text)

    if not sentences:
        return ""

    # 2. Process sentences in batches for efficiency
    humanized_sentences = []
    
    # Initialize a Streamlit spinner for overall processing feedback
    with st.spinner("Processing text... This may take a moment for longer inputs."):
        progress_text = "Operation in progress. Please wait."
        progress_bar = st.progress(0, text=progress_text)
        total_batches = (len(sentences) + batch_size - 1) // batch_size

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # --- MODIFIED PREFIX FOR BLOG STYLE ---
            # Instructions: rewrite for a blog, conversational, engaging, simpler language, 
            # address reader, avoid overly formal phrasing, focus on flow and readability.
            prefixed_batch = [
                f"rewrite this for a blog post. Make it conversational, engaging, "
                f"use simpler language, and address the reader directly (using 'you'). "
                f"Avoid overly formal phrasing and focus on natural flow and readability: {sentence}" 
                for sentence in batch
            ]
            # --- END MODIFIED PREFIX ---

            # 3. Paraphrase the batch
            try:
                results = paraphraser_pipeline(
                    prefixed_batch,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=100, # Increased top_k for more diverse sampling
                    top_p=0.95, # Increased top_p for more diverse sampling
                    temperature=0.7 # Add temperature for creativity, adjust as needed
                )
                # Extract the generated text from the results
                paraphrased_batch = [result['generated_text'] for result in results]
                humanized_sentences.extend(paraphrased_batch)
            except Exception as e:
                st.error(f"An error occurred during paraphrasing batch {i//batch_size + 1}: {e}")
                # In case of an error, append the original batch to avoid losing content.
                humanized_sentences.extend(batch)

            # 4. Update progress bar
            progress_percentage = min((i + batch_size) / len(sentences), 1.0)
            progress_bar.progress(progress_percentage, text=f"Humanizing... Batch {i//batch_size + 1}/{total_batches} complete.")

        progress_bar.empty() # Remove the progress bar after completion
    
    # 5. Join the humanized sentences back into a single text
    # This simple join still creates a continuous string. 
    # For more defined paragraphs (like in a blog), you might want to introduce
    # double newlines every X sentences. However, this is a heuristic and
    # not semantically driven, so it might break flow in some cases.
    # For now, we'll stick to a simple space join, letting the prompt guide sentence-level style.
    return " ".join(humanized_sentences)


# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Text Humanizer", page_icon="📝", layout="wide")

st.title("📝 AI Text Humanizer & Undetectable Rewriter (Blog Style)")
st.markdown(
    """
    Paste AI-generated content below (even long articles!), and I'll rewrite it to sound more natural, 
    human-like, and less detectable by AI content checkers. The output will aim for a **conversational, 
    engaging blog-post style**.
    """
)

# Input text
input_text = st.text_area("Enter AI-generated text here (up to 2000+ words recommended for optimal results):", height=300)

if st.button("Humanize & Rewrite", type="primary"):
    if input_text.strip():
        # Add a check for extremely long text that might hit Streamlit or model limits
        if len(input_text.split()) > 3000: # Example limit, adjust as needed
            st.warning("Input text is very long. Processing might take a significant amount of time or hit resource limits. Consider breaking it into smaller chunks.")

        humanized = humanize_long_text(input_text)

        st.subheader("✅ Humanized & Rewritten Output")
        st.markdown(humanized) # Use st.markdown for potentially better formatting if generated text includes it.

        st.download_button(
            label="Download Rewritten Text",
            data=humanized.encode("utf-8"), # Ensure data is bytes for download
            file_name="humanized_blog_style_text.txt",
            mime="text/plain"
        )
    else:
        st.warning("⚠️ Please enter some text first to humanize!")

st.markdown("---")
st.info(
    "**Note:** While this tool aims to significantly reduce AI detection scores and produce a blog-like style, "
    "no automated tool can guarantee 100% human-like text or complete undetectability. "
    "Human review remains the gold standard for critical content."
)
