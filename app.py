import streamlit as st
import re
from transformers import pipeline
import nltk

# --- NLTK SETUP ---
# Correctly handle the case where the resource is not found.
# nltk.data.find() raises a LookupError if it can't find the resource.
try:
    # We check for the main 'punkt' resource. If it's missing, we download both it
    # and its dependency 'punkt_tab'.
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    st.info("First-time setup: Downloading NLTK resources...")
    with st.spinner("Downloading... This may take a moment."):
        nltk.download('punkt')
        nltk.download('punkt_tab')
    st.success("Setup complete!")


# --- NLP MODEL ---
# Using st.cache_resource to load the model only once
@st.cache_resource
def load_paraphraser():
    """Loads the paraphrasing model and returns the pipeline."""
    return pipeline(
        "text2text-generation",
        model="Vamsi/T5_Paraphrase_Paws",
        device=-1  # Use -1 for CPU, or 0 for GPU if available
    )

paraphraser = load_paraphraser()


def clean_text(text: str) -> str:
    """
    Remove unwanted unicode characters and normalize whitespace.
    """
    text = text.encode("ascii", "ignore").decode()  # remove unicode
    text = re.sub(r'\s+', ' ', text).strip()        # normalize whitespace
    return text


def humanize_long_text(text: str, batch_size: int = 4) -> str:
    """
    Paraphrases long AI-generated text by processing it paragraph by paragraph
    to preserve content structure, headings, and lists for a blog-style output.

    Args:
        text (str): The AI-generated text to humanize.
        batch_size (int): The number of sentences to process in each batch.

    Returns:
        str: The humanized text with preserved paragraph structure.
    """
    # 1. Split the text into blocks (paragraphs, headings, list items, etc.)
    blocks = text.split('\n')
    humanized_blocks = []
    
    total_blocks = len(blocks)
    if total_blocks == 0:
        return ""

    progress_bar = st.progress(0, text="Humanizing text...")

    for i, block in enumerate(blocks):
        # 2. Process each block
        stripped_block = block.strip()

        # Keep empty lines to preserve paragraph spacing
        if not stripped_block:
            humanized_blocks.append("")
            continue

        # 3. Preserve Markdown structure: Don't paraphrase headings, lists, or blockquotes
        is_markdown_or_short = (
            stripped_block.startswith(('#', '*', '-', '>')) or
            re.match(r'^\d+\.\s', stripped_block) or
            len(stripped_block.split()) < 5
        )

        if is_markdown_or_short:
            humanized_blocks.append(block) # Keep the original block
        else:
            # This is a paragraph we should humanize
            sentences = nltk.sent_tokenize(stripped_block)
            if not sentences:
                continue

            paraphrased_sentences_for_block = []
            # Process the paragraph's sentences in batches
            for j in range(0, len(sentences), batch_size):
                batch = sentences[j:j+batch_size]
                prefixed_batch = [f"paraphrase: {clean_text(sentence)}" for sentence in batch]

                try:
                    results = paraphraser(
                        prefixed_batch,
                        max_new_tokens=128,
                        num_return_sequences=1,
                        do_sample=True,
                        top_k=100,
                        top_p=0.95,
                    )
                    paraphrased_batch = [result['generated_text'] for result in results]
                    paraphrased_sentences_for_block.extend(paraphrased_batch)
                except Exception as e:
                    st.error(f"An error occurred during paraphrasing: {e}")
                    paraphrased_sentences_for_block.extend(batch) # Fallback to original

            # 4. Reconstruct the humanized paragraph
            humanized_block = " ".join(paraphrased_sentences_for_block)
            humanized_blocks.append(humanized_block)
        
        # 5. Update progress bar based on blocks processed
        progress_percentage = (i + 1) / total_blocks
        progress_bar.progress(progress_percentage, text=f"Processing... Block {i+1}/{total_blocks}")

    progress_bar.empty() # Remove the progress bar after completion
    
    # 6. Join the processed blocks back together with newlines
    return "\n".join(humanized_blocks)


# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Text Humanizer", page_icon="📝", layout="wide")

st.title("📝 AI Text Humanizer")
st.markdown(
    "Paste AI-generated content below—including articles with **headings** and **lists**! "
    "I'll paraphrase the paragraphs to sound more human while preserving the original structure and length."
)

# Input text
input_text = st.text_area(
    "Enter AI-generated text here (supports Markdown for structure):", 
    height=300,
    placeholder="## My Blog Post Title\n\nThis is the first paragraph generated by an AI. It often sounds a bit robotic.\n\n* This is a list item.\n* And another one.\n\nThis second paragraph needs some human touch as well to make it more engaging for the reader."
)

if st.button("Humanize Text", type="primary", use_container_width=True):
    if input_text.strip():
        with st.spinner("Warming up the humanizer..."):
            humanized = humanize_long_text(input_text)

        st.subheader("✅ Humanized Blog-Style Output")
        
        with st.container(border=True):
            st.markdown(humanized)

        st.download_button(
            label="Download Result as a Markdown File",
            data=humanized,
            file_name="humanized_text.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.warning("⚠️ Please enter some text first!")
