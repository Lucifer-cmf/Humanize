import streamlit as st
import re
from transformers import pipeline
 
# --- NLP MODEL ---
# Force CPU usage (avoids meta tensor error on Windows without CUDA)
paraphraser = pipeline(
    "text2text-generation",
    model="Vamsi/T5_Paraphrase_Paws",
    device=-1  # ensures CPU is used
    
)
 
def clean_text(text: str) -> str:
    """
    Remove unwanted unicode characters and normalize whitespace
    """
    text = text.encode("ascii", "ignore").decode()  # remove unicode
    text = re.sub(r'\s+', ' ', text).strip()        # normalize whitespace
    return text
 
def humanize_text(text: str) -> str:
    """
    Paraphrase the AI-generated text to sound more human
    """
    cleaned = clean_text(text)
    result = paraphraser(
        f"paraphrase: {cleaned}",
        max_new_tokens=128,   # use only this (instead of max_length)
        num_return_sequences=1,
        do_sample=True,
        top_k=100,
        top_p=0.95,
    )
    return result[0]['generated_text']
 
 
# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Text Humanizer", page_icon="📝", layout="wide")
 
st.title("📝 AI Text Humanizer")
st.write("Paste AI-generated content below, and I'll clean + humanize it.")
 
# Input text
input_text = st.text_area("Enter AI-generated text:", height=200)
 
if st.button("Humanize"):
    if input_text.strip():
        with st.spinner("Processing..."):
            cleaned = clean_text(input_text)
            humanized = humanize_text(cleaned)
 
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
