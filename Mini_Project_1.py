import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tfidf_summarize(text, num_sentences=3):
    if not text.strip():
        return ""
    
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)

        vectorizer = TfidfVectorizer(stop_words='english', min_df=0.1, max_df=0.9)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        top_indices = sentence_scores.argsort()[-num_sentences:]
        return ' '.join([sentences[i] for i in sorted(top_indices)])
    except Exception as e:
        return text[:200]

def plot_statistics(original_length, summary_length):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(['Original', 'Summary'], [original_length, summary_length], color=['skyblue', 'lightgreen'])
    ax.set_ylabel('Word Count')
    ax.set_title(f'Compression: {(summary_length/original_length):.1%}')
    st.pyplot(fig)

def main():
    st.title("📰 Text Summarization App")
    st.write("Upload a CSV file with a column named 'text' to summarize the content.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        st.write(f"Loaded {len(df)} articles")

        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column.")
            return

        num_articles = min(5, len(df))
        for i in range(num_articles):
            sample = df.iloc[i]
            original_text = sample['text'] if pd.notna(sample['text']) else ""
            
            if original_text.strip():
                summary = tfidf_summarize(original_text)
                original_length = len(original_text.split())
                summary_length = len(summary.split())
                
                st.subheader(f"Article {i+1}")
                st.text_area("Original Text", original_text[:1000], height=200)
                st.text_area("Summary", summary, height=100)
                plot_statistics(original_length, summary_length)
            else:
                st.warning(f"Article {i+1} is empty.")

if __name__ == "__main__":
    main()
