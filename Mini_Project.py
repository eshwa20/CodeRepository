import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from textwrap import fill

# Download ALL required NLTK data (including punkt_tab)
import nltk
nltk.download('punkt')        # Main punkt tokenizer
nltk.download('punkt_tab')    # Supplemental tables
nltk.download('stopwords')

# Download required NLTK data
# nltk.download('punkt', quiet=True)

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
        print(f"Error in summarization: {e}")
        return text[:200]

def smart_summarize(text, target_ratio=0.50):
    if not isinstance(text, str) or not text.strip():
        return "[Empty input text]"

    display_text = text.strip()
    sentences = sent_tokenize(display_text)

    if not sentences:
        return display_text[:200] if len(display_text) > 200 else display_text

    word_count = len(display_text.split())
    target_length = max(10, min(50, int(word_count * target_ratio)))

    important_sentences = []
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[::-1]]

        current_length = 0
        for sent in ranked_sentences:
            sent_words = sent.split()
            if current_length + len(sent_words) <= target_length:
                important_sentences.append(sent)
                current_length += len(sent_words)
            elif not important_sentences:
                important_sentences.append(' '.join(sent_words[:target_length]))
                break
    except Exception as e:
        print(f"Error in summarization: {e}")
        important_sentences = sentences[:3]

    summary = ' '.join(important_sentences)

    if not summary.strip():
        summary = ' '.join(sentences[0].split()[:target_length])

    return summary

def plot_statistics(original_length, summary_length, article_num):
    fig, ax = plt.subplots(figsize=(8, 4))

    # Bar plot for word counts
    bars = ax.bar(['Original', 'Summary'], [original_length, summary_length],
                 color=['skyblue', 'lightgreen'])

    # Add text labels
    ax.bar_label(bars, padding=3)
    ax.set_ylabel('Word Count')
    ax.set_title(f'Article {article_num} Length Comparison\n'
                f'Compression: {(summary_length/original_length):.1%}')

    plt.tight_layout()
    plt.show()

def display_results(original, summary, headline, idx):
    print(f"\n{'='*80}")
    print(f"ARTICLE {idx + 1} RESULTS:")
    print(f"{'-'*80}")

    print("\nORIGINAL HEADLINE:")
    print(f"{'-'*40}")
    print(fill(headline, width=80))

    print("\nORIGINAL TEXT:")
    print(f"{'-'*40}")
    print(fill(original[:500], width=80))  # Show first 500 chars of original

    print("\nGENERATED SUMMARY:")
    print(f"{'-'*40}")
    print(fill(summary, width=80))

    # Calculate statistics
    original_words = len(original.split())
    summary_words = len(summary.split())
    ratio = summary_words / original_words

    print("\nSTATISTICS:")
    print(f"{'-'*40}")
    print(f"Original length: {original_words} words")
    print(f"Summary length: {summary_words} words")
    print(f"Compression ratio: {ratio:.1%}")
    print(f"{'='*80}\n")

    # Plot the statistics
    plot_statistics(original_words, summary_words, idx + 1)

def main():
    try:
        df = pd.read_csv('news_summary.csv', encoding='latin-1')
        print(f"Successfully loaded {len(df)} articles")

        # Process and display first 15 articles or all if less than 15
        num_articles = min(15, len(df))
        print(f"\nDisplaying {num_articles} articles with summaries:\n")

        for i in range(num_articles):
            sample = df.iloc[i]
            original_text = sample['text'] if pd.notna(sample['text']) else ""
            headline = sample['headlines'] if pd.notna(sample['headlines']) else ""

            summary = smart_summarize(original_text)
            display_results(original_text, summary, headline, i)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
