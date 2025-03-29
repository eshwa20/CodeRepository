# CodeRepository

# Text Summarization Using TF-IDF

## Overview
This project implements a text summarization algorithm using the **TF-IDF (Term Frequency-Inverse Document Frequency)** method. The goal is to extract the most important sentences from a given text while maintaining the overall meaning. The script processes textual data, generates concise summaries, and visualizes compression statistics.

## Features
- Cleans and preprocesses text for accurate analysis
- Uses **TF-IDF vectorization** to score and rank sentences
- Provides two summarization functions:
  - **tfidf_summarize:** Extracts the top-ranked sentences based on TF-IDF scores
  - **smart_summarize:** Generates a summary with a target word count ratio
- Displays statistics, including word count and compression ratio
- Visualizes the summarization performance using **matplotlib**

## Installation
To run the project, install the required dependencies:
```
pip install nltk pandas scikit-learn matplotlib

```

Download necessary NLTK datasets:
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage
### Running the Script
Ensure you have a CSV file named **news_summary.csv** containing text data. Then, execute the script:
```
python summarization.py
```

### Expected CSV Format
The input CSV file should contain at least the following columns:
- **headlines** (Title of the news article)
- **text** (Full text of the article)

### Output
The script will:
1. Read the dataset
2. Generate summaries for the first **15 articles** (or fewer if the dataset is smaller)
3. Print the original and summarized text along with statistics
4. Display a **bar chart** comparing the word count of the original and summarized text

## Code Breakdown
### Preprocessing
- Converts text to lowercase
- Removes punctuation and extra whitespace

### Summarization
- Tokenizes the text into sentences
- Computes **TF-IDF scores** for each sentence
- Selects the top-ranked sentences for the summary
- Ensures the summary maintains coherence and a target length

### Visualization
- A bar chart compares the word count before and after summarization

## Example Output
```
================================================================================
ARTICLE 1 RESULTS:
--------------------------------------------------------------------------------
ORIGINAL HEADLINE:
----------------------------------------
Breaking News: AI Revolution in 2025

ORIGINAL TEXT:
----------------------------------------
(First 500 characters of the article...)

GENERATED SUMMARY:
----------------------------------------
(Summary of the article...)

STATISTICS:
----------------------------------------
Original length: 350 words
Summary length: 90 words
Compression ratio: 25.7%
================================================================================
```

## Notes
- The **tfidf_summarize** function extracts key sentences based purely on scores.
- The **smart_summarize** function aims to generate a more structured and human-readable summary.
- If errors occur, the script gracefully handles them and provides a fallback summary.

## Future Improvements
- Implement **BERT-based summarization** for more accurate results
- Add **keyword-based summarization** using Named Entity Recognition (NER)
- Improve **sentence coherence** using NLP techniques

## Author
Developed by [Eshwa Vyas]

