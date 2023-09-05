# Automatic_answer_Evaluator
A BERT based NLP model to automatically evaluate answers

# Automatic_answer_Evaluator
A BERT based NLP model to automatically evaluate answers
# Automatic Answer Evaluator with BERT-Based Semantic Analysis

This project implements an automatic answer evaluator that leverages BERT-based semantic analysis, keyword matching, and optical character recognition (OCR) to assess the similarity between a student's answer and a model answer. The system provides an automated and objective evaluation of student responses, even when presented in image format.

## Components

### 1. `modelAnswer.py`

This module provides the function `get_answer_similarity(student_answer, model_answer)` that calculates the cosine similarity between BERT embeddings of the student's answer and the model answer. The BERT-based RoBERTa model is employed to encode and compare text segments.

### 2. `keywords.py`

This module contains functions related to keyword matching and semantic analysis:

- `checkForKeywords(keywords, student_answer, model_answer)`: Computes cosine similarity between student answer and model answer after converting them into vectors using CountVectorizer.
- `semanticA(text1, text2, aspect)`: Performs semantic analysis on two texts combined with a given aspect using a pre-trained BERT-based model.
- `get_synonyms(word)`: Retrieves synonyms of a given word from the WordNet database.
- `getRelevantSentences(text, word)`: Extracts sentences from a text containing a given word or its synonyms.
- `compare_sentences(sentences1, sentences2, word)`: Compares two sets of sentences based on a given word's presence, considering synonyms, and calculates a similarity score.
- `compareTexts(text1, text2, keywords)`: Compares two texts using relevant sentences for each keyword and returns a total similarity score.

### 3. `preprocess.py`

This module provides functions for text preprocessing:

- `get_wordnet_pos(treebank_tag)`: Converts Penn Treebank POS tags to WordNet POS tags for lemmatization.
- `preprocess_keys(keywords)`: Preprocesses a list of keywords by converting to lowercase, stripping whitespace, performing part-of-speech tagging, and lemmatizing.
- `preprocess_text(text)`: Preprocesses input text by tokenizing, removing stop words and punctuation, performing part-of-speech tagging, and lemmatizing.

### 4. `app.py`

This script sets up a Flask web application for user interaction. Users can submit model answers, student answers, and keywords for evaluation. It also supports processing student answer images to extract text using OCR and then evaluates the extracted text.

### 5. `getTextFromImage.py`

A utility script that uses the Google Cloud Vision API to extract text from images. It takes an image file path, performs OCR, and returns the extracted text content.

## Usage

1. Install the required libraries using `pip install transformers nltk scikit-learn google-cloud-vision flask pillow pytesseract`.

2. Configure API credentials: Replace `'./key.json'` in `gettextfromimage.py` with the path to your Google Cloud Vision API JSON key file.

3. Import the necessary functions from the modules and use them to process and evaluate student answers.

4. Run the Flask app using `python app.py`, and access it through a web browser.
