import nltk
#For preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string
from nltk.stem import PorterStemmer


# Helper function to get the WordNet POS tag from the Penn Treebank POS tag
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN

def preprocess_keys(keywords):
    
    keywords = [word.lower() for word in keywords]
    keywords = [word.strip() for word in keywords]
     # Perform POS tagging to get the part of speech of each word
    tagged_tokens = pos_tag(keywords)
    
    # Lemmatize the words based on their POS tag
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]
    
    return lemmatized_tokens

#For preprocessing text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words and punctuations
    stop_words = set(stopwords.words('english'))
    stop_words.update(list(string.punctuation))
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    
    # Perform POS tagging to get the part of speech of each word
    tagged_tokens = pos_tag(filtered_tokens)
    
    # Lemmatize the words based on their POS tag
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]
    
    # Join the lemmatized tokens into a single string
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    return preprocessed_text
    
