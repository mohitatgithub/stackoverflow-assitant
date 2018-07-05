import nltk
import pickle
import re
import numpy as np
import csv

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'models/intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'models/tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'models/tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'models/thread_embeddings_by_tags/',
    'WORD_EMBEDDINGS': 'datasets/starspace_embedding.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()

def load_embeddings(embeddings_path):
    embeddings = {}
    with open(embeddings_path, newline='') as embedding_file:
        reader = csv.reader(embedding_file, delimiter='\t')
        for line in reader:
            word = line[0]
            embedding = np.array(line[1:]).astype(np.float32)
            embeddings[word] = embedding
        dim = len(line) - 1
    return embeddings, dim

def question_to_vec(question, embeddings, dim):
    vec = np.zeros((dim,), dtype=np.float32)
    count = 0
    for w in question.split():
        if w in embeddings:
            count += 1
            vec += embeddings[w]
    if count == 0:
        return vec
    return vec/count

def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
