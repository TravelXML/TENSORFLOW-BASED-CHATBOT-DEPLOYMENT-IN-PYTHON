#This file will contain the data loading and preprocessing code.

import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def load_data(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    words = []
    classes = []
    documents = []
    ignore_words = set(stopwords.words('english'))

    for intent in data['intents']:
        for pattern in intent['patterns']:
            w = word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [w.lower() for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    return words, classes, documents
