#This file will handle user interaction.
import numpy as np
import random
import nltk
from tensorflow.keras.models import load_model
from data_preparation import preprocess_data, load_data

nltk.download('punkt')

data = load_data('intense.json')
words, classes, _ = preprocess_data(data)
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def classify(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

def chatbot_response(text):
    intents = classify(text)
    for intent in data['intents']:
        if intent['tag'] == intents[0][0]:
            return random.choice(intent['responses'])

# Test chatbot interaction
print(chatbot_response("Hi there!"))
print(chatbot_response("Tell me about your shipping options."))
