import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import pickle
import nltk

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path

# Ensure NLTK dependencies are available every run

nltk.data.path.append("/tmp")

# Download required NLTK resources every run
nltk.download("punkt", download_dir="/tmp")
nltk.download("stopwords", download_dir="/tmp")
nltk.download("wordnet", download_dir="/tmp")

try:
    nltk.data.find("tokenizers/punkt")
    print("Punkt tokenizer is already available.")
except LookupError:
    print("Downloading punkt tokenizer...")
    nltk.download("punkt")

# Load punkt tokenizer manually
punkt_tokenizer = nltk.tokenize.PunktSentenceTokenizer()

# Load other model dependencies
model_prediksi = keras.models.load_model('sentimen_model.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)
with open('maxlen.pkl', 'rb') as handle:
    maxlen = pickle.load(handle)

# Preprocessing function
# Preprocessing function
def preprocessing_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s\?!.,\'"]', '', text)

    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words or word in ['not', 'no', "n't"]]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Streamlit app
st.title('Klasifikasi Jenis Pertanyaan Menggunakan Machine Learning')
text = st.text_input("Masukkan Pertanyaan:", key="input1")

if text.strip():
    text_prepared = preprocessing_text(text)
    st.write("Processed Text:", text_prepared)
    sequence_testing = tokenizer.texts_to_sequences([text_prepared])
    padded_testing = pad_sequences(sequence_testing, maxlen=maxlen, padding='post')
    prediksi = model_prediksi.predict(padded_testing)
    predicted_class = np.argmax(prediksi, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    st.write("Hasil Prediksi (Class):", predicted_label)
