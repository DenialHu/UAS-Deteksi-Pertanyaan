import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import re
import pickle
import nltk

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set NLTK data path to /tmp (writable on Streamlit Cloud)
nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Force-download required NLTK resources every run
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)
nltk.download("wordnet", download_dir=nltk_data_path)


with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)
with open('maxlen.pkl', 'rb') as handle:
    maxlen = pickle.load(handle)

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
