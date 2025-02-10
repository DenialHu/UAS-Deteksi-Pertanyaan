import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import pickle
import nltk

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


import os
from pathlib import Path

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
    
# Set NLTK data path to a directory in your project
nltk_data_dir = Path('./nltk_data')
nltk_data_dir.mkdir(exist_ok=True)
nltk.data.path.append(str(nltk_data_dir))

# Download NLTK data
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=str(nltk_data_dir))
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=str(nltk_data_dir))
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', download_dir=str(nltk_data_dir))

# Call this function at startup
ensure_nltk_data()

# Rest of your code...

# Download NLTK dependencies jika belum ada
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load Model, Tokenizer, Class, & maxlen
model_prediksi = keras.models.load_model('sentimen_model.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)


with open('maxlen.pkl', 'rb') as handle:
    maxlen = pickle.load(handle)  



# Check if NLTK data exists, if not download it
if not os.path.exists('/root/nltk_data/tokenizers/punkt'):
    nltk.download('punkt')
if not os.path.exists('/root/nltk_data/corpora/stopwords'):
    nltk.download('stopwords')
if not os.path.exists('/root/nltk_data/corpora/wordnet'):
    nltk.download('wordnet')

# Preprocessing Text
def preprocessing_text(text):
    text = text.lower()  # Huruf kecil
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Hapus link
    text = re.sub(r'[^a-zA-Z0-9\s\?!.,\'"]', '', text)  # Hapus karakter khusus
    words = word_tokenize(text)  # Tokenisasi
    stop_words = set(stopwords.words('english'))  # Stopwords
    words = [word for word in words if word not in stop_words or word in ['not', 'no', "n't"] and word != '']
    lemmatizer = WordNetLemmatizer()  # Lematisasi
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)  # Gabung kembali

# Streamlit UI
st.title('Klasifikasi Jenis Pertanyaan Menggunakan Machine Learning')

tab1, tab2, tab3 = st.tabs(['Masukkan Pertanyaan', 'Presentase Prediksi Teks','Grafik Model'])

with tab1:
    text = st.text_input("Masukkan Pertanyaan:", key="input1")
    if text.strip():
        text_testing = [text]
        
        # Preprocessing
        text_prepared = preprocessing_text(text_testing[0])
        sequence_testing = tokenizer.texts_to_sequences([text_prepared])
        padded_testing = pad_sequences(sequence_testing, maxlen=maxlen, padding='post')

        # Prediksi
        prediksi = model_prediksi.predict(padded_testing)
        predicted_class = np.argmax(prediksi, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        # Label Descriptions
        label_descriptions = {
            'DESC': 'Class DESC untuk mendeskripsikan sesuatu.',
            'ENTY': 'Class ENTY untuk mengenali entitas atau kategori tertentu.',
            'ABBR': 'Class ABBR untuk mendeteksi singkatan atau akronim.',
            'HUM': 'Class HUM untuk mengenali pertanyaan yang berhubungan dengan manusia.',
            'NUM': 'Class NUM untuk mengenali pertanyaan yang membutuhkan jawaban berupa angka.',
            'LOC': 'Class LOC untuk menentukan suatu lokasi.'
        }

        st.write("Hasil Prediksi (Class):", predicted_label)
        st.write(f"Deskripsi Class: {label_descriptions.get(predicted_label, 'Tidak ada deskripsi tersedia.')}")

with tab2:
    if text.strip():  
        # Daftar kelas
        classes = label_encoder.classes_

        # Konversi ke persentase
        predictions_with_classes = {cls: f"{prob * 100:.2f}%" for cls, prob in zip(classes, prediksi[0])}

        # Tampilkan hasil
        for cls, prob in predictions_with_classes.items():
            st.write(f"{cls}: {prob}")
    else:
        st.write("Masukkan Pertanyaan Terlebih Dahulu!")

with tab3:
    from PIL import Image

    # Load the image
    image = Image.open(r"Grafik.png")


    # Display the image
    st.image(image, caption="Grafik Model", use_column_width=True)

