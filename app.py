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


# Ensure the nltk_data directory exists
nltk_data_dir = Path("./nltk_data")
nltk_data_dir.mkdir(exist_ok=True)
nltk.data.path.append(str(nltk_data_dir))

tokenizer_path = "punkt_tokenizer.pkl"


try:
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print("Loaded Punkt tokenizer from punkt_tokenizer.pkl")
except (FileNotFoundError, LookupError):
    print("punkt_tokenizer.pkl not found. Downloading and saving tokenizer...")
    nltk.download("punkt", download_dir=str(nltk_data_dir))

    # Create and save the tokenizer
    punkt_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    with open(tokenizer_path, "wb") as f:
        pickle.dump(punkt_tokenizer, f)

    tokenizer = punkt_tokenizer  # Assign it to tokenizer


print("Punkt tokenizer is ready to use!")

# Example usage
text = "This is a test sentence. Here is another one!"
sentences = tokenizer.tokenize(text)
st.title(sentences)
print(sentences)
    

# Load model, tokenizer, dan lain-lain
model_prediksi = keras.models.load_model('sentimen_model.h5')

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
    
    # Proper stopword filtering
    words = [word for word in words if (word not in stop_words or word in ['not', 'no', "n't"]) and word != '']
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Streamlit app
st.title('Klasifikasi Jenis Pertanyaan Menggunakan Machine Learning')
text = st.text_input("Masukkan Pertanyaan:", key="input1")
if text.strip():
    text_prepared = preprocessing_text(text)
    
    sequence_testing = tokenizer.texts_to_sequences([text_prepared])
    padded_testing = pad_sequences(sequence_testing, maxlen=maxlen, padding='post')
    
    prediksi = model_prediksi.predict(padded_testing)
    predicted_class = np.argmax(prediksi, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    st.write("Hasil Prediksi (Class):", predicted_label)
