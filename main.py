import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

model = load_model('imdb_rnn.h5')

# Decode review (not needed for prediction, but nice to keep)
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

# ✅ Correct preprocessing function
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [1]  # START token

    for word in words:
        if word in word_index:
            encoded_review.append(word_index[word] + 3)  # offset by +3
        else:
            encoded_review.append(2)  # unknown word
    
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(review):
    preprocessed_text = preprocess_text(review)
    prediction = model.predict(preprocessed_text)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit App
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    sentiment, score = predict_sentiment(user_input)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction score: {(score*100):.2f}%")
else:
    st.write("Please Enter movie review")
