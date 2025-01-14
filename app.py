import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import numpy as np

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.title(''':rainbow[Sentiment Analysis]''')
st.write("This is a machine learning based web application that detects polarity of sentiments in texts")
    

input = st.text_input("Enter the text")

if st.button('Predict'):
    # 1. preprocess
    transformed_text = transform_text(input)
    # 2. vectorize
    vector_input = tk.transform([transformed_text])
    # 3. predict
    result = model.predict(vector_input.toarray())[0]
    # 4. Display
    if result == 0:
        st.header("Negative")
    elif result == 1:
        st.header("Neutral")
    else:
        st.header("Positive :sunflower:")