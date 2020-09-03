import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title(" Machine Learning Major Project ")
st.subheader(" IMDB Review DataSet ")
st.write(" Classifier: NaiveBayes")
st.write(" Accuracy: 0.87 ")

data = pd.read_csv('https://drive.google.com/file/d/1SJPMM11AcA9D0fudo_VpzX9P8NxwYqP9/view?usp=sharing',usecols=['review','sentiment'])

x1 = data.iloc[:,0].values
y1  =data['sentiment'].values

sentence = st.text_input(" Write your review here : ")

model1 = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
model1.fit(x1,y1)

if sentence:
  y_pred = model1.predict([sentence])
  st.write(" Prediction:")
  st.write(y_pred)
