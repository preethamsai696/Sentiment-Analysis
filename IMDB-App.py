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

data = pd.read_csv('https://drive.google.com/file/d/1SJPMM11AcA9D0fudo_VpzX9P8NxwYqP9/view?usp=sharing.csv',header=None,error_bad_lines=False)

x2 = data.iloc[:,0]
x1 = data.iloc[:,0].values
y1  =data.iloc[:,1].values

sentence = st.text_input(" Write your review here : ")

v = TfidfVectorizer(decode_error='replace', encoding='utf-8')
x1 =  v.fit_transform(x2.values.astype('U'))
x1 = np.nan_to_num(x1)

model1 = MultinomialNB()
model1.fit(x1,y1)

if sentence:
  y_pred = model1.predict([sentence])
  st.write(" Prediction:")
  st.write(y_pred)
