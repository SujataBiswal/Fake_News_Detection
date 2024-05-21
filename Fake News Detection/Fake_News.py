import streamlit as st
import numpy as np 
import pandas as pd
import re
from nltk.corpus import stopwords #the for of in with
from nltk.stem.porter import PorterStemmer # loved loving == love
from sklearn.feature_extraction.text import TfidfVectorizer # loved =[0.0]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


news_dataset=pd.read_csv('train.csv_1/train.csv')
news_dataset.head()
#news_dataset.shape

news_dataset.isnull().sum()

news_dataset = news_dataset.fillna('')

news_dataset['content'] = news_dataset['author']+''+news_dataset['title']



X=news_dataset.drop(columns='label',axis=1)
Y=news_dataset ['label']




port_stem = PorterStemmer()
def stemming(content):

    # Remove non-alphabetic characters and tokenize
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)

    # Convert to lowercase
    stemmed_content = stemmed_content.lower()

    # Tokenize the text
    stemmed_content = stemmed_content.split()

    # Perform stemming and remove stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]

    # Join the stemmed words back into a single string
    stemmed_content = ' '.join(stemmed_content)


    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

X=news_dataset['content'].values
Y=news_dataset['label'].values

vector=TfidfVectorizer()
vector.fit(X)
X=vector.transform(X)




X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

#X_train.shape
#X_test.shape

model = LogisticRegression()
model.fit(X_train, Y_train)


#----------Web part-------------

st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred ==1:
        st.write('The News is Fake.')
    else:
        st.write('The News is Real.')