
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#adding stars column (train data)
file_path = '/content/drive/MyDrive/Colab datasets/SentimentAnalysis/train.csv'
df_train = pd.read_csv(file_path)
df_train = df_train.sample(frac=1).reset_index(drop=True)
#adding stars column (test data)
file_path = '/content/drive/MyDrive/Colab datasets/SentimentAnalysis/train.csv'
df_test = pd.read_csv(file_path)
df_test = df_test.sample(frac=1).reset_index(drop=True)
df = pd.concat([df_train,df_test],axis=0)
df['Sentiment'] = (df['Sentiment'] == 'pos').astype(int)
#splitting data
vectorizer = TfidfVectorizer()
x_reviews = df['Review'].values
y = df['Sentiment'].values
x_train, x_test, y_train, y_test = train_test_split(x_reviews, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('tftdf',TfidfVectorizer()),
    ('model',SVC()),
])
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
import joblib

# Save pipeline
joblib.dump(pipeline, '/content/drive/MyDrive/Colab datasets/sentiment_pipeline.joblib')

# Later: load it back
#pipeline = joblib.load('/content/drive/MyDrive/Colab datasets/SentimentAnalysis/sentiment_pipeline.joblib')

