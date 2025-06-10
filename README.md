# Dataset Description:
Large Movie Review Dataset v1.0
The Large Movie Review Dataset v1.0 (also known as the IMDB sentiment dataset) is designed to support binary sentiment classification—that is, determining whether a movie review is positive or negative. It is widely used as a benchmark for evaluating text classification models, especially in natural language processing (NLP).

<h2>Main Features</h2>
50,000 labeled reviews:
25k for training, 25k for testing.
Balanced: 25k positive and 25k negative reviews.
Positive: IMDb rating ≥ 7/10.
Negative: IMDb rating ≤ 4/10.
Neutral reviews are excluded.
No movie contributes more than 30 reviews to avoid dataset leakage.
Disjoint movie sets between train/test (prevents memorizing movie-specific terms).

<h2>Citation</h2>
When using this dataset, cite the ACL 2011 paper:
Learning Word Vectors for Sentiment Analysis
(Maas et al., 2011)

# Detailed Code Analysis
Let’s break down the code you provided step-by-step:
<h2>1. imports</h2>
<code>import pandas as pd : Data manipulation using DataFrames.</code>\n
import numpy as np : Numerical operations and arrays.
import matplotlib.pyplot as plt : Plotting and visualizing data.
from sklearn.feature_extraction.text import TfidfVectorizer : Converts text to TF-IDF features.
from sklearn.pipeline import Pipeline : Builds an end-to-end ML pipeline.
from sklearn.model_selection import train_test_split : Splits data into training/testing sets.
from sklearn.svm import SVC : Support Vector Machine classifier.
from sklearn.metrics import accuracy_score, classification_report : Metrics for evaluating classification.
<h2>2. Loading & Preparing Data</h2>
# Adding training data
file_path_train = 'train.csv'
df_train = pd.read_csv(file_path_train)
# Adding testing data
file_path_test = 'test.csv'
df_test = pd.read_csv(file_path_test)
# Merges the two DataFrames into one big df
df = pd.concat([df_train, df_test], axis=0)
# Encoding Sentiment column (necessary change because computer can understand only numbers)
df['Sentiment'] = (df['Sentiment'] == 'pos').astype(int)
Converts the "Sentiment" column to binary:
Positive → 1
Negative → 0
# Shuffling data
df = df.sample(frac=1).reset_index(deop=True)
<h2>3. Data Splitting and Vectorization</h2>
x_reviews = df['Review'].values
y = df['Sentiment'].values
Defines features (x_reviews) and labels (y).
vectorizer = TfidfVectorizer()
TF-IDF Vectorizer will turn the reviews (text) into numeric feature vectors.
x_train, x_test, y_train, y_test = train_test_split(
    x_reviews, y, test_size=0.2, random_state=42
)
Splits into training (80%) and testing (20%) sets with reproducibility (random_state=42).
<h2>4. ML Pipeline</h2>
pipeline = Pipeline([
    ('tftdf', TfidfVectorizer()),
    ('model', SVC()),
])
Pipeline simplifies the process:
Step 1: TfidfVectorizer() → converts raw text to TF-IDF features.
Step 2: SVC() → trains a Support Vector Classifier.

pipeline.fit(x_train, y_train)
Trains the entire pipeline:
Automatically vectorizes text.
Fits the SVM model on training data.
<h2>5. Evaluation</h2>
y_pred = pipeline.predict(x_test)
print(classification_report(y_test, y_pred))
Classification report: Precision, recall, F1-score for each class (positive/negative).
The results showed an Accuracy score of 97% which is very good result for a very basic model.
<h2>6. Saving the Pipeline</h2>
import joblib
joblib.dump(pipeline, '')
Saves the entire pipeline (vectorizer + model) to disk using joblib.
This allows you to reuse the trained model later without retraining.
# Loading back (optional)
pipeline = joblib.load('')
<h2>Summary of Pipeline Flow</h2>
Raw Movie Reviews → TF-IDF Vectorization → Support Vector Classifier → Predictions → Evaluation → Save Model
