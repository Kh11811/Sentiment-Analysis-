<h1>Sentiment Analysis on Large Movie Review Dataset</h1>
Large Movie Review Dataset v1.0
The Large Movie Review Dataset v1.0 (also known as the IMDB sentiment dataset) is designed to support binary sentiment classification—that is, determining whether a movie review is positive or negative. It is widely used as a benchmark for evaluating text classification models, especially in natural language processing (NLP).<br>

#  Objective

To build a text classification model that:<br>
Converts raw movie reviews into numerical feature representations using TF-IDF.<br>
Trains a Support Vector Machine (SVM) classifier to distinguish positive and negative sentiment.<br>
Evaluates model performance using accuracy, precision, recall, and F1-score.<br>
Saves the trained pipeline for easy reuse.<br>
# Dataset Description:
<h2>Main Features</h2>
50,000 labeled reviews:<br>
25k for training, 25k for testing.<br>
Balanced: 25k positive and 25k negative reviews.<br>
Positive: IMDb rating ≥ 7/10.<br>
Negative: IMDb rating ≤ 4/10.<br>
Neutral reviews are excluded.<br>
No movie contributes more than 30 reviews to avoid dataset leakage.<br>
Disjoint movie sets between train/test (prevents memorizing movie-specific terms).<br>

<h2>Citation</h2>
When using this dataset, cite the ACL 2011 paper:<br>
Learning Word Vectors for Sentiment Analysis<br>
(Maas et al., 2011)<br>

# Detailed Code Analysis
Let’s break down the code step-by-step:<br>
<h2>1. imports</h2><br>
<pre>
<code>import pandas as pd</code> : Data manipulation using DataFrames.<br>
<code>from sklearn.feature_extraction.text import TfidfVectorizer</code> : Converts text to TF-IDF features.<br>
<code>from sklearn.pipeline import Pipeline</code> : Builds an end-to-end ML pipeline.</code><br>
<code>from sklearn.model_selection import train_test_split</code> : Splits data into training/testing sets.<br>
<code>from sklearn.svm import SVC</code> : Support Vector Machine classifier.<br>
<code>from sklearn.metrics import classification_report</code> : Metrics for evaluating classification.<br>
</pre>
<h2>2. Loading & Preparing Data</h2>
<h4>Adding training data</h4>
<pre><code>file_path_train = 'train.csv'<br>
df_train = pd.read_csv(file_path_train)</code></pre><br>
<h4>Adding testing data</h4>
<pre><code>file_path_test = 'test.csv'<br>
df_test = pd.read_csv(file_path_test)</code></pre><br>
<h4>Merges the two DataFrames into one big df</h4>
<pre><code>df = pd.concat([df_train, df_test], axis=0)</code></pre><br>
<h4>Encoding Sentiment column (necessary change because computer can understand only numbers)</h4>
<pre><code>df['Sentiment'] = (df['Sentiment'] == 'pos').astype(int)</code></pre><br>

Converts the "Sentiment" column to binary:<br>
Positive → 1<br>
Negative → 0<br>
<h4>Shuffling data</h4>
<pre><code>df = df.sample(frac=1).reset_index(drop=True)</code></pre><br>

<h2>3. Data Splitting and Vectorization</h2>
features (x_reviews) and labels (y).<br>
<pre><code>x_reviews = df['Review'].values<br>
y = df['Sentiment'].values</code></pre><br>
TF-IDF Vectorizer will turn the reviews (text) into numeric feature vectors.<br>
<pre><code>vectorizer = TfidfVectorizer()</code></pre><br>

Splitting into training (80%) and testing (20%) sets with reproducibility (random_state=42).<br>
<pre><code>x_train, x_test, y_train, y_test = train_test_split(
    x_reviews, y, test_size=0.2, random_state=42)</code></pre><br>


<h2>4. ML Pipeline</h2>
Pipeline simplifies the process:<br>
Step 1: TfidfVectorizer() → converts raw text to TF-IDF features.<br>
Step 2: SVC() → trains a Support Vector Classifier.<br>
<pre><code>pipeline = Pipeline([<br>
    ('tftdf', TfidfVectorizer()),<br>
    ('model', SVC()),<br>
])</code></pre><br>

Training the entire pipeline:<br>
Automatically vectorizes text.<br>
Fits the SVM model on training data.<br>
<pre><code>pipeline.fit(x_train, y_train)</code></pre><br>



<h2>5. Evaluation</h2>
<pre><code>y_pred = pipeline.predict(x_test)<br>
print(classification_report(y_test, y_pred))</code></pre><br>

Classification report: Precision, recall, F1-score for each class (positive/negative).<br>
The results showed an Accuracy score of 97% which is very good result for a very basic model.<br>
<h2>6. Saving the Pipeline</h2>
Saving the entire pipeline (vectorizer + model) to disk using joblib.<br>
This allows you to reuse the trained model later without retraining.<br>
<pre><code>import joblib<br>
joblib.dump(pipeline, '')</code></pre><br>


<h2>Loading back (optional)</h2>
<pre><code>pipeline = joblib.load('')</code></pre><br>
<h2>Confusion Matrix</h2>
<pre><code>from sklearn.metrics import confusion_matrix<br>
import numpy as np<br>
y_pred = pipeline.predict(x_test)<br>
cm = confusion_matrix(y_test, y_pred)<br>
balanced_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]<br>
</code></pre><br>
<h3>Result</h3>
<pre><code>
Actual Positive : [0.94332392, 0.05667608]<br>
Actual Negative : [0.06783023, 0.93216977]<br>
</code></pre><br><br>

<h2>Predict your own text</h2>
<pre><code>
L = ["Give someone the cold shoulder",<br>
     "To fail spectacularly.",<br>
     "keep it up.",<br>
     "To achieve great success."]<br>
predictions = pipeline.predict(L)<br>
for i in predictions:<br>
  if i == 1:<br>
    print("Positive")<br>
  else:<br>
    print("Negative")<br>
</code></pre><br><br>
<h3>Results</h3>
<pre><code>
Negative<br>
Negative<br>
Positive<br>
Positive<br>
</code></pre><br><br>

<h2>Summary of Pipeline Flow</h2>
Raw Movie Reviews → TF-IDF Vectorization → Support Vector Classifier → Predictions → Evaluation → Save Model<br>
