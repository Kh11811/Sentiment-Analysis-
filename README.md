# Sentiment-Analysis
The Sentiment Analysis project consists of creating a machine learning model that can differenciate between positive and negative responses.For this example, I used a dataset of reviews for movies with their respective binary responses. You can find in the dataset the number of stars given to each movie, however it will not be used here since we are focusing on text analysis. Furthermore, the dataset is divided into training data and testing data, nevertheless I am going to mix both of datasets for a larger amount of data, nearly 50000 samples. Then, we will split the data as we would like using the train_test_split.
First, we have the imports : 
pandas : For data manipulation and analysis using DataFrames.
numpy : For numerical computing with powerful array and matrix operations.
matplotlib.pyplot : For creating static, animated, and interactive visualizations.
sklearn.feature_extraction.text : For converting text data into TF-IDF feature vectors.
sklearn.pipeline : For creating a machine learning workflow as a pipeline of steps.
sklearn.model_selection : For splitting data into training and testing sets.
sklearn.svm : For building Support Vector Machine (SVM) classifiers.
sklearn.metrics : For generating a detailed classification performance report.
