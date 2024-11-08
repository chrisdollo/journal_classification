# Naive Bayes Text Classification Project

Course: CMSC 478/678 – Machine Learning
Assignment: Homework 3
Topic: Naive Bayes Text Classification
Due Date: October 13, 2024

**Overview**
This project involves implementing a Naive Bayes classifier to distinguish between articles from The Economist and The Onion based on their text content. The goal is to understand and apply the Naive Bayes algorithm, utilizing log-space arithmetic to manage small probability values and avoid numerical underflow issues. The classification will rely on a processed vocabulary of words from the articles and binary feature vectors.

**Project Structure**
hw3_stub.ipynb: Jupyter notebook provided for implementing the Naive Bayes classifier functions and analyzing the results.
hw3.pdf: Assignment instructions and problem details, including theoretical concepts and implementation requirements.

**Dataset**
The dataset includes:

Vocabulary: A list of all unique words in the dataset.
XTrain and yTrain: The main training data, with feature vectors and class labels.
XTest and yTest: Test data for evaluating the classifier's performance.
XTrainSmall and yTrainSmall: A smaller subset of the training data for additional analysis.
Tasks


**The project consists of the following key steps:**
Log-Space Product Calculation: Implement logProd(x) to compute the product of probabilities in log-space.
Conditional Probability Estimation: Implement NB_XGivenY(XTrain, yTrain) to estimate conditional probabilities with a Beta(2,1) prior.
Class Prior Estimation: Implement NB_YPrior(yTrain) to calculate the Maximum Likelihood Estimate for the class priors.
Naive Bayes Classification: Implement NB_Classify(D, p, X) to classify test instances based on the learned model parameters.
Error Calculation: Implement ClassificationError(yHat, yTruth) to measure the error rate between predicted and actual labels.
Model Training and Testing: Train the Naive Bayes model on XTrain, evaluate on both XTrain and XTest, and analyze the results.
Model Analysis with Smaller Dataset: Repeat the training and evaluation steps using the smaller training set.
Parameter Interpretation: Identify words most indicative of each class, and analyze the model's interpretation.

**Requirements**
Python 3.8+
Jupyter Notebook for running hw3_stub.ipynb
Libraries: numpy, matplotlib (for potential visualization)

**How to Run**
Clone this repository and open hw3_stub.ipynb in Jupyter Notebook.
Follow the notebook’s sequential cells to implement each function.
Execute each cell to load data, train the model, make predictions, and evaluate performance.
Analyze the classifier's accuracy and interpret the parameters as outlined in the tasks.


**Evaluation**
The performance of the Naive Bayes classifier will be evaluated based on:

Training and Testing Errors: Using the ClassificationError function to assess prediction accuracy on both the main and smaller datasets.
Parameter Analysis: Interpretation of key words most indicative of each class, providing insights into the classifier's learned features.


**License**
This project is for educational purposes as part of the CMSC 478/678 Machine Learning course.
