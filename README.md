# ArXiv-Paper-Classification

This code repo was build as a part of the kaggle competition 1(https://www.kaggle.com/c/ift3395-6390-arxiv Classification of ArXiv Papers) for IFT 6390 Machine Learning course. This is a text classification task in which techniques like tf-idf, count vectorizer and models like Logistic Regression, Multinomial Naive Bayes and Support Vector Machines were used.

The code naive_bayes.py requires numpy, pandas and re libraries to run.

The train data(train.csv) and test data(test.csv) must be stored in a folder named as 'data'

Then you can run the code as:-
python naive_bayes.py

A nb_submission.csv file would be generated that can be uploaded to the Kaggle Competition

Model Accuracies
Public Score = 0.79088
Private Score = 0.78733

These scores cross the random and BernoulliNB(Numpy only) baselines.

Reference- Naïve Bayes from Scratch using Python only— No Fancy Frameworks
https://github.com/aishajv/Unfolding-Naive-Bayes-from-Scratch 
https://towardsdatascience.com/na%C3%AFve-bayes-from-scratch-using-python-only-no-fancy-frameworks-a1904b37222d
