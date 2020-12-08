import os
import csv
import re
import collections

import numpy as np
from itertools import zip_longest
import langid
from nltk import ngrams
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from langid.langid import LanguageIdentifier, model

from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import mean_absolute_error

""" Rough notes on the methodology
# 1. Build a representative sampling of the data set we'll end up testing (the universe of all English tweets)
#       -> Should this include tweets from the challenge data set? If so, how?
#       -> I think yes... maybe representative of their proportion of the general population
#       -> Or easier, just use samplings from http://help.sentiment140.com/for-students. Divide the
#       -> data set up into equal sized chunks and train on the accuracy on each chunk
#
# 2. Convert representative samples into the same feature set as the models
#       -> langid provides utilities to do this. It builds a final feature vector of most important
#       -> words. 
#       -> dl3 roughly describes what its feature space is. Mirror it to the best of our ability
#
# 3. Define meta-features of the representative sample and new data sets to compare against
#    "fraction of tokens appearing in the concerned dictionary"
#       -> For langid, I think this is the set of words selected for the feature vectors.
#          The langid features are n-grams ranked and selected by their information gain
#          So the metafeatures are
#          for each sample, (# of words in the sample that are in the dictionary)/(# of words in the sample)
#
# 4. Learn a regression model on predictive accuracy, where each training data input is a vector of the
#    metafeatures for each representative sample. Label with the accuracies of each representative sample
#    (run the models to get their accuracy). Use SVMs as the model.
#
# 5. Predict the accuracy on new data sets by running their meta-features as inputs to the model
#   -> Do so by drawing N equal sized samples from the new data set, and predicting accuracy on each. The overall
#   -> predicted accuracy is reported as the average
#   -> Use samples of size 4000, since that's what we traing the model on
#       -> But... we only showed experts a subset, so should just use the same 15 tweets?
#           -> I think yes, but we can also use the more general N equal sized samples approach to compare against also
"""

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
RAW_TRAINING_DATA_FILE = 'data/training_data.csv'
LABELLED_TRAINING_DATA_FILE = 'data/labelled_training_data_400.csv'

AAVE_DATA = 'data/twitteraae_all_aa.tsv'
SCOTTISH_DATA = 'data/scottish_english_tweets.csv'
GLOBAL_ENGLISH_DATA = 'data/global_english.tsv'

def read_data(file_name, tweet_index, tab_separated = False):
    result = []
    with open(os.path.join(THIS_FOLDER, file_name), 'r', errors='ignore') as data_file:
        reader = csv.reader(data_file) if not tab_separated else csv.reader(data_file, delimiter='\t')
        for row in reader:
            tweet = row[tweet_index]
            # Remove hashtags, mentions, and links in tweets
            tweet = re.sub(r"(?:\@|\#|https?\://)\S+", "", tweet)
            result.append(tweet)
    return result

def label_data(data, debug=False):
    accuracies = []
    for i, chunk in enumerate(data):
        correctly_classified = 0
        for tweet in chunk:
            classification = langid.classify(tweet)
            if classification[0] == 'en':
                correctly_classified += 1
        accuracy = correctly_classified/len(chunk)
        if debug: print('accuracy: {}'.format(accuracy))
        accuracies.append(accuracy)
        if i % 50 == 0: print('Labelled {} chunks...'.format(i))
    return data, np.array(accuracies)

def read_labelled_training_data(file):
    labels = []
    data = []
    with open(file, 'r', errors='ignore') as data_file:
        reader = csv.reader(data_file)
        for i, row in enumerate(reader):
            if i == 0:
                # First row are the labels
                labels.extend([float(acc) for acc in row])
                # Add an empty list for data entry
                data = [[] for _ in range(len(row))]
                continue
            for j, tweet in enumerate(row):
                data[j].append(tweet)
    return np.array(data), np.array(labels)

# Convert all inputs to meta-features
# use # of times the term occurred at least once in the data set for each feature
# and the mean number of occurrences
def convert_to_metafeatures(training_data):
    result = []
    lang_identifier = LanguageIdentifier.from_modelstring(model)
    for i, chunk in enumerate(training_data):
        langid_features = np.array([lang_identifier.instance2fv(tweet) for tweet in chunk])
        num_occurrences = langid_features.sum(axis=0)
        averages = [occurrences/len(chunk) for occurrences in num_occurrences]
        sparseness = (langid_features > 0).sum(axis=0)
        result.append(np.array(averages + sparseness))
        if i % 50 == 0: print('Converted {} inputs to meta-features...'.format(i))
    result = np.array(result)
    return result

def read_labelled_metafeatures_data(metafeatures_file):
    data = []
    labels = []
    with open(metafeatures_file, 'r', errors='ignore') as data_file:
        reader = csv.reader(data_file)
        for row in reader:
            data.append(list(map(float, row[:-1])))
            labels.append(float(row[-1]))
    return np.array(data), np.array(labels)

def read_global_english_data(global_english_file):
    global_english_tweets = []
    with open(global_english_file, 'r') as file:
        lineReader = csv.reader(file, delimiter='\t')
        for i, row in enumerate(lineReader):
            if i!= 0:
                tweet = row[3]
                is_english = int(row[4])
                if is_english == 1:
                    tweet = re.sub(r"(?:\@|\#|https?\://)\S+", "", tweet)
                    global_english_tweets.append(tweet)
    return global_english_tweets
    
if __name__ == '__main__':
    labelled_training_data_file = os.path.join(THIS_FOLDER, LABELLED_TRAINING_DATA_FILE)
    training_data = None
    labels = None
    if not os.path.exists(labelled_training_data_file):
        print('Creating new labelled data...')
        # Read the 1,600,000 tweets to use as training data, and split into chunks of roughly
        # 4000 tweets each, 400 chunks
        # NOTE: Some of the training data is incorrectly labelled as English. The incidence 
        #       of such errors seems small, though
        training_data = read_data(RAW_TRAINING_DATA_FILE, 5)
        training_data = np.array_split(training_data, 400)

        # Run the training data through langid to determine its accuracy on each
        training_data, labels = label_data(training_data, debug=False)
        with open(labelled_training_data_file, 'w') as csvfile:
            linewriter = csv.writer(csvfile)
            linewriter.writerow(labels)
            output_data = zip_longest(*training_data, fillvalue = '')
            linewriter.writerows(output_data)
    else:
        print('Reading labelled data from file...')
        training_data, labels = read_labelled_training_data(labelled_training_data_file)

    print('Converting inputs to metafeatures...')
    training_data = convert_to_metafeatures(training_data)

    # Train the SVM classifier on the meta features data set and predictive accuracy
    # Use 5 fold cross validation
    print('Training SVM regression model')

    # pass random_state for reproducible output
    # Tweaks so far: 
    # (1) using different kernels (no improvements), 
    # (2) using larger chunk sizes (improved training time, no accuracy improvement)
    # (3) Scaling feature and predictor variables
    # (4) Reduce number of features via recursive feature elimination
    X_train, X_test, y_train, y_test = train_test_split(training_data, labels, 
        test_size=0.4, random_state=0)

    regression = make_pipeline(
      StandardScaler(),
      RFE(estimator=svm.SVR(kernel='linear', C=1, cache_size=500), 
        step=0.33, n_features_to_select=100),
      svm.SVR(kernel='linear', C=1, cache_size=500)
    )
    regression = TransformedTargetRegressor(regressor=regression, transformer=StandardScaler())

    print('Computing 5-fold cross validation')
    cross_val_scores = cross_val_score(regression, X_train, y_train, 
        cv=5, verbose=1, scoring='neg_mean_absolute_error')
    print(cross_val_scores)

    print('Fitting model to training data')
    regression.fit(X_train, y_train)

    print('Running model on test data')
    predicted = regression.predict(X_test)
    print('SVM model predictions: {}'.format(predicted))
    print('SVM model MAE on test data: {}'.format(mean_absolute_error(y_test, predicted)))

    print('\nRunning model on challenge data sets\n')

    aave_file = os.path.join(THIS_FOLDER, AAVE_DATA)
    # split this into N samples of size 4000
    twitter_aave_data = read_data(aave_file, 5, tab_separated=True)

    # Only ~4000 tweets, so just use the whole sample
    scottish_file = os.path.join(THIS_FOLDER, SCOTTISH_DATA)
    scottish_data = read_data(scottish_file, 0)

    # Only ~5000 tweets, so just use the whole sample
    global_english_file = os.path.join(THIS_FOLDER, GLOBAL_ENGLISH_DATA)
    global_english_data = read_global_english_data(global_english_file)

    split_aave_data = np.array_split(twitter_aave_data, int(len(twitter_aave_data)/4000))
    aave_predictions = regression.predict(convert_to_metafeatures(split_aave_data))
    average_aave_prediction = sum(aave_predictions)/len(aave_predictions)
    print('Predicted accuracy for AAVE data {}'.format(average_aave_prediction))

    scottish_prediction = regression.predict(np.array(convert_to_metafeatures([scottish_data])))[0]
    print('Predicted accuracy for Scottish data {}'.format(scottish_prediction))

    global_english_prediction = regression.predict(np.array(convert_to_metafeatures([global_english_data])))[0]
    print('Predicted accuracy for global_english data {}'.format(global_english_prediction))
