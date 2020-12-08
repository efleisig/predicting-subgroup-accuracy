import os
import csv
import re
import collections
import copy
import random

import numpy as np
import matplotlib.pyplot as plt

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

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

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

def read_global_english_data(global_english_file):
    global_english_tweets = []
    locations = {}
    with open(global_english_file, 'r') as file:
        lineReader = csv.reader(file, delimiter='\t')
        for i, row in enumerate(lineReader):
            if i!= 0:
                tweet = row[3]
                is_english = int(row[4])
                if is_english == 1:
                    tweet = re.sub(r"(?:\@|\#|https?\://)\S+", "", tweet)
                    global_english_tweets.append(tweet)
                    locations[tweet] = row[1]
    return global_english_tweets, locations

def partition_by_location(tweets, locations):
    us_tweets = []
    other = []
    for chunk in tweets:
        for tweet in chunk:
            if locations[tweet] == 'US':
                us_tweets.append(tweet)
            else:
                other.append(tweet)
    return np.array(us_tweets), np.array(other)

def trim_to_short(tweets):
    result = []
    for chunk in tweets:
        for tweet in chunk:
            if len(tweet.split()) <= 5:
                result.append(tweet)
    return np.array(result)

def plot_challenge_data_performance_15(predicted, actual):
    N = 2

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(N)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, predicted, bar_width,
        alpha=opacity, color='b', label='Predicted Accuracy (with 15 Tweets)')

    rects2 = plt.bar(index + bar_width, actual, bar_width,
        alpha=opacity, color='g', label='Actual Accuracy')

    plt.xlabel('Challenge Data Set')
    plt.ylabel('Accuracy')
    plt.title('Predicted Accuracy vs Actual')
    plt.legend()

    plt.xticks(index, ['AAVE', 'Scottish English'])
    plt.yticks(np.arange(0, 1, 0.1))

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off

    plt.tight_layout()

def plot_challenge_data_performance(predicted, actual):
    N = 6

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(N)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, predicted, bar_width,
        alpha=opacity, color='b', label='Predicted Accuracy (with All Data)')

    rects2 = plt.bar(index + bar_width, actual, bar_width,
        alpha=opacity, color='g', label='Actual Accuracy')

    plt.xlabel('Challenge Data Set')
    plt.ylabel('Accuracy')
    plt.title('Predicted Accuracy vs Actual')
    plt.legend()

    plt.xticks(index, ['AAVE', 'Scottish English', 'Global English (All)', 
        'Global English (US)', 'Global English (Non-US)', 'Short Tweets'])
    plt.yticks(np.arange(0, 1, 0.1))

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off

    plt.tight_layout()

def plot_test_data(predicted, actual):
    random.seed(10)
    N = 10
    sample_indices = random.sample(range(0, len(predicted)), N)
    predicted_samples = [predicted[i] for i in sample_indices]
    actual_samples = [actual[i] for i in sample_indices]
    #data = [predicted_samples, actual_samples]

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(N)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, predicted_samples, bar_width,
        alpha=opacity, color='b', label='Predicted Accuracy')

    rects2 = plt.bar(index + bar_width, actual_samples, bar_width,
        alpha=opacity, color='g', label='Actual Accuracy')

    plt.xlabel('Tweet Set (Size 25)')
    plt.ylabel('Accuracy')
    plt.title('Test Data Predicted Accuracy vs Actual')
    plt.legend()

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    plt.tight_layout()

def plot_challenge_predictions(predictions):
    N = 6

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(N)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, predictions, bar_width,
        alpha=opacity, color='b')

    plt.xlabel('Challenge Dataset')
    plt.ylabel('Predicted Accuracy')
    plt.title('Predicted Accuracy on Challenge Datasets')
    plt.xticks(index, ['AAVE', 'Scottish English', 'Global English (All)', 
        'Global English (US)', 'Global English (Non-US)', 'Short Tweets'])
    plt.yticks(np.arange(0, 1, 0.1))

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,)         # ticks along the top edge are off

    plt.tight_layout()
    
if __name__ == '__main__':
    global_english_file = os.path.join(THIS_FOLDER, GLOBAL_ENGLISH_DATA)
    training_data, locations = read_global_english_data(global_english_file)
    print('Using global english data set with {} tweets'.format(len(training_data)))
    training_data = np.array_split(training_data, 200)
    print('Split into {} subsets with {} tweets each'.format(len(training_data), len(training_data[0])))
    training_data, labels = label_data(training_data)

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
        test_size=0.3, random_state=0)

    X_train = convert_to_metafeatures(X_train)
    X_test_og = copy.deepcopy(X_test)
    X_test = convert_to_metafeatures(X_test)

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

    avg_test_prediction = sum(predicted)/len(predicted)
    print('SVM model average predicted accuracy: {}'.format(avg_test_prediction))

    plot_test_data(predicted, y_test)

    print('\nRunning model on challenge data sets\n')
    challenge_predictions = []

    aave_file = os.path.join(THIS_FOLDER, AAVE_DATA)
    twitter_aave_data = read_data(aave_file, 5, tab_separated=True)
    print('Read {} AAVE tweets'.format(len(twitter_aave_data)))

    scottish_file = os.path.join(THIS_FOLDER, SCOTTISH_DATA)
    scottish_data = read_data(scottish_file, 0)
    print('Read {} scottish tweets'.format(len(scottish_data)))

    split_aave_data = np.array_split(twitter_aave_data, int(len(twitter_aave_data)/25))
    aave_predictions = regression.predict(convert_to_metafeatures(split_aave_data))
    average_aave_prediction = sum(aave_predictions)/len(aave_predictions)
    print('Predicted accuracy for AAVE data {}'.format(average_aave_prediction))
    challenge_predictions.append(average_aave_prediction)

    split_scottish_data = np.array_split(scottish_data, int(len(scottish_data)/25))
    scot_predictions = regression.predict(convert_to_metafeatures(split_scottish_data))
    average_scot_prediction = sum(scot_predictions)/len(scot_predictions)
    print('Predicted accuracy for Scottish data {}'.format(average_scot_prediction))
    challenge_predictions.append(average_scot_prediction)

    # Global English used as train and test data, so append the average test predicted accuracy
    # as the global english prediction
    challenge_predictions.append(avg_test_prediction)

    us_tweets, other_tweets = partition_by_location(X_test_og, locations)
    print('Partitioned into {} US tweets and {} non-US tweets'.format(len(us_tweets), len(other_tweets)))

    us_tweet_sets = np.array_split(us_tweets, int(len(us_tweets)/25))
    us_predictions = regression.predict(convert_to_metafeatures(us_tweet_sets))
    average_us_prediction = sum(us_predictions)/len(us_predictions)
    print('Predicted accuracy for Global English (US) tweet data {}'.format(average_us_prediction))
    challenge_predictions.append(average_us_prediction)

    other_tweet_sets = np.array_split(other_tweets, int(len(other_tweets)/25))
    other_predictions = regression.predict(convert_to_metafeatures(other_tweet_sets))
    average_other_prediction = sum(other_predictions)/len(other_predictions)
    print('Predicted accuracy for outside Global English (non-US) tweet data {}'.format(average_other_prediction))
    challenge_predictions.append(average_other_prediction)

    short_tweets = trim_to_short(X_test_og)
    print('Predicting accuracy on {} short tweets'.format(len(short_tweets)))
    short_tweet_sets = np.array_split(short_tweets, int(len(short_tweets)/25))
    short_predictions = regression.predict(convert_to_metafeatures(short_tweet_sets))
    average_short_prediction = sum(short_predictions)/len(short_predictions)
    print('Predicted accuracy for short tweet data {}'.format(average_short_prediction))
    challenge_predictions.append(average_short_prediction)

    plot_challenge_predictions(challenge_predictions)

    # Predictions using the same tweets for AAVE and Scottish English presented to human experts
    # in our survey
    survey_AAVE = np.array([
        "I'm nt understandinn yy icnt changee mi namee or changee mi profile pic on twitter anymoree!! Ugghh!!",
        "my full name is CHRISTOPHER DANE MOSS if you did not know",
        "Llf dont look at me dont look at me",
        "I wanna know why Emerald so MFn ashy tho",
        "Came home at 3am & left rite back out",
        "Thrax pack for da low, them 7s go for da huncho!",
        "Real Talk I Aint The Bitch Type But I Might Be Yo Bitch Type",
        "Bernie Was High After On The Phone lls he still my CRUSH ! (:",
        "I tried to break away like Kunta...",
        "Me and my baby >>>>>> hope it stays this way even though I always mess things up",
        "Shee Look Pretty With Her Like That...",
        "That girl at and was bad af tall dark skin and wish a nigga got her number",
        "Im acting up? You got my pw read the shit and stop assuming",
        "Dang I haven't got ah text from  since early this morning lls that's crazy but Hey"
        "I want my djs at vibe lounge tonight to play sum hot new reggae tonight ..I'm feelin da vibe"
    ])
    print('Prediction on survey AAVE data: {}'.format(regression.predict(convert_to_metafeatures([survey_AAVE]))[0]))

    survey_Scots = np.array([
        "GIT THE DIVING OOT AE EDINBURGH GIT IT TAE THE TIME CAPSULE",
        "See if that circus Vegas is 'Europes largest circus' WIT THEY DAIN IN THE PLAYDROME CARPARK!?",
        "8 mile is on. YASSSS",
        "What a SHITE night",
        "Freakishly large bottom lip and an almost non existent top lip. Pure GID balance, ok.",
        "Dj hixxy - discoland DAE ITTTT! ",
        "20 TIMES 20 TIMES YOUR STILL SHITE",
        "FUCK AYE",
        "I genuinely want to punch the fuck out my mum she's decided to come in n tidy my room, A WANTY SLEEP",
        "Aargh Labour Press team after my blood because we interviewed Gordon Brown against a Say AYE ( to a Killie pie sign..) ",
        "Fucksake a canny believe a neck nominated Boaby sands ..... YEV GOT 24 YA CUNT N A WANT THE VIDEO ON FACEBOOK NAE CHEATIN PAWL",
        "Watchin Art Attack lit aht AYE WAIT N ILL JIST PULL A BOTTO A PVA GLUE FAE MA ARSE NEIL x",
        "Had to do some serious grafting today , dno how I feel about this. Traumatised , shocked , wtf just happened.  I DIDNY GET MA NAP TIME !",
        "Hate when people are like 'I've done nothing'... AYE THATS THE PROBLEM",
        "Open the door I say for Iam a bogas gasman and am here TAE ransack YER HOOOSE!!! NOOO!!! Open the door plz...."
    ])

    print('Prediction on survey Scottish data: {}'.format(regression.predict(convert_to_metafeatures([survey_Scots]))[0]))

    actual_accuracies = [0.872, 0.936, 0.750, 0.892, 0.584, 0.704]
    print("Actual accuracies: {}".format(actual_accuracies))
    print('Mean absolute error: {}'.format(mean_absolute_error(actual_accuracies, challenge_predictions)))
    plot_challenge_data_performance(challenge_predictions, actual_accuracies)

    short_sample_predictions = [0.7934, 0.7884]
    actual = [0.872, 0.936]
    print('Mean absolute error on only 15 tweet samples: {}'.format(mean_absolute_error(actual, short_sample_predictions)))

    plot_challenge_data_performance_15(short_sample_predictions, actual)

    plt.show()