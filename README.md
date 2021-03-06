This repo contains code written for our COS597E final project.

# Files
1. fetch_tweets.py contains code used to rehydrate tweet data sets that only contained tweet IDs (the original Scotland geotagged data set)
2. data_collection.py contains code used to further clean up tweet data, such as narrowing the scotland tweets down to those containing scottish english vocabulary
3. accuracy-predictor.py implements our meta-learning model for predicting the accuracy of langid.py on sets of English language tweets
4. test_language_predictions.py finds the accuracy of langid.py and CLD3 on each dataset.

# Reproduction Instructions

To reproduce the meta-learning model and its results:

1. Unzip the zip files found in the data/ directory
    - twitteraae_all_aa.zip should be unzipped as twitteraae_all_aa.tsv
    - scottish_tweets.zip should be unzipped as scottish_english_tweets.csv
    - global_english.zip should be unzipped as global_english.tsv
2. Install the python dependencies specified in the Pipfile
3. Run `python accuracy-predictor.py`

To reproduce the testing code:
1. Unzip the data/ files as above
2. Run `python test_language_predictions.py`
