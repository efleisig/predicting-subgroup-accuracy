#This code creates the dataset from Corpus.csv which is downloadable from the
#internet well known dataset which is labeled manually by hand. But for the text
#of tweets you need to fetch them with their IDs.
import tweepy
import csv
import time
import os

from secrets import consumer_key, consumer_key_secret, access_token, access_token_secret

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# This method creates the training set
def loadTweetsFromIds(tweetFile, targetResultFile):

    counter = 0
    tweets = []

    with open(tweetFile, 'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter='\t')
        for row in lineReader:
            tweets.append({"tweet_id": row[1]})

    sleepTime = 2
    dataSet = []

    tweet_count = len(tweets)
    print 'Fetching {} tweets'.format(tweet_count)
    try:
        for i in range((tweet_count / 100) + 1):
            # Catch the last group if it is less than 100 tweets
            end_loc = min((i + 1) * 100, tweet_count)
            new_tweets = api.statuses_lookup([tweet["tweet_id"] for tweet in tweets[i * 100:end_loc]])
            print 'Fetched a group of tweets...'
            for tweet in new_tweets:
                print tweet.text
            dataSet.extend([tweet.text for tweet in new_tweets])
    except Exception as e:
        print 'Failed to fetch tweets'
        print e

    # for tweet in tweets:
    #     try:
    #         tweetFetched = api.get_status(tweet["tweet_id"])
    #         print("Tweet fetched" + tweetFetched.text)
    #         tweet["text"] = tweetFetched.text
    #         dataSet.append(tweet)
    #         time.sleep(sleepTime)
    #     except:
    #         print("Failed to fetch tweet with id {}".format(tweet["tweet_id"]))
    #         continue

    with open(targetResultFile, 'w') as csvfile:
        linewriter = csv.writer(csvfile)
        for tweet in dataSet:
            try:
                linewriter.writerow([tweet.encode('ascii', 'ignore')])
            except Exception as e:
                print(e)

    print 'Fetched and saved {} tweets'.format(len(dataSet))
    return dataSet

# Code starts here
# This is corpus dataset
corpusFile = os.path.join(THIS_FOLDER, 'GU.tsv')
# This is my target file
targetResultFile = os.path.join(THIS_FOLDER, 'UK_Tweets.csv')
# Call the method
resultFile = loadTweetsFromIds(corpusFile, targetResultFile)
