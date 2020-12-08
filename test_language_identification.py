
import cld3
import langid
import csv
import re
import matplotlib.pyplot as plt
import numpy as np

def get_misclassification_stats(dataset):
    total_tweets = 0
    misclassified_cld = 0
    misclassified_langid = 0
    for index, t in enumerate(dataset):
        if len(t) > 0:
            #print(index, t)
            # Remove mentions, URLs, and hashtags
            t = re.sub(r"(?:\@|\#|https?\://)\S+", "", t)
            t = " ".join(t.split()[:8])

            if len(t) > 0:

                cld_prediction = cld3.get_language(t)
                langid_prediction = langid.classify(t)

                if cld_prediction[0] != "en":
                    # print(t, cld_prediction[0])
                    misclassified_cld += 1
                if langid_prediction[0] != "en":
                    # print(t, langid_prediction[0])
                    misclassified_langid += 1

                total_tweets += 1

    print("CLD accuracy: ", total_tweets-misclassified_cld, "/", total_tweets, "=", (total_tweets-misclassified_cld) / total_tweets)
    print("Langid accuracy: ", total_tweets-misclassified_langid, "/", total_tweets, "=", (total_tweets-misclassified_langid) / total_tweets)

def get_avg_tweet_length(dataset):
    s = sum([len(re.sub(r"(?:\@|\#|https?\://)\S+", "", t).split()) for t in dataset])
    print(s/len(dataset))
    return s/len(dataset)


def plot_results():
    plt.figure()
    x = ["Global", "US", "Non-US", "AAVE", "Scottish", "Short"]
    y1 = [.75, .892, .584, .872, .936, .704]
    y2 = [.587, .726, .424, .529, .76, .317]
    x_pos = [i for i, _ in enumerate(x)]
    width = 0.35
    plt.bar(np.arange(6), y1, width, color='blue', label="langid")
    plt.bar(np.arange(6)+width, y2, width, color='green', label="CLD3")

    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title("Language Identification Accuracy by Dataset")

    plt.xticks(x_pos, x)
    plt.legend()
    plt.show()



# Load datasets

# Scottish English
scot_tweets = open("data/scottish_english_tweets.csv", "r").read().split('\n')#csv.reader("scottish_english_tweets.csv", delimiter = "\n"))

# AAVE
aave_0 = open("data/twitteraae_all_aa", "r").read().split('\n')
aave_tweets = []
for row in aave_0:
    if len(row.split("\t")) >= 5:
        #print("no go", row)
        aave_tweets.append(row.split("\t")[5])

#print(aave_tweets[:10])

# Global English
global_tweets = []
global_tweets_us = []
global_tweets_non_us = []
global_tweets_short = []
with open('data/all_annotated.tsv', newline='') as inputfile:
    for row in csv.reader(inputfile, delimiter = "\t"):
        if row[4] == "1": # Keep only tweets that are definitely English

            global_tweets.append(row[3])
            if row[1] == "US":
                global_tweets_us.append(row[3])
            else:
                global_tweets_non_us.append(row[3])

            words = row[3].split()
            if len(words) <= 5:
                global_tweets_short.append(row[3])

print("Scottish English:")
get_misclassification_stats(scot_tweets)
print("AAVE:")
get_misclassification_stats(aave_tweets)
print("Global English:")
get_misclassification_stats(global_tweets)
print("US English:")
get_misclassification_stats(global_tweets_us)
print("Non-US English:")
get_misclassification_stats(global_tweets_non_us)
print("Short Tweets:")
get_misclassification_stats(global_tweets_short)

plot_results()


