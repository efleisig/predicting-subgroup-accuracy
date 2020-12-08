import csv
import os
import math
import collections
import string
import nltk
import random

import numpy as np
from nltk.corpus import stopwords

from sage import sage

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
nltk.download('stopwords')
english_stopwords = stopwords.words('english')

SCOT_WORDS = """
    AULD AWFY BAIRNS BAW BAWS BELTER BELTERS BEVY BOABY BOKE BRAW BURD BURDS
    CRABBIT DAFTY DAIN DEFOS DOON DUGS DYNO ECCIES FANNYS FEART FITBA FUD GAD GAWN GEES GID GRANDA
    GREETIN HAME HAW HING HINK HOOSE HOWLIN HUNNERS JIST LADDIE LASSIE LASSIES MANKY MAW MAWS MORRA
    MONGO PISH PISHED PISHING RAGIN ROASTER SARE SHITE SHITEY STEAMIN SUHIN WEANS WHITEY
    ACH ANAW ANO AWRIGHT AWRITE AWRYT AYE EH NAE NAW OOFT YASS YASSS YASSSS YASSSSS YIP
    CANNAE CANNI CANY CANA DEH DINI DINNY DIDNY DOESNY GONNAE GONY ISNY WANTY YER YIR
    BOOT AE AFF ATS DAE FAE HAE MASEL MASELF OAN OOR OOT TAE WAE WAN WI WIS YERSEL YI YIN YOUS GTF MWI
    """
SCOT_WORDS = SCOT_WORDS.split()

def find_substring(needle, haystack):
    index = haystack.find(needle)
    if index == -1:
        return False
    if index != 0 and haystack[index-1] not in string.whitespace:
        return False
    L = index + len(needle)
    if L < len(haystack) and haystack[L] not in string.whitespace:
        return False
    return True

def accept_word(word):
   return (word not in english_stopwords) and ('#' not in word) and ('@' not in word) and ('http' not in word)

if __name__ == '__main__':
    # uk_tweets = []
    # with open(os.path.join(THIS_FOLDER, 'UK_Tweets.csv'), 'r') as uk_file:
    #     reader = csv.reader(uk_file)
    #     for row in reader:
    #         uk_tweets.append(row[0])

    scotland_tweets = []
    with open(os.path.join(THIS_FOLDER, 'data/Scotland_Tweets.csv'), 'r') as scotland_file:
        reader = csv.reader(scotland_file)
        for row in reader:
            scotland_tweets.append(row[0])
    
    print(len(scotland_tweets))
    
    scottish_tweets = []
    for tweet in scotland_tweets:
        if any(find_substring(scottish_word.lower(), tweet.lower()) for scottish_word in SCOT_WORDS):
            scottish_tweets.append(tweet)
    #print scottish_tweets
    print(len(scottish_tweets))

    print(random.sample(scottish_tweets, 10))

    # with open(os.path.join(THIS_FOLDER, 'scottish_english_tweets.csv'), 'r') as scotland_file:
    #     reader = csv.reader(scotland_file)
    #     for row in reader:
    #         import pdb; pdb.set_trace()
    #         scotland_tweets.append(row[0])

    with open(os.path.join(THIS_FOLDER, 'scottish_english_tweets.csv'), 'w') as csvfile:
        linewriter = csv.writer(csvfile)
        for tweet in scottish_tweets:
            try:
                linewriter.writerow([tweet.encode('ascii', 'ignore').decode()])
            except Exception as e:
                print(e)

    # aave_tweets = []
    # with open(os.path.join(THIS_FOLDER, 'twitteraae_all_aa'), 'r') as file:
    #     lineReader = csv.reader(file, delimiter='\t')
    #     for row in lineReader:
    #         if len(row[5].split()) > 6:
    #             aave_tweets.append(row[5])
    
    # print(len(aave_tweets))
    # print(random.sample(aave_tweets, 15))

    # global_english_tweets = []
    # with open(os.path.join(THIS_FOLDER, 'global_english.tsv'), 'r') as file:
    #     lineReader = csv.reader(file, delimiter='\t')
    #     for i, row in enumerate(lineReader):
    #         if i!= 0:
    #             tweet = row[3]
    #             is_english = int(row[4])
    #             if is_english == 1 and '@' not in tweet and 'http' not in tweet:
    #             # if len(row[5].split()) > 6:
    #                 global_english_tweets.append(tweet)
    
    # print(len(global_english_tweets))
    # random_sample = random.sample(global_english_tweets, 15)
    # for tweet in random_sample:
    #     print('- {}'.format(tweet))

    # with open(os.path.join(THIS_FOLDER, 'scottish_english_tweets.csv'), 'w') as csvfile:
    #     linewriter = csv.writer(csvfile)
    #     for tweet in scottish_tweets:
    #         try:
    #             linewriter.writerow([tweet.encode('ascii', 'ignore')])
    #         except Exception as e:
    #             print(e)
    
    # first compute log-probabilities of each word in the UK_tweets
    # I think this means:
    #
    # for each word in vocab:
    #       log(count / total_vocab_size)
    #
    # TODO: filter out stop words, hashtags, mentions, and URLs
    # counts = collections.Counter()
    # for tweet in uk_tweets:
    #     for word in tweet.split():
    #         if accept_word(word):
    #             curr_count = counts.get(word, 0)
    #             counts[word] = curr_count + 1
    
