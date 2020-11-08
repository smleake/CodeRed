import tweepy
import os
import sys 

import nltk
import sklearn
import random 
import re
import pickle
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
import string

consumer_key=os.environ.get('TWITTERKEY')
consumer_secret=os.environ.get('TWITTERSECRET')
access_token=os.environ.get('TWITTERTOKEN')
access_token_secret=os.environ.get('TWITTERTOKENSECRET')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

timeline = api.user_timeline(screen_name='realDonaldTrump', count=10)

tweets = []
all_words = []
doc = []

for tweet in timeline:
    tweets.append(tweet.text)
    print(tweet.text)

def StemNormalize(text):
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in nltk.word_tokenize(text) if word not in string.punctuation]
    return [word for word in words if word not in stopwords.words('english')]

postive_t = twitter_samples.strings('positive_tweets.json')
negative_t = twitter_samples.strings('negative_tweets.json')

for word in postive_t:
    doc.append((word, "pos"))
    
# sent = SentimentIntensityAnalyzer()





