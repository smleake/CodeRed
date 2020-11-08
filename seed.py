import tweepy
import os
import sys 

import nltk
import random 
import re
import pickle
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
import string 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
# consumer_key=os.environ.get('TWITTERKEY')
# consumer_secret=os.environ.get('TWITTERSECRET')
# access_token=os.environ.get('TWITTERTOKEN')
# access_token_secret=os.environ.get('TWITTERTOKENSECRET')
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
# api = tweepy.API(auth)
# timeline = api.user_timeline(screen_name='realDonaldTrump', count=10)
stopwords = stopwords.words('english')
tweets = []
all_words = []
doc = []

# for tweet in timeline:
#     tweets.append(tweet.text)
#     print(tweet.text)

def StemNormalize(text):
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in nltk.word_tokenize(text) if word not in string.punctuation]
    return [word for word in words if word not in stopwords.words('english')]

postive_t = twitter_samples.strings('positive_tweets.json') #5,000 pos
negative_t = twitter_samples.strings('negative_tweets.json') #5,000 neg
text = twitter_samples.strings('tweets.20150430-223406.json') #20,000 neutral
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

import re, string

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stopwords))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stopwords))


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos = get_all_words(positive_cleaned_tokens_list)
freq_dist = nltk.FreqDist(all_pos)

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[0:7000]
test_data = dataset[7000:10000]

classifier = nltk.NaiveBayesClassifier.train(train_data)

f = open('seeded_model.pickle', 'wb')
pickle.dump(classifier, f)
f.close()




