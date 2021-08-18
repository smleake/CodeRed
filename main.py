import pickle
import nltk
import random
from seed import test_data, remove_noise, word_tokenize, get_tweets_for_model
import matplotlib.pyplot as plt

f = open('seeded_model.pickle', 'rb')
classifier = pickle.load(f)
f.close()

queries = []
negativeQuery = []
positiveQuery = []
neutralQuery = []
cleanPosTokens = []
cleanNegTokens = []
queries.append("My girl will forever fascinate me... utterly beautiful")
queries.append("music is the answer")
queries.append("im top 5 iron 1 valorant players")

for word in queries:
    cus_tokens = remove_noise(word_tokenize(word))
    result = classifier.classify(dict([token, True] for token in cus_tokens))
    if result == 'Negative':
        print("Negative: ", word)
        negativeQuery.append(word)
        cleanNegTokens.append(cus_tokens)
    if result == 'Positive':
        print("Positive: ", word)
        positiveQuery.append(word)
        cleanPosTokens.append(cus_tokens)
    if result == 'Neutral':
        print("Neutral: ", word)
        neutralQuery.append(word) 
ratio = len(negativeQuery) / len(queries)
cleanPosTokens_M = get_tweets_for_model(cleanPosTokens)
cleanNegTokens_M = get_tweets_for_model(cleanNegTokens)
pos_data = [(tweet_dict, "Positive")
                     for tweet_dict in cleanPosTokens_M]
neg_data = [(tweet_dict, "Negative")
                     for tweet_dict in cleanNegTokens_M]    
accr_of_set = pos_data + neg_data                                      

random.shuffle(accr_of_set)

print("Accuracy:", nltk.classify.accuracy(classifier, accr_of_set)*100, "%")
print(ratio)

slices = [len(negativeQuery) / len(queries), len(positiveQuery) / len(queries), len(neutralQuery) / len(queries)]
labels = ['Negative', 'Positive', 'Neutral']
colors = ['red', 'blue', 'gray']

plt.pie(slices, labels=labels, colors=colors)
plt.title("Sentiment Results of Tweets")
plt.tight_layout()
plt.show()

