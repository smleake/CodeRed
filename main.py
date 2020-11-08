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
queries.append("@ oil field workers that put money over human rights.... i heard mcdonalds is hiring")
queries.append("This is so surreal. As a guy who spent most of his life in Georgia, I'm seeing a majority of the country rallying around us and the energy and the compassion behind it is undeniable. Change is coming and we got to give it everything we've got.")
queries.append("My girl will forever fascinate me... utterly beautiful")
queries.append("Up like Joe Biden")
queries.append("Just beat a Biden voter to death with a hammer")
queries.append("music is the answer")
queries.append("Main hobby.. bitin into pussy like it’s an apple")
queries.append("Big MILF guy")
queries.append("I'm so mentally unstable now it's ridiculous, I've never cried so much as I had the last 2 years shits ridiculous")
queries.append("im top 5 iron 1 valorant players")
queries.append("have no idea why conservatives are so scared of Biden?? you literally don’t have anything to lose??? unless you’re some insanely rich jeff bezos type mf. I can’t handle these cry baby straight white people who think they’re at risk. HOW YALL THINK MINORITIES FEEL 24/7??!!!")

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

