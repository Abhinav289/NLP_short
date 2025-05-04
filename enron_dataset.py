import os
import nltk
import collections
import sklearn.datasets

stop_words = [
    "a", "an", "the",
    "in", "of", "to", "for", "on", "at", "by", "with", "from", "up", "down", "out",
    "and", "but", "or", "if", "because", "as", "so",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their",
    "is", "are", "was", "were", "will", "would", "can", "could", "may", "might",
    "must", "should",
    "not", "no", "there", "this", "that", "these", "those", "then", "now"
]

# The provided list of words can be categorized into several groups based on their grammatical functions:

# Determiners:
#   Articles: a, an, the
#   Demonstratives: this, that, these, those

# Prepositions: in, of, to, for, on, at, by, with, from, up, down, out

# Conjunctions:and, but, or, if, because, as, so

# Pronouns:
#   Personal pronouns: i, you, he, she, it, we, they, me, him, her, us, them
#   Possessive pronouns: my, your, his, her, its, our, their

# Auxiliary verbs:is, are, was, were, will, would, can, could, may, might, must, should

# Adverbs: not, no, there, then, now

# Negation:not, no


 #
def load_files(directory):
    result=[]
    for fname in os.listdir(directory):
        with open(fname,'r') as f:
            result.append(f.read())
    
    return result

def preprocess_sentence(sentence):
    lemmatizer= nltk.WordNetLemmatizer()

    # preprocessing pipeline
    processed_tokens=nltk.word_tokenize(sentence)
    processed_tokens=[w.lower() for w in processed_tokens]

    # find the least common tokens
    word_counter=collections.Counter(processed_tokens)
    print(word_counter)

    # common words= more counts
    # uncommon words = least counts
    # least common 10 words0
    uncommon_words=word_counter.most_common()[:-10:-1]
    # remove these tokens
    processed_tokens= [w for w in processed_tokens if w not in stop_words]
    processed_tokens= [w for w in processed_tokens if w not in uncommon_words]
    # lemmatize

    processed_tokens=[lemmatizer.lemmatize(w) for w in processed_tokens]

    return processed_tokens

# word_tokenize(), most_common() ,lemmatize()
def feature_extraction(tokens):
     # turn each word into a feature. The feature value= word count
     return dict(collections.Counter(tokens))
# dict(), Counter(i/p is a list)

def train_test_split(dataset, train_size=0.8): # 80 % training set
    num_training_egs= int(len(dataset) *train_size)
    return dataset[:num_training_egs],dataset[num_training_egs:]

  # first return is training dataset and the other is test dataset





x= os.chdir('enron1\spam')


positive_egs = sklearn.datasets.load_files(os.getcwd())
x=os.chdir(os.path.dirname(os.getcwd())) # previous working directory
print(os.getcwd())
x=os.chdir('ham')
dir_list=os.listdir(os.getcwd())
print(dir_list)
negative_egs = sklearn.datasets.load_files(os.getcwd())


print(len(positive_egs) , len(negative_egs))

import random
# labelling the egs
positive_egs=[(email,1) for email in positive_egs]
negative_egs=[(email,0) for email in negative_egs]
all_egs=positive_egs+negative_egs
random.shuffle(all_egs)

print('{} emails processed.'.format(len(all_egs)))

# .format()

featurized= [(feature_extraction(corpus),label) for corpus,label in all_egs]
# lemmatization: process of bringing a word back to its normal form

# BOW- Bag of Word features from txt data: dictory of unique words to no of occurences
print(featurized)
training_set, test_set = train_test_split(featurized, train_size = 0.7)

# model = nltk.classify.NaiveBayesClassifier.train(training_set)
model = nltk.classify.DecisionTreeClassifier.train(training_set)
training_error= nltk.classify.accuracy(model , training_set)

print('Model training complete. Accuracy on training set: {}'.format(training_error))

test_error= nltk.classify.accuracy(model , test_set)

print('Model testing complete. Accuracy on test set: {}'.format(test_error))