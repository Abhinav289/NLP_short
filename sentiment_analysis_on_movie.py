import os
import nltk
nltk.download('stopwords')
import collections
import sklearn
from sklearn import (datasets,feature_extraction,model_selection,linear_model,naive_bayes,ensemble)


def extract_features(corpus):
    # extract TF-IDF features from corpus (Term frequency - Inverse document frequency)
    # vectorize means we turn non-numerical data into an array of numbers

    sa_stop_words=nltk.corpus.stopwords.words("english")

    # words that invert the meaning of a sentence

    white_list=[
        'what','but','if','because','as','until','against',
        'up','down','in','out','on','off','over','under','again',
        'further','then','once','here','there','why','how','all','any',
        'most','other','some','such','no','not','nor','only','own',
        'same','so','than','too','can','will','just','don','should'
    ]

    sa_stop_words=[sw for sw in sa_stop_words if sw not in white_list]
    count_vectorizer = feature_extraction.text.CountVectorizer(
        lowercase= True,
        tokenizer = None, # if we have custom tokenizer , we need to input the name of that function here
        stop_words=sa_stop_words, # default: "english"
        min_df=2 ,# minimum document frequency , i.e, the word must appear more than once
        ngram_range=(1,2),
        token_pattern=r"(?u)\b\w\w+\b"
    )
    processed_corpus = count_vectorizer.fit_transform(corpus)
    processed_corpus =feature_extraction.text.TfidfTransformer().fit_transform(processed_corpus)
    return processed_corpus


# data_directory = 'enron1'
# movie_sentiment_data = datasets.load_files(data_directory)
# print('{} files loaded.'.format(len(movie_sentiment_data)))
# print('They contain the following classes: {}.'.format(movie_sentiment_data.target_names))

# movie_tfidf= extract_features(movie_sentiment_data.data)

# X_train,Y_train,X_test,Y_test = model_selection.train_test_split(movie_tfidf,movie_sentiment_data.target,test_size=0.30,random_state=42)

# model = sklearn.linear_model.LogisticRegression()
# model.fit(X_train,Y_train)
# print('Model performance: {}'.format(model.score(X_test,Y_test)))

# Y_pred = model.predict(X_test)

# for i in range(5):
#     print('Review:\n{review}\n-\nCorrect label:{correct}: predicted={predict}'.format(review=X_test[i], correct=Y_test[i],predict=Y_pred[i]))

import tarfile
import numpy as np 

movie_reviews=[]
with tarfile.open("movie_review.tar.gz", "r:gz") as tar:
    for member in tar.getmembers():
         f = tar.extractfile(member)
         if f is not None:
             content = f.read()
             content_string = content.decode('utf-8')
             movie_reviews.append(content_string)
            #  print(f"First {100} characters of the movie review:")
            #  print(content_string[:100]) 
             
# print('{} files loaded.'.format(len(content)))
# print(type(content))

# The extract_features function returns a TF-IDF (Term Frequency-Inverse Document Frequency) representation of the input corpus. 
# This representation is a matrix where each row corresponds to a document and each column corresponds to a term (word). 
# The value at each cell represents the importance of that term in the corresponding document.

# CountVectorizer: Converts the text corpus into a matrix of word counts. 
# Each row represents a document, and each column represents a unique word. 
# The value at each cell is the number of times that word appears in the corresponding document.
# TfidfTransformer: Calculates the TF-IDF weight for each term in each document. 
# The TF-IDF weight is a measure of how important a term is in a document relative to the corpus as a whole.
# It is calculated by multiplying the term frequency (TF) by the inverse document frequency (IDF). 
# TF measures how frequently a term appears in a document, while IDF measures how rare the term is in the corpus.
print(movie_reviews[1])
movie_tfidf=extract_features(movie_reviews)

print(movie_tfidf[1])



X_train,Y_train,X_test,Y_test = model_selection.train_test_split(movie_tfidf,movie_reviews,test_size=0.30,random_state=42)

clf1 = linear_model.LogisticRegression()
clf1.fit(X_train[:10].toarray(),Y_train[:10].toarray())
print('Model performance: {}'.format(clf1.score(X_test[:10],Y_test[:10])))
Y_pred_log = clf1.predict(X_test)


# clf2 = linear_model.SGDClassifier()
# clf2.fit(X_train,Y_train)
# print('Model performance: {}'.format(clf2.score(X_test,Y_test)))
# Y_pred_sdg = clf2.predict(X_test)

# clf3 = naive_bayes.MultinomialNB()
# clf3.fit(X_train,Y_train)
# print('Model performance: {}'.format(clf3.score(X_test,Y_test)))
# Y_pred_multi = clf3.predict(X_test)

# clf4 = naive_bayes.BernoulliNB()
# clf4.fit(X_train,Y_train)
# print('Model performance: {}'.format(clf4.score(X_test,Y_test)))
# Y_pred_berno = clf4.predict(X_test)
# for i in range(5):
#     print('Review:\n{review}\n-\nCorrect label:{correct}: predicted={predict}'.format(review=X_test[i], correct=Y_test[i],predict=Y_pred[i]))