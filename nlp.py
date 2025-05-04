import re
import multiprocessing as mp # for parallel processing in python
# reading a simple text file in english into memory

# def process(line):
#     print(line)


f=open("C:\Desktop\cpp\python\About_me.txt","r") # file pointer
text=f.read() # string format
f.close()
print(text)


# # method 1: readlines()
# # with open("C:\Desktop\cpp\python\About_me.txt") as f:
# #     data =f.readlines()
# #     for line in data:
# #         process(line)

# # # method 2: use context managers to make sure file pointers are closed correctly.

# # with open("C:\Desktop\cpp\python\About_me.txt") as f:
# #    # to handle large text files we use file as an iterator
# #    for line in f:
# #        # each line is a wastage collected by python unless it is 
# #        #refernced elsewhere
# #        process(line)

# # # method 3: multiprocessing

# # pool = mp.Pool(1) # no of pools= no of CPU cores
# # jobs = []

# # with open("C:\Desktop\cpp\python\About_me.txt","r") as f:
# #     for line in f:
# #         jobs.append( pool.apply_async(process, (line)))

# # # process here is an analogous to OS processes where process is defined for each line in the text file
# # for job in jobs:
# #     job.get() # wait for all jobs to finish

# # pool.close()
# x=f.read()
# print("Splitting...")

# Splitting text into words with spaces
corpus=("Have a good night! It's raining here")

# method 1: direct split() splits the txt around white spaces
# print(corpus.split())

# method 2: removing punctuation marks 
# punctuation=".',?"

# for p in punctuation:
#     corpus=corpus.replace(p,'')
# print(corpus.split())

# method 3: regular expression (regex)
word_regex =r'\W+' # a raw string: one or more (+) non-word characters (\W)
split_corpus=re.split(word_regex, text)
print(split_corpus)

# observation: 's is treated as another word or it is joined as xxxs which are meaningless

# a better regex
# converting text into lists of lower case tokens

better_regex=r"(\w[\w']*\w|\w)"  # word character + 0 or more word characters or 's + word characters 
                                    # OR
                                # just a word character

word_matcher=re.compile(better_regex)
word_matcher.findall(corpus)

# same thing using function
print("\n THE DIFFERENCE...\n")

def split_into_words(line):
    better_regex=r"(\w[\w']*\w|\w)"  
    word_matcher=re.compile(better_regex)
    return word_matcher.findall(line)

processed_corpus=[]

with open('About_me.txt') as f:
    # to handle large text files , we use file as an iterator
    for line in f:
        processed_corpus.extend(split_into_words(line))

# all words are in lower case
processed_corpus=[w.lower() for w in processed_corpus]
print(processed_corpus)

# used functions: compile(), findall(), extend(), lower(),split(), Counter(), most_common()

import collections
# word count of each word in sorted manner

word_counter=collections.Counter(processed_corpus)
print(word_counter)

# common words= more counts
# uncommon words = least counts
# least common 10 words
uncommon_words=word_counter.most_common()[:-10:-1]
processed_corpus= [w for w in processed_corpus if w not in uncommon_words]
print("Uncommon words removal:\n")
print(len(processed_corpus))
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
processed_corpus= [w for w in processed_corpus if w not in stop_words]
print("stop words removal:\n")
print(len(processed_corpus))


