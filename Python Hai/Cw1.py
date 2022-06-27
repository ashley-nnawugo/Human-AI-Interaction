#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 09:52:38 2021

@author: ashleynnawugo
"""

import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
from urllib import request
import os
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from math import log10
from scipy import spatial
import numpy as np
import json
import tensorflow as tf
import tflearn

porter_stemmer = PorterStemmer()
sb_stemmer = SnowballStemmer('english')
lemmatiser = WordNetLemmatizer()

#log weighting method
def logfreq_weighting(vector):
    lf_vector = []
    for frequency in vector:
        lf_vector.append(log10(1+frequency))
    return np.array(lf_vector)

#cosine similarity 
def sim_cosine(vector_1, vector_2):
       similarity = 1 - spatial.distance.cosine(vector_1, vector_2) 
       return similarity
#opening json file
with open("Intent.json") as file:
    dataset = json.load(file)



labels = []
documents_x = []
documents_y = []
words = []       
xy = []

    
#Goes through intents list and looks through each individual intent
#tokenises and adds each intent to the words list
for intent in dataset["intents"]:
    for text in intent["text"]:
        tokenized_words = word_tokenize(text)
        words.extend(tokenized_words)
        documents_x.append(text) # adds all text to docuemnt
        documents_y.append(intent["intent"]) #adds all intents to a document
        xy.append((text, intent["intent"])) #creates list of intents to patterns of text corresponding to each intent 
    #adds intent to labels if it isn't in labels 
    if intent["intent"] not in labels:
        labels.append(intent["intent"])

#print(store_of)
#print(documents_y, documents_x)
#analyzer = CountVectorizer().build_analyzer()
#def stemmerfunction(doc):
#        return (sb_stemmer(doc) for w in analyzer(doc))

#Creating list of stemmed words
stemmed_words = []     
stemmed_words = [sb_stemmer.stem(w.lower()) for w in words]


vocabulary = [] 
vocabulary = sorted(list(set(stemmed_words))) #This is list of vocabulary from the dataset, contains each word only once

lower_words = [w.lower() for w in words] 

#print(stemmed_words) #print(vocabulary)

# 1) intents, 2)comparing input to types of input, 3) match input to intent of type of input 4) print a response from input 


#setting vectors to a list of 0 with length of vocab
vector = np.zeros(len(vocabulary))

#bag of words model
for wo in stemmed_words:
    bow = []
    lf_vector = []
    try:
        index = vocabulary.index(wo)
        vector[index] += 1
        bow.append(vector)
    except ValueError:
        continue
#print(bow)

#weigthing the bag of words model
lf_vector = logfreq_weighting(vector)    



training = []
output = []
out_vecotr = [0 for _ in range(len(labels))]
for x, doc in enumerate(documents_x):
    bag = []
    wrds = [sb_stemmer.stem(w.lower()) for w in [doc]]
    print(wrds)
    for w in words:
        if w in wrds:
            bag.append(1)
        else: 
            bag.append(0)
    outRow = out_vecotr[:]
    outRow[labels.index(documents_y[x])] = 1
    
    training.append(bag)
    output.append(outRow)
print(outRow)
print(bag)

training = np.array(training)
output = np.array(output)

query = 'how are you'

tok_query = word_tokenize(query)
stemmed_query = [sb_stemmer.stem(w.lower()) for w in tok_query]


vector_query = np.zeros(len(vocabulary))
for stem in stemmed_query:
    try:
        index_q = vocabulary.index(stem)
        vector_query[index_q] += 1
    except ValueError:
        continue
vector_query = logfreq_weighting(vector_query)   

simi = sim_cosine(vector_query, lf_vector)
print(simi)


tf.compat.v1.reset_default_graph()



#net = tflearn.input_data(shape=[None, len(training[0])])
#net - tflearn.fully_connected(net, 1000)
#net - tflearn.fully_connected(net, 1000)
#net - tflearn.fully_connected(net, len(output[0]), activation= "softmax")


#print(simi)

#print 

#X_train, X_test, y_train, y_test = train_test_split(documents_x, labels, stratify=labels, test_size=0.25, random_state=42)

#count_vect = CountVectorizer(stop_words=stopwords.words('english'))
#X_train_counts = count_vect.fit_transform(X_train)

#tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
#X_train_tf = tfidf_transformer.transform(X_train_counts)
#clf = LogisticRegression(random_state=0).fit(X_train_tf, y_train)
#X_new_counts = count_vect.transform(X_test)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#predicted = clf.predict(X_new_tfidf)k


def chat():
    return 
 


