#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:02:08 2021

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
import random


sb_stemmer = SnowballStemmer('english')

#opening json file
with open("Intent.json") as file:
    dataset = json.load(file)
    
    
words = []
labels = []
docs_x= []
docs_y = []
xy = []
#Goes through intents list and looks through each individual intent
#tokenises and adds each intent to the words list
for intent in dataset["intents"]:
    for pattern in intent["text"]:
        wrds = word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern) # adds all text to docuemnt
        docs_y.append(intent["intent"]) #adds all intents to a document
        xy.append((pattern, intent["intent"])) #creates list of intents to patterns of text corresponding to each intent 
    #adds intent to labels if it isn't in labels 
    if intent["intent"] not in labels:
        labels.append(intent["intent"])
        
words = [sb_stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))

labels = sorted(labels)


training = []
output = []
out_empty = [0 for _ in range(len(labels))]


for x, doc in enumerate(docs_x):
    bag = []
    wrds = [sb_stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else: 
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1


    training.append(bag)
    output.append(output_row)
    
training = np.array(training)
output = np.array(output)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation= "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch= 1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def bag_of(s, words):
    bag = [0 for _ in range(len(words))]
    global word_tokenize
    s_words = word_tokenize(s)
    s_words = [sb_stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = (1)
    return np.array(bag)
    
def chat():
    print("Talk with the bot!, type quit to stop")
    while True:
        inpu = input("You:")
        if inpu.lower() == "quit":
            break
        results = model.predict([bag_of(inpu, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]
        
        for tg in dataset["intents"]:
            if tg['intent'] == tag:
                response = tg['responses']
        print(random.choice(response))
        
chat()    
    
    
    

