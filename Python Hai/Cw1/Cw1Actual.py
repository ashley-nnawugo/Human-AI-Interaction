#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:53:18 2021

@author: ashleynnawugo
"""

from operator import itemgetter, pos
from os import EX_CANTCREAT, read
import warnings
import nltk
import pickle
from nltk import stem
from nltk import corpus
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
from urllib import request
import csv
import os
from nltk.util import pr
from numpy.lib.npyio import load
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from math import log10
from scipy import spatial
import random as rnd
import numpy as np

# Stemmer, analyzer
sb_stemmer = SnowballStemmer('english')
analyzer = CountVectorizer().build_analyzer()

#function for stemming documents 
def stemmed_words(doc):
    return (sb_stemmer.stem(w) for w in analyzer(doc))

#for stemming and removing stopwords
stem_vectorizer = CountVectorizer(analyzer= stemmed_words, stop_words=stopwords.words('english'))
#for stemming, not removing stopwords, will be used on small talk dataset
#stem_only_vectorizer = CountVectorizer(analyzer= stemmed_words)

docs_names = {
    "questions" : "QuestionAnswer.csv",
    "smallTalk" : "SmallTalkAlternate.csv",
    "name" : "name.csv"
}

#Questions & Patterns for small talk
QnP = []
qna = []
smallTalk = []

#Answers for Questions & Small talk
ans = []
ans_qna = []
ans_smallTalk = []

#store of just questions and small talk files
docs = {
    "questions" : {},
    "smallTalk": {}
}

change_name = []
show_name = []
labels = []  

with open(docs_names["questions"], encoding='utf8', errors='ignore', mode='r',newline='') as dataset:
    data = csv.reader(dataset)
    #skips the first line of excel file
    next(data)

    #ssss
    for row in data:
        docs["questions"][row[0]] = [row[1]]
        QnP.append(row[0])
        ans.append(row[1])
        labels.append('questions')
    qna = QnP

#print(qna)
with open(docs_names["smallTalk"], encoding='utf8', errors='ignore', mode='r',newline='') as dataset:
    data = csv.reader(dataset)

    next(data)

    for row in data:
        docs["smallTalk"][row[0]] = [row[1]]
        QnP.append(row[0])
        ans.append(row[1])
        smallTalk.append(row[0])
        ans_smallTalk.append(row[1])
        labels.append('smallTalk')

#print(smallTalk)
#print(docs["smallTalk"])

with open(docs_names["name"], encoding='utf8', errors='ignore', mode='r',newline='') as dataset:
    data = csv.reader(dataset)
    next(data)

    for row in data:
        change_name.append(row[0])
        show_name.append(row[1])


all_patterns = QnP
docs_copy = docs
#print(change_name)
#print(show_name)

#print(len(labels), len(QnP['questions']))
#print(len(all_patterns))

#X_train, X_test, y_train, y_test = train_test_split(all_patterns, labels, stratify=labels, test_size=0.15)
#count_vect = stem_vectorizer
#print(count_vect)
#tfidf_vec = TfidfVectorizer(analyzer= stemmed_words, stop_words= stopwords.words('english'), use_idf=True, sublinear_tf=True)
#X_train_counts = tfidf_vec.fit_transform(X_train)

#tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)
#X_train_tf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tf)
#print(X_train_counts.shape)



try:
    with open("data.pickle", "rb") as tfidf_v:
        tfidf_vec = pickle.load(tfidf_v)
    with open("classifier.pickle", "rb") as f:
        clf = pickle.load(f)
except:
    X_train, X_test, y_train, y_test = train_test_split(all_patterns, labels, stratify=labels, test_size=0.15)
    tfidf_vec = TfidfVectorizer(analyzer= stemmed_words, stop_words= stopwords.words('english'), use_idf=True, sublinear_tf=True)
    X_train_counts = tfidf_vec.fit_transform(X_train)
    with open("data.pickle", "wb") as tfidf_v:
        pickle.dump(tfidf_vec, tfidf_v)
    clf = LogisticRegression(random_state=0).fit(X_train_counts, y_train)
    #print(clf)
    with open("classifier.pickle", "wb") as f:
        pickle.dump(clf, f)
        
# THIS CODE IS FOR CONFUSION MATRIX, ACCURACY SCORE AND F1SCORE
#X_new_counts = count_vect.transform(X_test)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts) 
#predicted = clf.predict(X_new_tfidf)

#print(confusion_matrix(y_test, predicted))
#print(accuracy_score(y_test, predicted))
#print(f1_score(y_test, predicted, pos_label='smallTalk'))

def transformation(doc_set):
    doc = stem_vectorizer
    tf = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True)
    tdm = doc.fit_transform(doc_set)
    tfidf = tf.fit_transform(tdm)
    return tfidf, doc, tf

def similarity(QuestionPattern, all_data, processed_q):

    sim_book = [] 
    position = []

    for i in range(len(all_data)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                sim_1 = 1 - spatial.distance.cosine(processed_q, QuestionPattern[i])
                sim_book.append(sim_1)
                if sim_1 > 0.50:
                    position.append([all_data[i], sim_1])
            except: FloatingPointError
        
    return position, sim_book


stop = False
first_q = 0 #move outside the while loop
name = ""
while not stop:
    
    if name == "":
        name = input("Hello, please tell me your name, if you want to exit please type 'stop','quit'\n")

    if name == 'stop' or name == 'Stop' or name == 'quit': 
        print("Farewell " + name + " it was a pleasure to meet you \n")
        stop = True
        break

    if first_q == 0: 
        query = input("Talk to me " + name +", if you want to change your name type 'change my name' \nTo see your name say 'who am I' \n")
        first_q =+ 1 
    if query == 'stop' or query == 'Stop' or query == 'quit': 
        print("Farewell " + name + " it was a pleasure to meet you \n")
        stop = True
        break
    if any(sentence in query for sentence in change_name):
        #print(query)
        nameStore = name
        name = input("Enter your new identity \n")
        if name == 'stop' or name == 'Stop' or name == 'quit': 
            print("Farewell " + name + " it was a pleasure to meet you \n")
            stop = True
            break
        query = input("Your old name was " + nameStore +" new name is now " + name + "\n\n")
        
    if any(sentence in query for sentence in show_name):
        query = input("Your name is " +name +"\n") 
        sentence = []
        
        
    else:
        query = query.lower()
        #query = query.strip("!?.,;)(@[]{}\|/><")
        query = [query]
        #print(query)
    
        processed_q = tfidf_vec.transform(query) # stems query
        #print(processed_q)
       # processed_q = tfidf_transformer.transform(processed_q) #term frequency of transformer

        #print(processed_q)
        clasification = clf.predict(processed_q)
        #print(clasification)
        #processed_q = processed_q.toarray()
        #print(X_train_tf)
        #tfidf_d = X_train_tf.toarray()
        #print(len(tfidf_d))
        if clasification == 'smallTalk':
            tfidf, cv, tft = transformation(smallTalk)
            tfidf_array = tfidf.toarray()
            #print(tfidf_array)
            processed_q = cv.transform(query)
            processed_q = tft.transform(processed_q)
            processed_q_a = processed_q.toarray()[0]
            #print(processed_q_a)
            position, sim_book = similarity(tfidf_array, smallTalk, processed_q_a)
            #print(position)
            position.sort(key= itemgetter(1), reverse= True) 
            #print(position)
            sim_book.sort( reverse= True)
            accuracy = sim_book[0]
            valueList = []
            for x in sim_book:
                if x == accuracy:
                    valueList.append(x)
                    #print(x)
            #accuracy = position[0][1]
            #print(accuracy
            if valueList != []:
                picker = len(valueList) - 1
                #print(picker)

                #print(valueList)
                answer_store = len(position)
                random_ans = rnd.randint(0, picker)
            #print(random_ans)
            if position == []:
                    query = input("Sorry, I don't understand \n")
            else:
                position = position[0][0]

                #print(position)
                #print(position, sorted_sim_book)
                #print(sim_book)
                #answer_pos = []
                query = input(docs['smallTalk'][position][0] + "\n")

            #answer_pos = max(position, key = lambda item: item[1])
            #answer = answer_pos[0] 
            #reply = ans[answer]
        elif clasification == 'questions':
            tfidf, cv, tft = transformation(qna)
            tfidf_array = tfidf.toarray()
            processed_q = cv.transform(query)
            processed_q = tft.transform(processed_q)
            processed_q_a = processed_q.toarray()[0]
            #print(processed_q_a)
            position, sim_book = similarity(tfidf_array, qna, processed_q_a)
            #print(position)
            position.sort(key= itemgetter(1), reverse= True)
            accuracy = sim_book[0]
            listValues = []
            for x in sim_book:
                if x == accuracy:
                    listValues.append(x)
                    #print(x)
            #accuracy = position[0][1]
            #print(accuracy
            if listValues != []:
                picker = len(listValues) - 1
                #print(picker)
                #print(listValues)
                answer_store = len(position)
                random_ans = rnd.randint(0, picker)
            #print(position)
            if position == []:
                    query = input("Sorry, I don't understand \n")
            else:
                try:
                    position = position[0][0]
                    query = input(docs['questions'][position][0] + "\n")
                except: KeyError
                query = input("Sorry, I don't understand \n")

#need to map similarity to the position of it etc, 0.9 has a positon of 100

    
