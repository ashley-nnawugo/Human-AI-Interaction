#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:11:16 2021

@author: ashleynnawugo
"""

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
from urllib import request
url = "http://www.gutenberg.org/files/84/84-0.txt"
content = request.urlopen(url).read().decode('utf8', errors='ignore')



with open('rural.txt', 'r', encoding= 'utf-8') as f:
    content = f.read()
    print(len(content))
f.close()

url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = request.urlopen(url).read().decode('utf8')
html[:60]

from bs4 import BeautifulSoup
content = BeautifulSoup(html, 'html.parser').get_text()
tokens = word_tokenize(content)
tokens

text = "Artificial intelligence is cool but I am not too keen on Skynet."
text_tokens = word_tokenize(text)
tokens_without_sw = [word.lower() for word in text_tokens if not word in stopwords.words()]
print(tokens_without_sw)
filtered_sentence = (" ").join(tokens_without_sw)
print(filtered_sentence)

text = nltk.Text(tokens_without_sw)
print(text.count('cool'))


p_stemmer = PorterStemmer()
sb_stemmer = SnowballStemmer('english')
sentence = "This is a test sentence, and I am hoping it doesn't get chopped up too much."
print(sentence)
for token in word_tokenize(sentence):
    print(p_stemmer.stem(token))
    print(sb_stemmer.stem(token))
    print("---")
    
lemmatiser = WordNetLemmatizer()
posmap = {
    'ADJ': 'j',
    'ADV': 'r',
    'NOUN': 'n',
    'VERB': 'v'
}

post = nltk.pos_tag(word_tokenize(sentence), tagset='universal')
print(post)
for token in post:
    word = token[0]
    tag = token[1]
    if tag in posmap.keys():
        print(lemmatiser.lemmatize(word,posmap[tag]))
    else:
        print(lemmatiser.lemmatize(word))
        
    print("---")
    
tokens = word_tokenize(sentence)
count = 0
for token in tokens:
    if token.endswith("ing"): 
        count += 1
print(count)


"Exercise 1"

s = 'colorless'
print("Exercise 1:")
print(s[0:4]+'u'+ s[4:])

"Exercise 2"

sample = "dishes"
sample1 = "running"
sample2 = "nationality"
sample3 = "undo"
sample4 = "preheat"

print("Excercise 2")
print(sample[:-2] + ' ' +sample1[:-4] + ' ' + sample2[:-5] + ' ' + sample3[:-2] + ' ' + sample4[:-4])

    
"Exercise 4"

print(sentence[1:12:2])

"Exercise 5"

print(sentence[::-1])
"goes through whole list as no numbers are specified and reverses it "

"Exercise 6"

sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
result = [n for n in sent]
print(result)


"Exercise 7"

raw = "goes through whole list as no numbers are specified and reverses it "
print(raw.split('s'))

"Exercies 8"

y = "Chimezie"

for char in y:
    print(char)
    

"Task 1"


url = "https://en.wikipedia.org/wiki/Main_Page"
html = request.urlopen(url).read().decode('utf8', errors = 'ignore')
text = BeautifulSoup(html, 'html.parser').get_text()
print(text)

"Task 2"
print(stopwords.words('English'))
      
sentence1 = "I will eat chicken tomorrow but i will consequently end up with measles"

sentence_tokenize = word_tokenize(sentence1)  
extra_sw = ['consequently','end', 'will', 'i']  

sentence_tokenize_nosw = [word.lower() for word in sentence_tokenize if not word in stopwords.words('English') + extra_sw]
print(sentence_tokenize_nosw)
"Task 3"


"Task 4"





