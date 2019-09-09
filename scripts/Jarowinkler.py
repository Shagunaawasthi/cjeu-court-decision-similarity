#!/usr/bin/env python
# coding: utf-8

# In[2]:

import gensim
from gensim.models import Word2Vec
import os
import pickle
import nltk
from nltk.corpus import stopwords
import time

start = time.time()

print("Jarowinkler")
print("-----------")
print()

print("* Building index of documents...")

# List all documents in directory
path = "../inputdata/full_texts_all_cases/"

# Import stopwords           
stopwordsfile = "../script_resources/stopwords.pickle"
stopwords_full = []
with open(stopwordsfile, "rb") as f:
    tmp = pickle.load(f)
    stopwords_full.extend(list(tmp))
    stopwords_full.extend(stopwords.words('english'))
    
stopwords_full = list(set(stopwords_full))

#print(stopwords_full)

# Only keep celex number from filename
def cleanfilename(name):
    result = ""
    result = name.replace("full_text_","")
    result = result.replace(".txt","")
    return result

def removeStopWords(text, stopwords_list):
    text = text.lower()
    for item in stopwords_list:
        text = text.replace(" " + item.lower() + " "," ")
        text = text.replace(" " + item.lower() + ","," ")
        text = text.replace(" " + item.lower() + "."," ")
        text = text.replace(" " + item.lower() + ";"," ")
    text = text.replace("+","")
    return text

# Import files and define mapping between case IDS and full texts   
files = []
index_to_celex = {}
index_to_value = {}
celex_to_value = {}
datafortraining = []
index = 0
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
            celexnum = cleanfilename(os.path.basename(file))
            with open (path+file, "r", encoding="utf-8") as myfile:

                data = myfile.read().replace('\n', '')

                data = removeStopWords(data,stopwords_full)
                datafortraining.append(data)
                index_to_celex[index] = file
                index_to_value[index] = data
                celex_to_value[celexnum] = data
                index += 1

print(" Index successfully built!")
print()


import numpy as np    
from scipy.spatial.distance import pdist, squareform
from similarity.jarowinkler import JaroWinkler
jarowinkler = JaroWinkler()
print(jarowinkler.similarity(str2, str3))

# my list of strings
strings = ["hello","hallo","choco"]

# prepare 2 dimensional array M x N (M entries (3) with N dimensions (1)) 
transformed_docs = np.array(datafortraining).reshape(-1,1)

# calculate condensed distance matrix by wrapping the Levenshtein distance function
distance_matrix = pdist(transformed_docs,lambda x,y: jarowinkler.similarity(x[0], y[0]))
