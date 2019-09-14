#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from __future__ import division
import gensim
from gensim.models import Word2Vec
import os
import pickle
import nltk
from nltk.corpus import stopwords
import time
import string
import math
import operator
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

start = time.time()

print("Jaccard Stem")
print("------------")
print()

tokenize = lambda doc: doc.lower().split(" ")

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

porter=PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

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
value_to_celex = {}
celex_to_tokenized = {}
value_to_tokenized = {}
index = 0
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
            celexnum = cleanfilename(os.path.basename(file))
            with open (path+file, "r", encoding="utf-8") as myfile:
                data = myfile.read().replace('\n', '')
                data = data.replace("  "," ")
                data = removeStopWords(data,stopwords_full)
                sent_text = nltk.sent_tokenize(data)
                lemmatized_doc = ""
                for sentence in sent_text:
                    lemmatized_doc += stemSentence(sentence)
                celex_to_value[celexnum] = lemmatized_doc
                index_to_celex[index] = file
                index_to_value[index] = lemmatized_doc
                value_to_celex[lemmatized_doc] = celexnum
                tknzd = tokenize(lemmatized_doc)
                value_to_tokenized[lemmatized_doc] = tknzd
                celex_to_tokenized[celexnum] = tknzd
                index += 1

print(" Index successfully built!")
print()

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

# Import sample cases
import pandas as pd

# Fetch sample cases from file
def get_sample_cases(topic):
    data = pd.read_csv("../inputdata/sampled_cases.csv")
    relevant_rows = data[data['topic'] == topic]
    return relevant_rows['source'].tolist()

print("* Importing sample cases...")
# Celex numbers of reference cases
publichealth = get_sample_cases('public health')
socialpolicy = get_sample_cases('social policy')
dataprotection = get_sample_cases('data protection')
print(" Successfully imported sample cases!")
print()

# Import citations
print("* Import citations for cases...")
citations = pd.read_csv('../inputdata/all_cases_citations.csv')
print(" Successfully imported citations!")

def find_cited_cases(celexnumber):
    global citations
    relevantsource = citations[citations['source'] == celexnumber]
    return relevantsource['target'].tolist()

def exists_citation_link_between(celexnumber1,celexnumber2):
    global citations
    relevantsource1 = citations[citations['source'] == celexnumber1]
    relevantsource2 = citations[citations['source'] == celexnumber2]
    if celexnumber2 in relevantsource1['target'].tolist() or celexnumber1 in relevantsource2['target'].tolist():
        return True
    return False

# In[ ]:

results = []

def lookup_similar_cases(sample_cases, n, topic):
    global results
    global celex_to_value
    global celex_to_tokenized
    global value_to_tokenized

    count = 1
    num = len(sample_cases)
    for item in sample_cases:
        print(count,"/",num,item)
        count+=1
        current_dict = {}
        for k,v in celex_to_value.items():
            if k != item:
                start = time.time()
                current_sim_val = jaccard_similarity(celex_to_tokenized[item], value_to_tokenized[v])
                end = time.time()
                current_dict[k] = current_sim_val
                timetaken = end - start
                print((k,current_sim_val),timetaken,"s")
        sorted_dict = sorted(current_dict.items(), key=operator.itemgetter(1))
        topn = sorted_dict[-n:]
        for reference in topn:
            results.append([item,reference[0],reference[1],'jaccard-stem',exists_citation_link_between(item,reference[0]),topic])

print("* Computing similar cases...")
print("* Public health")
# 1. Public Health
lookup_similar_cases(publichealth,20,'public health')
print("* Social policy")
# 2. Social Policy
lookup_similar_cases(socialpolicy,20,'social policy')
# 3. Data Protection
print("* Data protection")
lookup_similar_cases(dataprotection,20,'data protection')
print(" Successfully computed similar cases!")
print()


# In[ ]:


print("* Writing results to file...")
import csv
import os.path

if os.path.exists('../outputdata/results_jaccardstem.csv') == False:
    results.insert(0,['source_case','similar_case','similarity_score','method','citation_link','source_case_topic'])
    
with open('../outputdata/results_jaccardstem.csv', 'a', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerows(results)
    
end = time.time()

print(" Successfully wrote results to file!")
print()
print(" Done!")
print()
print("* Time taken:",(end-start),"s")

