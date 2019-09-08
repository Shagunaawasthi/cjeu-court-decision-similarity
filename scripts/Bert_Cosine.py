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

print("Bert + Cosine document similarity analyser")
print("-------------------------------------------")
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

# In[3]:

import os.path
print("* Loading / training Bert model...")
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

fname = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "bert.vec"))
model = KeyedVectors.load_word2vec_format(fname)

# In[4]:

path = "../inputdata/full_texts_all_cases/"

def getfile(celexnumber):
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if celexnumber in file:
                return file
    return None

def celex_to_index(celexnumber):
    file = getfile(celexnumber)
    for k, v in index_to_celex.items():
        if v == file:
            return k
    return -1

import pandas as pd

print("* Importing sample cases...")
# Fetch sample cases from file
def get_sample_cases(topic):
    data = pd.read_csv("../inputdata/sampled_cases.csv")
    relevant_rows = data[data['topic'] == topic]
    return relevant_rows['source'].tolist()

# Celex numbers of reference cases
publichealth = get_sample_cases('public health')
socialpolicy = get_sample_cases('social policy')
dataprotection = get_sample_cases('data protection')

print(" Successfully imported sample cases!")
print()


# In[5]:


# Function to convert entire similarity results to case ID references
def convert_to_case_references(wmd_result):
    global index_to_celex
    result = []
    for item in wmd_result:
        case_reference = cleanfilename(index_to_celex[item[0]]) # convert to case reference
        similarity_value = item[1]
        result.append((case_reference,similarity_value))
    return result


# In[6]:


print("* Import citations for cases...")
citations = pd.read_csv('../inputdata/all_cases_citations.csv')
print(" Successfully imported citations!")
print()

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


# In[16]:


from gensim.similarities import WmdSimilarity

results = []

def lookup_similar_cases(sample_cases, n, topic, model, modelfilename):
    global results
    global celex_to_value
    global datafortraining
    global sim

    count = 1
    
    num_samples = len(sample_cases)
    for item in sample_cases:
        print("(",count,"/",num_samples,")","-","computing similarity for",str(item),"...")
        count += 1
        similar_cases = sim[celex_to_value[item]]
        similar_cases_references = convert_to_case_references(similar_cases)
        for reference in similar_cases_references:
            path = str(os.path.join(os.path.realpath('..'), "script_resources"))
            method = modelfilename.replace(path,"")
            method = method.replace(".model","")
            method = method.replace('\\',"")
            method = method.replace('/',"")
            results.append([item,reference[0],reference[1],method,exists_citation_link_between(item,reference[0]),topic])

# In[17]:

print("* Computing similar cases...")

print(" Building WMD document similarity matrix...")
sim = MatrixSimilarity(model[datafortraining], num_features=548704, num_best=20)
print(" Successfully built the WMD similarity matrix!")
print()
print(" Computing similar cases for PUBLIC HEALTH samples...")
lookup_similar_cases(publichealth,20,'public health', model, fname)
print(" Successfully computed similarities for PUBLIC HEALTH samples...")
print()
print(" Computing similar cases for SOCIAL POLICY samples...")
lookup_similar_cases(socialpolicy,20,'social policy', model, fname)
print(" Successfully computed similarities for SOCIAL POLICY samples...")
print()
print(" Computing similar cases for DATA PROTECTION samples...")
lookup_similar_cases(dataprotection,20,'data protection', model, fname)
print(" Successfully computed similarities for DATA PROTECTION samples...")
print()
print(" Successfully computed similar cases!")
print()

# In[18]:

print("* Writing results to file...")
import csv
import os.path

if os.path.exists('../outputdata/results_bert_cosine.csv') == False:
    results.insert(0,['source_case','similar_case','similarity_score','method','citation_link','source_case_topic'])
    
with open('../outputdata/results.csv', 'a', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerows(results)
    
end = time.time()

print(" Successfully wrote results to file!")
print()
print(" Done!")
print()
print("* Time taken:",(end-start),"s")

