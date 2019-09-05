#!/usr/bin/env python
# coding: utf-8

# In[2]:


from gensim.models import Word2Vec
import os
import pickle
import nltk
from nltk.corpus import stopwords
import time

start = time.time()

print("Word2Vec + Word Mover's Distance document similarity analyser")
print("-------------------------------------------------------------")
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
from gensim.test.utils import get_tmpfile

print("* Loading / training Doc2Vec models...")

model_64 = None
model_128 = None
model_256 = None

model_64_10 = None
model_128_10 = None
model_256_10 = None

fname_64 = None
fname_128 = None
fname_256 = None

fname_64_10 = None
fname_128_10 = None
fname_256_10 = None

print()
print("* Window size: 5")
print()
print("* Vector size 64")
if os.path.exists(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_64.model")):
    print(" loading model from file...")
    fname_64 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_64.model"))
    model_64 = Word2Vec.load(fname_64)
    print(" successfully loaded model!")
else:
    print(" training model...")
    model_64 = Word2Vec(datafortraining, size=64, window=5, min_count=1, workers=16)
    model_64.train(datafortraining, total_examples=model_64.corpus_count,epochs=20)
    model_64.init_sims(replace=True)
    print(" successfully trained model!")
    print(" saving model to file...")
    fname_64 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_64.model"))
    model_64.save(fname_64)
    print(" successfully saved model!")
    
print()

print("* Vector size 128")
if os.path.exists(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_128.model")):
    print(" loading model from file...")
    fname_128 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_128.model"))
    model_128 = Word2Vec.load(fname_128)
    print(" successfully loaded model!")
else:
    print(" training model...")
    model_128 = Word2Vec(datafortraining, size=128, window=5, min_count=1, workers=16)
    model_128.train(datafortraining, total_examples=model_128.corpus_count,epochs=20)
    model_128.init_sims(replace=True)
    print(" successfully trained model!")
    print(" saving model to file...")
    fname_128 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_128.model"))
    model_128.save(fname_128)
    print(" successfully saved model!")

print()
    
print("* Vector size 256")
if os.path.exists(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_256.model")):
    print(" loading model from file...")
    fname_256 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_256.model"))
    model_256 = Word2Vec.load(fname_256)
    print(" successfully loaded model!")
else:
    print(" training model...")
    model_256 = Word2Vec(datafortraining, size=256, window=5, min_count=1, workers=16)
    model_256.train(datafortraining, total_examples=model_256.corpus_count,epochs=20)
    model_256.init_sims(replace=True)
    print(" successfully trained model!")
    print(" saving model to file...")
    fname_256 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_256.model"))
    model_256.save(fname_256)
    print(" successfully saved model!")
print()


print()
print("* Window size: 10")
print()
print("* Vector size 64")
if os.path.exists(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_64_10.model")):
    print(" loading model from file...")
    fname_64_10 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_64_10.model"))
    model_64_10 = Word2Vec.load(fname_64_10)
    print(" successfully loaded model!")
else:
    print(" training model...")
    model_64_10 = Word2Vec(datafortraining, size=64, window=10, min_count=1, workers=16)
    model_64_10.train(datafortraining, total_examples=model_64_10.corpus_count,epochs=20)
    model_64_10.init_sims(replace=True)
    print(" successfully trained model!")
    print(" saving model to file...")
    fname_64_10 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_64_10.model"))
    model_64_10.save(fname_64_10)
    print(" successfully saved model!")
    
print()

print("* Vector size 128")
if os.path.exists(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_128_10.model")):
    print(" loading model from file...")
    fname_128_10 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_128_10.model"))
    model_128_10 = Word2Vec.load(fname_128_10)
    print(" successfully loaded model!")
else:
    print(" training model...")
    model_128_10 = Word2Vec(datafortraining, size=128, window=10, min_count=1, workers=16)
    model_128_10.train(datafortraining, total_examples=model_128_10.corpus_count,epochs=20)
    model_128_10.init_sims(replace=True)
    print(" successfully trained model!")
    print(" saving model to file...")
    fname_128_10 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_128_10.model"))
    model_128_10.save(fname_128_10)
    print(" successfully saved model!")

print()
    
print("* Vector size 256")
if os.path.exists(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_256_10.model")):
    print(" loading model from file...")
    fname_256_10 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_256_10.model"))
    model_256_10 = Word2Vec.load(fname_256_10)
    print(" successfully loaded model!")
else:
    print(" training model...")
    model_256_10 = Word2Vec(datafortraining, size=256, window=10, min_count=1, workers=16)
    model_256_10.train(datafortraining, total_examples=model_256_10.corpus_count,epochs=20)
    model_256_10.init_sims(replace=True)
    print(" successfully trained model!")
    print(" saving model to file...")
    fname_256_10 = get_tmpfile(os.path.join(os.path.join(os.path.realpath('..'), "script_resources"), "WMD_256_10.model"))
    model_256_10.save(fname_256_10)
    print(" successfully saved model!")
print()


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
    
    count = 1
    sim = WmdSimilarity(datafortraining, model, num_best=n)
    for item in sample_cases:
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

lookup_similar_cases(publichealth,20,'public health', model_64, fname_64)
print(" 1 ")
lookup_similar_cases(publichealth,20,'public health', model_128, fname_128)
print(" 2 ")
lookup_similar_cases(publichealth,20,'public health', model_256, fname_256)
print(" 3 ")
lookup_similar_cases(publichealth,20,'public health', model_64_10, fname_64_10)
print(" 4 ")
lookup_similar_cases(publichealth,20,'public health', model_128_10, fname_128_10)
print(" 5 ")
lookup_similar_cases(publichealth,20,'public health', model_256_10, fname_256_10)
print(" 6 ")

lookup_similar_cases(socialpolicy,20,'social policy', model_64, fname_64)
print(" 7 ")
lookup_similar_cases(socialpolicy,20,'social policy', model_128, fname_128)
print(" 8 ")
lookup_similar_cases(socialpolicy,20,'social policy', model_256, fname_256)
print(" 9 ")
lookup_similar_cases(socialpolicy,20,'social policy', model_64_10, fname_64_10)
print(" 10 ")
lookup_similar_cases(socialpolicy,20,'social policy', model_128_10, fname_128_10)
print(" 11 ")
lookup_similar_cases(socialpolicy,20,'social policy', model_256_10, fname_256_10)
print(" 12 ")

lookup_similar_cases(dataprotection,20,'data protection', model_64, fname_64)
print(" 13 ")
lookup_similar_cases(dataprotection,20,'data protection', model_128, fname_128)
print(" 14 ")
lookup_similar_cases(dataprotection,20,'data protection', model_256, fname_256)
print(" 15 ")
lookup_similar_cases(dataprotection,20,'data protection', model_64_10, fname_64_10)
print(" 16 ")
lookup_similar_cases(dataprotection,20,'data protection', model_128_10, fname_128_10)
print(" 17 ")
lookup_similar_cases(dataprotection,20,'data protection', model_256_10, fname_256_10)
print(" 18 ")

print(" Successfully computed similar cases!")
print()


# In[18]:


print("* Writing results to file...")
import csv
import os.path

if os.path.exists('../outputdata/results.csv') == False:
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

