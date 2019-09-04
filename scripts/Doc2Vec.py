#!/usr/bin/env python
# coding: utf-8

# # EUR-LEX Case Similarity Notebook: Doc2Vec

# ### Abstract: 
# #### This notebook implements Doc2Vec document similarity measures (based on Word2Vec) on EUR-LEX judgements and orders

# ### Step 1. Import data & resources

# #### I.e. import full texts of cases from file, define a mapping to case IDs so we can lookup Doc2Vec similarity values by caseID, import stopwords file

# In[23]:


from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import pickle
import nltk
from nltk.corpus import stopwords

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

# Only keep celex number from filename
def cleanfilename(name):
    result = ""
    result = name.replace("full_text_","")
    result = result.replace(".txt","")
    return result

def remove_stopwords(text):
    global stopwords_full
    for item in stopwords_full:
        text = text.replace(str(item),'')
    return text

# Import files and define mapping between case IDS and full texts   
files = []
index_to_celex = {}
datafortraining = []
index = 0
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
            celexnum = cleanfilename(os.path.basename(file))
            with open (path+file, "r", encoding="utf-8" ) as myfile:
                data = myfile.read().replace('\n', '')
                data = remove_stopwords(data)
                datafortraining.append(data)
                index_to_celex[index] = file
                index += 1

#datafortraining = datafortraining[1200:1250]
documents = [TaggedDocument(file, [i]) for i, file in enumerate(datafortraining)]

# ### Step 2. Train Doc2Vec model on input case texts

# In[24]:


# This will take a while... a few hours
from gensim.test.utils import get_tmpfile
fname_32 = get_tmpfile("../script_resources/doc2vec_model_32.model")
fname_64 = get_tmpfile("../script_resources/doc2vec_model_64.model")
fname_128 = get_tmpfile("../script_resources/doc2vec_model_128.model")
fname_256 = get_tmpfile("../script_resources/doc2vec_model_256.model")

model = Doc2Vec(documents, vector_size=128, window=5, min_count=1, workers=4)
model.train(documents, total_examples=model.corpus_count,epochs=20)
model = Doc2Vec(documents, vector_size=128, window=5, min_count=1, workers=4)
model.train(documents, total_examples=model.corpus_count,epochs=20)
model = Doc2Vec(documents, vector_size=128, window=5, min_count=1, workers=4)
model.train(documents, total_examples=model.corpus_count,epochs=20)


# ### Step 3. Import sample cases

# In[30]:


import pandas as pd

# Fetch sample cases from file
def get_sample_cases(topic):
    data = pd.read_csv("../inputdata/sampled_cases.csv")
    relevant_rows = data[data['topic'] == topic]
    return relevant_rows['source'].tolist()

# Celex numbers of reference cases
publichealth = get_sample_cases('public health')
socialpolicy = get_sample_cases('social policy')
dataprotection = get_sample_cases('data protection')

# print(publichealth)
# print(socialpolicy)
# print(dataprotection)


# ### Step 4. Define functions mapping between case ID, filename, and associated full text (references to same document)

# In[31]:


# Function to convert entire similarity results to case ID references
def convert_to_case_references(doc2vec_result):
    global index_to_celex
    result = []
    for item in doc2vec_result:
        case_reference = cleanfilename(index_to_celex[item[0]]) # convert to case reference
        similarity_value = item[1]
        result.append((case_reference,similarity_value))
    return result

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


# ### Step 5. Define functions to look up citations for a given case, and to check if there is a citation link between two cases

# In[32]:


citations = pd.read_csv('../inputdata/all_cases_citations.csv')

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


# ### Step 6. Look up top n similar cases per sample case

# In[36]:


results = []

def lookup_similar_cases(sample_cases, n, topic):
    global results
    global model
    for item in sample_cases:
        similar_cases = model.docvecs.most_similar(celex_to_index(item), topn=n)
        similar_cases_references = convert_to_case_references(similar_cases)
        for reference in similar_cases_references:
            results.append([item,reference[0],reference[1],'doc2vec - 128',exists_citation_link_between(item,reference[0]),topic])
            
# 1. Public Health
lookup_similar_cases(publichealth,20,'public health')
# 2. Social Policy
lookup_similar_cases(socialpolicy,20,'social policy')
# 3. Data Protection
lookup_similar_cases(dataprotection,20,'data protection')


# ### Step 7. Write results to file

# In[37]:


import csv
import os.path

if os.path.exists('../outputdata/results.csv') == False:
    results.insert(0,['source_case','similar_case','similarity_score','method','citation_link','source_case_topic'])
    
with open('../outputdata/results.csv', 'a', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerows(results)

