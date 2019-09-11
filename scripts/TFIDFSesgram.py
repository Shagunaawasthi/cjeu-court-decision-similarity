#!/usr/bin/env python
# coding: utf-8

# # EUR-LEX Case Similarity Notebook: TF-IDF

# ### Abstract: 
# #### This notebook implements the TF/IDF document similarity measure on EUR-LEX judgements

# ### Step 1. Import data and resources

# #### I.e. import full texts of cases from file, define a mapping to case IDs so we can lookup TF/IDF similarity values by caseID, import stopwords file

# In[74]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

# Import stopwords           
stopwordsfile = "../script_resources/stopwords.pickle"
stopwords_full = []
with open(stopwordsfile, "rb") as f:
    tmp = pickle.load(f)
    stopwords_full.extend(list(tmp))
    stopwords_full.extend(stopwords.words('english'))
stopwords_full = list(set(stopwords_full))

#print(stopwords_full)

porter=PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def removeStopWords(text, stopwords_list):
    text = text.lower()
    for item in stopwords_list:
        text = text.replace(" " + item + " "," ")
        text = text.replace(" " + item + ","," ")
        text = text.replace(" " + item + "."," ")
        text = text.replace(" " + item + ";"," ")
    text = text.replace("+","")
    return text

# Only keep celex number from filename
def cleanfilename(name):
    result = ""
    result = name.replace("full_text_","")
    result = result.replace(".txt","")
    return result

# List all documents in directory
path = "../inputdata/full_texts_all_cases/"

files = []
file_dict = {}
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    count = 1
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
            celexnum = cleanfilename(os.path.basename(file))
            with open (path+file, "r", encoding="utf-8" ) as myfile:
                data = myfile.read().replace('\n','')
                data = data.replace("  "," ")
                data = removeStopWords(data,stopwords_full)
                sent_text = nltk.sent_tokenize(data)
                lemmatized_doc = ""
                for sentence in sent_text:
                    lemmatized_doc += stemSentence(sentence)
                file_dict[celexnum] = lemmatized_doc

# Mapping values (text of cases), to filename/case_id (k)
values = []
key = {}
counter = 0
for k,v in file_dict.items():
    values.append(v)
    key[k] = counter
    counter+=1


# ### Step 2. Compute TFIDF similarity scores for input data

# In[ ]:


# Create a TF-IDF vectoriser
# Use custom legal text stopwords
# Apply the tfidf model to the input files
tfidfvect = TfidfVectorizer(analyzer='word', ngram_range=(6,6), use_idf=True, stop_words=stopwords_full)
tfidf_data = tfidfvect.fit_transform(values)


# In[ ]:


pickle.dump(tfidf_data, open("tfidfmodel_sesgram.pickle", "wb"))


# ### Step 3. Import sample cases from file

# In[ ]:


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


# ### Step 4. Define function to lookup similarity values in TF/IDF matrix

# In[ ]:


def find_similar(tfidf_matrix, index, top_n):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]


# ### Step 5. Define functions mapping between case ID, filename, and associated full text (references to same document)

# In[ ]:


# Keep a record of document to index
def get_doc_index(docid):
    global key
    global tfidf_data
    rowid = key[docid]
    return rowid
    
# Keep a record of document to index
def get_doc_row(docid):
    global key
    global tfidf_data
    rowid = key[docid]
    row = tfidf_data[rowid,:]
    return row

# Keep a record of document to index
def get_doc_id(rowid):
    global key
    global tfidf_data
    for k, v in key.items():    
        if v == rowid:
            return k
    return -1


# ### Step 6. Define functions to look up citations for a given case, and to check if there is a citation link between two cases

# In[ ]:


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


# ### Step 7. Lookup top n similar cases per sample case

# In[ ]:


results = []

# Function to convert entire similarity results to case ID references
def convert_to_case_references(tfidf_result):
    result = []
    for item in tfidf_result:
        case_reference = get_doc_id(item[0]) # convert to case reference
        similarity_value = item[1]
        result.append((case_reference,similarity_value))
    return result

def lookup_similar_cases(sample_cases, n, topic):
    global results
    global tfidf_data
    for item in sample_cases:
        index = get_doc_index(item)                         # Look up this cases index in the TFIDF matrix
        similar_cases = find_similar(tfidf_data, index, n)  # Look up top n similar cases for this case
        similar_cases_references = convert_to_case_references(similar_cases)
        for reference in similar_cases_references:
            results.append([item,reference[0],reference[1],'tfidf-sesgram',exists_citation_link_between(item,reference[0]),topic])

# 1. Public Health
lookup_similar_cases(publichealth,20,'public health')
# 2. Social Policy
lookup_similar_cases(socialpolicy,20,'social policy')
# 3. Data Protection
lookup_similar_cases(dataprotection,20,'data protection')


# ### Step 8. Write results to file

# In[ ]:


import csv
import os.path

if os.path.exists('../outputdata/results_sesgram.csv') == False:
    results.insert(0,['source_case','similar_case','similarity_score','method','citation_link','source_case_topic'])
    
with open('../outputdata/results_sesgram.csv', 'a', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerows(results)

