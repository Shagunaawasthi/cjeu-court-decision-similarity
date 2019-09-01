# Usage:
# python extract_case_citations.py

import sys
import getopt
import re
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv

def get_celex_numbers(file):
    candidates = []
    with open(file, 'r') as fh:
        for line in fh:
            line = line.rstrip('\n')
            candidates.append(line)

    errors = []
    with open('../inputdata/errors_while_extracting_citations.csv', 'r') as ef:
        for line in ef:
            line = line.rstrip('\n')
            errors.append(line)

    result = list(set(candidates).difference(set(errors)))
    return result

def get_html_page_from_url(url):
    try:
        page = urlopen(url)
    except Exception as e:
        print('Request has failed for this URL:')
        print(url)
        print()
        # #all_citations.insert(0,['source','target'])
        # with open('../data/extracted_citations_tmp.csv', 'w', newline='') as outfile:
        #     writer = csv.writer(outfile, delimiter=',')
        #     writer.writerows(all_citations)
        raise(e)
    soup = BeautifulSoup(page, "lxml")
    return soup

def clean_celex_number(text):
    if ")" in text:
        result = text.split(")")
        result[0] += ")"
        return result[0]
    else:
        return text[:11]

def is_valid_celex(celex):
    alphabet_count = 0
    for i in range(len(celex)):
        if(celex[i].isalpha()):
            alphabet_count += 1            
    if ("CJ" in celex.upper() or "CO" in celex.upper()) and alphabet_count == 2:
        return True
    else:
        return False

# Extract citations for the case given the BeautifulSoup format of it's HTML page
def extractCitations(soup_judgement_page, celexNumber):
    # Citations array
    citations = []
    # Get all list items in this web page (the citations are in one of the list items in the HTML source)
    li_results = soup_judgement_page.find_all('li')

    # loop through items until you find the citation list item
    for result in li_results:
        for link in result.find_all('a',href=True):
            if "./../../../legal-content/EN/AUTO/?uri=CELEX:6" in link['href']:
                currentrow = []
                citedcelex = clean_celex_number(link.text)
                if is_valid_celex(citedcelex) and is_valid_celex(celexNumber):
                    currentrow.append(celexNumber)
                    currentrow.append(citedcelex)
                    citations.append(currentrow)
    return citations

base_url = 'http://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:'
celexnumbers = get_celex_numbers('../inputdata/celex_numbers_citations.txt')
errors = []

all_citations = []
length = len(celexnumbers)
index = 1
for celexnumber in celexnumbers:
    print(index,"/",length)
    index+=1
    url = base_url + celexnumber
    try:
        page = get_html_page_from_url(url)
        citations = extractCitations(page, celexnumber)
        if (len(subjects) == 0):
            item = []
            item.append(celexnumber)
            errors.append(celexnumber)
        all_citations.extend(citations)
    except Exception as e:
        item = []
        item.append(celexnumber)
        errors.append(item)
    
with open('../inputdata/extracted_citations_failed.csv', 'a', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerows(errors)

all_citations.insert(0,['source','target'])
with open('../inputdata/extracted_citations.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerows(all_citations)

print()
print('Done!')
