# Usage:
# python extract_case_text.py

import sys
import getopt
import re
import requests
from bs4 import BeautifulSoup

def get_parsed_html_text(page):
	soup = BeautifulSoup(page.content, 'html.parser')
	text = soup.find('div', {'id': 'text'})
	if text is not None:
		return text.text
	else:
		return False

def get_celex_numbers(file):
	result = []
	with open(file, 'r') as fh:
		for line in fh:
			line = line.rstrip('\n')
			result.append(line)
	return result

def get_html_page_from_url(url):
	try:
		page = requests.get(url)
	except Exception as e:
		print('Request has failed for this URL:')
		print(url)
		print()
		raise(e)
	return page

def remove_unwanted_artefacts(text):
	text = text.replace(u'\xa0', ' ')
	text = re.sub(r'\n+', '\n', text)
	text = re.sub(r'\n\d+\n', '\n', text)
	text = re.sub(r'\d+\s+', '', text)
	return text

base_url = 'http://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:'
celexnumbers = get_celex_numbers('../inputdata/celex_numbers_texts.txt')
errors = []
index = 1
length = len(celexnumbers)

for celexnumber in celexnumbers:
	print(index,"/",length)
	index += 1
	url = base_url + celexnumber
	page = get_html_page_from_url(url)
	case_text = get_parsed_html_text(page)
	if case_text:
		case_text = remove_unwanted_artefacts(case_text)
		file_name = '../inputdata/full_texts_all_cases/full_text_' + celexnumber + '.txt'
		with open(file_name, 'w', encoding='utf-8') as out:
			out.write(case_text)
	else:
		errors.append(celexnumber)

if len(errors) > 0:
	print('Failed to extract ', len(errors), ' cases')
	with open('../inputdata/extracted_casetexts_failed.txt', 'w') as out:
		for error in errors:
			out.write(error + '\n')
	out.close()

print()
print('Done!')
