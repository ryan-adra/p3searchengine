import json
import nltk
import pymongo
import re
import math
import multiprocessing as mp
from collections import defaultdict
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from math import log
STOPWORDS = set(stopwords.words('english'))
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["invertedIndex"]
mycol = mydb["words"]
inverted_index_dict = {}

def isvalid(s):
    return all(ord(c) < 128 and ord(c) not in range(48, 58) for c in s)

def preprocess_tokens(extracted_words):
	tokenize_words = word_tokenize(extracted_words)
	text = [word for word in tokenize_words if word.lower() not in STOPWORDS and isvalid(word) and len(word) > 1]
	return text

def calculate_tf(num):
	return 1+log(num)

def calculate_idf(num):
	return log(37497/num)

def strip_raw_html_text(key):
	file_path = '/Users/filoprince/Documents/cs121_project3/WEBPAGES_RAW/' + key
	with open(file_path, 'r', encoding='utf8') as file_html:
		html_file = BeautifulSoup(file_html, features='lxml')

	for script in html_file(["script","style"]):
		script.decompose()

	html_text = html_file.get_text()
	lines = (line.strip() for line in html_text.splitlines())
	separate_lines = (phrase.strip() for line in lines for phrase in line.split(" "))
	html_text = ' '.join(chunk for chunk in separate_lines if chunk)
	return html_text

def get_important_words(key):
	file_path = '/Users/filoprince/Documents/cs121_project3/WEBPAGES_RAW/' + key
	with open(file_path, 'r', encoding='utf8') as file_html:
		html_page = BeautifulSoup(file_html, features='lxml')
	meta_tags = [item.get_text() for item in html_page.findAll('meta')]
	title = [item.get_text() for item in html_page.findAll('title')]
	headers = [item.get_text() for item in html_page.findAll('h')]
	bold_tags = [item.get_text() for item in html_page.findAll('b')]
	meta_tags = [WordNetLemmatizer().lemmatize(token).lower() for i in meta_tags for token in preprocess_tokens(i)]
	title = [WordNetLemmatizer().lemmatize(token).lower() for i in title for token in preprocess_tokens(i)]
	headers = [WordNetLemmatizer().lemmatize(token).lower() for i in headers for token in preprocess_tokens(i)]
	bold_tags = [WordNetLemmatizer().lemmatize(token).lower() for i in bold_tags for token in preprocess_tokens(i)]
	return [meta_tags,title,headers,bold_tags]

def build_postings(key):
	print('GOING THROUGH DOCID ' + key)
	extracted_html_text = strip_raw_html_text(key)
	text = preprocess_tokens(extracted_html_text)
	postings = dict()
	for token in text:
		word = WordNetLemmatizer().lemmatize(token).lower()
		if word not in postings:
			postings[word] = {'docID': key, 'occurrences': 1, 'tf_idf': 0, 'tag_score': 0}
		elif word in postings:
			postings[word]['occurrences'] +=1

	doc_length = 0
	for word,value in postings.items():
		postings[word]['tf_idf'] = calculate_tf(value['occurrences'])
		doc_length += math.pow(postings[word]['tf_idf'],2)
	doc_length = math.sqrt(doc_length)
	for word in postings.keys():
		postings[word]['doc_length'] = doc_length

	important_words_list = get_important_words(key)
	for j in important_words_list:
		for i in j:
			if i in postings:
				postings[i]['tag_score'] += 0.01

	return postings

def create_index():
	inverted_index_dict = dict()
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["invertedIndex"]
	mycol = mydb["words"]

	with open("/Users/filoprince/Documents/cs121_project3/WEBPAGES_RAW/bookkeeping.json") as f:
		htmlPageData = json.load(f)
	for key in htmlPageData:
		postings = build_postings(key)
		for word in postings.keys():
			if word not in inverted_index_dict.keys():
				inverted_index_dict[word] = [postings[word]]
			else:
				inverted_index_dict[word].append(postings[word])
	for word in inverted_index_dict.keys():
		mycol.insert_one({'word': word, 'metadata': inverted_index_dict[word], 'idf': calculate_idf(len(inverted_index_dict[word]))})

if __name__ == '__main__':
	create_index()

	

	
	
