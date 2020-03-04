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
inverted_index_dict = dict()
PATH = '/Users/filoprince/Documents/cs121_project3/WEBPAGES_RAW/'

#checks string to see if it is valid (ascii + alphanumeric)
def isvalid(s):
    return all(ord(c) < 128 and ord(c) not in range(48, 58) for c in s)

#tokenizes words using word tokenize, removes stopwords, invalid words, and single char words
def preprocess_tokens(extracted_words):
	tokenize_words = word_tokenize(extracted_words)
	text = [word for word in tokenize_words if word.lower() not in STOPWORDS and isvalid(word) and len(word) > 1]
	return text

#calculates the term frequency based on its occurrences in the doc
def calculate_tf(num):
	return 1+log(num)

#calculates the inverse document frequency based on the number of docs the words 
# is in relative to the document number
def calculate_idf(num):
	return log(37497/num)

#Uses bs4 library to get visible text and remove markup that includes
#style and script
def strip_raw_html_text(key):
	file_path = PATH + key
	with open(file_path, 'r', encoding='utf8') as file_html:
		html_file = BeautifulSoup(file_html,'lxml')

	for script in html_file(["script","style"]):
		script.decompose()

	html_text = html_file.get_text(separator=' ')
	lines = (line.strip() for line in html_text.splitlines())
	separate_lines = (phrase.strip() for line in lines for phrase in line.split(" "))
	html_text = ' '.join(chunk for chunk in separate_lines if chunk)
	return html_text

#Uses the beautiful soup library to get lists of all important words
# in each document
def get_important_words(key):
	file_path = PATH + key
	with open(file_path, 'r', encoding='utf8') as file_html:
		html_page = BeautifulSoup(file_html, "lxml")
	meta_tags = [item.get_text() for item in html_page.findAll('meta')]
	title = [item.get_text() for item in html_page.findAll('title')]
	headers = [item.get_text() for item in html_page.findAll('h')]
	bold_tags = [item.get_text() for item in html_page.findAll('b')]
	meta_tags = [WordNetLemmatizer().lemmatize(token).lower() for i in meta_tags for token in preprocess_tokens(i)]
	title = [WordNetLemmatizer().lemmatize(token).lower() for i in title for token in preprocess_tokens(i)]
	headers = [WordNetLemmatizer().lemmatize(token).lower() for i in headers for token in preprocess_tokens(i)]
	bold_tags = [WordNetLemmatizer().lemmatize(token).lower() for i in bold_tags for token in preprocess_tokens(i)]
	return [meta_tags,title,headers,bold_tags]


#Builds the posting for each word in the document,
#Storing its occurrences, tf_idf, document_length, and additional tag score
#based on the number of times it appears in an important tag
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

#Stores each of the postings created into an inverted_index dictionary
def create_index():
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["invertedIndex"]
	mycol = mydb["words"]

	with open(PATH + "bookkeeping.json") as f:
		htmlPageData = json.load(f)
	for key in htmlPageData:
		postings = build_postings(key)
		for word in postings.keys():
			if word not in inverted_index_dict.keys():
				inverted_index_dict[word] = [postings[word]]
			else:
				inverted_index_dict[word].append(postings[word])
	with mp.Pool() as pool:
		pool.map(insert_words_into_db,inverted_index_dict.keys())
		
	myclient.close()

#Inserts each word in the inverted index into mongodb
def insert_words_into_db(word):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["invertedIndex"]
	mycol = mydb["words"]
	mycol.insert_one({'word': word, 'metadata': inverted_index_dict[word], 'idf': calculate_idf(len(inverted_index_dict[word]))})
	myclient.close()



if __name__ == '__main__':
	create_index()
