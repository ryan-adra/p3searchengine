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

def isvalid(s):
    return all(ord(c) < 128 and ord(c) not in range(48, 58) for c in s)

def preprocess_tokens(extracted_words):
	tokenize_words = word_tokenize(extracted_words)
	text = [word for word in tokenize_words if word.lower() not in STOPWORDS and isvalid(word) and len(word) > 1]
	#tagged = nltk.pos_tag(text)
	return text

def get_database():
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["invertedIndex"]
	mycol = mydb["words"]
	return mycol

def calculate_tf(num):
	return 1+log(num)

def calculate_idf(posting):
	return log(37497/len(posting['postingList']))

def calculate(posting_dict):
	tf = calculate_tf(posting_dict['doc']['occurrences'])
	mycol = get_database()
	mycol.update_one({"word": posting_dict['word'], "postingList.docID": posting_dict['doc']['docID']}, {'$set':{'postingList.$.tf_idf':tf}})

def calculate_tfidf_document(query):
	mycol = get_database()
	for word in query:
		if 'tf_idf' not in mycol.find_one({'word':word})['postingList'][0].keys():
			alist = [{'word':word,'doc':d} for d in mycol.find_one({'word':word})['postingList']]
			with mp.Pool() as pool:
				pool.map(calculate,alist)
				pool.close()
			'''for posting in mycol.find_one({'word':word})['postingList']:
				tf = calculate_tf(posting['occurrences'])
				mycol.update_one({"word": mycol.find_one({'word':word})['word'], "postingList.docID": posting['docID']}, {'$set':{'postingList.$.tf_idf':tf}})'''
		else:
			continue

def build_postings(key):
	file_path = '/Users/filoprince/Documents/cs121_project3/WEBPAGES_RAW/' + key
	with open(file_path, 'r', encoding='utf-8') as file:
		html_page = BeautifulSoup(file, features='lxml')
	extracted_words = html_page.get_text()
	text = preprocess_tokens(extracted_words)
	postings = dict()
	for token in text:
		docExistsFlag = True
		word = WordNetLemmatizer().lemmatize(token).lower()
		if word not in postings:
			postings[word] = {'postingList' : {'docID': key, 'occurrences': 1, 'term_frequency': 0, 'tf_idf':0 }}
		elif word in postings:
			postings[word]['postingList']['occurrences'] +=1
	for word,value in postings.items():
		tf = calculate_tf(extracted_words,value['postingList']['occurrences'])
		postings[word]['postingList']['term_frequency'] = tf
	return postings

def insert_index(key):
	print('GOING THROUGH FILE ' + key)
	postings = build_postings(key)
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["invertedIndex"]
	mycol = mydb["words"]
	for word in postings:
		myQuery = mycol.find_one({"word": word})
		if not myQuery:
			document = {"word": word, "postingList": [postings[word]['postingList']]}
			mycol.insert_one(document)
		else:
			mycol.update_one(myQuery, {'$addToSet': {'postingList': postings[word]['postingList']}})		
	myclient.close()
	
def multi_build():
	with open("/Users/filoprince/Documents/cs121_project3/WEBPAGES_RAW/bookkeeping.json") as f:
		htmlPageData = json.load(f)
	with mp.Pool() as pool:
		pool.map(insert_index,htmlPageData.keys())
		pool.close()

def query(word):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["invertedIndex"]
	mycol = mydb["words"]

	with open("/Users/filoprince/Documents/cs121_project3/WEBPAGES_RAW/bookkeeping.json") as f:
		htmlPageData = json.load(f)

	myQuery = mycol.find_one({"word": word.lower()})

	for docID in (d["docID"] for d in myQuery["postingList"]):
		print(htmlPageData[docID])

	print("Num URLS: " + str(len(myQuery["postingList"])))
	
def w_intersection(list1, list2):
	return list(set(list1).intersection(set(list2)))

def query_dict(user_query):
	q_dict = dict()
	for w in user_query:
		if w not in q_dict:
			q_dict[w] = 1
		else:
			q_dict[w] += 1
	return q_dict

def normalize_q(num_list):
	result = []
	sum = 0
	for num in num_list:
		sum += math.pow(num,2)
	doc_length = math.sqrt(sum)
	for num in num_list:
		result.append(num/doc_length)
	return result

def normalize_d(num_list1,num_list2):
	result = []
	sum = 0
	for num in num_list1:
		sum += math.pow(num,2)
	doc_length = math.sqrt(sum)
	for num in num_list2:
		result.append(num/doc_length)
	return result

def calculate_cosine(q,d):
	result = 0
	for i in range(0,len(q)):
		result += q[i]*d[i]
	return result

def get_tfidf_document(user_query):
	d_dict = defaultdict(dict)
	mycol = get_database()
	for w in user_query:
		query = mycol.find_one({'word':w})
		for posting in query['postingList']:
			d_dict[w][posting['docID']] = posting['tf_idf']
	return d_dict

def tfidf_query_list(word_intersection,query_dict):
	q_list = []
	mycol = get_database()
	for word in word_intersection:
		mycol = get_database()
		query = mycol.find_one({'word':word})
		tf_idf = calculate_tf(query_dict[word]) * calculate_idf(query)
		q_list.append(tf_idf)
	return q_list

def get_doc_ids(user_query):
	result = []
	mycol = get_database()
	for w in user_query:
		word = mycol.find_one({'word':w})
		if word != None:
			result = [i['docID'] for i in word['postingList']]
	return list(set(result))

def get_document_text(docID):
	file_path = '/Users/filoprince/Documents/cs121_project3/WEBPAGES_CLEAN/' + docID
	with open(file_path, 'r' , encoding='utf8') as file:
		html_page = BeautifulSoup(file, features='lxml')
	extracted_words = html_page.get_text().lower()
	extracted_words = preprocess_tokens(extracted_words)
	return extracted_words

def get_tag_scores(query, docID):
	lemmatized = [WordNetLemmatizer().lemmatize(token).lower() for token in query]
	file_path = '/Users/filoprince/Documents/cs121_project3/WEBPAGES_CLEAN/' + docID
	with open(file_path, 'r' , encoding='utf8') as file:
		html_page = BeautifulSoup(file, features='lxml')
	meta_tags = html_page.findAll('meta')
	title = html_page.findAll('title')
	headers = html_page.findAll('h')
	bolded = html_page.findAll('b')
	score = 0
	for w in lemmatized:
		if any(w for w in (meta_tags,title,headers,bolded)):
			score+=0.01
	return score

def doc_id(scores):
	return scores['score']

def tfidf_document_list(tfidf_dict,docid):
	result_list = []
	for word,value in tfidf_dict.items():
		docidfound = False
		for v in value.keys():
			if v == docid:
				docidfound = True
				result_list.append(value[v])
		if not docidfound:
			result_list.append(0)
	return result_list

def prompt_query():
	user_query = input("Enter query: ")
	user_query = preprocess_tokens(user_query)
	q_dict = query_dict(user_query)
	docid_list = get_doc_ids(user_query)
	calculate_tfidf_document(user_query)
	tfidf_document_dict = get_tfidf_document(user_query)
	q_list = tfidf_query_list(user_query,q_dict)
	scores = []
	for docid in docid_list:
		text = get_document_text(docid)
		word_intersection = w_intersection(user_query,text)
		calculate_tfidf_document(text)
		tfidf_document_dict1 = get_tfidf_document(text)
		d_normalize_list = tfidf_document_list(tfidf_document_dict1,docid)	
		tfidf_document_dict2 = get_tfidf_document(user_query)
		d_list = tfidf_document_list(tfidf_document_dict2,docid)
		tag_score = get_tag_scores(user_query,docid)
		q_normalize = normalize_q(q_list)
		d_normalize = normalize_d(d_normalize_list,d_list)
		scores.append({'docID': docid, 'score':calculate_cosine(q_normalize,d_normalize)+tag_score})
	scores = sorted(scores,key=doc_id,reverse=True)
	scores_length = len(scores)
	if scores_length > 20:
		scores_length = 20
	for i in range(0,scores_length):
		print('/Users/filoprince/Documents/cs121_project3/WEBPAGES_RAW/' + scores[i]['docID'] 
			+ ' score: ' + str(scores[i]['score']))

if __name__ == '__main__':
	#multi_build()
	prompt_query()

	

	
	