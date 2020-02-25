import json
import nltk
import pymongo
import re
import math
import multiprocessing as mp
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

def translate_tag(tag):
  if tag.startswith('J'):
    return wordnet.ADJ
  elif tag.startswith('V'):
    return wordnet.VERB
  elif tag.startswith('N'):
    return wordnet.NOUN
  elif tag.startswith('R'):
    return wordnet.ADV
  else:          
    return None

def preprocess_tokens(extracted_words):
	tokenize_words = word_tokenize(extracted_words)
	text = [word for word in tokenize_words if word.lower() not in STOPWORDS and isvalid(word) and len(word) > 1]
	#tagged = nltk.pos_tag(text)
	return text

def get_database():
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["invertedIndexTest"]
	mycol = mydb["words"]
	return mycol

def calculate_tf(num):
	return 1+log(num)

def calculate_idf(posting):
	return log(37497/len(posting['postingList']))

def calculate_tfidf_document():
	mycol = get_database()
	for document in mycol.find():
		for post in document['postingList']:
			tf = calculate_tf(post['occurrences'])
			mycol.update_one({"word": document['word'], "postingList.docID": post['docID']}, {'$set':{'postingList.$.tf_idf':tf}})

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

def build_index(key):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["invertedIndexTest"]
	mycol = mydb["words"]
	file_path = '/Users/filoprince/Documents/cs121_project3/WEBPAGES_CLEAN/' + key
	print('GOING THROUGH FILE ' + key)
	with open(file_path, 'r' , encoding='utf8') as file:
		html_page = BeautifulSoup(file, features='lxml')
	extracted_words = html_page.get_text()
	tagged = preprocess_tokens(extracted_words)
	for token in tagged:
		docExistsFlag = True
		word = WordNetLemmatizer().lemmatize(token).lower()
		myQuery = mycol.find_one({"word": word})
		if not myQuery:
			document = {"word": word, "postingList": [{"docID": key, "occurrences": 1}]}
			mycol.insert_one(document)
		else:
			docExistsFlag = False
			for docID in [d["docID"] for d in myQuery["postingList"]]:
				if docID == key:
					mycol.update_one({"word": word, "postingList.docID": docID},{'$inc': {'postingList.$.occurrences':1}})
					docExistsFlag = True
					break
			if not docExistsFlag:
				mycol.update_one(myQuery, {'$addToSet' : {'postingList' : {'docID': key, 'occurrences':1}}})
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
	
def union(list1, list2):
	return list(set(list1) | set(list2))

def query_dict(user_query):
	q_dict = dict()
	for w in user_query:
		if w not in q_dict:
			q_dict[w] = 1
		else:
			q_dict[w] += 1
	return q_dict

def normalize(num_list):
	result = []
	sum = 0
	for num in num_list:
		sum += math.pow(num,2)
	doc_length = math.sqrt(sum)
	for num in num_list:
		result.append(num/doc_length)
	return result

def calculate_cosine(q,d):
	result = 0
	for i in range(0,len(q)):
		result += q[i]*d[i]
	return result

def tfidf_document_list(word_union):
	mycol = get_database()
	d_list = []
	for word in word_union:
		query = mycol.find_one({'word':word})
		if not query:
			d_list.append(0)
		else:
			tf_idf = 0
			for i in query['postingList']:
				tf_idf += i['tf_idf']
			d_list.append(tf_idf)
	return d_list 

def tfidf_query_list(word_union,query_dict):
	q_list = []
	for word in word_union:
		if word not in query_dict.keys():
			q_list.append(0)
		else:
			mycol = get_database()
			query = mycol.find_one({'word':word})
			if query != None:
				tf_idf = calculate_tf(query_dict[word]) * calculate_idf(query)
				q_list.append(tf_idf)
			else:
				q_list.append(0)
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
			score+=0.1
	return score


def doc_id(scores):
	return scores['score']

def prompt_query():
	user_query = input("Enter query: ")
	user_query = preprocess_tokens(user_query)
	q_dict = query_dict(user_query)
	docid_list = get_doc_ids(user_query)
	scores = []
	for docid in docid_list:
		text = get_document_text(docid)
		word_union = union(user_query,text)
		d_list = tfidf_document_list(word_union)
		q_list = tfidf_query_list(word_union,q_dict)
		tag_score = get_tag_scores(user_query,docid)
		scores.append({'docID': docid, 'score':calculate_cosine(normalize(q_list),normalize(d_list))+tag_score})
	scores = sorted(scores,key=doc_id,reverse=True)
	scores_length = len(scores)
	for i in range(0,scores_length):
		print('/Users/filoprince/Documents/cs121_project3/WEBPAGES_CLEAN/' + scores[i]['docID'] 
			+ ' score: ' + str(scores[i]['score']))

if __name__ == '__main__':
	#multi_build()
	prompt_query()

	

	
	