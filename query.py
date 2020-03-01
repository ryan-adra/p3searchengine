import json
import nltk
import pymongo
import re
import math
import multiprocessing as mp
import inverted_index
from collections import defaultdict
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from math import log
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
STOPWORDS = set(stopwords.words('english'))

def isvalid(s):
  return all(ord(c) < 128 and ord(c) not in range(48, 58) for c in s)

def preprocess_tokens(extracted_words):
  tokenize_words = word_tokenize(extracted_words)
  text = [word for word in tokenize_words if word.lower() not in STOPWORDS and isvalid(word) and len(word) > 1]
  return text

def calculate_tf(num):
  return 1+log(num)

def calculate_idf(posting):
  return log(37497/len(posting['postingList']))

def get_doc_ids(user_query):
  result = []
  myclient = pymongo.MongoClient("mongodb://localhost:27017/")
  mydb = myclient["invertedIndex"]
  mycol = mydb["words"]
  for w in user_query:
    word = mycol.find_one({'word':w})
    if word != None:
      tf_dict = {info['docID']: info['tf_idf']+info['tag_score'] for info in word['metadata']}
      top_docs_list = sorted(tf_dict,key=tf_dict.__getitem__,reverse=True)
      for num in range(0,50):
        if top_docs_list[num] not in result:
          result.append(top_docs_list[num])
  return result

def query_dict(user_query):
  q_dict = dict()
  for w in user_query:
    if w not in q_dict:
      q_dict[w] = 1
    else:
      q_dict[w] += 1
  return q_dict

def normalize(num_list,doc_length):
  result = []
  for num in num_list:
    result.append(num/doc_length)
  return result

def calculate_cosine(q,d):
  result = 0
  for i in range(0,len(q)):
    result += q[i]*d[i]
  return result

def get_tfidf_document(text,docid):
  result_list = []
  myclient = pymongo.MongoClient("mongodb://localhost:27017/")
  mydb = myclient["invertedIndex"]
  mycol = mydb["words"]
  for token in text:
    isInDocId = False
    query = mycol.find_one({'word':token})
    for p in query['metadata']:
      if p['docID'] == docid:
        isInDocId = True
        result_list.append(p['tf_idf']+p['tag_score'])
    if not isInDocId:
      result_list.append(0)
  return result_list

def tfidf_query_list(word_intersection,query_dict):
  q_list = []
  myclient = pymongo.MongoClient("mongodb://localhost:27017/")
  mydb = myclient["invertedIndex"]
  mycol = mydb["words"]
  for word in word_intersection:
    query = mycol.find_one({'word':word})
    tf_idf = calculate_tf(query_dict[word]) * query['idf']
    q_list.append(tf_idf)
  return q_list

def get_doc_length(user_query,docid):
  myclient = pymongo.MongoClient("mongodb://localhost:27017/")
  mydb = myclient["invertedIndex"]
  mycol = mydb["words"]
  for w in user_query:
    query = mycol.find_one({'word':w})['metadata']
    for item in query:
      if item['docID'] == docid:
        return item['doc_length']

def doc_id(scores):
  return scores['score']

def find_cosine_score(user_query,docid,q_list,q_list_length):
  print('GOING THROUGH DOCID ' + docid)
  d_list_length = get_doc_length(user_query,docid)
  d_list = get_tfidf_document(user_query,docid)
  q_normalize = normalize(q_list,q_list_length)
  d_normalize = normalize(d_list,d_list_length)
  return {'docID': docid, 'score':calculate_cosine(q_normalize,d_normalize)}

if __name__ == "__main__":
  user_query = input("Enter query: ")
  user_query = preprocess_tokens(user_query)
  q_dict = query_dict(user_query)
  docid_list = get_doc_ids(user_query)
  q_list = tfidf_query_list(user_query,q_dict)
  q_list_length = 0
  for item in q_list:
    q_list_length += math.pow(item,2)
  q_list_length = math.sqrt(q_list_length)
  scores = []

  for docid in docid_list:
    s_dict = find_cosine_score(user_query,docid,q_list,q_list_length)
    scores.append(s_dict)

  scores = sorted(scores,key=doc_id,reverse=True)
  scores_length = len(scores)
  if scores_length > 20:
    scores_length = 20
  with open("/Users/filoprince/Documents/cs121_project3/WEBPAGES_RAW/bookkeeping.json") as f:
    htmlPageData = json.load(f)
  for i in range(0,scores_length):
    print(htmlPageData[scores[i]['docID']])
    print('/Users/filoprince/Documents/cs121_project3/WEBPAGES_RAW/' + scores[i]['docID'] 
      + ' score: ' + str(scores[i]['score']))







