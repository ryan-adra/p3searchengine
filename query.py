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
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
STOPWORDS = set(stopwords.words('english'))

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["invertedIndex"]
mycol = mydb["words"]

def isvalid(s):
  return all(ord(c) < 128 and ord(c) not in range(48, 58) for c in s)

def preprocess_tokens(extracted_words):
	tokenize_words = word_tokenize(extracted_words)
	text = (word for word in tokenize_words if word.lower() not in STOPWORDS and isvalid(word) and len(word) > 1)
	return text

def get_doc_ids(user_query):
  result = dict()
  count = 0
  for w in user_query:
    count+=1
    word = mycol.find_one({'word':w})
    if word != None:
      for d in word['metadata']['postingList']:
        if d['docID'] not in result.keys():
          result[d['docID']] = {w: d['tf_idf']}
        else:
          result[d['docID']][w] = d['tf_idf']
  filtered_results = dict()
  if(count>1):
    for k in result.keys():
      if len(result[k]) > 1:
        filtered_results[k]=result[k]
  else:
    filtered_results = result
  return filtered_results

def add_tfidf(doc):
  print('updating.. ' + doc['word'])
  for d in doc['postingList']:
    tf = 1 + log(d['occurrences'])
    tfidf = tf*doc['idf']
    mycol.update_one({"word": doc['word'], "postingList.docID": d['docID']}, {'$set':{'postingList.$.tf_idf': tfidf}})

def prompt_query():
  user_query = input("Enter query: ")
  user_query = preprocess_tokens(user_query)
  print(get_doc_ids(user_query))

if __name__ == "__main__":
  prompt_query()
  myclient.close()