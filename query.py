import json
import nltk
import pymongo
import math
import inverted_index
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
PATH = 'C:/Users/nickj/Desktop/CS 121/webpages_clean/'

def get_doc_ids(user_query):
  myclient = pymongo.MongoClient("mongodb://localhost:27017/")
  mydb = myclient["invertedIndex"]
  mycol = mydb["words"]
  result = dict()
  for w in user_query:
    word = mycol.find_one({'word':w})
    if word != None:
      for d in word['metadata']:
        score = d['tf_idf'] + d['tag_score']
        if d['docID'] not in result.keys():
          result[d['docID']] = {w: {'score': score, 'doc_length': d['doc_length']}}
        else:
           result[d['docID']].update({w: {'score': score, 'doc_length': d['doc_length']}})
  if(len(user_query)>1):
    result = {k:v for k,v in result.items() if len(v.keys()) > 1}

  sorted_results = sorted(result, key=lambda x: sum([v['score'] for k,v in result[x].items()]),reverse=True)

  return sorted_results[0:100]

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

def tfidf_query_list(words,query_dict):
  q_list = []
  myclient = pymongo.MongoClient("mongodb://localhost:27017/")
  mydb = myclient["invertedIndex"]
  mycol = mydb["words"]
  for word in words:
    query = mycol.find_one({'word':word})
    tf_idf = inverted_index.calculate_tf(query_dict[word]) * query['idf']
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

def find_cosine_score(user_query,docid,q_list,q_list_length):
  print('GOING THROUGH DOCID ' + docid)
  d_list_length = get_doc_length(user_query,docid)
  d_list = get_tfidf_document(user_query,docid)
  q_normalize = normalize(q_list,q_list_length)
  d_normalize = normalize(d_list,d_list_length)
  return {'docID': docid, 'score':calculate_cosine(q_normalize,d_normalize)}

def prompt_query():
  user_query = input("Enter query: ")
  user_query = inverted_index.preprocess_tokens(user_query)
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
  return sorted(scores,key=lambda scores: scores['score'],reverse=True)

def print_top_20_scores(scores):
  scores_length = len(scores)
  if scores_length > 20:
    scores_length = 20
  with open(PATH + "bookkeeping.json") as f:
    htmlPageData = json.load(f)
  for i in range(0,scores_length):
    print(htmlPageData[scores[i]['docID']])
    print(PATH + scores[i]['docID'] 
      + ' score: ' + str(scores[i]['score']))

if __name__ == "__main__":
  #print(get_doc_ids(['bren', 'school']))
  print_top_20_scores(prompt_query())

  







