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
PATH = '/Users/filoprince/Documents/cs121_project3/WEBPAGES_CLEAN/'

'''
For each token in user query, find the token in the database and for each document id the word is associated with,
retrieve its tf_idf score and tag score. Store this information into a dictionary. If the user query list is longer
than one, retrieve document ids that contain more than one of the user query terms. Sort the results based on their 
total score and return the top 100 scores.
'''
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

#stores the user query into a dictionary with the word as a key and number of occurrences in the query as a value.
def query_dict(user_query):
  q_dict = dict()
  for w in user_query:
    if w not in q_dict:
      q_dict[w] = 1
    else:
      q_dict[w] += 1
  return q_dict

#normalize the tfidf scores with the document length
def normalize(num_list,doc_length):
  result = []
  for num in num_list:
    result.append(num/doc_length)
  return result

#calculates cosine score by doing the dot product of two vectors
def calculate_cosine(q,d):
  result = 0
  for i in range(0,len(q)):
    result += q[i]*d[i]
  return result

#gets the tfidf scores + tag scores from the database only for the tokens in the user query for the document id specified
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

#creates a list that stores the tf_idf score for the word in the query
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

#retrieve the document length of a specified document id for a word
def get_doc_length(user_query,docid):
  myclient = pymongo.MongoClient("mongodb://localhost:27017/")
  mydb = myclient["invertedIndex"]
  mycol = mydb["words"]
  for w in user_query:
    query = mycol.find_one({'word':w})['metadata']
    for item in query:
      if item['docID'] == docid:
        return item['doc_length']

#Gets the document length and the tfidf scores for the words in user query. normalize the scores using the document length and the query length
def find_cosine_score(user_query,docid,q_list,q_list_length):
  d_list_length = get_doc_length(user_query,docid)
  d_list = get_tfidf_document(user_query,docid)
  q_normalize = normalize(q_list,q_list_length)
  d_normalize = normalize(d_list,d_list_length)
  return {'docID': docid, 'score':calculate_cosine(q_normalize,d_normalize)}

'''prompts the user for a query and processes the query. creates a dictionary of the query and a list of the documents that have the words from the user query. 
for each docid in the docidlist, creates two vectors containing normalized scores and calculates the cosine score for the query with each doc id. sorts the 
cosine scores at the end from highest to lowest.
'''
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

#prints the top 20 urls along with their scores
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

  







