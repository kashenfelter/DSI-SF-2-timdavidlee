import pandas as pd
import datetime
import numpy as np
import cPickle as pickle
from gensim import corpora, models, matutils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from topicModelingClass import topicModeling


#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
# main run time

with open('../jNotebooks/master_total_df.p','rb') as f:
    master_total_df = pickle.load(f)
master_total_df.head(2)
alltext = master_total_df['jobdesc'].values


print '\n============= Count Vec'
tM = topicModeling(alltext[:100],verbose=1)
vec = CountVectorizer(stop_words='english', ngram_range=(1,1), min_df =2,max_features=50000)
tM.setVectorizer(vec)
tM.fitText()
tM.fitTopics(topic_ct=3,passes=20)


print '\n============ word frequencies =================='
print tM.wordFreq[:20]
print '\n============ topics found === =================='
tM.printTopics()


print '\n============= Tfidf'
tM = topicModeling(alltext[:100],verbose=1)
vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1), min_df =2,max_features=50000)
tM.setVectorizer(vec)
tM.fitText()
tM.fitTopics(topic_ct=3,passes=3)

print '\n============ word frequencies =================='
print tM.wordFreq[:20]
print '\n============ topics found === =================='
tM.printTopics()



print '\n============= Count X 22,000'
tM = topicModeling(alltext,verbose=1)
vec = CountVectorizer(stop_words='english', ngram_range=(1,1), min_df =2,max_features=50000)
tM.setVectorizer(vec)
tM.fitText()
tM.fitTopics(topic_ct=5,passes=25)
tM.printTopics()
tM.save('ALL22k05')


print '\n============= Count X 10,000'
tM.fitTopics(topic_ct=10,passes=25)


print '\n============ word frequencies =================='
print tM.wordFreq[:20]
print '\n============ topics found === =================='
tM.printTopics()
tM.save('ALL22k10')



print '\n============= Count X 10,000'
tM.fitTopics(topic_ct=25,passes=25)

print '\n============ word frequencies =================='
print tM.wordFreq[:20]
print '\n============ topics found === =================='
tM.printTopics()
tM.save('ALL22k25')


print '\n============= Count X 10,00 ngram 2'
tM = topicModeling(alltext,verbose=1)
vec = CountVectorizer(stop_words='english', ngram_range=(2,2), min_df =2,max_features=50000)
tM.setVectorizer(vec)
tM.fitText()
tM.fitTopics(topic_ct=10,passes=25)

print '\n============ word frequencies =================='
print tM.wordFreq[:20]
print '\n============ topics found === =================='
tM.printTopics()
tM.save('ALL22k2G25')