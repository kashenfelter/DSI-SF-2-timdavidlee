import pandas as pd
import datetime
import numpy as np
import time
import gzip
import cPickle as pickle
from gensim import corpora, models, matutils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class topicModeling(object):
    margin = '\t\t'
    def verboseMsg(self, msg):
        if self.verbose:
            print self.margin, msg
        
    def __init__(self, alltext, verbose=0):
        self.sometext = alltext
        self.verbose = verbose
    
    def loadText(self,alltext):
        self.sometext = alltext
    
    def setVectorizer(self,vectorizer):
        self.vectorizer = vectorizer
    
    def setCvec(self,ngram_range):
        self.vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range)
    
    def setTvec(self,ngram_range):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
        
    def fitText(self):
        start = datetime.datetime.now()
        self.verboseMsg('text===>word start: fitting text of size %d documents' % (len(self.sometext)))
            
        self.X = self.vectorizer.fit_transform(self.sometext)
        
        self.verboseMsg('text===>word summarizing vocab and creating corpus ')
        
        #prepping summary
        self.df_X = pd.DataFrame(self.X.toarray(), columns=self.vectorizer.get_feature_names())
        self.wordFreq = self.df_X.sum().sort_values(ascending = False)        
    
        self.vocab = {v: k for k, v in self.vectorizer.vocabulary_.iteritems()}
        self.corpus = matutils.Sparse2Corpus(self.X, documents_columns=False)    
        
        self.verboseMsg('text===>word complete. ')
        print datetime.datetime.now() - start
            
    def fitTopics(self,topic_ct,passes):
        start = datetime.datetime.now()        
        self.topic_ct = topic_ct
        self.passes = passes
        
        self.verboseMsg('worp===>%d topics, %d passes: start ' %(topic_ct,passes))
        self.lda = models.LdaMulticore(
            self.corpus,
            num_topics  =  self.topic_ct,
            passes      =  passes,
            id2word     =  self.vocab,
            workers = 4,
            iterations = 2500,
            eval_every = 100,
            chunksize = 2000
        )
        self.verboseMsg('worp===>%d topics, %d passes: lda model complete ' %(topic_ct,passes))
        
        self.topic_vectors = self.lda.print_topics(num_topics=self.topic_ct, num_words=8)

        self.topic_proba = []
        for x in self.corpus:
            local = self.lda.get_document_topics(x)
            row = { x:float(0) for x in range(self.topic_ct)}
            for y in local:
                row[y[0]] = y[1]
            self.topic_proba.append(row)

        self.verboseMsg('worp===>%d topics, %d passes: creating probabilities in dataframe ' %(topic_ct,passes))
        
        self.topic_proba_df = pd.DataFrame(self.topic_proba)
    
        self.verboseMsg('worp===>%d topics, %d passes: complete ' %(topic_ct,passes))
        print datetime.datetime.now() - start
    def printTopics(self):
        for topic in self.topic_vectors:
            print '========================================'
            print 'topic number:', topic[0]
            for y in topic[1].split('+'):
                print '\t',y

    def save(self,prefix='_'):
        suffix = str(int(time.mktime(datetime.datetime.now().timetuple())))[-6:]
        with open("../datastorage/z003_"+prefix+"_topic_proba_df_dict_"+suffix+".p",'wb') as f:
            pickle.dump(self.topic_proba_df,f)
        with open("../datastorage/z003_"+prefix+"_topic_vectors_"+suffix+".p",'wb') as f:
            pickle.dump(self.topic_vectors,f)
        with open("../datastorage/z003_"+prefix+"_X_"+suffix+".p",'wb') as f:
            pickle.dump(self.df_X,f)
        