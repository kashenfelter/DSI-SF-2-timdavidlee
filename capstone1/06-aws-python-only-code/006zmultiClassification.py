#load the data
import datetime
import pandas as pd
import cPickle as pickle
import patsy
import unidecode
import numpy as np
from altair import *

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

global_start = datetime.datetime.now()
local_start = datetime.datetime.now()

def printProgress(messages):
	global global_start
	global local_start
	current = datetime.datetime.now()
	print 'task time: ', current-local_start, 'overall', current-global_start
	print '='*100
	print messages
	local_start = current

def uncode(x):
    try:
        return unidecode.unidecode(x.decode('utf-8')).replace('\n',' -').lower()

    except:
        try:
            return unidecode.unidecode(x).replace('\n',' -').lower()
        except:
            print x
            return x 

def load_master():
	with open('../jNotebooks/master_total_df.p','rb') as f:
		master_total_df = pickle.load(f)
	return master_total_df

def munge(master_total_df):
	master_total_df.city = master_total_df.city.map(lambda x :'other' if type(x) == dict else x)
	master_total_df.city.fillna('other', inplace=True)
	master_total_df.state.fillna('other', inplace=True)
	master_total_df.company.fillna('other',inplace=True)
	master_total_df.company = master_total_df.company.map(uncode)
	return master_total_df

def truncateTopTitles(master_total_df):
	alltitles = master_total_df.expanded_title.value_counts()
	limittitles = alltitles[alltitles>100].index

	limited_postings = master_total_df.loc[master_total_df['expanded_title'].isin(limittitles)].copy()
	limited_postings.reset_index(inplace=True)
	
	def removeTitles(y):
	    y = y.lower()
	    for z in limittitles:
	        y = y.replace(z,' ')
	    return y
	limited_postings['orig_jobdesc'] = limited_postings['jobdesc']
	limited_postings['jobdesc'] = limited_postings['jobdesc'].map(removeTitles)
	limited_postings['jobdesc'] = limited_postings['jobdesc'].map(lambda x : x.replace('\n',' '))
	limited_postings['desc_len'] = limited_postings['jobdesc'].map(lambda x: len(x))

	limited_postings['cleandesc'] = limited_postings['jobdesc'].map(uncode) 
	
	limited_postings = limited_postings[['expanded_title','base_title','company','city','state','jobdesc','desc_len','cleandesc','orig_jobdesc']]
	return limited_postings



def addStatFields(limited_postings):
	
	dsi_terms = ['machine learning'
	             ,'data science'
	             ,'model'
	             ,'predict'
	             ,'regression'
	             ,'bayes'
	             ,'sklearn'
	             ,'data scientist'
	             ,'neural network'
	             , ' R '
	             ,'SQL'
	            ]
	for y in dsi_terms:
	    limited_postings['term_'+y.strip()] = limited_postings['cleandesc'].map(lambda x : len(x.split(y))-1)

	    
	limited_postings['totalterms'] = sum([limited_postings[x] for x in limited_postings.columns if 'term_' in x])
	limited_postings['totalterms'] = limited_postings['totalterms'].astype(float)

	company_posting_summary = limited_postings.company.value_counts()
	top_co_post = company_posting_summary[:25]
	
	company_term_summary = limited_postings.groupby('company')[['totalterms']].sum().reset_index().sort_values('totalterms', ascending=False)
	top_co_terms = company_term_summary[:25]

	limited_postings['term_company'] = limited_postings['company'].map(lambda x: x if x in top_co_terms['company'].values else 'other_co')
	limited_postings['post_company'] = limited_postings['company'].map(lambda x: x if x in top_co_post else 'other_co')

	return limited_postings

import pandas as pd
import numpy as np
import datetime
import cPickle as pickle
import unidecode


# takes in list of text fields, manipulates them

class hardSkillParser(object):


    def __init__(self,list_of_text):
        #expecting a single value data frame
        self.text_dict = [{'text':self._fixText(x)} for x in list_of_text] 
        
    def load(self,list_of_text):
        self.text_dict = [{'text':self._fixText(x)} for x in list_of_text]
        
#===========================================================================        

    # some of hte HTML data that was pulled needs to be 
    # decoded from unicode and either falls in one of the
    # two combinations: decode UTF8 then UNICODE decode
    
    def _fixText(self, input_string):
        prevlower = False
        newText = ''
        for x in input_string:
            if x=='*':
                newText +='\n'
            elif x.isalpha()==False:
                prevlower=0
            elif x.islower():
                prevlower = True

            if (prevlower == True) & (x.isupper()):
                newText += '.\n' + x
                prevlower = False        
            else:
                newText += x
        return newText
            
            
    # similar to the previous function but 
    # designed for pandas MAP and APPLY functions
    
    def _cleanLI(self, yy):
        newlist = []
        for y in yy:
            try:
                newlist.append(unidecode.unidecode(y.decode('utf-8')).lower())
            except:
                try:
                    newlist.append(y.decode('utf-8').lower())
                except:
                    newlist.append(y.lower())
        return newlist       
   
    
    # =========================================================
    # pulls newline delinated lines, since these are typically
    # the bullet point descriptions
    # and have a higher percentage of relevant details
    # relating to job instead of disclaimers, or 
    # company background or descriptions
    
    def _parseLI(self,y):
        LI = []
        y  = y.replace('i.e.','-ie-').replace('e.g.','-eg-')
        for x in y.split('\n'):
            ct = len(x.strip().split('.'))
            if ct ==1:
                if len(x)>0:
                    LI.append(x)
            elif len(x.strip().split('.')[1])<=1:
                if len(x.strip().split('.')[0])>0:
                    LI.append(x.strip().split('.')[0])
        return LI

    # =========================================================    
    # this will provide a second level of detail
    # from the LI items pulled, this will go through and further
    # pull out more hard topics
    
    def _splitLI(self,y,phrase):
        try:
            if phrase in ('to','in'):
                return y.split(' '+phrase+' ')[1]
            else:
                return y.split(phrase)[1]
        except:
            return ''

#===========================================================================        

     
    def fit(self):
        start = datetime.datetime.now()
        terms = ['in','including','knowledge of','experience with', 'understanding of', 'to','develop ','design ','requirements']
        for x in self.text_dict:
            x['LI'] =self._cleanLI(self._parseLI(x['text']))
            hardskill = []
            for y in terms:
                x[y] = [self._splitLI(LI,y) for LI in x['LI']]
                x[y] = [z for z in x[y] if z!='']
                hardskill.extend(x[y])
            x['hardskill'] = hardskill
            x['LI_text'] = ' '.join(x['LI'])
            x['hardskill_text'] = ' '.join(x['hardskill'])
        print datetime.datetime.now()-start
        
        self.parsed_df = pd.DataFrame(self.text_dict)[['text','LI','LI_text','hardskill','hardskill_text']]
    
    def save(self,prefix='_'):
        suffix = str(int(time.mktime(datetime.datetime.now().timetuple())))[-6:]
        with open("008_"+prefix+"_parsed_text_dict_"+suffix+".p",'wb') as f:
            pickle.dump(self.text_dict,f)
        with open("008_"+prefix+"_parsed_df_"+suffix+".p",'wb') as f:
            pickle.dump(self.parsed_df,f)


def NLP_parser(text):
    # using count vectorizer to turn the cleaned job descriptions into feadutes
    cvec = CountVectorizer(stop_words='english', lowercase=True,ngram_range=(1,1))
    start_time = datetime.datetime.now()
    cvec.fit(text)


    cdf  = pd.DataFrame(cvec.transform(text).todense(),
                 columns=cvec.get_feature_names())

    #keeping the top 20,000 features (or words)
    summary = cdf.sum().sort_values(ascending = False)
    keep_cols = summary[:20000].index
    cdf_to_merge = cdf[keep_cols]


    # since some of the words are reserved, adding NLP_ as a prefix to stop any type of compiler issues
    cdf_to_merge.columns = ['nlp_'+x for x in cdf_to_merge.columns]
    return cdf_to_merge




def generateXY(base_dataframe, word_df):
    formula = 'expanded_title ~ company + city + state + desc_len -1'
    print formula
    y, X = patsy.dmatrices(formula, base_dataframe, return_type='dataframe')
    # will ignore the y value for the matrix below

    X = X.merge(word_df, how = 'left', left_index=True,right_index=True)
    print X.shape

    # turns y into a multi class matrix
    # 2 classes require a single vector n multiclassification requires n-1 matrix
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform([[x] for x in limited_postings['expanded_title']])
    print mlb.classes_
    print y.shape
    return y, X, mlb



def multiGres(model, X,y, gs='no'):
    
    #splitting data for cross validation
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    start = datetime.datetime.now()
    if  gs=='no':
        
        #non-grid search model
        #initialize the one vs. rest classifier, fit and score
        
        print 'initialzing One vs. Rest'
        OVRC = OneVsRestClassifier(model,n_jobs=-1)
        print 'Fitting training data...'
        OVRC.fit(X_train,y_train)
        print datetime.datetime.now()-start
        print 'scoring....'
        train_score = OVRC.score(X_train,y_train)
        test_score = OVRC.score(X_test,y_test)
        print 'train score'
        print train_score
        print 'test score'
        print test_score
        return train_score, test_score, OVRC
    else:
        #non-grid search model
        #initialize the one vs. rest classifier, fit and score
        
        print 'initialzing One vs. Rest'
        OVRC = OneVsRestClassifier(model)
        print 'Fitting training data...'
        OVRC.fit(X_train,y_train)
        print datetime.datetime.now()-start
        print 'scoring....'
        train_score = OVRC.score(X_train,y_train)
        test_score = OVRC.score(X_test,y_test)
        print 'train score'
        print train_score
        print 'test score'
        print test_score
        return train_score, test_score, OVRC


def topFeatMulti(OVRC, bin_class_list, X_columns):
    coefs = []
    for x in OVRC.estimators_:
        coefs.append(x.coef_) 
    fitfeat_dict = {}
    for x,y in zip(bin_class_list, coefs):
        coef_lkup = {abs(m):(m,n) for m,n in zip(y[0],X_columns)}
        top5 = np.sort(abs(y))[0][::-1][:10]
        top5_labeld = [coef_lkup[m] for m in top5]
        print '='*60
        print x
        for m in top5_labeld:
            print m
        fitfeat_dict[x] = top5_labeld
    
    return fitfeat_dict


def topGuesses(OVRC, X, original, mlb):
    proba = OVRC.predict_proba(X)
    print proba.shape
    df = pd.DataFrame(proba)
    df.columns = mlb.classes_
    def topPredict(y):
        top3 = y.transpose().sort_values(ascending=False)[:4]
        return [(x,z) for x,z in zip(top3.index, top3)]

    df['top_pred'] = df.apply(topPredict,axis=1)
    results = original.merge(df[['top_pred']], how='left', left_index=True, right_index=True)
    results['yhat'] = results['top_pred'].map(lambda x: x[0][0]) 
    return results






#==============================================================================================================
#==============================================================================================================
#==============================================================================================================


# initialize the storage dictionary
results_storage = {}




def preprocess():
	printProgress('loading data')
	master_total_df = load_master()

	printProgress('subsetting by expanded titles that have > 100 occr')	
	limited_postings = truncateTopTitles(munge(master_total_df))

	printProgress('adding EDA ML stat fields')
	limited_postings = addStatFields(limited_postings)

	printProgress('creating subtexts')
	hSP = hardSkillParser(limited_postings['orig_jobdesc'])
	hSP.fit()
	fulltext = limited_postings['cleandesc']
	LItext = hSP.parsed_df['LI_text']
	HStext = hSP.parsed_df['hardskill_text']
	full_df = NLP_parser(fulltext)
	LI_df = NLP_parser(LItext)
	HS_df = NLP_parser(HStext)
	print full_df.shape, LI_df.shape, HS_df.shape
	return limited_postings, full_df, LI_df, HS_df


limited_postings, full_df, LI_df, HS_df = preprocess()

# =============================================================
printProgress('Generating X and y for full text')
y, X, mlb = generateXY(limited_postings, full_df)


printProgress('start modeling - starting with full text')
printProgress('Logistic Regression - vanilla')
lgr = LogisticRegression(n_jobs=-1, max_iter=5000)
train_score, test_score, OVRC = multiGres(lgr, X, y)

fitfeat_dict = topFeatMulti(OVRC, mlb.classes_, X.columns)
predicted = topGuesses(OVRC, X, limited_postings,mlb)


# saving results
results_storage['FULL_LOG'] = {
	'train_score' : train_score
	, 'test_score' : test_score
	, 'fitfeature_dict':fitfeat_dict
	, 'predicted': predicted['top_pred']
}



for C in [.01,.1,1.,10,100]:
	printProgress('Logistic Regression - with l1 loss %f' % C)
	lgr = LogisticRegression(n_jobs=-1, penalty='l1',max_iter = 5000, C=C)
	train_score, test_score , OVRC = multiGres(lgr, X, y)
	
	fitfeat_dict = topFeatMulti(OVRC, mlb.classes_, X.columns)
	predicted = topGuesses(OVRC, X, limited_postings,mlb)

	#Saving results
	results_storage['FULL_LOGL1'+str(C)] = {
		'train_score' : train_score
		, 'test_score' : test_score
		, 'fitfeature_dict':fitfeat_dict
		, 'predicted': predicted['top_pred']
	}


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
print X_train.shape, X_test.shape, y_train.shape, y_test.shape
printProgress('Random Forest Classifier')
rfc = RandomForestClassifier(n_estimators=1000, n_jobs = -1)
rfc.fit(X_train, y_train)
train_score = rfc.score(X_train, y_train)
test_score = rfc.score(X_test, y_test)
print train_score, test_score

results_storage['FULL_RFC'] = {
	'train_score' : train_score
	, 'test_score' : test_score
	, 'fitfeature_dict': None
	, 'predicted': None
}


# =============================================================
printProgress('Generating X and y for line item text')
y, X, mlb = generateXY(limited_postings, LI_df)


printProgress('start modeling - starting with full text')
printProgress('Logistic Regression - vanilla')
lgr = LogisticRegression(n_jobs=-1, max_iter=5000)
train_score, test_score , OVRC = multiGres(lgr, X, y)


fitfeat_dict = topFeatMulti(OVRC, mlb.classes_, X.columns)
predicted = topGuesses(OVRC, X, limited_postings,mlb)


# saving results
results_storage['LI_LOG'] = {
	'train_score' : train_score
	, 'test_score' : test_score
	, 'fitfeature_dict':fitfeat_dict
	, 'predicted': predicted['top_pred']
}


for C in [.01,.1,1.,10,100]:
	printProgress('Logistic Regression - with l1 loss %f' % C)
	lgr = LogisticRegression(n_jobs=-1, penalty='l1',max_iter = 5000, C=C)
	train_score, test_score , OVRC = multiGres(lgr, X, y)

	fitfeat_dict = topFeatMulti(OVRC, mlb.classes_, X.columns)
	predicted = topGuesses(OVRC, X, limited_postings,mlb)

	#Saving results
	results_storage['LI_LOGL1'+str(C)] = {
		'train_score' : train_score
		, 'test_score' : test_score
		, 'fitfeature_dict':fitfeat_dict
		, 'predicted': predicted['top_pred']
	}

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
print X_train.shape, X_test.shape, y_train.shape, y_test.shape
printProgress('Random Forest Classifier')
rfc = RandomForestClassifier(n_estimators=1000, n_jobs = -1)
rfc.fit(X_train, y_train)
train_score = rfc.score(X_train, y_train)
test_score = rfc.score(X_test, y_test)
print train_score, test_score

results_storage['LI_RFC'] = {
	'train_score' : train_score
	, 'test_score' : test_score
	, 'fitfeature_dict': None
	, 'predicted': None
}




# =============================================================
printProgress('Generating X and y for HS text')
y, X, mlb = generateXY(limited_postings, HS_df)


printProgress('start modeling - starting with full text')
printProgress('Logistic Regression - vanilla')
lgr = LogisticRegression(n_jobs=-1, max_iter=5000)
train_score, test_score , OVRC = multiGres(lgr, X, y)


fitfeat_dict = topFeatMulti(OVRC, mlb.classes_, X.columns)
predicted = topGuesses(OVRC, X, limited_postings,mlb)

# saving results
results_storage['HS_LOG'] = {
	'train_score' : train_score
	, 'test_score' : test_score
	, 'fitfeature_dict':fitfeat_dict
	, 'predicted': predicted['top_pred']
}

for C in [.01,.1,1.,10,100]:
	printProgress('Logistic Regression - with l1 loss %f' % C)
	lgr = LogisticRegression(n_jobs=-1, penalty='l1',max_iter = 5000, C=C)
	train_score, test_score , OVRC = multiGres(lgr, X, y)


	#Saving results
	results_storage['HS_LOGL1'+str(C)] = {
		'train_score' : train_score
		, 'test_score' : test_score
		, 'fitfeature_dict':fitfeat_dict
		, 'predicted': predicted['top_pred']
	}

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
print X_train.shape, X_test.shape, y_train.shape, y_test.shape
printProgress('Random Forest Classifier')
rfc = RandomForestClassifier(n_estimators=1000, n_jobs = -1)
rfc.fit(X_train, y_train)
train_score = rfc.score(X_train, y_train)
test_score = rfc.score(X_test, y_test)
print train_score, test_score

results_storage['HS_RFC'] = {
	'train_score' : train_score
	, 'test_score' : test_score
	, 'fitfeature_dict': None
	, 'predicted': None
}



print 'saving stored results'
with open('../datastorage/MULTILOG_RESULTS.p','wb') as f:
	pickle.dump(results_storage, f)






