#load the data
import datetime
import pandas as pd
import cPickle as pickle
import patsy
import unidecode
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer

# loading data from the pickled data frame

with open('../jNotebooks/master_total_df.p','rb') as f:
    master_total_df = pickle.load(f)

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


#========================================================================
#========================================================================
printProgress('0. loading data')
master_total_df.city = master_total_df.city.map(lambda x :'other' if type(x) == dict else x)
master_total_df.city.fillna('other', inplace=True)
master_total_df.state.fillna('other', inplace=True)
master_total_df.company.fillna('other',inplace=True)
master_total_df.company = master_total_df.company.map(uncode)



#========================================================================
#========================================================================
printProgress('1. start data cleaning')

alltitles = master_total_df.expanded_title.value_counts()
limittitles = alltitles[alltitles>100].index

limited_postings = master_total_df.loc[master_total_df['expanded_title'].isin(limittitles)].copy()
limited_postings.reset_index(inplace=True)


def removeTitles(y):
    y = y.lower()
    for z in limittitles:
        y = y.replace(z,' ')
    return y

limited_postings['jobdesc'] = limited_postings['jobdesc'].map(removeTitles)
limited_postings['jobdesc'] = limited_postings['jobdesc'].map(lambda x : x.replace('\n',' '))
limited_postings['desc_len'] = limited_postings['jobdesc'].map(lambda x: len(x))


        
limited_postings['cleandesc'] = limited_postings['jobdesc'].map(uncode) 

limited_postings = limited_postings[['expanded_title','company','city','state','jobdesc','desc_len','cleandesc']]

text = limited_postings['cleandesc']

#========================================================================
#========================================================================
printProgress('2. process the job description word data for NLP')


cvec = CountVectorizer(stop_words='english', lowercase=True,ngram_range=(1,1))
start_time = datetime.datetime.now()
cvec.fit(limited_postings['cleandesc'])


cdf  = pd.DataFrame(cvec.transform(text).todense(),
             columns=cvec.get_feature_names())

summary = cdf.sum().sort_values(ascending = False)
keep_cols = summary[:10000].index
cdf_to_merge = cdf[keep_cols]
cdf_to_merge.columns = ['nlp_'+x for x in cdf_to_merge.columns]


#========================================================================
#========================================================================
printProgress('3. word vectorizing and processing complete')



formula = 'expanded_title ~ company + city + state + desc_len -1'
print formula
y, X = patsy.dmatrices(formula, limited_postings, return_type='dataframe')
# will ignore the y value for the matrix below

X = X.merge(cdf_to_merge, how = 'left', left_index=True,right_index=True)
print X.shape


mlb = MultiLabelBinarizer()
y = mlb.fit_transform([[x] for x in limited_postings['expanded_title']])
print mlb.classes_
print y.shape

#========================================================================
#========================================================================
printProgress('4. train test and split the data')

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
print X_train.shape, X_test.shape, y_train.shape, y_test.shape

premodel_data ={
    'X_train' : X_train
    ,'y_train' : y_train
    , 'X_test' : X_test
    , 'y_test' : y_test
}

with open('../mount_point/006_classification_matrix.p','wb') as f:
    pickle.dump(premodel_data,f)

#========================================================================
#========================================================================
printProgress('5. start basic logistic regression')

def multiGres(model, X_train, y_train,X_test,y_test, gs='no',params=None, cv=5):
	start = datetime.datetime.now()
	if (params==None) or (gs=='no'):
		OVRC = OneVsRestClassifier(model,n_jobs=-1)
		OVRC.fit(X_train,y_train)
		print datetime.datetime.now()-start
		scores = OVRC.score(X_test,y_test)
		return scores, OVRC
	else:
		OVRC = OneVsRestClassifier(model,n_jobs=-1)	    
		srchr = GridSearchCV(OVRC,cv=cv, param_grid=params, n_jobs=-1)
		srchr.fit(X_train,y_train)
		
		print 'gridscearch score'
		print srchr.best_score_
		print srchr.best_params_
		OVRC_best = srchr.best_estimator_
		OVRC_best.fit(X_train,y_train)
		best_score = OVRC_best.score(X_test,y_test)
		print 'actual test score'
		print best_score
		return best_score, OVRC_best 
    
lgr = LogisticRegression(n_jobs=-1)
score , OVRC = multiGres(lgr, X_train, y_train,X_test,y_test)
print 'test score:', score

#========================================================================
#========================================================================
printProgress('6. create feature importance by models')

def topFeatMulti(OVRC, bin_class_list, X_columns):
    coefs = []
    for x in OVRC.estimators_:
        coefs.append(x.coef_) 
    fitfeat_dict = {}
    for x,y in zip(bin_class_list[:5], coefs[:5]):
        coef_lkup = {abs(m):(m,n) for m,n in zip(y[0],X_columns)}
        top5 = np.sort(abs(y))[0][::-1][:5]
        top5_labeld = [coef_lkup[m] for m in top5]
        print '='*60
        print x
        for m in top5_labeld:
            print m
        fitfeat_dict[x] = top5_labeld
    
    return fitfeat_dict
    

fitfeat_dict = topFeatMulti(OVRC, mlb.classes_, X.columns)

#========================================================================
#========================================================================
printProgress('7. identify top 3 titles per ')


def topGuesses(OVRC, X, original):
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



predicted = topGuesses(OVRC, X, limited_postings)
# will print the top 6 postings and the predicted titles
for i in range(6):
    print predicted['expanded_title'][i], 
    print '\n\t'.join([str(x) for x in predicted['top_pred'][i]])


#========================================================================
#========================================================================
printProgress('8. Getting top word count by listing')

topwords_limit = 15
j = 0
def getTopWords(y):
    global j
    j+=1
    if j % 100 == 0:
        print j,
    sublist = [(str(i), x) for i, x in zip(y.transpose().sort_values(ascending=False)[1:topwords_limit].index, X.iloc[1,:].transpose().sort_values(ascending=False)[1:topwords_limit])]
    return sublist
topwords_perdoc = X.apply(getTopWords,axis=1)


predicted['topword_ct'] = topwords_perdoc
topwords_perdoc[:5]

printProgress('9. SGDClassifier - logres with gridsearch')

params = {
	'estimator__alpha' : [0.00001,0.0001, 0.001,0.01,0.1,1,10 ]
	, 'estimator__n_jobs':[-1]
	, 'estimator__loss' :['log']
	, 'estimator__penalty':['l1']
}
sgdc = SGDClassifier()
score , OVRC = multiGres(sgdc, X_train, y_train,X_test,y_test,gs='yes',params=params, cv=5)
print 'score:', score

fitfeat_dict = topFeatMulti(OVRC, mlb.classes_, X.columns)

predicted = topGuesses(OVRC, X, limited_postings)

printProgress('9. completed analysis')


