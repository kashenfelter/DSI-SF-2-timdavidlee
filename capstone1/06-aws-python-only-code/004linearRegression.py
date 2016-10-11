#load libraries
import datetime
import pandas as pd
import numpy as np
import cPickle as pickle
import patsy
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, Ridge
from sklearn.grid_search import GridSearchCV
import unidecode

global_start = datetime.datetime.now()
start = datetime.datetime.now()

def printProgress():
    global start
    currentTime = datetime.datetime.now()
    print currentTime - start, ' of ', currentTime - global_start 
    start = currentTime


#loading data from the pickled data frame

# ======================================================================================
banner = '======================================================================================\n======================================================================================'
print banner
print '1. loading data'
with open('../jNotebooks/master_total_df.p','rb') as f:
    master_total_df = pickle.load(f)
    
printProgress()

# ======================================================================================
print banner
print '2. cleaning data'
print master_total_df.shape
# only one data source has view data, will sub-set here.
jobview_data = master_total_df[master_total_df.sourcesite =='lnk'].copy()
print jobview_data.shape
jobview_data.reset_index(inplace=True)

jobview_data['expanded_title'].fillna('other',inplace=True)
jobview_data['prefix_title'].fillna('other',inplace=True)
jobview_data['state'].fillna('other',inplace=True)
jobview_data['base_title'].replace(to_replace='', value='other',inplace=True)

jobview_data['post_month'] = jobview_data['post_start_date'].map(lambda x : x[:3] )
jobview_data['days_posted'] = jobview_data['days_posted'].map(lambda x: x.split(' ')[1])
jobview_data['description_length'] = jobview_data['jobdesc'].map(lambda x: len(x))

text = jobview_data['jobdesc'].map(lambda x : unidecode.unidecode(x).replace('\n',' '))

printProgress()

# ======================================================================================
print banner
print '3. scaling data data'
# initialize the scaler and create a normalized dataset
sclr = StandardScaler()
jobview_data_norm = jobview_data.copy()

# convert to float and remove any words in the column
jobview_data_norm['days_posted'].replace(to_replace=unicode('less'), value=0. , inplace=True)
jobview_data_norm['days_posted'] = jobview_data_norm['days_posted'].astype(float)

jobview_data_norm['description_length'].replace(to_replace='less', value=0. , inplace=True)
jobview_data_norm['description_length'] = jobview_data_norm['description_length'].astype(float)


# normalize the two number features
norm_cols = ['days_posted','description_length']
jobview_data_norm[norm_cols].mean()
jobview_data_norm[norm_cols] = sclr.fit_transform(jobview_data_norm[norm_cols])

printProgress()


job_columns = ['company','city','state','views','days_posted','base_title','expanded_title', 'post_month','description_length']
# ======================================================================================
print banner
print '4. splitting X and Y datasets'
formula = 'views ~ ' + ' + '.join([x for x in jobview_data_norm[job_columns].columns if x != 'views']) + '-1'
y, X = patsy.dmatrices(formula, jobview_data_norm,return_type='dataframe')
#X = X.merge(cdf_for_merge, how='left', left_index=True, right_index=True)
y = np.ravel(y)
print X.shape, y.shape

printProgress()
# ======================================================================================
print banner
print '4a. train_test_split'
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

printProgress()
# ======================================================================================
print banner
print 'begin basic and baseline regressions (without NLP additions) ========= '
print '5.Performing lasso'
lm = LassoCV(cv=5, n_alphas=200, n_jobs=-1, max_iter=10000)
lm.fit(X,y)
score = lm.score(X,y)
print score, lm.alpha_

print 'train / test score'
lm.fit(X_train,y_train)
score = lm.score(X_test,y_test)
print score, lm.alpha_

printProgress()
# ======================================================================================
print banner
print '6. Performing Ridge'
rm = RidgeCV(cv=5, alphas=[0.1,1.0,3,10,30,100,300, 1000,3000])
rm.fit(X,y)
score = rm.score(X,y)
print score, rm.alpha_

print 'train / test score'
rm.fit(X_train,y_train)
score = rm.score(X_test,y_test)
print score, rm.alpha_

printProgress()
# ======================================================================================
print banner
print '7. Performing Decision Tree Regressor'
from sklearn.tree import DecisionTreeRegressor
dtc = DecisionTreeRegressor(max_depth=10, min_samples_split=20)
dtc.fit(X,y)
mscore = dtc.score(X,y)
print mscore

print 'test train score'
dtc.fit(X_train,y_train)
mscore = dtc.score(X_test,y_test)
print mscore

printProgress()
# ======================================================================================
print banner
print '8. prepping the word data'
cvec = CountVectorizer(stop_words='english', lowercase=True,ngram_range=(1,1))
cvec.fit(text)

printProgress()

cdf  = pd.DataFrame(cvec.transform(text).todense(),
             columns=cvec.get_feature_names())

printProgress()
# ======================================================================================
print banner
print 'begin add in NLP features ========= '
print '9. merging the top 10,000 most common words; merging with X'
summary = cdf.sum().sort_values(ascending=False)
word_features = summary[:10000].index
word_features

cdf_for_merge = cdf[word_features].copy()
cdf_for_merge.columns = ['nlp_'+x for x in cdf_for_merge.columns]

X_plus = X.copy()
X_plus = X_plus.merge(cdf_for_merge, how='left', left_index=True, right_index=True)
print X_plus.shape

X_plus_train = X_train.merge(cdf_for_merge, how='left', left_index=True, right_index=True)
X_plus_test = X_test.merge(cdf_for_merge, how='left', left_index=True, right_index=True)


printProgress()
# ======================================================================================
print banner
print '10. using enhanced data with Ridge'

rm = RidgeCV(cv=5, alphas=[0.1,1.0,3,10,30,100,300, 1000,3000])
rm.fit(X_plus_train,y_train)

print 'ridgeCV - training score'
score = rm.score(X_plus_train,y_train)
print score, rm.alpha_

print 'ridgeCV - test score'
score = rm.score(X_plus_test,y_test)
print score, rm.alpha_


printProgress()

# ======================================================================================
print banner
print '11. using enhanced data with decision tree'
dtc.fit(X,y)
mscore = dtc.score(X,y)
print mscore

print 'test train score'
dtc.fit(X_plus_train,y_train)
mscore = dtc.score(X_plus_test,y_test)
print mscore