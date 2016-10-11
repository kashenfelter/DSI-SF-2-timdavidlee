import pandas as pd
import datetime
import numpy as np
import cPickle as pickle
from gensim import corpora, models, matutils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
from lda2vec import preprocess, Corpus
from topicModelingClass import topicModeling

with open('../jNotebooks/master_total_df.p','rb') as f:
    master_total_df = pickle.load(f)

alltext = master_total_df['jobdesc'].values
def uniuncode(x):
    try:
        return unicode(x.decode('utf-8')).replace('\n',' -').lower()

    except:
        try:
            return unicode(x).replace('\n',' -').lower()
        except:
            print x
            return x 
sometext = [uniuncode(x) for x in alltext[:100]]

# max words grabbed per document
max_words = 10000

start = datetime.datetime.now()

# convert text to unicode (if not already)
# in my case text is already in unicode
# tokenize uses spacy under the hood
tokens, vocab = preprocess.tokenize(sometext, max_words, merge=False, n_threads=4)

print '1. tokens made'
# he made a generic corpus based on default dictionary
# see documentation
corpus = Corpus()

# Make a ranked list of rare vs frequent words
corpus.update_word_count(tokens)
corpus.finalize()

print '2. corpus updated'

# The tokenization uses spaCy indices, and so may have gaps
# between indices for words that aren't present in our dataset.
# This builds a new compact index
compact = corpus.to_compact(tokens)

print '3. corpus compacted'


# Remove extremely rare words
pruned = corpus.filter_count(compact, min_count=10)

# Convert the compactified arrays into bag of words arrays
bow = corpus.compact_to_bow(pruned)

print '4. bag of words (BOW) made'

# Words tend to have power law frequency, so selectively
# downsample the most prevalent words
clean = corpus.subsample_frequent(pruned)
doc_ids = np.arange(pruned.shape[0])
flattened, (doc_ids,) = corpus.compact_to_flat(pruned, doc_ids)
assert flattened.min() >= 0

print '5. corpus flattened'

# Fill in the pretrained word vectors
n_dim = 300
fn_wordvc = '..mount_point/08-wordvectors-pretrained/GoogleNews-vectors-negative300.bin'
vectors, s, f = corpus.compact_word_vectors(vocab, filename=fn_wordvc)

print '6. pretrained words loaded'

# Save all of the preprocessed files
pickle.dump(vocab, open('vocab.pkl', 'w'))
pickle.dump(corpus, open('corpus.pkl', 'w'))
np.save("flattened", flattened)
np.save("doc_ids", doc_ids)
np.save("pruned", pruned)
np.save("bow", bow)
np.save("vectors", vectors)

print '7. files saved'