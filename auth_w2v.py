import nltk
import re

from gensim.models import Word2Vec
from nltk.corpus import gutenberg, stopwords

def preprocessing_text(sents_token):
    ''' preprocessing of text '''

    #removing stopwords
    stpwds = stopwords.words('english')
    for ind in range(len(sents_token)):
	    sents_token[ind] = [w.lower() for w in sents_token[ind] if not w in stpwds]

    #removing anything other than alphabetial words
    for ind in range(len(sents_token)):
	    sents_token[ind] = [re.sub(r'[^a-zA-Z]', '', w) for w in sents_token[ind]]


#pre processing of corpus
sents_token1 = list(gutenberg.sents('austen-emma.txt'))
sents_token1 = sents_token1[3:]

sents_token2 = list(gutenberg.sents('chesterton-thursday.txt'))
sents_token2 = sents_token2[3:]

preprocessing_text(sents_token1)
preprocessing_text(sents_token2)

#training word2vec CBOW network for gutenberg corpus
model1 = Word2Vec(sents_token2, min_count = 2, size = 300, workers = 3, sg = 0)

# summarize the loaded model
print(model1)
# summarize vocabulary
words = list(model1.wv.vocab)
print(words)
# access vector for one word
print(model1['mother'])
# save model
model1.save('model1.bin')

vector_w = model1.wv['destroy']	#word vector
print type(vector_w)
print(len(vector_w))
sim_words = model1.wv.most_similar('destroy')	#similar words
print(sim_words)
