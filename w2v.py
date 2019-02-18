import nltk
import re

from gensim.models import Word2Vec
from nltk.corpus import gutenberg, stopwords



#pre processing of corpus
sents_token = list(gutenberg.sents())
sents_token = sents_token[3:]

#removing stopwords
stpwds = stopwords.words('english')
for ind in range(len(sents_token)):
	sents_token[ind] = [w.lower() for w in sents_token[ind] if not w in stpwds]
#not removing stopwords
''' decreases overall probability for similarity measurement '''
''' for ind in range(len(sents_token)):
	    sents_token[ind] = [w.lower() for w in sents_token[ind]]'''

#removing anything other than alphabetial words
for ind in range(len(sents_token)):
	sents_token[ind] = [re.sub(r'[^a-zA-Z]', '', w) for w in sents_token[ind]]


#training word2vec CBOW network for gutenberg corpus
model1 = Word2Vec(sents_token, min_count = 2, size = 300, workers = 3, sg = 0)

# summarize the loaded model
print(model1)
# summarize vocabulary
words = list(model1.wv.vocab)
print(words)
# access vector for one word
print(model1['sentence'])
# save model
model1.save('model1.bin')
# load model
new_model = Word2Vec.load('model1.bin')
print(new_model)

vector_w = model1.wv['brave']	#word vector
print type(vector_w)
sim_words = model1.wv.most_similar('mountain')	#similar words
print(sim_words)
