from gensim.models import Word2Vec
from nltk.corpus import gutenberg, stopwords
import nltk
import re


#pre processing of corpus
words_raw = gutenberg.raw()
ind = words_raw.index('CHAPTER')
words_raw = words_raw[ind+10:]
words_raw = words_raw.lower()   #lowercasing every word
words_raw = re.sub(r'[^a-zA-Z]', ' ', words_raw)	#removing anything other than alphabetical words
sents_token = nltk.sent_tokenize(words_raw)	#forming snetences
words_token = [nltk.word_tokenize(sent) for sent in sents_token]	#creating word tokens
stpwds = stopwords.words('english')
#removing stopwords
for ind in range(len(words_token)):
	words_token[ind] = [w for w in words_token[ind] if not w in stpwds]
#print(words_token[:50])

#training word2vec CBOW network for gutenberg corpus
model1 = Word2Vec(words_token, min_count = 2)

vocab = model1.wv.vocab
#all the unique words
'''for v in vocab:
	print(v)'''

vector_w = model1.wv['brave']	#word vector
print type(vector_w)
sim_words = model1.wv.most_similar('mountain')	#similar words

print(sim_words)
