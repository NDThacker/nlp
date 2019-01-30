import nltk
import re

from gensim.models import Word2Vec
from matplotlib import pyplot
from nltk.corpus import gutenberg, stopwords
from sklearn.decomposition import PCA


''' pre processing of corpus '''
sents_token = list(gutenberg.sents())
sents_token = sents_token[3:10]
stpwds = stopwords.words('english')
#removing stopwords
for ind in range(len(sents_token)):
	sents_token[ind] = [w.lower() for w in sents_token[ind] if not w in stpwds]

#removing anything other than alphabetial words
for ind in range(len(sents_token)):
	sents_token[ind] = [re.sub(r'[^a-zA-Z]', '', w) for w in sents_token[ind]]


''' training word2vec CBOW network for gutenberg corpus '''
model1 = Word2Vec(sents_token, min_count = 2)

X = model1[model1.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model1.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

