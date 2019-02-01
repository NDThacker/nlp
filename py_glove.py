from __future__ import print_function
import gensim
import nltk
import pprint
import re


from nltk.corpus import gutenberg, stopwords
from glove import Glove
from glove import Corpus


def read_corpus():
	''' reading and preproccessing the corpus '''

	sents_token = list(gutenberg.sents())
	sents_token = sents_token[3:]
	stpwds = stopwords.words('english')
	#removing stopwords
	for ind in range(len(sents_token)):
		sents_token[ind] = [w.lower() for w in sents_token[ind] if not w in stpwds]

	#removing anything other than alphabetial words
	for ind in range(len(sents_token)):
		sents_token[ind] = [re.sub(r'[^a-zA-Z]', '', w) for w in sents_token[ind]]
	
	return sents_token


if __name__ == '__main__':
	
	# Set up parameters.
	train = 1
	parallelism = 1
	query = 'brave'

	# Build the corpus dictionary and the cooccurrence matrix.
	print('Pre-processing corpus')
	corpus_model = Corpus()
	corpus_model.fit(read_corpus(), window=10)
	corpus_model.save('corpus.model')
	
	print('Dict size: %s' % len(corpus_model.dictionary))
	print('Collocations: %s' % corpus_model.matrix.nnz)

	# Train the GloVe model and save it to disk.
	print('Training the GloVe model')

	glove = Glove(no_components=100, learning_rate=0.05)
	glove.fit(corpus_model.matrix, epochs=train,
				no_threads=parallelism, verbose=True)
	glove.add_dictionary(corpus_model.dictionary)
	glove.save('glove.model')

	print('Querying for %s' % query)
	pprint.pprint(glove.most_similar(query, number=10))