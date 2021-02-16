from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('wordEmbeddings.txt', binary=False)
vector = model.wv['ve']
print(vector)
