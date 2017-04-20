from bunch import Bunch
from corpus import Corpus

en_corpus_config = Bunch()
en_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
en_corpus_config.data_addr = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\etest'
en_corpus_config.name = 'ettest'
en_corpus_config.vocab_size = 100
en_corpus_config.divs_size = [3 * i for i in range(10)]
en_corpus = Corpus(en_corpus_config)

fa_corpus_config = Bunch()
fa_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
fa_corpus_config.data_addr = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\ftest'
fa_corpus_config.name = 'ftest'
fa_corpus_config.vocab_size = 100
fa_corpus_config.encoding = 'utf-8'
fa_corpus_config.divs_size = en_corpus.divs_size
fa_corpus = Corpus(fa_corpus_config, en_corpus.inds)

while True:
    input('next_batch')
    tmp = en_corpus.next_batch(5)
    for s in tmp:
        print(en_corpus.to_words(s))

print('fin')