from bunch import Bunch
from corpus import Corpus

# fa_corpus_config = Bunch()
# fa_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
# fa_corpus_config.data_addr = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\fa1'
# fa_corpus_config.name = 'fa_corpus2'
# fa_corpus_config.vocab_size = 500
# fa_corpus_config.encoding = 'utf-8'
# fa_corpus = Corpus(fa_corpus_config)

en_corpus_config = Bunch()
en_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
en_corpus_config.data_addr = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\test'
en_corpus_config.name = 'test'
en_corpus_config.vocab_size = 100
en_corpus_config.divs_size = [5, 8, 11, 17]
en_corpus = Corpus(en_corpus_config)

while(True):
    input('next_batch')
    for s in en_corpus.next_batch(5):
        print(en_corpus.to_words(s))


print('fin')