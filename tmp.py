from bunch import Bunch
from corpus import Corpus

fa_corpus_config = Bunch()
fa_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
fa_corpus_config.data_addr = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\fa1'
fa_corpus_config.name = 'fa_corpus2'
fa_corpus_config.vocab_size = 500
fa_corpus_config.encoding = 'utf-8'
fa_corpus = Corpus(fa_corpus_config)

en_corpus_config = Bunch()
en_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
en_corpus_config.data_addr = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\en1'
en_corpus_config.name = 'en_corpus2'
en_corpus_config.vocab_size = 500
en_corpus = Corpus(en_corpus_config)

print(fa_corpus)