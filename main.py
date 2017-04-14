from translator import Translator
from corpus import Corpus
from bunch import Bunch

tr_config = Bunch()
tr_config.inp_vocab_size = 50
tr_config.out_vocab_size = 50
tr_config.encoder_layers = 2
tr_config.encoder_hidden_size = 10
tr_config.attention_hidden_size = 10
tr_config.decoder_layers = 2
tr_config.decoder_hidden_size = 10
tr_config.rate = 0.5
tr_config.name = 'en_fa_nmt'
tr_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\model'
translator = Translator(tr_config)

en_corpus_config = Bunch()
en_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
en_corpus_config.data_addt = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\en1'
en_corpus_config.name = 'en_corpus'
en_corpus_config.vocab_size = 100
en_corpus = Corpus(en_corpus_config)

fa_corpus_config = Bunch()
fa_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
fa_corpus_config.data_addt = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\fa1'
fa_corpus_config.name = 'fa_corpus'
fa_corpus_config.vocab_size = 100
fa_corpus = Corpus(en_corpus_config)

EPOCH_SIZE = 1000
BATCH_SIZE = 10
for i in range(EPOCH_SIZE):
    x = en_corpus.next_batch(BATCH_SIZE)
    y = fa_corpus.next_batch(BATCH_SIZE)
    print(translator.train_model(x, y))

translator.save_model()