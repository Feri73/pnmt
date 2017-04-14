import numpy as np

from translator import Translator
from corpus import Corpus
from bunch import Bunch


en_corpus_config = Bunch()
en_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
en_corpus_config.data_addr = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\en1'
en_corpus_config.name = 'en_corpus'
en_corpus_config.vocab_size = 100
en_corpus = Corpus(en_corpus_config)

fa_corpus_config = Bunch()
fa_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
fa_corpus_config.data_addr = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\fa1'
fa_corpus_config.name = 'fa_corpus'
fa_corpus_config.vocab_size = 100
fa_corpus = Corpus(fa_corpus_config)

print(en_corpus.index, fa_corpus.index)

tr_config = Bunch()
tr_config.inp_vocab_size = en_corpus.vocab_size
tr_config.out_vocab_size = fa_corpus.vocab_size
tr_config.encoder_layers = 2
tr_config.encoder_hidden_size = 10
tr_config.attention_hidden_size = 10
tr_config.decoder_layers = 2
tr_config.decoder_hidden_size = 10
tr_config.rate = 5
tr_config.name = 'en_fa_nmt'
tr_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\model'
translator = Translator(tr_config)

EPOCH_SIZE = 10000
BATCH_SIZE = 1
PRINT_PER = 100
for i in range(EPOCH_SIZE * int(en_corpus.data.shape[0] / BATCH_SIZE)):  # use batch==>also in model
    try:
        x = en_corpus.next_batch(BATCH_SIZE)
        x = np.reshape(x[0], (x.shape[1], x.shape[2], 1))
        y = fa_corpus.next_batch(BATCH_SIZE)
        y = np.reshape(y[0], (y.shape[1], y.shape[2], 1))
        lost = translator.train_model(x, y)
        if i % PRINT_PER == 0:
            print(i, ': ', lost)
    except KeyboardInterrupt:
        break
print('saving...')
en_corpus.save_corpus_state()
fa_corpus.save_corpus_state()
translator.save_model()
print('finished')