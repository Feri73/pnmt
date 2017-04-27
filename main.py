from translator import Translator
from corpus import Corpus
from bunch import Bunch
import os.path

print('reading config file...')
conf = {}
f = open('config.cfg')
for line in f.readlines():
    k, v = line[:-1].split('=')
    conf[k] = v
f.close()

print('preparing english corpus...')
en_corpus_config = Bunch()
en_corpus_config.save_dir = 'cache' + os.path.sep + 'data'
en_corpus_config.data_addr = 'data' + os.path.sep + conf['english data filename']
en_corpus_config.name = conf['english data filename']
en_corpus_config.vocab_size = int(conf['english data vocab size'])
en_corpus_config.divs_size = [int(conf['english data division difference']) * i for i in
                              range(int(conf['english division number']))]
en_corpus = Corpus(en_corpus_config)
print('preparing farsi corpus...')
fa_corpus_config = Bunch()
fa_corpus_config.save_dir = 'cache' + os.path.sep + 'data'
fa_corpus_config.data_addr = 'data' + os.path.sep + conf['farsi data filename']
fa_corpus_config.name = conf['farsi data filename']
fa_corpus_config.vocab_size = int(conf['farsi data vocab size'])
fa_corpus_config.encoding = 'utf-8'
fa_corpus_config.divs_size = en_corpus.divs_size
fa_corpus = Corpus(fa_corpus_config, en_corpus.inds)
print('div_index:', en_corpus.div_index, fa_corpus.div_index)
print('indexes:', en_corpus.indexes, fa_corpus.indexes)
print('preparing translator...')
tr_config = Bunch()
tr_config.inp_vocab_size = en_corpus.vocab_size
tr_config.out_vocab_size = fa_corpus.vocab_size
tr_config.encoder_layers = int(conf['translator encoder layer number'])
tr_config.encoder_hidden_size = int(conf['translator encoder hidden size'])
tr_config.attention_hidden_size = int(conf['translator attention hidden size'])
tr_config.decoder_layers = int(conf['translator decoder layer number'])
tr_config.decoder_hidden_size = int(conf['translator decoder hidden size'])
tr_config.rate = float(conf['translator initial learning rate'])
tr_config.name = conf['translator name']
tr_config.save_dir = 'cache' + os.path.sep + 'model'
tr_config.residuals = False
translator = Translator(tr_config)
print('training started:')
EPOCH_NUM = 10000
BATCH_SIZE = 100
PRINT_PER = 100
SAVE_PER = 100

data_so_far = 0
data_size = int(sum([len(x) for x in en_corpus.datas]))
batch_index = 0

while data_so_far < EPOCH_NUM * data_size:
    try:
        x = en_corpus.next_batch(BATCH_SIZE)
        y = fa_corpus.next_batch(BATCH_SIZE)
        loss = translator.train_model(x, y)
        if batch_index % PRINT_PER == 0:
            assert en_corpus.indexes == fa_corpus.indexes
            assert en_corpus.div_index == fa_corpus.div_index
            print(batch_index, ': ', loss)
        if batch_index % SAVE_PER == SAVE_PER - 1:
            print('saving...')
            en_corpus.save_corpus_state()
            fa_corpus.save_corpus_state()
            translator.save_model()
            print('finished')
        batch_index += 1
    except KeyboardInterrupt:
        break
print('finished')