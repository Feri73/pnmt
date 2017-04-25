from translator import Translator
from corpus import Corpus
from bunch import Bunch


en_corpus_config = Bunch()
en_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
en_corpus_config.data_addr = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\en'
en_corpus_config.name = 'en1_corpus'
en_corpus_config.vocab_size = 500
en_corpus_config.divs_size = [3 * i for i in range(10)]
en_corpus = Corpus(en_corpus_config)
fa_corpus_config = Bunch()
fa_corpus_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\data'
fa_corpus_config.data_addr = 'F:\\Faraz\\University\\Thesis\\pnmt\\data\\fa'
fa_corpus_config.name = 'fa1_corpus'
fa_corpus_config.vocab_size = 500
fa_corpus_config.encoding = 'utf-8'
fa_corpus_config.divs_size = en_corpus.divs_size
fa_corpus = Corpus(fa_corpus_config, en_corpus.inds)

print(en_corpus.div_index, fa_corpus.div_index)
print(en_corpus.indexes, fa_corpus.indexes)

tr_config = Bunch()
tr_config.inp_vocab_size = en_corpus.vocab_size
tr_config.out_vocab_size = fa_corpus.vocab_size
tr_config.encoder_layers = 2
tr_config.encoder_hidden_size = 5
tr_config.attention_hidden_size = 5
tr_config.decoder_layers = 2
tr_config.decoder_hidden_size = 5
tr_config.rate = 5
tr_config.name = 'en_fa_nmt'
tr_config.save_dir = 'F:\\Faraz\\University\\Thesis\\pnmt\\cache\\model'
translator = Translator(tr_config)

EPOCH_SIZE = 10000
BATCH_SIZE = 100
PRINT_PER = 100
SAVE_PER = 100
for i in range(EPOCH_SIZE * int(sum([len(x) for x in en_corpus.datas]) / BATCH_SIZE)):  # use batch==>also in model
    try:
        x = en_corpus.next_batch(BATCH_SIZE)
        y = fa_corpus.next_batch(BATCH_SIZE)

        # x1 = np.zeros([x.shape[0], en_corpus.vocab_size, x.shape[1]], dtype='float32')
        # for k in range(x.shape[0]):
        # for j in range(x.shape[1]):
        #         x1[k, x[k, j], j] = 1
        #
        # x=x1

        # y1 = np.zeros([y.shape[0], fa_corpus.vocab_size, y.shape[1]], dtype='float32')
        # for k in range(y.shape[0]):
        #     for j in range(y.shape[1]):
        #         y1[k, y[k, j], j] = 1
        #
        # y=y1

        lost = translator.train_model(x, y)
        if i % PRINT_PER == 0:
            assert en_corpus.indexes == fa_corpus.indexes
            assert en_corpus.div_index == fa_corpus.div_index
            print(i, ': ', lost)
            # print(en_corpus.to_words(x[:, :, 0]))
            # print(fa_corpus.to_words(y[:, :, 0]))
            # print(fa_corpus.to_words(translator.helped_translate(x[:, :, 0:1], y[:, :, 0:1])))
        if i % SAVE_PER == SAVE_PER - 1:
            print('saving...')
            en_corpus.save_corpus_state()
            fa_corpus.save_corpus_state()
            translator.save_model()
            print('finished')
    except KeyboardInterrupt:
        break
print('saving...')
en_corpus.save_corpus_state()
fa_corpus.save_corpus_state()
translator.save_model()
print('finished')