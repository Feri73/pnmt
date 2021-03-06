import string
import pickle

import numpy as np

import codecs
import os.path


class Corpus:
    def __init__(self, config, inds=None):
        self.PUNCS = config.puncs or list(string.punctuation) + [' ', '\n']
        self.SAVE_DIR = config.save_dir
        self.name = config.name
        self.data_addr = config.data_addr
        self.vocab_size = config.vocab_size
        self.encoding = config.encoding or 'ascii'
        if not inds:
            self.divs_size = config.divs_size

        self.words = []
        self.tokens = []
        self.datas = []
        self.indexes = []
        self.sizes = []
        self.inds = []
        self.div_index = 0

        self.load_data(inds)

    def load_data(self, inds):
        path = self.SAVE_DIR + os.path.sep + self.name
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
        if os.path.isfile(path + 'tokens'):
            self._load_property('tokens')
            self._load_property('words')
            self._load_property('datas')
            self._load_property('indexes')
            self._load_property('sizes')
            self._load_property('div_index')
            self._load_property('inds')
            self._load_property('divs_size')
            self.vocab_size = len(self.words)
        else:
            file = codecs.open(self.data_addr, 'r', self.encoding)
            sentences = []
            for line in file.readlines():
                sentences.append(line[0:-1])
            file.close()
            self.tokens, sentences = self.find_tokens(sentences)
            self.vocab_size = min(self.vocab_size, len(self.tokens))
            self.datas, self.sizes, self.inds = self.tokenize(sentences, inds)
            self.words = [x[0] for x in sorted(list(self.tokens.items()), key=lambda x: x[1])]
            self.indexes = [0 for _ in self.divs_size]
            self.div_index = 0
            self._save_property('tokens')
            self._save_property('words')
            self._save_property('datas')
            self._save_property('indexes')
            self._save_property('sizes')
            self._save_property('div_index')
            self._save_property('inds')
            self._save_property('divs_size')
        return self.vocab_size

    def find_words(self, str):
        words = []
        tmp = ''
        for c in str:
            if c in self.PUNCS:
                if not (c == ' '):
                    words.append(c)
                if len(tmp) > 0:
                    words.append(tmp)
                tmp = ''
            else:
                tmp = tmp + c
        if len(tmp) > 0:
            words.append(tmp)
        return words

    def find_tokens(self, sentences):
        tokens = {}
        sents = []
        for s in sentences:
            words = self.find_words(s)
            words.insert(0, 'SOS')
            words.append('EOS')
            sents.append(words)
            for w in words:
                if w in tokens:
                    tokens[w] += 1
                else:
                    tokens[w] = 1
        tokens = dict(
            [(x[0], i) for i, x in enumerate(sorted(list(tokens.items()), key=lambda x: -x[1])[0:self.vocab_size - 1])])
        tokens['UNK'] = len(tokens)
        return tokens, sents

    def tokenize(self, sentences, inds):
        sizes = [len(s) for s in sentences]
        mmax = max(sizes)
        if inds:
            self.divs_size = [max([len(sentences[i]) for i in iss] or [0]) for iss in inds]
        else:
            self.divs_size = self.divs_size + [mmax] if mmax > self.divs_size[-1] else self.divs_size
            inds = [[] for _ in range(len(self.divs_size))]
            for i, s in enumerate(sentences):
                for j, n in enumerate(self.divs_size):
                    if len(s) <= n:
                        inds[j].append(i)
                        break
        result = [np.zeros([n, len(inds[j])], dtype='int32') for j, n in enumerate(self.divs_size)]
        for j, n in enumerate(self.divs_size):
            for i, si in enumerate(inds[j]):
                for k in range(n):
                    w = sentences[si][k] if k < len(sentences[si]) else 'EOS'
                    w_index = self.tokens[w] if w in self.tokens else self.tokens['UNK']
                    result[j][k, i] = w_index
        return result, sizes, inds

    def save_corpus_state(self):
        self._save_property('indexes')
        self._save_property('div_index')

    def to_words(self, token_list):
        numbers = [np.argmax(x) for x in token_list]
        return ' '.join([self.words[i] for i in numbers])

    def next_batch(self, size, save=False):  # poor performance
        while len(self.datas[self.div_index]) == 0:
            self.div_index = (self.div_index + 1) % len(self.divs_size)
        d_size = self.datas[self.div_index].shape[1]
        start = self.indexes[self.div_index]
        end = min(start + min(size, d_size), d_size)  # I don't do (start+size)%d_size
        res = self.datas[self.div_index][:, start:end]
        self.indexes[self.div_index] = end % d_size
        self.div_index = (self.div_index + 1) % len(self.divs_size)
        if save:
            self.save_corpus_state()
        return res

    def _save_property(self, p_name):
        with open(self.SAVE_DIR + os.path.sep + self.name + p_name, 'wb+') as file:
            pickle.dump(getattr(self, p_name), file)

    def _load_property(self, p_name):
        with open(self.SAVE_DIR + os.path.sep + self.name + p_name, 'rb') as file:
            setattr(self, p_name, pickle.load(file))