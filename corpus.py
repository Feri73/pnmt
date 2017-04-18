import string
import pickle

import numpy as np

import codecs

import os.path


class Corpus:
    def __init__(self, config):
        # self.EOS_PUNCS = config.eos_puncs or ['.', '!', '?', '\n']
        self.PUNCS = config.puncs or list(string.punctuation) + [' ', '\n']
        self.SAVE_DIR = config.save_dir
        self.name = config.name
        self.data_addr = config.data_addr
        self.vocab_size = config.vocab_size
        self.encoding = config.encoding or 'ascii'

        self.words = None
        self.tokens = None
        self.data = None
        self.index = None

        self.load_data()

    # def find_sentences(self, str):
    #     sentences = []
    #     tmp = ''
    #     for c in str:
    #         tmp = tmp + c
    #         if c in self.EOS_PUNCS:
    #             sentences.append(tmp)
    #             tmp = ''
    #     if len(tmp) > 1:
    #         sentences.append(tmp)
    #     return sentences

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

    def load_data(self):
        path = self.SAVE_DIR + '\\' + self.name
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
        if os.path.isfile(path + 'tokens'):
            with open(path + 'tokens', 'rb') as file:
                self.tokens = pickle.load(file)
            with open(path + 'words', 'rb') as file:
                words = pickle.load(file)
            with open(path + 'data', 'rb') as file:
                data = pickle.load(file)
            with open(path + 'index', 'rb') as file:
                index = pickle.load(file)
            self.vocab_size = len(words)
        else:
            file = codecs.open(self.data_addr, 'r', self.encoding)
            sentences = []
            for line in file.readlines():
                # sentences.extend(self.find_sentences(line[0:-1]))
                sentences.append(line[0:-1])
            file.close()
            self.tokens, sentences = self.find_tokens(sentences)
            self.vocab_size = min(self.vocab_size, len(self.tokens))
            data = self.tokenize(sentences)
            words = [x[0] for x in sorted(list(self.tokens.items()), key=lambda x: x[1])]
            index = 0
            with open(path + 'tokens', 'wb+') as file:
                pickle.dump(self.tokens, file)
            with open(path + 'words', 'wb+') as file:
                pickle.dump(words, file)
            with open(path + 'data', 'wb+') as file:
                pickle.dump(data, file)
            with open(path + 'index', 'wb+') as file:
                pickle.dump(index, file)
        self.words = words
        self.data = data
        self.index = index
        return self.vocab_size

    def next_batch(self, size, save=False):  # poor performance
        d_size = len(self.data)
        assert size <= d_size
        end = (self.index + size) % d_size
        # i=self.index
        # size=self.data
        # result=[]
        # while(i!=end):


        # extend for size


        if end <= self.index:
            tmp1 = self.data[self.index:]
            tmp2 = self.data[0:end]
            # if tmp1.shape[0] == 0:
            #     res = tmp2
            # elif tmp2.shape[0] == 0:
            #     res = tmp1
            # else:
            res = tmp1 + tmp2
        else:
            res = self.data[self.index:end]
        self.index = end
        if save:
            self.save_corpus_state()
        return np.array(res, dtype='float32')

    def tokenize(self, sentences):
        result = []
        for i, s in enumerate(sentences):
            mm = len(s)
            tmp = np.zeros([mm, self.vocab_size], dtype='float32')
            for j in range(0, mm):
                w = s[j]  # if j < len(s) else 'EOS'
                if w in self.tokens:
                    tmp[j, self.tokens[w]] = 1
                else:
                    tmp[j, self.tokens['UNK']] = 1
            result.append(tmp)
        return result

    def save_corpus_state(self):
        path = self.SAVE_DIR + '\\' + self.name
        with open(path + 'index', 'wb+') as file:
            pickle.dump(self.index, file)

    def to_words(self, token_list):
        numbers = [np.argmax(x) for x in token_list]
        return ' '.join([self.words[i] for i in numbers])