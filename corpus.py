import string
import pickle

import numpy as np

import os.path


class Corpus:
    def __init__(self, config):
        self.EOS_PUNCS = config.eos_puncs or ['.', '!', '?', '\n']
        self.PUNCS = config.puncs or list(string.punctuation) + [' ', '\n']
        self.SAVE_DIR = config.save_dir
        self.name = config.name
        self.data_addr = config.data_addr
        self.vocab_size = config.vocab_size

        self.words = None
        self.tokens = None
        self.data = None
        self.index = None

        self.load_data()

    def find_sentences(self, str):
        sentences = []
        tmp = ''
        for c in str:
            tmp = tmp + c
            if c in self.EOS_PUNCS:
                sentences.append(tmp)
                tmp = ''
        if len(tmp) > 1:
            sentences.append(tmp)
        return sentences

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

    def find_tokens(self, sentences, vocab_size):
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
        tokens = dict([(x[0], i) for i, x in enumerate(sorted(list(tokens.items()), key=lambda x: x[1])[0:vocab_size])])
        tokens['UNK'] = len(tokens) + 1
        return tokens, sents

    def load_data(self):
        path = self.SAVE_DIR + '\\' + self.name
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
        if os.path.isfile(path + 'tokens'):
            with open(path + 'tokens', 'rb') as file:
                tokens = pickle.load(file)
            with open(path + 'words', 'rb') as file:
                words = pickle.load(file)
            with open(path + 'data', 'rb') as file:
                data = pickle.load(file)
            with open(path + 'index', 'rb') as file:
                index = pickle.load(file)
            self.vocab_size = len(words)
        else:
            file = open(self.data_addr, 'r')
            sentences = []
            for line in file.readlines():
                sentences.extend(self.find_sentences(line))
            file.close()
            tokens, sentences = self.find_tokens(self.vocab_size, sentences)
            self.vocab_size = min(self.vocab_size, len(tokens))
            data = tokenize(sentences, tokens)
            words = [x[0] for x in sorted(list(tokens.items()), key=lambda x: x[1])]
            index = 0
            with open(path + 'tokens', 'wb+') as file:
                pickle.dump(tokens, file)
            with open(path + 'words', 'wb+') as file:
                pickle.dump(words, file)
            with open(path + 'data', 'wb+') as file:
                pickle.dump(data, file)
            with open(path + 'index', 'wb+') as file:
                pickle.dump(index, file)
        self.words = words
        self.tokens = tokens
        self.data = data
        self.index = index
        return self.vocab_size

    def next_batch(self, size, save=False):
        d_size = self.data.shape[0]
        print(d_size)
        assert size <= d_size
        end = (self.index + size - 1) % d_size
        if end <= self.index:
            res = self.data[self.index:-1] + self.data[0:end]
        else:
            res = self.data[self.index:self.index + size - 1]
        self.index = end
        if save:
            path = self.SAVE_DIR + '\\' + self.name
            with open(path + 'index', 'wb+') as file:
                pickle.dump(self.index, file)
        return res


def tokenize(sentences, vocab_size, tokens):  # it fucks the sentences (because of extendingg them)
    mm = max([len(s) for s in sentences])
    result = np.zeros([len(sentences), mm, vocab_size])
    for i, s in enumerate(sentences):
        for j in range(0, mm):
            w = s[j] if j < len(s) else 'EOS'
            if w in tokens:
                result[i, j, tokens[w]] = 1
            else:
                result[i, j, tokens['UNK']] = 1
    return result