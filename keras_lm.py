from keras.models import *
from keras.layers import *
import string
import numpy as np
import pickle
import os.path


def find_sentences(str):
    sentences = []
    tmp = ''
    for c in str:
        tmp = tmp + c
        if c in EOS_PUNCS:
            sentences.append(tmp)
            tmp = ''
    if len(tmp) > 1:
        sentences.append(tmp)
    return sentences


def find_words(str):
    words = []
    tmp = ''
    for c in str:
        if c in PUNCS:
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


def find_tokens(sentences):
    tokens = {}
    sents = []
    for s in sentences:
        words = find_words(s)
        words.insert(0, 'SOS')
        words.append('EOS')
        sents.append(words)
        for w in words:
            if w in tokens:
                tokens[w] += 1
            else:
                tokens[w] = 1
    tokens = dict([(x[0], i) for i, x in enumerate(sorted(list(tokens.items()), key=lambda x: x[1])[0:VOCAB_SIZE])])
    tokens['UNK'] = len(tokens) + 1
    return tokens, sents


def tokenize(sentences, tokens):
    # return [([tokens[w] if w in tokens else tokens['UNK'] for w in s]) for s in sentences]
    mm = max([len(s) for s in sentences])
    result = np.zeros([len(sentences), mm, VOCAB_SIZE])
    for i, s in enumerate(sentences):
        for j in range(0, mm):
            w = s[j] if j < len(s) else 'EOS'
            if w in tokens:
                result[i, j, tokens[w]] = 1
            else:
                result[i, j, tokens['UNK']] = 1
    return result


HIDDEN_SIZE = 10
BATCH_SIZE = 128
DATA = 'data\\en.y'
EOS_PUNCS = ['.', '!', '?', '\n']
PUNCS = list(string.punctuation)
PUNCS.extend([' ', '\n'])
CACHE = 'cache' + '\\' + DATA + '\\'
if not os.path.exists(CACHE):
    os.makedirs(CACHE)

if os.path.isfile(CACHE + 'tokens'):
    with open(CACHE + 'tokens', 'rb') as file:
        tokens = pickle.load(file)
    with open(CACHE + 'words', 'rb') as file:
        words = pickle.load(file)
    with open(CACHE + 'X', 'rb') as file:
        X = pickle.load(file)
    with open(CACHE + 'Y', 'rb') as file:
        Y = pickle.load(file)
    VOCAB_SIZE = len(words)
else:
    VOCAB_SIZE = 10000
    file = open(DATA, 'r')
    sentences = []
    for line in file.readlines():
        sentences.extend(find_sentences(line))
    file.close()
    tokens, sentences = find_tokens(sentences)
    VOCAB_SIZE = min(VOCAB_SIZE, len(tokens))
    train_data = tokenize(sentences, tokens)
    words = [x[0] for x in sorted(list(tokens.items()), key=lambda x: x[1])]
    X = np.array([s[0:-1] for s in train_data])
    Y = np.array([s[1:] for s in train_data])
    with open(CACHE + 'tokens', 'wb+') as file:
        pickle.dump(tokens, file)
    with open(CACHE + 'words', 'wb+') as file:
        pickle.dump(words, file)
    with open(CACHE + 'X', 'wb+') as file:
        pickle.dump(X, file)
    with open(CACHE + 'Y', 'wb+') as file:
        pickle.dump(Y, file)

if os.path.isfile(CACHE + 'model'):
    model = load_model(CACHE + 'model')
else:
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE, input_shape=(None, VOCAB_SIZE), return_sequences=True))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    for i in range(20):
        model.fit(X, Y, epochs=1, batch_size=BATCH_SIZE)
    model.save(CACHE + 'model')
# for i in range(1, 200):
#         model.fit(X, Y, epochs=1, batch_size=BATCH_SIZE)
# model.save(CACHE + 'model')
preds = [['SOS']]
K = 20
MMAX = 10
for i in range(1, MMAX+1):
    u = np.zeros([len(preds), i, VOCAB_SIZE])
    for j, p in enumerate(preds):
        for k in range(0,i):
            u[j, k, tokens[p[k]]] = 1
    ps = np.resize(model.predict(u)[:, i - 1, :], VOCAB_SIZE * len(preds)).argsort()[-K:]
    pred2 = []
    for p in ps:
        tmp = preds[int(p / VOCAB_SIZE)][:]
        tmp.append(words[p % VOCAB_SIZE])
        pred2.append(tmp)
    preds = pred2

for p in preds:
    print(p)
