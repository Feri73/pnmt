import tensorflow as tf
import numpy as np


def np_func(fn, inp):
    return fn(inp, dtype='float32')


INP_VOCAB_SIZE = 10000
OUT_VOCAB_SIZE = 20000
ENCODER_LAYERS = 8
ENCODER_HIDDEN_SIZE = 1000

ATTENTION_HIDDEN_SIZE = 1000

DECODER_LAYERS = 8
DECODER_HIDDEN_SIZE = 1000

enc_Wf = []
enc_bf = []
enc_Wi = []
enc_bi = []
enc_Wc = []
enc_bc = []
enc_Wo = []
enc_bo = []

att_W = []
att_b = []

dec_Wf = []
dec_bf = []
dec_Wi = []
dec_bi = []
dec_Wc = []
dec_bc = []
dec_Wo = []
dec_bo = []


# def dot(a, b):
# return tf.reduce_sum(a * b, -1, keep_dims=True)


def new_var(size):
    return tf.Variable(tf.zeros(size))


def add_to_enc_weights(inp_size, hid_size):
    enc_Wf.append(new_var([hid_size, hid_size + inp_size]))
    enc_Wi.append(new_var([hid_size, hid_size + inp_size]))
    enc_Wc.append(new_var([hid_size, hid_size + inp_size]))
    enc_Wo.append(new_var([hid_size, hid_size + inp_size]))
    enc_bf.append(new_var([hid_size, 1]))
    enc_bi.append(new_var([hid_size, 1]))
    enc_bc.append(new_var([hid_size, 1]))
    enc_bo.append(new_var([hid_size, 1]))


def NN(inp, W, b):
    return tf.matmul(W, inp) + b
    # return tf.matmul(W, tf.expand_dims(inp, -1)) + b
    # return dot(W, inp) + b
    # return tf.reduce_sum(W * inp, 1, keep_dims=True) + b


def sigmoidNN(inp, W, b):
    return tf.sigmoid(NN(inp, W, b))


def tanhNN(inp, W, b):
    return tf.tanh(NN(inp, W, b))


def enc_elem(C_prev, h_prev, x, enc_layer_index):
    Wf = enc_Wf[enc_layer_index]
    bf = enc_bf[enc_layer_index]
    Wi = enc_Wi[enc_layer_index]
    bi = enc_bi[enc_layer_index]
    Wc = enc_Wc[enc_layer_index]
    bc = enc_bc[enc_layer_index]
    Wo = enc_Wo[enc_layer_index]
    bo = enc_bo[enc_layer_index]
    tmp = tf.concat([x, h_prev], 0)  # axis is 1??
    f = sigmoidNN(tmp, Wf, bf)
    i = sigmoidNN(tmp, Wi, bi)
    C_h = tanhNN(tmp, Wc, bc)
    C = f * C_prev + i * C_h
    o = sigmoidNN(tmp, Wo, bo)
    h = o * tf.tanh(C)
    return C, h


def add_encoder_layer(inp, hid_size):
    add_to_enc_weights(int(inp.shape[1]), hid_size)
    initial = 2 * (np_func(np.zeros, [hid_size, 1]),)  # why not tf.zeros
    return tf.scan(lambda a, x: enc_elem(a[0], a[1], x, len(enc_Wf) - 1), inp, initializer=initial)[1]


inp_seq = tf.placeholder(tf.float32, [None, INP_VOCAB_SIZE, 1])
enc_layers = []

lstmf = add_encoder_layer(inp_seq, int(ENCODER_HIDDEN_SIZE / 2))
lstmb = add_encoder_layer(tf.reverse(inp_seq, [-1]), int(ENCODER_HIDDEN_SIZE / 2))  # HIDDEN_SIZE/2 is OK????
enc_layers.append(tf.concat([lstmf, lstmb], 1))

for i in range(ENCODER_LAYERS - 1):
    inp = enc_layers[-1]
    if i > 0:
        inp = inp + enc_layers[-2]
    enc_layers.append(add_encoder_layer(inp, ENCODER_HIDDEN_SIZE))


def dec_elem(C_prev, h_prev, c, x, dec_layer_index):
    Wf = dec_Wf[dec_layer_index]
    bf = dec_bf[dec_layer_index]
    Wi = dec_Wi[dec_layer_index]
    bi = dec_bi[dec_layer_index]
    Wc = dec_Wc[dec_layer_index]
    bc = dec_bc[dec_layer_index]
    Wo = dec_Wo[dec_layer_index]
    bo = dec_bo[dec_layer_index]
    tmp = tf.concat([x, h_prev, c], 0)  # axis is 1??
    f = sigmoidNN(tmp, Wf, bf)
    i = sigmoidNN(tmp, Wi, bi)
    C_h = tanhNN(tmp, Wc, bc)
    C = f * C_prev + i * C_h
    o = sigmoidNN(tmp, Wo, bo)
    h = o * tf.tanh(C)
    return C, h  # , c


def dec_first_layer_elem(C_prev, h_prev, c, x, dec_layer_index):
    C, h = dec_elem(C_prev, h_prev, c, x, dec_layer_index)
    c = get_context(h)
    return C, h, c


def add_decoder_layer(inp, contexts, hid_size):
    add_to_dec_weights(int(inp.shape[1]), hid_size)
    initial = 2 * (tf.zeros([hid_size, 1]),)
    return tf.scan(lambda a, x: dec_elem(a[0], a[1], x[:, 1], x[:, 0], len(dec_Wf) - 1),
                   tf.expand_dims(tf.concat([inp, contexts], -1), -1), initializer=initial)[1]


def add_to_dec_weights(inp_size, hid_size):
    dec_Wf.append(new_var([hid_size, 2 * hid_size + inp_size]))
    dec_Wi.append(new_var([hid_size, 2 * hid_size + inp_size]))
    dec_Wc.append(new_var([hid_size, 2 * hid_size + inp_size]))
    dec_Wo.append(new_var([hid_size, 2 * hid_size + inp_size]))
    dec_bf.append(new_var([hid_size, 1]))
    dec_bi.append(new_var([hid_size, 1]))
    dec_bc.append(new_var([hid_size, 1]))
    dec_bo.append(new_var([hid_size, 1]))


# def repeat(row, shape):
# return tf.scan(lambda a, x: a, shape, initializer=row)


def attention_function(st_y_prev, st_x):
    W0 = att_W[0]
    W1 = att_W[1]
    b0 = att_b[0]
    b1 = att_b[1]
    # return NN(NN(tf.concat([repeat(y_prev, x), x], -2), W0, b0), W1, b1)  # concat dim, how to use y and x here??
    return NN(NN(tf.concat([st_y_prev, st_x], 0), W0, b0), W1, b1)  # concat dim, how to use y and x here??,


def get_context(st_prev):
    res = tf.scan(lambda a, x: attention_function(st_prev, x)[0, 0], enc_layers[-1], initializer=float(0))
    # res = attention_function(st_prev, enc_layers[-1])
    return tf.matmul(tf.transpose(enc_layers[-1][:, :, 0]), tf.expand_dims(tf.nn.softmax(res), 1))  # better


def add_attention_weights(inp_size, hid_size):
    att_W.append(new_var([hid_size, inp_size]))
    att_W.append(new_var([1, hid_size]))
    att_b.append(new_var([hid_size, 1]))
    att_b.append(new_var([1, 1]))


out_seq = tf.placeholder(tf.float32, [None, OUT_VOCAB_SIZE, 1])
add_attention_weights(DECODER_HIDDEN_SIZE + ENCODER_HIDDEN_SIZE, ATTENTION_HIDDEN_SIZE)
dec_layers = []

add_to_dec_weights(int(out_seq.shape[1]), DECODER_HIDDEN_SIZE)
initial = (tf.zeros([DECODER_HIDDEN_SIZE, 1]), tf.zeros([DECODER_HIDDEN_SIZE, 1]), tf.zeros([ENCODER_HIDDEN_SIZE, 1]))
_, h, contexts = tf.scan(lambda a, x: dec_first_layer_elem(a[0], a[1], a[2], x, len(dec_Wf) - 1), out_seq,
                         initializer=initial)
dec_layers.append(h)
for i in range(DECODER_LAYERS - 1):
    inp = dec_layers[-1]
    if i > 0:
        inp = inp + dec_layers[-2]
    hid_size = DECODER_HIDDEN_SIZE if i < DECODER_LAYERS - 1 else OUT_VOCAB_SIZE
    dec_layers.append(add_decoder_layer(inp, contexts, hid_size))

tf.scan(lambda a, x: tf.nn.softmax(x), dec_layers[-1])

sess = tf.Session()
file_writer = tf.summary.FileWriter('mammad', sess.graph)
sess.close()















# contexts.append(get_context(tf.zeros([DECODER_HIDDEN_SIZE, 1])))
# dec_layers_C_prev = []
# dec_layers_h_prev = []
# dec_layers_inp = []
# dec_layers_C_prev.append(DECODER_LAYERS * [tf.zeros([DECODER_HIDDEN_SIZE, 1])])
# dec_layers_h_prev.append(DECODER_LAYERS * [tf.zeros([DECODER_HIDDEN_SIZE, 1])])
# dec_layers_inp.append([tf.zeros([DECODER_HIDDEN_SIZE, 1])])
#
# Index = 0
# for i in range(DECODER_LAYERS):
# inp = dec_layers_inp[Index][-1]
# if i > 0:
# inp = inp + dec_layers_inp[Index][-2]
# add_to_dec_weights(int(inp.shape[0]), DECODER_HIDDEN_SIZE)
# C, h = dec_elem(dec_layers_C_prev[Index][-1], dec_layers_h_prev[Index][-1], contexts[-1], inp, i)
# dec_layers_C_prev.append([C])
# dec_layers_h_prev.append([h])
# dec_layers_inp[Index].append(h)
# Y = tf.nn.softmax(dec_layers_inp[-1])
# dec_layers_inp.append([])

















#
#
# def dec_elem(C_prev, h_prev, y_prev, context, dec_layer_index):
# Wf = dec_Wf[dec_layer_index]
# bf = dec_bf[dec_layer_index]
# Wi = dec_Wi[dec_layer_index]
# bi = dec_bi[dec_layer_index]
# Wc = dec_Wc[dec_layer_index]
# bc = dec_bc[dec_layer_index]
# Wo = dec_Wo[dec_layer_index]
# bo = dec_bo[dec_layer_index]
# tmp = tf.concat([y_prev, h_prev, context], 0)  # axis is 1??
# f = sigmoidNN(tmp, Wf, bf)
# i = sigmoidNN(tmp, Wi, bi)
# C_h = tanhNN(tmp, Wc, bc)
# C = f * tf.reshape(C_prev, [int(C_prev.shape[0]), 1]) + i * C_h
# o = sigmoidNN(tmp, Wo, bo)
# h = o * tf.tanh(C)
# return tf.transpose(tf.concat([C, h], 1))
#
#
#
#
# def dec_first_layer_elem(C_prev, h_prev, context, y_prev, dec_layer_index):
# tmp = dec_elem(C_prev, h_prev, y_prev, context, dec_layer_index)
# c = tf.transpose(get_context(tmp[1]))
# return tf.concat([tmp, c], 0)
#
#
# def add_decoder_layer(inp, context, hid_size):
# add_to_dec_weights(int(inp.shape[1]), hid_size)
# initial = np_func(np.zeros, [2, hid_size])
# return tf.scan(lambda a, x: dec_elem(a[0], a[1], x[0], x[1], len(dec_Wf) - 1), tf.concat([inp, context], -1),
# initializer=initial)[:, 1, :]
#
#
#
# dec_layers = []
#
# add_to_dec_weights(int(inp.shape[1]), DECODER_HIDDEN_SIZE)
# initial = np_func(np.zeros, [3, DECODER_HIDDEN_SIZE])
# tmp = tf.scan(lambda a, x: dec_first_layer_elem(a[0], a[1], x, a[2], len(dec_Wf) - 1), enc_layers[-1],
# initializer=initial)
# context = tmp[:, 2, :]
# dec_layers.append(tmp[:, 1, :])
#
# for i in range(DECODER_LAYERS - 1):
# inp = dec_layers[-1]
# if i > 0:
#         inp = inp + dec_layers[-2]
#     dec_layers.append(add_decoder_layer(inp, context, DECODER_HIDDEN_SIZE))