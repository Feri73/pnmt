import tensorflow as tf
import numpy as np

import os.path


class Translator:
    def __init__(self, config):
        self.INP_VOCAB_SIZE = config.inp_vocab_size
        self.OUT_VOCAB_SIZE = config.out_vocab_size
        self.ENCODER_LAYERS = config.encoder_layers
        self.ENCODER_HIDDEN_SIZE = config.encoder_hidden_size
        self.ATTENTION_HIDDEN_SIZE = config.attention_hidden_size
        self.DECODER_LAYERS = config.decoder_layers
        self.DECODER_HIDDEN_SIZE = config.decoder_hidden_size
        self.RATE = config.rate
        self.name = config.name
        self.SAVE_DIR = config.save_dir

        self.enc_Wf = []
        self.enc_bf = []
        self.enc_Wi = []
        self.enc_bi = []
        self.enc_Wc = []
        self.enc_bc = []
        self.enc_Wo = []
        self.enc_bo = []

        self.att_W = []
        self.att_b = []

        self.dec_Wf = []
        self.dec_bf = []
        self.dec_Wi = []
        self.dec_bi = []
        self.dec_Wc = []
        self.dec_bc = []
        self.dec_Wo = []
        self.dec_bo = []

        self.trans = None
        self.cross_entropy = None
        self.train_step = None
        self.init_op = None
        self.sess = None

        self.inp_seq = None
        self.out_seq = None

        self._load_model()

    def __del__(self):
        self.sess.close()

    def add_to_enc_weights(self, inp_size, hid_size):
        self.enc_Wf.append(new_var([hid_size, hid_size + inp_size]))
        self.enc_Wi.append(new_var([hid_size, hid_size + inp_size]))
        self.enc_Wc.append(new_var([hid_size, hid_size + inp_size]))
        self.enc_Wo.append(new_var([hid_size, hid_size + inp_size]))
        self.enc_bf.append(new_var([hid_size, 1]))
        self.enc_bi.append(new_var([hid_size, 1]))
        self.enc_bc.append(new_var([hid_size, 1]))
        self.enc_bo.append(new_var([hid_size, 1]))

    def enc_elem(self, C_prev, h_prev, x, enc_layer_index):
        Wf = self.enc_Wf[enc_layer_index]
        bf = self.enc_bf[enc_layer_index]
        Wi = self.enc_Wi[enc_layer_index]
        bi = self.enc_bi[enc_layer_index]
        Wc = self.enc_Wc[enc_layer_index]
        bc = self.enc_bc[enc_layer_index]
        Wo = self.enc_Wo[enc_layer_index]
        bo = self.enc_bo[enc_layer_index]
        tmp = tf.concat([x, h_prev], 0)  # axis is 1??
        f = sigmoidNN(tmp, Wf, bf)
        i = sigmoidNN(tmp, Wi, bi)
        C_h = tanhNN(tmp, Wc, bc)
        C = f * C_prev + i * C_h
        o = sigmoidNN(tmp, Wo, bo)
        h = o * tf.tanh(C)
        return C, h

    def dec_elem(self, C_prev, h_prev, c, x, dec_layer_index):
        Wf = self.dec_Wf[dec_layer_index]
        bf = self.dec_bf[dec_layer_index]
        Wi = self.dec_Wi[dec_layer_index]
        bi = self.dec_bi[dec_layer_index]
        Wc = self.dec_Wc[dec_layer_index]
        bc = self.dec_bc[dec_layer_index]
        Wo = self.dec_Wo[dec_layer_index]
        bo = self.dec_bo[dec_layer_index]
        tmp = tf.concat([x, h_prev, c], 0)  # axis is 1??
        f = sigmoidNN(tmp, Wf, bf)
        i = sigmoidNN(tmp, Wi, bi)
        C_h = tanhNN(tmp, Wc, bc)
        C = f * C_prev + i * C_h
        o = sigmoidNN(tmp, Wo, bo)
        h = o * tf.tanh(C)
        return C, h  # , c

    def add_to_dec_weights(self, inp_size, hid_size, last_hid_size=-1):
        if last_hid_size == -1:
            last_hid_size = hid_size
        tmp = hid_size + last_hid_size
        self.dec_Wf.append(new_var([hid_size, tmp + inp_size]))
        self.dec_Wi.append(new_var([hid_size, tmp + inp_size]))
        self.dec_Wc.append(new_var([hid_size, tmp + inp_size]))
        self.dec_Wo.append(new_var([hid_size, tmp + inp_size]))
        self.dec_bf.append(new_var([hid_size, 1]))
        self.dec_bi.append(new_var([hid_size, 1]))
        self.dec_bc.append(new_var([hid_size, 1]))
        self.dec_bo.append(new_var([hid_size, 1]))

    def add_encoder_layer(self, inp, hid_size):
        self.add_to_enc_weights(int(inp.shape[1]), hid_size)
        initial = 2 * (np_func(np.zeros, [hid_size, 1]),)  # why not tf.zeros
        return tf.scan(lambda a, x: self.enc_elem(a[0], a[1], x, len(self.enc_Wf) - 1), inp, initializer=initial)[1]

    def add_decoder_layer(self, inp, contexts, hid_size, last_hid_size=-1):
        self.add_to_dec_weights(int(inp.shape[1]), hid_size, last_hid_size)
        initial = 2 * (tf.zeros([hid_size, 1]),)
        return tf.scan(lambda a, x: self.dec_elem(a[0], a[1], x[:, 1], x[:, 0], len(self.dec_Wf) - 1),
                       tf.expand_dims(tf.concat([inp, contexts], -1), -1), initializer=initial)[1]

    def attention_function(self, st_y_prev, st_x):
        W0 = self.att_W[0]
        W1 = self.att_W[1]
        b0 = self.att_b[0]
        b1 = self.att_b[1]
        return NN(NN(tf.concat([st_y_prev, st_x], 0), W0, b0), W1, b1)  # concat dim, how to use y and x here??,

    def add_attention_weights(self, inp_size, hid_size):
        self.att_W.append(new_var([hid_size, inp_size]))
        self.att_W.append(new_var([1, hid_size]))
        self.att_b.append(new_var([hid_size, 1]))
        self.att_b.append(new_var([1, 1]))

    def create_graph(self):
        def dec_first_layer_elem(C_prev, h_prev, c, x, dec_layer_index):
            C, h = self.dec_elem(C_prev, h_prev, c, x, dec_layer_index)
            c = get_context(h)
            return C, h, c

        def get_context(st_prev):
            res = tf.scan(lambda a, x: self.attention_function(st_prev, x)[0, 0], enc_layers[-1], initializer=float(0))
            return tf.matmul(tf.transpose(enc_layers[-1][:, :, 0]), tf.expand_dims(tf.nn.softmax(res), 1))  # better

        self.inp_seq = tf.placeholder(tf.float32, [None, self.INP_VOCAB_SIZE, 1])
        self.out_seq = tf.placeholder(tf.float32, [None, self.OUT_VOCAB_SIZE, 1])

        enc_layers = []

        lstmf = self.add_encoder_layer(self.inp_seq, int(self.ENCODER_HIDDEN_SIZE / 2))
        lstmb = self.add_encoder_layer(tf.reverse(self.inp_seq, [-1]),
                                       int(self.ENCODER_HIDDEN_SIZE / 2))  # HIDDEN_SIZE/2 is OK????
        enc_layers.append(tf.concat([lstmf, lstmb], 1))

        for i in range(self.ENCODER_LAYERS - 1):
            inp = enc_layers[-1]
            if i > 0:
                inp = inp + enc_layers[-2]
            enc_layers.append(self.add_encoder_layer(inp, self.ENCODER_HIDDEN_SIZE))

        self.add_attention_weights(self.DECODER_HIDDEN_SIZE + self.ENCODER_HIDDEN_SIZE, self.ATTENTION_HIDDEN_SIZE)
        dec_layers = []

        self.add_to_dec_weights(int(self.out_seq.shape[1]), self.DECODER_HIDDEN_SIZE)
        initial = (tf.zeros([self.DECODER_HIDDEN_SIZE, 1]), tf.zeros([self.DECODER_HIDDEN_SIZE, 1]),
                   tf.zeros([self.ENCODER_HIDDEN_SIZE, 1]))
        _, h, contexts = tf.scan(lambda a, x: dec_first_layer_elem(a[0], a[1], a[2], x, len(self.dec_Wf) - 1),
                                 self.out_seq,
                                 initializer=initial)
        dec_layers.append(h)
        for i in range(self.DECODER_LAYERS - 1):
            inp = dec_layers[-1]
            if i > 0:
                inp = inp + dec_layers[-2]
            if i < self.DECODER_LAYERS - 2:
                last_hid_size = -1
                hid_size = self.DECODER_HIDDEN_SIZE
            else:
                last_hid_size = self.DECODER_HIDDEN_SIZE
                hid_size = self.OUT_VOCAB_SIZE
            dec_layers.append(self.add_decoder_layer(inp, contexts, hid_size, last_hid_size))

        self.trans = tf.scan(lambda a, x: tf.nn.softmax(x), dec_layers[-1])

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.out_seq * tf.log(self.trans), reduction_indices=[
            1]))  # use built in tensorflow methods, use indexing instead of multiplying
        self.train_step = tf.train.GradientDescentOptimizer(self.RATE).minimize(self.cross_entropy)
        self.init_op = tf.global_variables_initializer()

        self.sess = tf.Session()

    def train_model(self, x, y):  # use batches
        self.sess.run(self.train_step, feed_dict={self.inp_seq: x, self.out_seq: y})
        return self.sess.run(self.cross_entropy, feed_dict={self.inp_seq: x, self.out_seq: y})

    def _load_model(self):
        self.create_graph()
        self.sess.run(self.init_op)
        path = self.SAVE_DIR + "\\" + self.name
        if os.path.isfile(path + '.index'):
            saver = tf.train.Saver()
            saver.restore(self.sess, path)

    def save_model(self):
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
        saver = tf.train.Saver()
        path = self.SAVE_DIR + "\\" + self.name
        saver.save(self.sess, path)


def np_func(fn, inp):
    return fn(inp, dtype='float32')


def new_var(size):
    return tf.Variable(tf.random_uniform(size, 0, 1))


def NN(inp, W, b):
    return tf.matmul(W, inp) + b


def sigmoidNN(inp, W, b):
    return tf.sigmoid(NN(inp, W, b))


def tanhNN(inp, W, b):
    return tf.tanh(NN(inp, W, b))