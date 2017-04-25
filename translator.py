import tensorflow as tf

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

    def enc_elem(self, C_prev, h_prev, x, enc_layer_index, one_hot=False, one_hot_size=None):
        Wf = self.enc_Wf[enc_layer_index]
        bf = self.enc_bf[enc_layer_index]
        Wi = self.enc_Wi[enc_layer_index]
        bi = self.enc_bi[enc_layer_index]
        Wc = self.enc_Wc[enc_layer_index]
        bc = self.enc_bc[enc_layer_index]
        Wo = self.enc_Wo[enc_layer_index]
        bo = self.enc_bo[enc_layer_index]
        if one_hot:
            tmp = h_prev
            f = tf.matmul(Wf[:, one_hot_size:], tmp)
            f = tf.sigmoid(f + tf.transpose(tf.gather(tf.transpose(Wf[:, :one_hot_size]), x)) + bf)
            i = tf.matmul(Wi[:, one_hot_size:], tmp)
            i = tf.sigmoid(i + tf.transpose(tf.gather(tf.transpose(Wi[:, :one_hot_size]), x)) + bi)
            C_h = tf.matmul(Wc[:, one_hot_size:], tmp)
            C_h = tf.sigmoid(C_h + tf.transpose(tf.gather(tf.transpose(Wc[:, :one_hot_size]), x)) + bc)
            o = tf.matmul(Wo[:, one_hot_size:], tmp)
            o = tf.sigmoid(o + tf.transpose(tf.gather(tf.transpose(Wo[:, :one_hot_size]), x)) + bo)
        else:
            tmp = tf.concat([x, h_prev], 0)  # axis is 1??
            f = sigmoidNN(tmp, Wf, bf)
            i = sigmoidNN(tmp, Wi, bi)
            C_h = tanhNN(tmp, Wc, bc)
            o = sigmoidNN(tmp, Wo, bo)
        C = f * C_prev + i * C_h
        h = o * tf.tanh(C)
        return C, h

    def dec_elem(self, C_prev, h_prev, c, x, dec_layer_index, one_hot=False, one_hot_size=None):
        Wf = self.dec_Wf[dec_layer_index]
        bf = self.dec_bf[dec_layer_index]
        Wi = self.dec_Wi[dec_layer_index]
        bi = self.dec_bi[dec_layer_index]
        Wc = self.dec_Wc[dec_layer_index]
        bc = self.dec_bc[dec_layer_index]
        Wo = self.dec_Wo[dec_layer_index]
        bo = self.dec_bo[dec_layer_index]
        if one_hot:
            tmp = tf.concat([h_prev, c], 0)
            f = tf.matmul(Wf[:, one_hot_size:], tmp)
            f = tf.sigmoid(f + tf.transpose(tf.gather(tf.transpose(Wf[:, :one_hot_size]), x)) + bf)
            i = tf.matmul(Wi[:, one_hot_size:], tmp)
            i = tf.sigmoid(i + tf.transpose(tf.gather(tf.transpose(Wi[:, :one_hot_size]), x)) + bi)
            C_h = tf.matmul(Wc[:, one_hot_size:], tmp)
            C_h = tf.sigmoid(C_h + tf.transpose(tf.gather(tf.transpose(Wc[:, :one_hot_size]), x)) + bc)
            o = tf.matmul(Wo[:, one_hot_size:], tmp)
            o = tf.sigmoid(o + tf.transpose(tf.gather(tf.transpose(Wo[:, :one_hot_size]), x)) + bo)
        else:
            tmp = tf.concat([x, h_prev, c], 0)  # axis is 1??
            f = sigmoidNN(tmp, Wf, bf)
            i = sigmoidNN(tmp, Wi, bi)
            C_h = tanhNN(tmp, Wc, bc)
            o = sigmoidNN(tmp, Wo, bo)
        C = f * C_prev + i * C_h
        h = o * tf.tanh(C)
        return C, h  # , c

    def add_to_dec_weights(self, inp_size, hid_size, context_size):
        # if last_hid_size == -1:
        # last_hid_size = hid_size
        tmp = context_size + hid_size  # last_hid_size
        self.dec_Wf.append(new_var([hid_size, tmp + inp_size]))
        self.dec_Wi.append(new_var([hid_size, tmp + inp_size]))
        self.dec_Wc.append(new_var([hid_size, tmp + inp_size]))
        self.dec_Wo.append(new_var([hid_size, tmp + inp_size]))
        self.dec_bf.append(new_var([hid_size, 1]))
        self.dec_bi.append(new_var([hid_size, 1]))
        self.dec_bc.append(new_var([hid_size, 1]))
        self.dec_bo.append(new_var([hid_size, 1]))

    def add_encoder_layer(self, inp, hid_size, one_hot=False, one_hot_size=None):
        sz = one_hot_size or int(inp.shape[1])
        self.add_to_enc_weights(sz, hid_size)
        initial = 2 * (zeros([hid_size, tf.shape(inp)[-1]]),)
        return tf.scan(lambda a, x: self.enc_elem(a[0], a[1], x, len(self.enc_Wf) - 1, one_hot, one_hot_size), inp,
                       initializer=initial)[1]

    def add_decoder_layer(self, inp, contexts, hid_size):
        self.add_to_dec_weights(int(inp.shape[1]), hid_size, int(contexts[0].shape[0]))
        initial = 2 * (zeros([hid_size, tf.shape(inp)[2]]),)
        return tf.scan(lambda a, x: self.dec_elem(a[0], a[1], x[1], x[0], len(self.dec_Wf) - 1), (inp, contexts),
                       initializer=initial)[1]

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
            C, h = self.dec_elem(C_prev, h_prev, c, x, dec_layer_index, True, self.OUT_VOCAB_SIZE)
            c = get_context(h)
            return C, h, c

        def get_context(st_prev):
            res = tf.scan(lambda a, x: self.attention_function(st_prev, x), enc_layers[-1],
                          initializer=zeros([1, tf.shape(enc_layers[-1])[2]]))
            return \
                tf.transpose(
                    tf.matmul(tf.transpose(enc_layers[-1]), tf.transpose(tf.nn.softmax(res, dim=0)), False, True))[
                    0]  # better #softmax dimension==>REALY IMPORTANT

        self.inp_seq = tf.placeholder(tf.int32, [None, None])  # words_n, word_i, sentences_n
        self.out_seq = tf.placeholder(tf.int32, [None, None])

        enc_layers = []

        lstmf = self.add_encoder_layer(self.inp_seq, int(self.ENCODER_HIDDEN_SIZE / 2), True, self.INP_VOCAB_SIZE)
        lstmb = self.add_encoder_layer(tf.reverse(self.inp_seq, [-1]), int(self.ENCODER_HIDDEN_SIZE / 2), True,
                                       self.INP_VOCAB_SIZE)  # HIDDEN_SIZE/2 is OK????
        enc_layers.append(tf.concat([lstmf, lstmb], 1))

        for i in range(self.ENCODER_LAYERS - 1):
            inp = enc_layers[-1]
            if i > 0:
                inp = inp + enc_layers[-2]
            enc_layers.append(self.add_encoder_layer(inp, self.ENCODER_HIDDEN_SIZE))

        self.add_attention_weights(self.DECODER_HIDDEN_SIZE + self.ENCODER_HIDDEN_SIZE, self.ATTENTION_HIDDEN_SIZE)
        dec_layers = []

        self.add_to_dec_weights(self.OUT_VOCAB_SIZE, self.DECODER_HIDDEN_SIZE, self.ENCODER_HIDDEN_SIZE)
        initial = (zeros([self.DECODER_HIDDEN_SIZE, tf.shape(self.out_seq)[-1]]),
                   zeros([self.DECODER_HIDDEN_SIZE, tf.shape(self.out_seq)[-1]]),
                   zeros([self.ENCODER_HIDDEN_SIZE, tf.shape(self.out_seq)[-1]]))
        _, h, contexts = tf.scan(lambda a, x: dec_first_layer_elem(a[0], a[1], a[2], x, len(self.dec_Wf) - 1),
                                 self.out_seq, initializer=initial)

        dec_layers.append(h)
        for i in range(self.DECODER_LAYERS - 1):
            inp = dec_layers[-1]
            if i > 0:
                inp = inp + dec_layers[-2]
            if i < self.DECODER_LAYERS - 2:
                hid_size = self.DECODER_HIDDEN_SIZE
            else:
                hid_size = self.OUT_VOCAB_SIZE
            dec_layers.append(self.add_decoder_layer(inp, contexts, hid_size))

        self.trans = tf.nn.softmax(dec_layers[-1], dim=1)

        self.cross_entropy = -tf.reduce_mean(tf.scan(lambda a, x: (a[0] + 1, tf.reduce_mean(
            tf.scan(lambda a2, x2: (a2[0] + 1, tf.log(x2[self.out_seq[a2[0], a[0]]])), tf.transpose(x), (0, 0.0))[1])),
                                                     tf.transpose(self.trans), (0, 0.0))[
            1])  # is indexing of out_seq OK??

        # self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.out_seq * tf.log(self.trans), reduction_indices=[
        #     1]))  # use built in tensorflow methods, use indexing instead of multiplying
        self.train_step = tf.train.GradientDescentOptimizer(self.RATE).minimize(self.cross_entropy)
        # self.train_step = tf.train.AdamOptimizer(self.RATE).minimize(self.cross_entropy)
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

    def helped_translate(self, input, output):
        return self.sess.run(self.trans, feed_dict={self.inp_seq: input, self.out_seq: output})


def np_func(fn, inp):
    return fn(inp, dtype='float32')


def new_var(size):
    return tf.Variable(tf.random_uniform(size, -1, 1))


def NN(inp, W, b):
    return tf.matmul(W, inp) + b


def sigmoidNN(inp, W, b):
    return tf.sigmoid(NN(inp, W, b))


def tanhNN(inp, W, b):
    return tf.tanh(NN(inp, W, b))


def zeros(shape):
    res = tf.stack(shape)
    return tf.fill(res, 0.0)