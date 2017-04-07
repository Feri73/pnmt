from keras.layers import *
from keras.models import *
from keras.layers.wrappers import *

VOCAB_SIZE = 10000
HIDDEN_SIZE = 128
ENCODER_LAYER = 8
DECODER_LAYER = 8

inp = Input(shape=(None, VOCAB_SIZE))
lstm = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True), merge_mode='concat')(inp)
for i in range(ENCODER_LAYER - 1):
    rs = True if i < ENCODER_LAYER - 2 else False
    tmp = LSTM(2 * HIDDEN_SIZE, return_sequences=rs)(lstm)  # 2*HIDDEN_SIZE is OK??
    lstm = merge([tmp, lstm], mode='sum')
encoder=RepeatVector(None)