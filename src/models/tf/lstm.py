#!/usr/bin/env python3

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

class lstm:
    def __init__(self, PERIOD, UNITS):
        self.model = None
        input_layer = Input(shape=(PERIOD, 3))
        lstm_layer = LSTM(UNITS) (input_layer)
        dense_layer = Dense(1) (lstm_layer)
        self.model = Model(inputs=input_layer, outputs=dense_layer)
