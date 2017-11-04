#!/usr/bin/env python

"""
Toy charNN example
"""

from load_data import CharNNData
from keras.layers import Input, LSTM, Dropout, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import numpy as np


class CharNNModel:
    def __init__(self):
        self.dl = CharNNData()
        self.trainX, self.trainY = self.dl.load()
        self.inp_shape = (self.trainX.shape[1], self.trainX.shape[2])
        self.out_shape = self.trainY.shape[1]
        self.gen_model = None
        self.batch_size = 256
        self.nb_epochs = 200
        self.gen_text = []
        self.dataX = self.dl.data()

    def build_model(self):
        inp = Input(shape = self.inp_shape)
        x = LSTM(256)(inp)
        x = Dropout(0.2)(x)
        out = Dense(self.out_shape, activation='softmax')(x)
        self.gen_model = Model(inp, out)
        self.gen_model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.gen_model.summary()

    def _checkpoints(self):
        filepath="checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-gentext-CharRNN-simple.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        return callbacks_list

    def run(self):
        self.gen_model.fit(self.trainX, self.trainY, nb_epoch=self.nb_epochs, batch_size=self.batch_size, callbacks=self._checkpoints() )
        self.gen_model.save('checkpoints/Text_gen_01-CharRNN_no_embedding-simple')

    def generate(self):
        start = np.random.randint(0, len(self.dataX)-1)
        pattern = self.dataX[start]
        seed = self.dataX[start]
        for i in range(100):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self.dl.nvocab)
            prediction = self.gen_model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = self.dl.int2char[index]
            seq_in = [self.dl.int2char[value] for value in pattern]
            pattern.append(index)
            self.gen_text.append(index)
            pattern = pattern[1:len(pattern)]

        print("\nDone.")
        print(pattern)
        print("\"", ''.join([self.dl.int2char[value] for value in seed]), "\"")
        print("\"", ''.join([self.dl.int2char[value] for value in self.gen_text]), "\"")

def main():
    cn = CharNNModel()
    cn.build_model()
    cn.run()
    cn.generate()

if __name__ == "__main__":
    main()
