#!/usr/bin/env python

"""
Toy charNN generation example
"""

import numpy as np
from keras.utils import np_utils

class CharNNData:

    def __init__(self):
        self.filename = "./data/wonderland.txt"
        self.raw_text = None
        self.chars = None
        self.char2int = None
        self.int2char = None
        self.nchars = None
        self.nvocab = None
        self.npattern = None
        self.seqlen = 50
        self.dataX = []
        self.dataY = []
        self._load_file()

    def _load_file(self):
        self.raw_text = open(self.filename, encoding='utf-8').read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char2int = dict((c, i) for i, c in enumerate(self.chars))
        self.int2char = dict((i, c) for i, c in enumerate(self.chars))
        self.nchars = len(self.raw_text)
        self.nvocab = len(self.chars)

    def _create_seq(self):
        for i in range(0, self.nchars - self.seqlen, 1):
            seq_in = self.raw_text[i:i+self.seqlen]
            seq_out = self.raw_text[i+self.seqlen]
            self.dataX.append([self.char2int[ch] for ch in seq_in])
            self.dataY.append(self.char2int[seq_out])
        self.npattern = len(self.dataX)

    def load(self):
        self._create_seq()
        trainX = np.reshape(self.dataX, (self.npattern, self.seqlen, 1))
        trainX = trainX/float(self.nvocab)
        trainY = np_utils.to_categorical(self.dataY)
        return  trainX, trainY

    def data(self):
        return self.dataX







