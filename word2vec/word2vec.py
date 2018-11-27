# coding:utf-8
import fasttext
import os
import numpy as np
import torch

from utils.util import print_info

transformer = None


class Word2vec:
    def __init__(self, path):
        print_info("begin to load word embedding")

        self.model = fasttext.load_model(path)

        print_info("load word embedding succeed")

    def load(self, word):
        return self.model[word]


def init_transformer(config):
    global transformer
    if config.get("data", "need_word2vec"):
        transformer = Word2vec(config.get("data", "word2vec"))


def load(word):
    global transformer
    return transformer.load(word)


def transform_sentence(sentence):
    result = []

    for word in sentence:
        result.append(load(word))

    result = np.array(result, dtype=np.float32)

    return result


def pad_trans(data, length):
    arr = []
    for a in range(0, len(data)):
        arr.append(load(data[a]))

    while len(arr) < length:
        arr.append(load("BLANK"))

    arr = np.array(arr, dtype=np.float32)
    arr = torch.from_numpy(arr)

    return arr[0:length]
