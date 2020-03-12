import numpy as np
from keras.layers import Input, Embedding, Lambda
from keras.models import Model, load_model
from keras.utils import plot_model
import keras.backend as K
import pandas as pd
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()


word_size = 30  
window = 4 
num_negative = 15 
min_count = 2  
num_epoch = 1 
subsample_t = 5e-5  
num_sentence_per_batch = 16
fname = R'C:\Users\xiong35\Desktop\corpus\Holmose.txt'


def preprocess(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    sentences = []
    data = []
    stopwords = get_stopwords()
    for line in lines:
        data.append(line.strip())
        line = line.replace(',', '')
        line = line.replace('.', '')
        line = line.replace('-', ' ')
        line = line.replace('?', '')
        line = line.replace('”', '')
        line = line.replace('“', '')
        line = line.replace('!', '')
        line = line.lower()
        line = line.strip()
        sts = line.split(' ')
        splits = []
        for w in sts:
            if w not in stopwords:
                splits.append(w)
        sentences.append(splits)
    f.close()
    return data, sentences


def get_stopwords():
    stopwords = []
    stopwords = [line.strip() for line in open(
        R'C:\Users\xiong35\Desktop\corpus\stopped_word.txt').readlines()]
    return stopwords


def bulid_dic(sentences):
    words = {}
    num_sentence = 0
    total = 0.

    for d in sentences:
        num_sentence += 1
        for w in d:
            if w not in words:
                words[w] = 0
            words[w] += 1
            total += 1

    words = {i: j for i, j in words.items() if j >= min_count}
    id2word = {i+1: j for i, j in enumerate(words)}
    word2id = {j: i for i, j in id2word.items()}
    num_word = len(words)+1

    subsamples = {i: j/total for i, j in words.items() if j/total >
                  subsample_t}
    subsamples = {i: subsample_t/j +
                  (subsample_t/j)**0.5 for i, j in subsamples.items()}
    subsamples = {word2id[i]: j for i,
                  j in subsamples.items() if j < 1.}
    return num_sentence, id2word, word2id, num_word, subsamples


def data_generator():
    while True:
        x, y = [], []
        sentence_num = 0
        for d in data:
            d = [0]*window + [word2id[w]
                              for w in d if w in word2id] + [0]*window
            r = np.random.random(len(d))
            has_result = False
            for i in range(window, len(d)-window):
                if d[i] in subsamples and r[i] > subsamples[d[i]]:
                    continue
                temp = d[i-window:i]+d[i+1:i+1+window]
                if len(temp) != window * 2:
                    continue
                has_result = True
                x.append(temp)
                y.append([d[i]])
            if has_result:
                sentence_num += 1
            if sentence_num == num_sentence_per_batch:
                x, y = np.array(x), np.array(y)
                z = np.zeros((len(x), 1))
                yield [x, y], z
                x, y = [], []
                sentence_num = 0


def build_w2vm(word_size, window, num_word, num_negative):
    input_words = Input(shape=(window*2,), dtype='int32')
    input_vecs = Embedding(num_word, word_size, name='word2vec')(input_words)
    input_vecs_sum = Lambda(lambda x: K.sum(x, axis=1))(input_vecs)

    target_word = Input(shape=(1,), dtype='int32', name='target_word')
    negatives = Lambda(lambda x: K.random_uniform(
        (K.shape(x)[0], num_negative), 0, num_word, 'int32'), name='negatives')(target_word)
    samples = Lambda(lambda x: K.concatenate(x))([target_word, negatives])

    softmax_weights = Embedding(num_word, word_size, name='W')(samples)
    softmax_biases = Embedding(num_word, 1, name='b')(samples)
    my_softmax = Lambda(lambda x:
                        K.softmax(
                            (K.batch_dot(x[0], K.expand_dims(x[1], 2))+x[2])[:, :, 0])
                        )([softmax_weights, input_vecs_sum, softmax_biases])

    model = Model(inputs=[input_words, target_word], outputs=my_softmax)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def most_similar(word2id, w, k=10):
    embeddings = model.get_weights()[0]
    normalized_embeddings = embeddings / \
        (embeddings**2).sum(axis=1).reshape((-1, 1))**0.5
    v = normalized_embeddings[word2id[w]]
    sims = np.dot(normalized_embeddings, v)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[i], sims[i]) for i in sort[:k]]


def predict():
    while True:
        word = input("enter a word: ")
        if word == 'q':
            break
        if word in word2id:
            print(pd.Series(most_similar(word2id, word)))
        else:
            print('not in')

data, sentences = preprocess(fname)  
num_sentence, id2word, word2id, num_word, subsamples = bulid_dic(
    sentences)  # FIXME: change to new_h.txt
model = build_w2vm(word_size, window, num_word, num_negative)  
model.fit_generator(data_generator(),
                    steps_per_epoch=int(
                        num_sentence/num_sentence_per_batch),
                    epochs=num_epoch,)
# model.load_weights('model_weights.h5')

predict()
model.save('model.h5')

