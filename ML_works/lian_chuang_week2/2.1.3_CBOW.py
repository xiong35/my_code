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

filename = R'C:\Users\xiong35\Desktop\t_asv.csv'
df = pd.read_csv(filename, header=0, sep=",", dtype=str)
description = df['t']


def bulid_dic(sentences):
    words = {}
    num_sentence = 0
    total = 0.

    for d in sentences:
        d = d.replace(',', '')
        d = d.replace('.', '')
        d = d.split(' ')[1:]
        num_sentence += 1
        for w in d:
            if w not in words:
                words[w] = 0
            words[w] += 1
            total += 1
        if num_sentence % 100 == 0:
            print(u'已经找到%s个句子' % num_sentence)

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


def data_generator(word2id, subsamples, data):
    x, y = [], []
    sentence_num = 0
    for d in data:
        d = [0]*window + [word2id[w] for w in d if w in word2id] + [0]*window
        r = np.random.random(len(d))
        for i in range(window, len(d)-window):
            if d[i] in subsamples and r[i] > subsamples[d[i]]:
                continue
            x.append(d[i-window:i]+d[i+1:i+1+window])
            y.append([d[i]])
        sentence_num += 1
        if sentence_num == num_sentence_per_batch:
            x, y = np.array(x), np.array(y)
            z = np.zeros((len(x), 1))
            return [x, y], z


def build_w2vm(word_size, window, num_word, num_negative):
    # CBOW输入
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

    # 留意到，我们构造抽样时，把目标放在了第一位，也就是说，softmax的目标id总是0，这可以从data_generator中的z变量的写法可以看出

    model = Model(inputs=[input_words, target_word], outputs=my_softmax)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # 请留意用的是sparse_categorical_crossentropy而不是categorical_crossentropy
    model.summary()
    return model


def most_similar(word2id, w, k=10):
    # model = load_model('./word2vec.h5')  # 载入模型 在数据集较大的时候用空间换时间
    # weights = model.get_weights()#可以顺便看看每层的权重
    embeddings = model.get_weights()[0]
    normalized_embeddings = embeddings / \
        (embeddings**2).sum(axis=1).reshape((-1, 1))**0.5  # 词向量归一化，即将模定为1
    v = normalized_embeddings[word2id[w]]
    sims = np.dot(normalized_embeddings, v)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[i], sims[i]) for i in sort[:k]]


if __name__ == '__main__':
    word_size = 60  # 词向量维度
    window = 5  # 窗口大小
    num_negative = 15  # 随机负采样的样本数
    min_count = 0  # 频数少于min_count的词将会被抛弃
    num_worker = 4  # 读取数据的并发数
    num_epoch = 2  # 迭代次数，由于使用了adam，迭代次数1～2次效果就相当不错
    subsample_t = 1e-5  # 词频大于subsample_t的词语，会被降采样，这是提高速度和词向量质量的有效方案
    num_sentence_per_batch = 20

    # data,sentences = getdata(fname) #读原始数据
    num_sentence, id2word, word2id, num_word, subsamples = bulid_dic(
        description)  # 建字典
    ipt, opt = data_generator(word2id, subsamples, description)  # 构造训练数据
    model = build_w2vm(word_size, window, num_word, num_negative)  # 搭模型
    model.fit(ipt, opt,
              steps_per_epoch=int(num_sentence/num_sentence_per_batch),
              epochs=num_epoch,
              workers=num_worker,
              use_multiprocessing=True
              )
    print(pd.Series(most_similar(word2id, 'father')))
    print(pd.Series(most_similar(word2id, 'glory')))
    print(pd.Series(most_similar(word2id, 'heaven')))

    model.save_weights('model_weights.h5')
    plot_model(model, to_file='./word2vec.png',
               show_shapes=True, dpi=300)  # 输出框架图
