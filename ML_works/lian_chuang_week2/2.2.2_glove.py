
import keras.backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Reshape, Add
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

word2id = pd.read_csv(R'C:\Users\xiong35\Desktop\word2id.csv')
id2word = pd.read_csv(R'C:\Users\xiong35\Desktop\id2word.csv')
num_of_words = word2id.shape[1]


def get_co_occur():
    import os
    filename = R'C:\Users\xiong35\Desktop\co_occur.csv'
    if os.path.exists(filename):
        print('reading')
        co_occur = pd.read_csv(filename, dtype=int, index_col=0)
        co_occur = co_occur.values
    else:
        co_occur = np.zeros((num_of_words, num_of_words))

        window = 12
        corpus = R'C:\Users\xiong35\Desktop\corpus\new_h.txt'
        with open(corpus) as fr:
            lines = fr.readlines()

        for j, line in enumerate(lines):
            line = line.strip('\n').split()
            for index, central_word in enumerate(line):
                if central_word not in word2id:
                    continue
                central_id = word2id[central_word][0]
                bg = max(0, index-window)
                ed = min(len(line), index+window+1)
                for i in range(bg, ed):
                    context_word = line[i]
                    if context_word not in word2id:
                        continue
                    context_id = word2id[context_word][0]
                    co_occur[context_id, central_id] += 1
            print('%d is done!' % j)

        co_occur = pd.DataFrame(co_occur)
        co_occur.to_csv(R'C:\Users\xiong35\Desktop\co_occur.csv', index=False)
    return co_occur


co_occur = get_co_occur()

print(co_occur)


X_MAX = 100
a = 3.0 / 4.0


def glove_model(vocab_size=num_of_words, vector_dim=64):

    input_target = Input((1,), name='central_word_id')
    input_context = Input((1,), name='context_word_id')

    central_embedding = Embedding(
        vocab_size, vector_dim, input_length=1, name='central_embedding')
    central_bias = Embedding(
        vocab_size, 1, input_length=1, name='central_bias')

    context_embedding = Embedding(
        vocab_size, vector_dim, input_length=1, name='context_embedding')
    context_bias = Embedding(
        vocab_size, 1, input_length=1, name='context_bias')

    vector_target = central_embedding(input_target)
    vector_context = context_embedding(input_context)

    bias_target = central_bias(input_target)
    bias_context = context_bias(input_context)

    dot_product = Dot(axes=-1)([vector_target, vector_context])
    dot_product = Reshape((1,))(dot_product)
    bias_target = Reshape((1,))(bias_target)
    bias_context = Reshape((1,))(bias_context)

    prediction = Add()([dot_product, bias_target, bias_context])

    model = Model(inputs=[input_target, input_context], outputs=prediction)
    model.compile(loss=my_loss, optimizer=Adam(lr=0.01))

    model.summary()
    return model


def my_loss(y_true, y_pred):
    return K.sum(K.pow(K.clip(y_true / X_MAX, 0.0, 1.0), a) *
                 K.square(y_pred - K.log(y_true+1e-6)), axis=-1)


def gen_data():
    while True:
        centrals = []
        contexts = []
        outputs = []
        to_train = np.random.randint(0, num_of_words, int(num_of_words*0.01))
        central_id = to_train[0]
        for context_id in to_train[1:]:
            centrals.append(central_id)
            contexts.append(context_id)
            out = co_occur[central_id, context_id]
            outputs.append(out)
        yield [np.array(centrals), np.array(contexts)], np.array(outputs)


def most_similar(word, k=10):
    embeddings = model.get_weights()[0]+model.get_weights()[1]
    normalized_embeddings = embeddings / \
        (embeddings**2).sum(axis=1).reshape((-1, 1))**0.5
    v = normalized_embeddings[int(word2id[word])]
    sims = np.dot(normalized_embeddings, v)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[str(i)], sims[i]) for i in sort[:k]]


def predict():
    while True:
        word = input("enter a word: ")
        if word == 'q':
            break
        if word in word2id:
            print(pd.Series(most_similar(word)))
        else:
            print('not in')


model = glove_model()
history = model.fit_generator(gen_data(),
                              steps_per_epoch=500,
                              epochs=40,)
try:
    loss = history.history['loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.show()
except:
    pass

model.save_weights('my_glove_w.h5')
predict()
