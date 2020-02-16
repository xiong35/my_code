
# keras的奇技淫巧

## 1 函数式API

e.g.

``` python
from keras import Input, layers
from keras.models import Model

input_tensor = Input(shape=(64,))
x = layers.Dense(32,activation='relu')(input_tensor)
x = layers.Dense(32,activation='relu')(x)
output_tensor = layers.Dense(10，activation='softmax')(x)

model = Model(input_tensor, output_tensor)

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy')
model.fit(x_train, y_train,epochs=10,batch_size=128)
```

以上案例用函数式API构建了一个2层全连接网络  
Model接受一个输入向量和一个输出向量，在后台自动找两者间的通路  

### 1.1 多输入模型

e.g.双输入问答模型

```python
from keras.models import Model
from keras import Input, layers
import numpy as np

### build the model ###
text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,),
                   dtype='int32', name='text')
embedded_text = layers.Embedding(
    text_vocabulary_size,64)(text_input)
encoded_text = layers.LSTM(32)(enbedded_text)

question_input = Input(shape=(None,),
                       dtype='int32', name='question')
embedded_question = layers.Embedding(
    question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(enbedded_question)

concatenated = layers.concatenate(
    [encoded_text,encoded_question],axis=-1)

answer = layers.Dense(answer_vocabulary_size,
                      activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              matrics=['acc'])

## assume we have texts, questions, and answers
model.fit({'text': text,'question': question}, answers,
          epochs=10,batch_size=128)
```

### 1.2 多输出模型

e.g.：给一系列博客，预测年龄、收入、性别

```python
from keras import Input, layers
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,),dtype='int32',name='posts')
embedded_posts = layers.Embedding(256,vocabulary_size)(posts_input)

x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_pre = layers.Dense(1, name='age')(x)
income_pre = layers.Dense(num_income_groups,
                          activation='softmax', name='income')(x)
gender_pre = layers.Dense(1, activation='sigmoid', name='gender')

model = Model(posts_input, [age_pre, income_pre, gender_pre])

model.compile(optimizer='rmsprop',
              loss={'age': 'mse',
                    'income': 'categorical_crossentropy',
                    'gender': 'binary_crossentropy'},
              loss_weights={'age': 0.25,
                           'income':1. ,
                           'gender': 10.})

model.fit(posts,{'age':age_targets,
                 'income':income_targets,
                 'gender':gender_targets},
          epochs=10, batch_size=64)
```

### 1.3 层组成的有向无环图

- Inception：用不同方式（卷积核大小/步长）卷积，汇总结果
- residual connection：直接把开头几层输入传给后面的层

### 1.4 共享权重

### 1.5 将『模型』作为层

---

## 2 回调函数

### 2.1 常用函数

```python
from keras import callbacks as cb

callbacks_list = [
    # 解决过拟合的问题
    cb.EarlyStopping(   # 提前终止
        monitor='acc',  # 监控指标
        patience=2      # 如果指标在3轮内没降就中断
    )，
    cb.ModelCheckpoint(         # 每轮后都保存权重
        filepath='my_model.h5', # 保存路径
        monitor='val_loss',
        save_best_only=True     # 如果指标没改善就不保存
    )
    # 解决碰到平台的问题
    cb.ReduceLROnPlateau(       # 调整LR
        monitor='val_loss',
        factor=0.1,
        patience=10
    )
]

model.compile(balabala)

model.fit(x, y, epochs=10, batch_size=128
    callbacks=callbacks_list,validation_data=(xVal, yVal))
```

### 2.2 编写自己的回调函数

略，自己搜去

---

## 3 TensorBoard

