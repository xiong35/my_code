# 逻辑回归  
## 介绍  
既然已经有线性回归了，为什么还要逻辑回归呢?
  
- 线性回归适合解决生成问题，而在分类问题上会有一些问题  
e.g.  输入肿瘤大小，判断恶性(1)或良性(0)，一个特别大的肿瘤是恶性，这个肿瘤有多大本不该影响模型判断，但离群数据却会影响线性回归的结果，这显然是不合理的
  
那逻辑回归要怎么改进呢?
- 总的来说，假设样本服从某一个分布(一般选择正态分布)，设置一系列分布核心和弥散区域，假设实际情况就是这个高斯分布，计算基于 产生如此样本的概率，选出最大可能的一个分别作为最终模型(最大似然)  
  
可是这样选这么多次太蠢了吧!
- 所以我们还是用梯度下降的方法!

##MATH WARNING
![maximum likelihood](images/maximum_likelihood.jpg)  
![goodness of a function](images/cross_entropy.png)
![derivation](images/logistic_regression.png)

## demo  
引自[github](https://github.com/aymericdamien/TensorFlow-Examples)
``` python
'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

```