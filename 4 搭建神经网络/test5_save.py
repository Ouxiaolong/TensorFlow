"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""

import tensorflow as tf
import numpy as np

## 【1】保存文件
# remember to define the same dtype and shape when restore
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init= tf.initialize_all_variables()

saver = tf.train.Saver()

# 用 saver 将所有的 variable 保存到定义的路径
with tf.Session() as sess:
   sess.run(init)
   save_path = saver.save(sess, "my_net/save_net.ckpt")
   print("Save to path: ", save_path)


#【2】存储数据
# redefine the same shape and same type for your variables
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# not need init step
saver = tf.train.Saver()
# 用 saver 从路径中将 save_net.ckpt 保存的 W 和 b restore 进来
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))
