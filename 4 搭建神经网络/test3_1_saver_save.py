"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np

# 建立零时的W 和 b容器. 找到文件目录, 并用saver.restore()我们放在这个目录的变量.
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# not need init step
saver = tf.train.Saver()
# 用 saver 从路径中将 save_net.ckpt 保存的 W 和 b restore 进来
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))
