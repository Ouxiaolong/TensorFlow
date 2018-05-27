"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(4.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)
#mul = tf.mul(input1, intermed)#最新的TensorFlow已经tf.mul请读者朋友注意咯

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)
