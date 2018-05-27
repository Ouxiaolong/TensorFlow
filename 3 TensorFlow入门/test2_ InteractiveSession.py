"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0]);

# 使用初始化器initializer op的run()方法初始化x
x.initializer.run()

# 增加一个减法sub op，从x减去a。运行减法op，输出结果
sub = tf.subtract(x,a)
#sub = tf.sub(x,a)#最新的TensorFlow已经弃用tf.sub
print(sub.eval())
