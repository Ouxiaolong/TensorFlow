"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf

# 创建两个变量
a=tf.constant([1.0,2.0],name='a')
b=tf.constant([2.0,3.0],name='b')

result=tf.add(a,b,name='add')

print(result)