"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt#需要安装才可使用

#添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    
    #线性模型
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        
    return outputs

#【1】创建原始数据，及要训练的数据
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#【2】定义节点准备接收数据
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#【3】定义神经层：隐藏层和预测层
#添加隐藏层，输入值是 xs，在隐藏层有 10 个神经元  
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.tanh)
#添加输出层，输入值是隐藏层 l1，在预测层输出 1 个结果
prediction = add_layer(l1, 10, 1, activation_function=tf.nn.tanh)

#【4】定义损失函数，误差的均方差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
#【5】选择 optimizer 使 loss 达到最小，选择梯度下降的方法训练数据
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#【6】初始化数据，tf 的必备步骤，主要声明了变量，就必须初始化才能用
init = tf.initialize_all_variables()

#【7】创建Session会话。启动图
sess = tf.Session()
#上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)


#【8】训练模型的到结果，迭代，反复执行上面的最小化损失函数这一操作，拟合数据
for _ in range(2000):
    #train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    
prediction_value = sess.run(prediction, feed_dict={xs: x_data})
#画图
plt.figure()
plt.scatter(x_data,y_data)
plt.plot(x_data,prediction_value,'r-',lw=5)
plt.show()
