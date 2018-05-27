"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np

#【第一步】准备数据
#创建原始数据,使用 NumPy 生成假数据(phony data), 总共 100 个点
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.6 + 0.6

#【第二步】构造线性模型
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases

#【第三步】求解模型
#定义损失函数，误差的均方差
loss = tf.reduce_mean(tf.square(y-y_data))
# 选择梯度下降的方法
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 迭代的目标：最小化损失函数
train = optimizer.minimize(loss)

#【第四步】初始化数据，tf 的必备步骤，主要声明了变量，就必须初始化才能用
init = tf.initialize_all_variables()


# 设置tensorflow对GPU的使用按需分配，配置好GPU才能使用以下两行代码
#config  = tf.ConfigProto()
#config.gpu_options.allow_growth = True

#【第五步】创建Session会话。启动图
sess = tf.Session()
#sess = tf.Session(config = config)配置好GPU才能使用以下代码
sess.run(init)          

#【第六步】训练模型的到结果，迭代，反复执行上面的最小化损失函数这一操作（train op）,拟合平面
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))


