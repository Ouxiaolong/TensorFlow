"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf

# 创建一个变量，初始为标量0
state = tf.Variable(0, name="counter")

# 创建一个op，其作用是使`state`增加1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后，变量必须先经过init op初始化
# 首先先增加一个初始化op到图中
init_op = tf.initialize_all_variables()

# 启动图
with tf.Session() as sess:
    # 运行init op
    sess.run(init_op)
    # 打印 state 的初始值
    print(sess.run(state))
    # 运行op， 更新state 并打印
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

