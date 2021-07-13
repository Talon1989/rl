import numpy as np
import tensorflow as tf


#  this is custom coded to work for the original tensorflow 1


# with tf.compat.v1.Session() as session:
#     hello = tf.constant('Hello tf')
#     print(session.run(hello))


# graph = tf.Graph()
# with graph.as_default():
#     a = tf.multiply(8, 5)
#     b = tf.multiply(a, 1)
#     z = tf.add(
#         a,
#         b,
#         name='Add'
#     )


# session = tf.compat.v1.Session()
# a = tf.multiply(3, 3)
# print(a)


# with tf.compat.v1.Session() as session:
#     a = tf.multiply(3, 3)
#     print(session.run(a))


#  VARIABLES, CONSTANTS, AND PLACEHOLDERS  #################################


# with tf.compat.v1.Session() as session:
#     x = tf.Variable(13)
#     W = tf.Variable(tf.compat.v1.random_normal([500, 111], mean=0, stddev=0.35), name='weights')
#     session.run(tf.compat.v1.global_variables_initializer())
#     print(session.run(x))
#     print(session.run(tf.constant(np.pi)))


#  TENSORBOARD  #################################


# with tf.compat.v1.Session() as session:
#     x = tf.constant(1, name='x')
#     y = tf.constant(1, name='y')
#     a = tf.constant(3, name='a')
#     b = tf.constant(3, name='b')
#     prod_1 = tf.multiply(x, y, name='prod1')
#     prod_2 = tf.multiply(a, b, name='prod2')
#     summ = tf.add(prod_1, prod_2, name='sum')
#     writer = tf.compat.v1.summary.FileWriter(
#         logdir='./data/graphs',
#         graph=session.graph
#     )
#     print(session.run(summ))


































































































































































































































