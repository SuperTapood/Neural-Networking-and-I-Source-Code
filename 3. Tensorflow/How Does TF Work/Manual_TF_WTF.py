import tensorflow.compat.v1 as tf


tf.disable_eager_execution()
a = tf.placeholder(dtype=float)
b = tf.Variable(7.0)
init = tf.global_variables_initializer()
c = a * b
d = a / b
e = c + d
graph = e - a

with tf.Session() as sess:
	sess.run(init)
	feed_dict = {a: 5.0}
	result = sess.run(graph, feed_dict=feed_dict)
	print(result)