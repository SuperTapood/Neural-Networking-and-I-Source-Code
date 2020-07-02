import tensorflow.compat.v1 as tf


tf.disable_eager_execution()
a = tf.Variable(5.0)
b = tf.Variable(7.0)
init = tf.global_variables_initializer()
c = a * b
d = a / b
e = c + d
graph = e - a

with tf.Session() as sess:
	sess.run(init)
	result = sess.run(graph)
	print(result)