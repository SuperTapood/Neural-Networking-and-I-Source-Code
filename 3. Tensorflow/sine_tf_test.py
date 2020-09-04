import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_eager_execution()


x_test = np.arange(10, 20, 0.1)
y_test = np.sin(x_test)

num_inputs = 1
num_outputs = 1
hidden_layers = 3
hidden_units = 16
learning_rate = 0.01
batch_size = 1024

inputs = tf.placeholder(tf.float32 ,shape=(None, 1))
targets = tf.placeholder(tf.float32 ,shape=(None, 1))

w = tf.get_variable("weight-1", shape=(num_inputs, hidden_units))
b = tf.get_variable("bias-1", shape=(hidden_units))
output = tf.matmul(inputs, w) + b

for i in range(hidden_layers):
	w = tf.get_variable(f"weight{i}", shape=(hidden_units, hidden_units))
	b = tf.get_variable(f"bias{i}", shape=(hidden_units))
	output = tf.matmul(output, w) + b

w = tf.get_variable("weight-fin", shape=(hidden_units, num_outputs))
b = tf.get_variable("bias-fin", shape=(num_outputs))
output = tf.matmul(output, w) + b


loss = tf.square(output - targets)
mean_loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(mean_loss)

init = tf.global_variables_initializer()
batch_count = 5000

with tf.Session() as sess:
	sess.run(init)
	for i in range(batch_count):
		print(f"batch {i}")
		x = np.random.randint(0, 100, size=(batch_size, 1)) / 10
		print(x)
		feed_dict = {inputs: x, targets: np.sin(x)}
		_, curr_loss = sess.run(fetches=[train, mean_loss], feed_dict=feed_dict)
		print(curr_loss)
	y_hat = sess.run(fetches=[output], feed_dict={inputs: x_test.reshape(100, 1)})

plt.plot(x_test, y_test)
plt.plot(x_test, y_hat[0], color="red")
plt.show()