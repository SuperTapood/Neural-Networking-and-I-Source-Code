from tensorflow.keras.datasets import mnist
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_inputs = 28 * 28
num_outputs = 1
hidden_layers = 7
hidden_units = 128
batch_size = 1024
batch_count = 4096
learning_rate = 0.001

inputs = tf.placeholder(dtype=tf.float32, shape=(None, num_inputs))
targets = tf.placeholder(dtype=tf.float32, shape=(None, num_outputs))

w = tf.get_variable("weight", shape=(num_inputs, hidden_units))
b = tf.get_variable("bias", shape=(hidden_units))
layer_output = tf.sigmoid(tf.matmul(inputs, w) + b)

for i in range(hidden_layers):
	w = tf.get_variable(f"weight{i}", shape=(hidden_units, hidden_units))
	b = tf.get_variable(f"bias{i}", shape=(hidden_units))
	layer_output = tf.sigmoid(tf.matmul(layer_output, w) + b)

w = tf.get_variable("fin_weight", shape=(hidden_units, num_outputs))
b = tf.get_variable("fin_bias", shape=(num_outputs))
output = tf.sigmoid(tf.matmul(layer_output, w) + b) * 10

loss = tf.square(output - targets)
mean_loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(mean_loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for batch in range(batch_count):
		print(f"Batch {batch}")
		chosen = np.random.randint(0, len(x_train), size=(batch_size))
		x_batch = []
		y_batch = []
		for i in chosen:
			x_batch.append(x_train[i].flatten())
			y_batch.append(y_train[i])
		feed_dict = {inputs: x_batch, targets: np.array(y_batch).reshape(batch_size, 1)}
		_, curr_loss = sess.run([train, mean_loss], feed_dict=feed_dict)
		print(curr_loss)
	losses = []
	for x, y in zip(x_test, y_test):
		loss = sess.run(mean_loss, feed_dict={inputs: x.flatten().reshape(1, 784), targets: y.reshape(1, 1)})
		losses.append(loss)
	print(f"Mean test loss: {np.mean(losses)}")