import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt


tf.disable_eager_execution()


x_data = np.arange(0, 50, 0.1)
y_data = 186 * x_data + 255
batch_size = 64
learning_rate = 0.1

x = tf.placeholder(tf.float64)
y = tf.placeholder(tf.float64)
m = tf.Variable(np.random.randn(1)[0])
b = tf.Variable(np.random.randn(1)[0])
model = m * x + b

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
loss = tf.square(y - model)
sum_loss = tf.reduce_sum(loss)
train = optimizer.minimize(sum_loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	batches = 20000
	for i in range(batches):
		print(f"Batch {i}")
		random_index = np.random.randint(len(x_data), size=batch_size)
		feed_dict = {x: x_data[random_index], y: y_data[random_index]}
		sess.run(train, feed_dict=feed_dict)
		model_m, model_b = sess.run([m, b])
		print(model_m)
		print(model_b)

plt.plot(x_data, y_data)
plt.plot(x_data, x_data * model_m + model_b, color="red")
plt.show()