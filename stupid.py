import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class Neuron:
	def __init__(self):
		self.weight = np.random.uniform(-1, 1)
		return

	def feed(self, inputs):
		return self.weight * inputs

	def optimize(self, cost):
		self.weight += cost * np.random.uniform(-1, 1)
		return

class Layer:
	def __init__(self, nodes, output):
		self.nodes = np.array([Neuron() for i in range(nodes)])
		return

	def feed(self, inputs):
		out_arr = []
		for node in self.nodes:
			out_arr.append(node.feed(inputs))
		return sigmoid(np.sum(out_arr))

	def backwards(self, cost):
		return [node.optimize(cost) for node in self.nodes]

class Overlord:
	def __init__(self, inputs, layers, nodes, output):
		self.inputs = inputs
		self.outputs = output
		self.layers = []
		layer_array = [inputs]
		for i in range(layers):
			layer_array.append(nodes)
		layer_array.append(output)
		for i in range(1, len(layer_array) - 1, 1):
			self.layers.append(Layer(layer_array[i], layer_array[i + 1]))
		return

	def __feed_forward(self, inputs):
		out = inputs
		for layer in self.layers:
			out = layer.feed(out)
		return out

	def __get_cost(self, out, labels):
		return np.sum(np.square(labels - out))

	def __backwards(self, cost):
		for layer in self.layers:
			layer.backwards(cost)
		return

	def train(self, input_arr, output_arr):
		global iterations
		cost_arr = []
		for i in range(iterations):
			output = self.__feed_forward(input_arr)
			cost = self.__get_cost(output, output_arr)
			self.__backwards(cost)
			cost_arr.append(cost)
		return np.mean(cost_arr)



if __name__ == "__main__":
	nn = Overlord(inputs=1, layers=3, nodes=5, output=1)
	batch_size = 64
	iterations = 4096
	input_arr = np.zeros(batch_size)
	out_arr = np.ones(batch_size)
	print(nn.train(input_arr, out_arr))

