class Constant:
	def __init__(self, value):
		self.value = value
		return

	def compute(self):
		return self.value
	pass

class Variable:
	def __init__(self, value):
		self.value = value
		return

	def change_value(self, new_value):
		self.value = new_value
		return

	def compute(self):
		return self.value
	pass

class Placeholder:
	def __init__(self):
		return

	def feed(self, value):
		self.value = value
		return

	def compute(self):
		return self.value
	pass

class Operation:
	def __init__(self, a, b):
		self.a = a
		self.b = b
		return
	def compute(self):
		pass
	pass

class Add(Operation):
	def compute(self):
		return self.a.compute() + self.b.compute()
	pass

class Substract(Operation):
	def compute(self):
		return self.a.compute() - self.b.compute()
	pass

class Multiply(Operation):
	def compute(self):
		return self.a.compute() * self.b.compute()
	pass

class Divide(Operation):
	def compute(self):
		return self.a.compute() / self.b.compute()
	pass

class Session:
	def __init__(self, graph):
		self.graph = graph
		return

	def run(self, feed_dict={}):
		for ph in feed_dict:
			print(ph)
			ph.feed(feed_dict[ph])
		return self.graph.compute()
	pass

a = Placeholder()
b = Variable(7)
graph = Substract(Add(Multiply(a, b), Divide(a, b)), a)
sess = Session(graph)
feed_dict = {a: 5}
print(sess.run(feed_dict=feed_dict))