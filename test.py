import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x= tf.placeholder(tf.float32, [None, 784], name="x")
y= tf.placeholder(tf.float32, [None, 10], name="y")

W1= tf.get_variable("W1", [784, 25], initializer = tf.contrib.layers.xavier_initializer())
b1= tf.get_variable("b1", [1, 25], initializer = tf.zeros_initializer())

W2= tf.get_variable("W2", [25, 19], initializer = tf.contrib.layers.xavier_initializer())
b2= tf.get_variable("b2", [1, 19], initializer = tf.zeros_initializer())

W3= tf.get_variable("W3", [19, 14], initializer = tf.contrib.layers.xavier_initializer())
b3= tf.get_variable("b3", [1, 14], initializer = tf.zeros_initializer())

W4= tf.get_variable("W4", [14, 10], initializer = tf.contrib.layers.xavier_initializer())
b4= tf.get_variable("b4", [1, 10], initializer = tf.zeros_initializer())

Z1= tf.add(tf.matmul(x,W1), b1)
A1= tf.nn.relu(Z1)

Z2= tf.add(tf.matmul(A1,W2), b2)
A2= tf.nn.relu(Z2)

Z3= tf.add(tf.matmul(A2,W3), b3)
A3= tf.nn.relu(Z3)

Z4= tf.add(tf.matmul(A3,W4), b4)
cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= Z4, labels= y))
optimizer= tf.train.AdamOptimizer(learning_rate= 0.001).minimize(cost)

num_epochs= 100
minibatch_size= 64
costs= []

init= tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(num_epochs):
		epoch_cost= 0
		num_minibatches= int(mnist.train.num_examples/minibatch_size)
		#minibatches= random_mini_batches(mnist.train.images, mnist.train.labels, minibatch_size)

		for minibatch in range(num_minibatches):
			minibatch_X, minibatch_Y= mnist.train.next_batch(minibatch_size)
			_ , minibatch_cost = sess.run([optimizer, cost], feed_dict={x: minibatch_X, y: minibatch_Y})
			epoch_cost += minibatch_cost / num_minibatches

		if epoch % 10 == 0:
			print("Cost after epoch %i: %f" % (epoch, epoch_cost))
		if epoch % 5 == 0:
			costs.append(epoch_cost)

	correct_prediction = tf.equal(tf.argmax(Z4, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print("Test Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))