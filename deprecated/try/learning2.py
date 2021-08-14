import numpy
import tensorflow as tf


class StockLearning:
	def __init__(self, dateSize = 188):
		self.__dateSize = dateSize

		#定义买方网络
		self.buyDataInput = tf.placeholder(tf.float32, [None, dateSize, 7], 'buyDataInput')
		self.buyQout = self.__buildNetWork(self.buyDataInput, 'buyside')

		#定义loss
		self.buyDataNextPrice = tf.placeholder(tf.float32, [None, 1], 'buyDataNextPrice')
		with tf.name_scope('sp_loss'):
			self.loss = tf.reduce_sum(
				tf.squared_difference(self.buyQout, self.buyDataNextPrice)/tf.square(self.buyDataNextPrice))
			tf.summary.scalar('tb_loss', self.loss)
			self.trainer = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())

		self.summaryMerged = tf.summary.merge_all()
		self.summaryWriter = tf.summary.FileWriter('/tf_log/train', self.session.graph)
		

	def __buildNetWork(self, input, name):
		with tf.name_scope(name):
			conv1 = tf.layers.conv1d(inputs = input, filters = 20, kernel_size = 5, strides = 3,
													 activation = tf.nn.relu)

			conv2 = tf.layers.conv1d(inputs = conv1, filters = 40, kernel_size = 4, strides = 2,
													 activation = tf.nn.relu)

			conv3 = tf.layers.conv1d(inputs = conv2, filters = 200, kernel_size = 6, strides = 4,
													 activation = tf.nn.relu)

			conv4 = tf.layers.conv1d(inputs = conv3, filters = 500, kernel_size = 6, strides = 1,
													 activation = tf.nn.relu)

			stream = tf.layers.flatten(conv4)
			layer5 = tf.layers.dense(stream, 1)

			return layer5


	def learn(self, buyDataInput, buyDataNextPrice, step):
		_, merged = self.session.run([self.trainer, self.summaryMerged],
									 feed_dict = {self.buyDataInput:buyDataInput,	self.buyDataNextPrice: buyDataNextPrice})

		self.summaryWriter.add_summary(merged, step)
