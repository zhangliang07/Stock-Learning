import numpy
import tensorflow as tf


class StockLearning:
	def __init__(self):
		self.__dateSize = 30

		#定义网络
		self.dataInput = tf.placeholder(tf.float32, [None, self.__dateSize, 8], 'buyDataInput')
		#normalInput = tf.layers.batch_normalization(self.dataInput, axis = -1) #数据正则化，可选用

		self.qOut = self.__buildNetWork(self.dataInput, 'buyside')
		self.predict = tf.argmax(self.qOut, 1)

		#监视
		qTest = tf.reduce_mean(self.qOut, axis = 0)
		tf.summary.scalar('mean profit[0]', qTest[0])
		tf.summary.scalar('mean profit[1]', qTest[1])
		tf.summary.scalar('mean profit[2]', qTest[2])
	
		#定义loss
		self.actions = tf.placeholder(tf.int32, [None], 'inputActions')
		self.reward = tf.placeholder(tf.float32, [None], 'buyDataNextPrice')
		tf.summary.scalar('mean profit', tf.reduce_mean(self.reward))
		with tf.name_scope('sp_loss'):
			actions_onehot = tf.one_hot(self.actions, 3, dtype = tf.float32)
			qInAction = tf.reduce_sum(self.qOut * actions_onehot, axis = 1)

			different = -tf.log(tf.squared_difference(qInAction, self.reward)) #负数
			self.loss = tf.reduce_mean(different)
			tf.summary.scalar('tb_loss', self.loss)
			self.trainer = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())

		self.summaryMerged = tf.summary.merge_all()
		self.summaryWriter = tf.summary.FileWriter('/tf_log/train', self.session.graph)
		

	def __buildNetWork(self, input, name):
		with tf.name_scope(name):
			init = tf.random_normal_initializer(mean=0, stddev=0.3)
			conv1 = tf.layers.conv1d(input, filters = 8, kernel_size = 3, strides = 1,
													activation = tf.nn.relu, kernel_initializer = init)

			conv2 = tf.layers.conv1d(conv1, filters = 16, kernel_size = 3, strides = 1,
													activation = tf.nn.relu, kernel_initializer = init)

			conv3 = tf.layers.conv1d(conv2, filters = 32, kernel_size = 3, strides = 1,
													activation = tf.nn.relu, kernel_initializer = init)

			#conv4 = tf.layers.conv1d(conv2, filters = 512, kernel_size = 6, strides = 1,
			#										activation = tf.nn.relu, kernel_initializer = init)

			streamAC, streamVC = tf.split(conv3, 2, 1)
			streamA = tf.layers.flatten(streamAC)
			streamV = tf.layers.flatten(streamVC)

			layer5A_i = tf.layers.dense(streamA, units = 600,
													activation = tf.nn.relu, kernel_initializer = init)
			layer5V_i = tf.layers.dense(streamV, units = 600,
													activation = tf.nn.relu, kernel_initializer = init)
			#layer5A_d = tf.layers.dropout(streamA, 0.1)
			#layer5V_d = tf.layers.dropout(streamV, 0.1)

			layer5A = tf.layers.dense(layer5A_i, units = 3,
													activation = tf.nn.relu, kernel_initializer = init)
			layer5V = tf.layers.dense(layer5V_i, units = 1,
													activation = tf.nn.relu, kernel_initializer = init)

			Qout = layer5V + (layer5A - tf.reduce_mean(layer5A, axis = 1, keepdims = True))

			return Qout


	def learn(self, dataInput, actions, reward, step):
		_, merged = self.session.run([self.trainer, self.summaryMerged],
									 feed_dict = {self.dataInput:dataInput,
										 self.actions: actions,
										 self.reward: reward})

		self.summaryWriter.add_summary(merged, step)


	def chooseAction(self, dataInput, randomRate):
		if numpy.random.random() > randomRate:
			num = self.session.run(self.predict, feed_dict = {self.dataInput:[dataInput]})
			predict = int(num[0])
		else:
			predict = numpy.random.randint(0, 3)

		return predict
