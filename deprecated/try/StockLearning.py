import numpy
import tensorflow as tf


class StockLearning:
	def __init__(self, dateSize = 188):
		self.__dateSize = dateSize

		#定义买方网络
		self.buyDataInput = tf.placeholder(tf.float32, [None, dateSize, 7], 'buyDataInput')
		self.buyQout = self.__buildNetWork(self.buyDataInput, 'buyside')
		self.buyDate = tf.argmax(self.buyQout, 0, name = 'buyPredict')
		tf.summary.scalar('buyDate', self.buyDate)
		self.buyDate_onehot = tf.one_hot(self.buyDate, tf.size(self.buyDate), dtype = tf.float32)

		#定义卖方网络
		self.sellDataInput = tf.placeholder(tf.float32, [None, dateSize, 7], 'buyDataInput')
		self.sellQout = self.__buildNetWork(self.sellDataInput, 'sellside')
		self.sellDate = tf.argmax(self.sellQout, 0, name = 'sellPredict')
		tf.summary.scalar('sellDate', self.sellDate)
		self.sellDate_onehot = tf.one_hot(self.sellDate, tf.size(self.sellDate), dtype = tf.float32)

		#定义loss
		self.buyDataNextPrice = tf.placeholder(tf.float32, [None], 'buyDataNextPrice')
		self.sellDataNextPrice = tf.placeholder(tf.float32, [None], 'sellDataNextPrice')
		with tf.name_scope('sp_loss'):
			buyPrice = tf.reduce_max(self.buyDate_onehot * self.buyDataNextPrice)
			tf.summary.scalar('buyPrice', buyPrice)
			sellPrice = tf.reduce_max(self.sellDate_onehot * self.sellDataNextPrice)
			tf.summary.scalar('sellPrice', sellPrice)

			self.profit = 10000 / buyPrice * sellPrice
			tf.summary.scalar('tb_profit', self.profit)
			self.loss = -tf.log(self.profit)
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


	def learn(self, buyDataInput, buyDataNextPrice, sellDataInput, sellDataNextPrice):
		buydate, buyQ, selldate, sellQ, buyDate_onehot, sellDate_onehot = self.session.run(
			[self.buyDate, self.buyQout, self.sellDate, self.sellQout, self.buyDate_onehot, self.sellDate_onehot],
										feed_dict = {self.buyDataInput:buyDataInput,
											self.buyDataNextPrice: buyDataNextPrice, 
											self.sellDataInput: sellDataInput, 
											self.sellDataNextPrice: sellDataNextPrice})
		a = 100
		b = 100 +a


		a, b, size = self.session.run([self.trainer, self.summaryMerged, self,size],
									 feed_dict = {self.buyDataInput:buyDataInput,
											self.buyDataNextPrice: buyDataNextPrice, 
											self.sellDataInput: sellDataInput, 
											self.sellDataNextPrice: sellDataNextPrice})