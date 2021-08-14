import numpy
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" #不使用Gpu反而快，特别是rnn


class StockLearning:
	def __init__(self):
		self.__dateSize = 30
		self.__actionNum = 3
		self.__learningRate = 0.01 #这个参数很重要

		#定义主网络和目标网络
		self.dataInput = tf.placeholder(tf.float32, [None, self.__dateSize, 8], 'dataInput')
		self.batchSize = tf.placeholder(tf.int32, []) #仅在使用rnn的时候有用
		#normalInput = tf.layers.batch_normalization(self.dataInput, axis = -1) #数据正则化，可选用

		self.mainQout = self.__buildNetWork(self.dataInput, 'mainNetWork')
		self.targetQout = self.__buildNetWork(self.dataInput, 'targetNetwork')
		#self.mainQout = self.__buildNetWork_rnn(self.dataInput, 'mainNetWork')
		#self.targetQout = self.__buildNetWork_rnn(self.dataInput, 'targetNetwork')


		#监视
		mainTest = tf.reduce_mean(self.mainQout, axis = 0)
		targetTest = tf.reduce_mean(self.targetQout, axis = 0)
		tf.summary.scalar('mainQout[0]', mainTest[0])
		tf.summary.scalar('mainQout[1]', mainTest[1])
		tf.summary.scalar('mainQout[2]', mainTest[2])
		tf.summary.scalar('targetQout[0]',targetTest[0])
		tf.summary.scalar('targetQout[1]', targetTest[1])
		tf.summary.scalar('targetQout[2]', targetTest[2])

		#计算下一个环境下的主网络的最佳行为的目标网络的对应Q值
		with tf.name_scope('sp_targetQ'):
			self.predict = tf.argmax(self.mainQout, 1, name = 'mainPredict')
			actionsNext_onehot = tf.one_hot(self.predict, 3, dtype = tf.float32)
			self.targetQ = tf.reduce_sum(self.targetQout * actionsNext_onehot, axis = 1)
			tf.summary.scalar('tb_targetQNext', tf.reduce_mean(self.targetQ))

		#计算现在环境下给定行为的主网络的对应Q值
		with tf.name_scope('sp_mainQ'):
			self.actions = tf.placeholder(tf.int32, [None], 'actions')
			actions_onehot = tf.one_hot(self.actions, self.__actionNum, dtype = tf.float32)
			self.mainQ = tf.reduce_sum(self.mainQout * actions_onehot, axis = 1)
			tf.summary.scalar('tb_mainQ', tf.reduce_mean(self.mainQ))

		#从副DQN输入加工后tartgetQ, 计算两个主副DQN的Q值相似度
		with tf.name_scope('sp_loss'):
			self.processedTargetQ = tf.placeholder(tf.float32, [None], 'processedTargetQ')
			tf.summary.scalar('tb_processedTargetQ', tf.reduce_mean(self.processedTargetQ))
			self.loss = tf.reduce_mean(tf.squared_difference(self.mainQ, self.processedTargetQ))
			tf.summary.scalar('tb_loss', self.loss)

		self.trainer = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

		#定义target网络缓慢学习main网络的参数
		with tf.name_scope('sp_replace'):
			self.replace = []
			mainVariables = tf.trainable_variables('mainNetWork')
			targetVariables = tf.trainable_variables('targetNetwork')
			for t, e in zip(targetVariables, mainVariables):
				self.replace.append(
					tf.assign(t, t * (1-self.__learningRate) + e * self.__learningRate))

		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())

		#summary
		self.summaryMerged = tf.summary.merge_all()
		self.summaryWriter = tf.summary.FileWriter('/tf_log/train', self.session.graph)


	def learn(self, dataInput, actions, rewards, nextDataInput, step):
		#获取下一步的行为的Q值
		targetQ = self.session.run(self.targetQ,
						feed_dict = {self.dataInput: nextDataInput, self.batchSize: 100})
		processedTargetQ = targetQ * 0.99 + rewards

		_, summary = self.session.run([self.trainer, self.summaryMerged],
																feed_dict = {self.dataInput: dataInput,
														self.processedTargetQ: processedTargetQ,
													 self.actions: actions,
													  self.batchSize: 100})

		self.summaryWriter.add_summary(summary, step)

		self.session.run(self.replace)


	def chooseAction(self, dataInput, randomRate):
		if numpy.random.random() > randomRate:
			num = self.session.run(self.predict,
												 feed_dict = {self.dataInput:[dataInput], self.batchSize: 1})
			predict = int(num[0])
		else:
			predict = numpy.random.randint(0, 3)

		return predict
		
	
	def __buildNetWork(self, input, name):
		#对于这里的三层卷积网络，常用的设置[k,s]有
		#[3,1],[4,2],[3,2]或
		#[2,1],[3,2],[4,2]或
		#[2,1],[3,2],[5,3]或
		#[3,1],[3,1],[3,1]好像这几种都差不多
		#提高网络单元数量有助于改善结果, 但也会导致过拟合？
		with tf.name_scope(name):
			init = tf.random_normal_initializer(mean=0, stddev=0.3)

			conv1 = tf.layers.conv1d(input, filters = 8, kernel_size = 3, strides = 1,
													activation = tf.nn.relu, kernel_initializer = init)

			conv2 = tf.layers.conv1d(conv1, filters = 16, kernel_size = 3, strides = 1,
													activation = tf.nn.relu, kernel_initializer = init)

			conv3 = tf.layers.conv1d(conv2, filters = 32, kernel_size = 3, strides = 1,
													activation = tf.nn.relu, kernel_initializer = init)

			#streamAC, streamVC = tf.split(conv3, 2, 1)
			#streamA = tf.layers.flatten(streamAC)
			#streamV = tf.layers.flatten(streamVC)
			stream = tf.layers.flatten(conv3)

			layer5A_i = tf.layers.dense(stream, units = 600,
													activation = tf.nn.relu, kernel_initializer = init)
			layer5V_i = tf.layers.dense(stream, units = 600,
													activation = tf.nn.relu, kernel_initializer = init)
			#layer5A_d = tf.layers.dropout(layer5A_i, 0.6)
			#layer5V_d = tf.layers.dropout(layer5V_i, 0.6)

			layer5A = tf.layers.dense(layer5A_i, units = 3,
													activation = tf.nn.relu, kernel_initializer = init)
			layer5V = tf.layers.dense(layer5V_i, units = 1,
													activation = tf.nn.relu, kernel_initializer = init)

			Qout = layer5V + (layer5A - tf.reduce_mean(layer5A, axis = 1, keepdims = True))

			return Qout


	def __buildNetWork_rnn(self, input, name):
		with tf.name_scope(name):
			init = tf.random_normal_initializer(mean=0, stddev=0.3)

			#BasicLSTMCell和GRUCell返回的final_state的shape不同
			#lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(8, activation = tf.nn.tanh,  kernel_initializer = init) 
			lstm_cell = tf.nn.rnn_cell.GRUCell(8, activation = tf.nn.tanh, kernel_initializer = init)
			lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 3)  #只用一层rnn效果很差
			init_state = lstm_cell.zero_state(self.batchSize, dtype=tf.float32) #这个batchsize要和前面呼应

			outputs, final_state = tf.nn.dynamic_rnn(
				lstm_cell, input, initial_state=init_state, time_major=False, scope =  name + 'rnn')

			#BasicLSTMCell使用final_state[-1][1], GRUCell使用final_state[-1]
			stream = tf.layers.flatten(final_state[-1]) 

			layer5A_i = tf.layers.dense(stream, units = 600,
													activation = tf.nn.relu, kernel_initializer = init)
			layer5V_i = tf.layers.dense(stream, units = 600,
													activation = tf.nn.relu, kernel_initializer = init)
			#layer5A_d = tf.layers.dropout(layer5A_i, 0.6)
			#layer5V_d = tf.layers.dropout(layer5V_i, 0.6)

			layer5A = tf.layers.dense(layer5A_i, units = 3,
													activation = tf.nn.relu, kernel_initializer = init)
			layer5V = tf.layers.dense(layer5V_i, units = 1,
													activation = tf.nn.relu, kernel_initializer = init)

			Qout = layer5V + (layer5A - tf.reduce_mean(layer5A, axis = 1, keepdims = True))

			return Qout