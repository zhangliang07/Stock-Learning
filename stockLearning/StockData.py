import csv
import numpy
import glob		#文件搜索库
import os


#_rootDataPath = os.path.realpath(os.getcwd() + '/../../stockData/') + '\\'

#if the code downloaded from github, use this path, due to the whole stock data wasn't uploaded.
_rootDataPath = os.path.realpath(os.getcwd() + '/../stockData/') + '\\'


#获取目录下所有的股票代码
def getStockRandomList():
	stockFile = _rootDataPath + '上证A股/*.SH.CSV'
	#stockFile = _rootDataPath + '深证A股/*.SZ.CSV'
	#stockFile = _rootDataPath + '*.SH.CSV'		//used for github

	filenameList = glob.glob(stockFile)
	idlist = []
	for name in filenameList:
		list = name.split('\\')
		idstr = list[-1].split('.')[0]
		#if int(idstr) > 300000:	#剔除不是创业板的
		idlist.append(int(idstr))

	numpy.random.shuffle(idlist)
	return idlist


class StockData:
	def __init__(self, stockId, dateSize):
		self.__stockId = stockId
		self.__dateSize = dateSize #数据集固定的采样大小

		#获取文件名
		if stockId > 0 and stockId < 600000:
			stockFile = format('深证A股/%0*d.SZ.CSV' % (6, stockId))
			fileName = _rootDataPath + stockFile
		else:
			stockFile = format('上证A股/%0*d.SH.CSV' % (6, stockId))
			fileName = _rootDataPath + stockFile

		file =  open(fileName, 'r')
		reader = csv.DictReader(file)

		#读取数据
		array = []
		days = []
		for row in reader:
			rowData = (row['开盘价(元)'], row['收盘价(元)'],
							row['最高价(元)'], row['最低价(元)'],
							row['涨跌幅(%)'], row['成交量(股)'],
							row['市盈率'], row['市净率'])

			#去掉数据中有na的行
			if 'N/A' not in rowData:
				days.append(row['日期'])
				array.append(rowData)

		file.close()
		self.__days = numpy.array(days)
		self.__data = numpy.array(array, dtype = numpy.float32)

	
	def getDateRange(self):
		return self.__days[0], self.__days[-1]


	def getDataLength(self):
		return len(self.__data)


	#返回一条股票数据和下一天的股票数据。不提取最后一行数据。shape = [dateSize, 7]
	def getData(self, startIndex):
		if len(self.__data) - 1 > startIndex + self.__dateSize:
			data = self.__data[startIndex : startIndex + self.__dateSize, :]
			nextData = self.__data[1 + startIndex : 1 + startIndex + self.__dateSize, :]
			return data, nextData
		else:
		 return None, None


	#返回乱序的股票数据和下一天的股票数据，以及他们的索引。不提取最后一行数据
	def getDataRandomBatch(self, startIndex, batchSize):
		if len(self.__data) > startIndex + self.__dateSize + batchSize:
			allIndex = range(startIndex, startIndex + batchSize)
			indexList = numpy.array(allIndex)
			numpy.random.shuffle(indexList)

			data = []
			nextData = []
			for i in indexList:
				data.append(self.__data[i : i + self.__dateSize, :])
				nextData.append(self.__data[1 + i: 1 + i + self.__dateSize, :]) #下一天的股票数据

			return indexList, numpy.array(data), numpy.array(nextData)
		else:
			return None, None, None


	#重置内部仓位
	def reset(self, index, moneyBase = 10000):
		self.__index = index #目前操作的天数 
		self.__moneyBase = moneyBase #1万元本钱
		self.__money = moneyBase
		self.__lastMoney = moneyBase #1上一次空仓时的资金
		self.__stock = 0 
		self.__state = 0 #仓位状态。0空仓，1满仓

		if self.__index + self.__dateSize >= len(self.__data) -1:
			return None  #读到末尾结束
		else:
			return self.__data[self.__index : self.__index + self.__dateSize, :]


	def getCurrentIndex(self):
		return self.__index

	
	def getCurrentMoney(self):
		nextPrice = self.__data[self.__index + self.__dateSize, 0]
		return self.__money + self.__stock * nextPrice


	def takeAction(self, action):
		if self.__index + self.__dateSize >= len(self.__data) -2: #为最后一次结算预留一位
			return None, None, True #读到末尾结束

		self.__index += 1
		nextData = self.__data[self.__index : self.__index + self.__dateSize, :]
		nextPrice = nextData[-1, 0]

		#定义的操作是0为买入，1为不动，2为卖出
		reward = 0.0	#仅在卖出时结算奖励
		if action < 0.5 and self.__state == 0:
			count = (self.__money * 0.999) //nextPrice
			self.__stock += count
			self.__money -= (count * nextPrice) * 1.001 #手续费
			self.__state = 1
		elif action > 1.5 and self.__state == 1:
			self.__money += self.__stock * nextPrice * 0.998 #手续费
			self.__stock = 0
			self.__state = 0
			reward = (self.__money - self.__lastMoney) / self.__lastMoney * 10000.0 #奖励为利润万分数
			self.__lastMoney = self.__money #为下一次结算更新空仓时的资金

		if self.__index + self.__dateSize >= len(self.__data) - 2:	#判断结束，为最后一次结算预留一位
			done = True
		elif self.__state == 0 and self.__money < self.__moneyBase * 0.9: #亏损到一定百分比结束
			done = True
		else:
			done = False

		return nextData, reward, done


class StockBuffer:
		def __init__(self, bufferSize = 50000):
			self.__bufferSize = bufferSize #数据集缓存容量
			self.__size = 0
			self.__dataList = []
			self.__nextDataList = []
			self.__actionList = []
			self.__rewardList = []


		#输入变量为当前股票信息，下一天的股票信息，当前采取的行动(0买入，1不动，2卖出), 行动获得的利润率
		def saveAction(self, data, action, reward, nextData):
			self.__dataList.append(data)
			self.__nextDataList.append(nextData)
			self.__actionList.append(action)
			self.__rewardList.append(reward)

			if self.__size > self.__bufferSize:
				self.__dataList.pop(0)
				self.__nextDataList.pop(0)
				self.__actionList.pop(0)
				self.__rewardList.pop(0)
			else:
				self.__size += 1


		def getSampleData(self, size):
			size = size if size < self.__size else self.__size
			allIndex = range(self.__size)
			indexList = numpy.random.choice(allIndex, size)

			dataList = []
			nextDataList = []
			actionList = []
			rewardList = []
			for i in indexList:
				dataList.append(self.__dataList[i])
				nextDataList.append(self.__nextDataList[i])
				actionList.append(self.__actionList[i])
				rewardList.append(self.__rewardList[i])

			return numpy.array(dataList), numpy.array(actionList),\
				numpy.array(rewardList), numpy.array(nextDataList)



#记录股票操作的类
class Log:
	def __init__(self):
		self.__breakTimes = 0
		self.__profitList = []
		self.__totalProfit = []
		self.__endMoneyList = []
		self.__totalEndMoney = []


	def logBreak(self):
		self.__breakTimes += 1


	def addProfit(self, profit):
		self.__profitList.append(profit)


	def addEndMoney(self, money):
		self.__endMoneyList.append(money)


	def printInfo(self, step):
		if len(self.__profitList) == 0 or len(self.__endMoneyList) == 0:
			return

		meanProfit = numpy.mean(self.__profitList)
		self.__totalProfit.append(meanProfit)
		if len(self.__totalProfit) > 100:
			self.__totalProfit.pop(0)

		meanEndMoney = numpy.mean(self.__endMoneyList)
		self.__totalEndMoney.append(meanEndMoney)
		if len(self.__totalEndMoney) > 100:
			self.__totalEndMoney.pop(0)

		#print('total step %d, mean porfit: %f, totalProfit: %f, mean endMoney: %f, totalEndMoney: %f, break times: %d'
		#	% (step, meanProfit, numpy.mean(self.__totalProfit),
		# meanEndMoney, numpy.mean(self.__totalEndMoney), self.__breakTimes))
		print('total step %d, mean porfit: %f, mean endMoney: %f, break times: %d'
			% (step, meanProfit, meanEndMoney, self.__breakTimes))
		self.__profitList.clear()
		self.__endMoneyList.clear()
		self.__breakTimes = 0


	def restoreVariables(self):
		pass


	def saveVaribles(self, step):
		pass