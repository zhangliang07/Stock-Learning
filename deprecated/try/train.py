import StockData
import learning2
import numpy
import glob


dateSize = 188 #该数值要符合卷积的要求


#获取目录下所有的股票代码
filenameList = glob.glob('F:\deeplearning\数据集\上证A股\*.SH.CSV')
idlist = []
for name in filenameList:
	list = name.split('\\')
	idstr = list[-1].split('.')[0]
	idlist.append(int(idstr))


stockLearning = learning2.StockLearning(dateSize)

with stockLearning.session:
	totalStep = 0
	for id in idlist:
		print('stock ', id, ':')
		stockData = StockData.StockData(id, dateSize)
		stockSize =  (stockData.getDataLength() - 201 - dateSize)//100

		for i in range(stockSize):
			index, data, nextPrice = stockData.getDataRandomBatch(i*100, 200)
			#price = data[:,187, 1]
			#rate = numpy.reshape((price - nextPrice)/price * 100, [-1,1])
			#stockLearning.learn(data, rate, totalStep)

			price = numpy.reshape(nextPrice, [-1, 1])
			stockLearning.learn(data, price, totalStep)

			totalStep += 1

			if i % 1000 == 0:
				print('step ', i, ', total step ', totalStep)

