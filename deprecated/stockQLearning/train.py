import StockData
import StockQLearning
import numpy


#dateSize = 188 #该数值要符合卷积的要求
dateSize = 30
batchSize = 100
moneyBase = 10000

idlist = StockData.getStockRandomList()
stockLearning = StockQLearning.StockLearning()
stockBuffer = StockData.StockBuffer(10000)
log = StockData.Log()

with stockLearning.session:
	#log.restoreVariables()

	learnCount = 0
	totalStep = 0
	for id in idlist:
		print('stock ', id, ':')
		stockData = StockData.StockData(id, dateSize)
		stockSize = stockData.getDataLength() - 2 #为结算空余两天 
		if stockSize <= 0 :
			continue

		index = 0 #扫描到多少天
		while index + dateSize <= stockSize - 2:
			data = stockData.reset(index)
			for step in range(245) : #最多操作1年
				randomRate = 0.05 if totalStep > 10000 else (1.05 - totalStep/3000.0)
				action = stockLearning.chooseAction(data, randomRate)

				nextData, reward, done = stockData.takeAction(action)
				stockBuffer.saveAction(data, action, reward, nextData)
				data = nextData

				log.addProfit(reward/100)
				totalStep += 1

				if done == True:
					log.logBreak()
					break

				if totalStep > 0 and totalStep % 1000 == 0: #每1000步进行学习
					dataArray, actionArray, rewardArray, nextDataArray = \
						stockBuffer.getSampleData(batchSize)
					stockLearning.learn(dataArray, actionArray, rewardArray, nextDataArray, learnCount)
					learnCount += 1

				if totalStep > 0 and totalStep % 980 == 0: #4年
					log.printInfo(totalStep)

				#if totalStep > 0 and totalStep % 100000 == 0:
				#	log.saveVaribles()

			#强制结算
			index  = stockData.getCurrentIndex()
			log.addEndMoney(stockData.getCurrentMoney())
