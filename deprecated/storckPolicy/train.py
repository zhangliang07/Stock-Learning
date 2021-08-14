import StockData
import StockPolicyLearning
import numpy


#dateSize = 188 #该数值要符合卷积的要求
dateSize = 30
operationCount = 20
moneyBase = 10000

idlist = StockData.getStockRandomList()
stockLearning = StockPolicyLearning.StockLearning()
log = StockData.Log()


with stockLearning.session:
	#log.restoreVariables()

	totalStep = 0
	for id in idlist:
		print('stock ', id, ':')
		stockData = StockData.StockData(id, dateSize)
		stockSize = stockData.getDataLength() - 2 #为结算空余两天 
		if stockSize <= 0 :
			continue

		index = 0 #扫描到多少天
		while index + dateSize <= stockSize - 2:
			dataList = []
			actionList = []
			opCount = 0
			data = stockData.reset(index, moneyBase)
			while opCount < 20 : #最多操作20天
				randomRate = 0.05 if totalStep > 10000 else (1.05 - totalStep/3000.0)
				action = stockLearning.chooseAction(data, randomRate)

				nextData, reward, done = stockData.takeAction(action)
				dataList.append(data)
				actionList.append(action)
				data = nextData
				opCount += 1

				if done == True:
					log.logBreak()
					break

			#强制结算
			index  = stockData.getCurrentIndex()
			money = stockData.getCurrentMoney()

			#设置奖励:
			profit = (money - moneyBase) / moneyBase * 100 / opCount
			rewardArray = numpy.empty([opCount])
			rewardArray.fill(profit) #每笔操作的平均盈利
			log.addProfit(profit)
			log.addEndMoney(money)

			actionArray = numpy.array(actionList)
			dataArray = numpy.array(dataList)
			stockLearning.learn(dataArray, actionArray, rewardArray, totalStep)

			totalStep += 1

			if totalStep % 50 == 0:
				log.printInfo(totalStep)

