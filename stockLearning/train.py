import StockData
import PolicyModel


#the days of past stock data to be watched, this will change the layers' output shape
dateSize = 30

#maximum of days to deal
operationCount = 30

#the original money the model have
moneyBase = 10000


idlist = StockData.getStockRandomList()
stockLearning = PolicyModel.PolicyCnnNetwork(6)
log = StockData.Log()


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
		while opCount < operationCount : #maximum of days to deal
			randomRate = 0.6 if totalStep < 3000 else 0.1 #if totalStep < 10000 else 0.0
			action = stockLearning.predict(data, randomRate)

			nextData, done = stockData.takeAction(action)
			dataList.append(data)
			actionList.append(action)
			data = nextData
			opCount += 1

			if done == True:
				log.logBreak()
				break

		#强制结算
		stockData.takeAction(2)
		index  = stockData.getCurrentIndex()
		money = stockData.getCurrentMoney()

		#设置奖励:
		#reward should be a coefficient in (--, ++)
    #if rewead is 0.3, it means the final money is 1.3 times of original money
    #if rewead is -0.2, it means the final money is 0.8 times of original money
		reward = (money - moneyBase)/ moneyBase # couldn't to '/ opCount'
		log.addProfit(reward)
		log.addEndMoney(money)
		rewardList = [reward] * len(actionList)

		#the more previous action is more important, this step can skip
		for i in range(0, len(actionList)):
			rewardList[i] = reward
			reward *= 0.98    #factor

		stockLearning.learn(dataList, actionList, rewardList, totalStep)

		totalStep += 1

		if totalStep % 50 == 0:
			log.printInfo(totalStep)
