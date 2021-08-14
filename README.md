# stockLearning
AI for stock market

It operate automatically in the stock market by reinforcement learning.
The model can make suggestions to buy in or sell for tomorrow, based on past several days' stock data.

It seems able to make profit.
But I think nobody will apply it for real stock market.
Because the model based on the trainment only with past years' stock data.
The performance for current real market is uncertain.

Several years ago it was written by tensorflow. Recently I rewrote it with pytorch.

As I think the most important part is to write the correct loss function.
Firstly it should be a convex function.
Secondly it approaches the minimum (or negative maximum) along your desired direction or at your desired point.
Thirdly it covers the input domain of your usage.

For reinforcement learning, it's necessary to add random actions to avoid local convergence.
