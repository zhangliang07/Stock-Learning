# Stock Learning
AI for stock market

Please run the "stockLearning/train.py".
If the model become steady and non-actions, the training falls in local convergence.
Please retry the script again.

It operates automatically in stock markets by reinforcement learning.
The model can suggest to buy in or sell for tomorrow, based on past several days' stock data.

It seems able to make profits. But I think nobody will apply it to real stock markets.
Because the model is based on the training only with past years' stock data.
The performance for current real data is uncertain.

Several years ago I wrote it by TensorFlow. Recently I rewrote it with PyTorch.

As I think the most important part is to write a correct loss function.
Firstly it should be a convex function.
Secondly it approaches the minimum (or negative maximum) along your desired direction or at your desired point.
Thirdly it covers the input domain of your usage.
For reinforcement learning, it's necessary to add random actions to avoid local convergence.

The model simply uses 5 layers (3 convolutions and 2 linears).
(As I tested, 2 convolutions and 2 linears are already enough.)
Now the model's mean profit is about 20% per year (0.2% of transaction tax).
Maybe the performance will improve with more amount of neural units.
