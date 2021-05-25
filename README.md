# FEAR - Feature Engineered Augumented Returns

This project uses a multivariate deep neural network to predict stock price direction.

```
[channels.alpaca] INFO 2021-05-23 21:19:58,917 Fetched 5457 bars for 'iht' from 2021-05-09T21:19:58Z to 2021-05-20T21:19:58Z with freq TimeFrame.Minute
[dnn] INFO 2021-05-23 21:19:58,918 Train: 2021-05-10 08:00:00+00:00 - 2021-05-18 16:05:00+00:00 (4365, 1)
[dnn] INFO 2021-05-23 21:19:58,918 Test: 2021-05-18 16:06:00+00:00 - 2021-05-20 21:19:00+00:00 (1092, 1)
[dnn] INFO 2021-05-23 21:19:59,197 Building model using features ['lag_3', 'lag_5', 'return', 'distance', '14 period RSI', 'volatility', 'lag_1', 'momentum', 'lag_2', 'lag_4']
[dnn] INFO 2021-05-23 21:20:05,270 Training model on 4161 records
131/131 [==============================] - 0s 529us/step - loss: 5.7715 - accuracy: 0.7642
[dnn] INFO 2021-05-23 21:20:13,379 Trades made: 240
[dnn] INFO 2021-05-23 21:20:13,387 Returns [iht]:
                            return  strategy
timestamp                                   
2021-05-20 21:19:00+00:00  0.99661  7.966899
```
