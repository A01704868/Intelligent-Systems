import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

dataset = pd.read_csv("dataset.csv")

dataset = dataset.filter(['Volume','20_sma', 'upper_bb','lower_bb', 'rsi','macd', 'signal', 'hist','Close', 'target'])
n = len(dataset)
train_df = dataset[0:int(n*0.7)]
val_df = dataset[int(n*0.7):int(n*0.9)]
test_df = dataset[int(n*0.9):]
scaler = MinMaxScaler()
scaler.fit(train_df)
train = scaler.transform(train_df)
val = scaler.transform(val_df)
test = scaler.transform(test_df)
X_train = train[:,:-2]
y_train = train[:,[8,9]]

X_val = val[:,:-2]
y_val = val[:,[8,9]]

X_test = test[:,:-2]
y_test = test[:,[8,9]]

window_length = 4
batch_size = 32
num_features = 9
train_generator = TimeseriesGenerator(X_train, y_train, length=window_length, sampling_rate=1, batch_size=batch_size)
val_generator = TimeseriesGenerator(X_val, y_val, length=window_length, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(X_test, y_test, length=window_length, sampling_rate=1, batch_size=batch_size)

model = Sequential()

model.add(LSTM(units=1000, return_sequences=True, input_shape=(window_length,num_features)))
model.add(LSTM(units=1000,return_sequences=True))
model.add(LSTM(units=100,return_sequences=False))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=2, activation='linear'))

model.summary()

model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
history = model.fit(train_generator, epochs=50, validation_data=val_generator, shuffle=False)
plt.plot(history.history['loss'])
plt.show()
model.evaluate(test_generator, verbose=1)
predictions = model.predict(test_generator)

inv_transform = pd.DataFrame(data=X_test, columns=['Volume','20_sma', 'upper_bb','lower_bb', 'rsi','macd', 'signal', 'hist'])
predicted_stock_price = pd.DataFrame(data=predictions, columns=['predictedClose', 'predictedTarget'])

inv_transform = inv_transform.join(predicted_stock_price)

inv_transform = scaler.inverse_transform(inv_transform)
predictions = pd.DataFrame(data=inv_transform, columns=['Volume','20_sma', 'upper_bb','lower_bb', 'rsi','macd', 'signal', 'hist', 'predictedClose', 'predictedTarget'])
target = pd.DataFrame(data=X_test, columns=['Volume','20_sma', 'upper_bb','lower_bb', 'rsi','macd', 'signal', 'hist'])
y_test = pd.DataFrame(data=y_test, columns=['Close', 'target'])
target = target.join(y_test)
target = scaler.inverse_transform(target)
target = pd.DataFrame(data=target, columns=['Volume','20_sma', 'upper_bb','lower_bb', 'rsi','macd', 'signal', 'hist', 'Close', 'target'])
predictions = predictions.join(target['Close'])
predictions = predictions.join(target['target'])
print(predictions.head())

plt.plot(predictions['predictedClose'], color='red')
plt.plot(predictions['Close'], color='blue')
plt.show()
plt.plot(predictions['predictedTarget'], color='green')
plt.plot(predictions['target'], color='orange')
plt.show()
plot_all = scaler.transform(dataset)
plot_all_X = plot_all[:,:-2]
plot_all_Y = plot_all[:,[8,9]]
plot_all_gen = TimeseriesGenerator(plot_all_X, plot_all_Y, length=window_length, sampling_rate=1, batch_size=batch_size)
all_predict = model.predict(plot_all_gen)
line_predict = pd.DataFrame(plot_all_X, columns=['Volume','20_sma', 'upper_bb','lower_bb', 'rsi','macd', 'signal', 'hist'])
predicted__lvls = pd.DataFrame(data=all_predict, columns=['predictedClose', 'predictedTarget'])
line_predict = line_predict.join(predicted__lvls)
line_predict = scaler.inverse_transform(line_predict)
predictions2 = pd.DataFrame(data=line_predict, columns=['Volume','20_sma', 'upper_bb','lower_bb', 'rsi','macd', 'signal', 'hist', 'predictedClose', 'predictedTarget'])
plt.plot(predictions2['predictedClose'], color='purple')
plt.plot(dataset['Close'], color='black')
plt.show()
plt.plot(dataset['target'], color='purple')
plt.plot(predictions2['predictedTarget'], color='black')
plt.show()
realTime = pd.read_csv("real_dataset.csv")
realTime = realTime.filter(['Volume','20_sma', 'upper_bb','lower_bb', 'rsi','macd', 'signal', 'hist','Close', 'target'])
realTime = realTime.iloc[22:,:]
real_time = scaler.transform(realTime)
real_time_X = real_time[:,:-2]
real_time_y = real_time[:,[8,9]]
generator = TimeseriesGenerator(real_time_X, real_time_y, length=window_length, sampling_rate=1, batch_size=1)
real_predict = model.predict(generator)
real_time_X = real_time_X[4:]
predict_real = pd.DataFrame(real_time_X, columns=['Volume','20_sma', 'upper_bb','lower_bb', 'rsi','macd', 'signal', 'hist'])
predicted_Rlvls = pd.DataFrame(data=real_predict, columns=['predictedClose', 'predictedTarget'])
predict_real = predict_real.join(predicted_Rlvls)
predict_real = scaler.inverse_transform(predict_real)
predict_real = pd.DataFrame(data=predict_real, columns=['Volume','20_sma', 'upper_bb','lower_bb', 'rsi','macd', 'signal', 'hist', 'predictedClose', 'predictedTarget'])
print(predict_real[['predictedClose', 'predictedTarget']])