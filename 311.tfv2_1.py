import time
import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf


### data preprocessing ###

def read(filename, date_idx, date_parse, year=None, bucket=7):
    days_in_year = 365

    freq = {}
    if year != None:
        for period in range(0, int(days_in_year / bucket)):
            freq[period] = 0

    with open(filename, 'r', encoding='utf-8') as csvfile: # 注意編碼
        csvreader = csv.reader(csvfile)
        next(csvreader) # 使下方的for loop不要讀到標題行
        for row in csvreader:
            if row[date_idx] == '':
                continue
            t = time.strptime(row[date_idx], date_parse)
            if year == None:
                if not t.tm_year in freq:
                    freq[t.tm_year] = {}
                    for period in range(0, int(days_in_year / bucket)):
                        freq[t.tm_year][period] = 0

                if t.tm_yday < (days_in_year - 1):
                    freq[t.tm_year][int(t.tm_yday / bucket)] += 1

            else:
                if t.tm_year == year and t.tm_yday < (days_in_year - 1):
                    freq[int(t.tm_yday / bucket)] += 1

    return freq

freq = read('311.csv', 1, '%m/%d/%Y %H:%M:%S %p', 2014)
#print(freq)

X_train = np.asarray(list(freq.keys()))
Y_train = np.asarray(list(freq.values()))
#print("number of samples:", str(len(X_train)))
maxY = np.max(Y_train)
nY_train = Y_train / maxY
#plt.scatter(X_train, nY_train)
#plt.show()

######


### Train the model ###

learning_rate = 1.5
training_epochs = 5000

X_train = tf.constant(X_train, dtype=tf.float32)
nY_train = tf.constant(nY_train, dtype=tf.float32)
'''
mu = tf.Variable(1., name="mu")
sig = tf.Variable(1., name="sig")

def model(X, _mu, _sig):
    return tf.math.exp(tf.math.divide(tf.math.negative(tf.math.pow(tf.math.subtract(tf.cast(X, tf.float32), _mu), 2.)), \
                       tf.math.multiply(2., tf.math.pow(_sig, 2.))))

def cost(y_hat, y):
    return tf.math.square(tf.cast(y_hat, tf.float32) - tf.cast(y, tf.float32))

optimizer = tf.keras.optimizers.SGD(learning_rate)

def train_step(inputs, _mu, _sig, outputs):
    with tf.GradientTape() as t:
        current_cost = cost(model(inputs, _mu, _sig), outputs)

    grads = t.gradient(current_cost, [_mu, _sig])
    optimizer.apply_gradients(zip(grads, [_mu, _sig]))

    return current_cost

for epoch in tqdm(range(training_epochs)):
    for i in range(0, len(X_train)):
        _cost = train_step(X_train[i], mu, sig, nY_train[i])

mu_val = mu
sig_val = sig

print(mu_val.numpy())
print(sig_val.numpy())

plt.scatter(X_train, Y_train)
trY2 = maxY * (np.exp(-np.power(X_train - mu_val, 2.) / (2 * np.power(sig_val, 2.))))
plt.plot(X_train, trY2, 'r')
plt.show()

'''
class Model:
    def __init__(self):
        self.mu = tf.Variable(1., name="mu")
        self.sig = tf.Variable(1., name="sig")

    def __call__(self, X):
        return tf.math.exp(tf.math.divide(tf.math.negative(tf.math.pow(tf.math.subtract(tf.cast(X, tf.float32), self.mu), 2.)), \
                           tf.math.multiply(2., tf.math.pow(self.sig, 2.))))

def cost(y_hat, y):
    return tf.math.reduce_sum(tf.math.square(tf.cast(y_hat, tf.float32) - tf.cast(y, tf.float32)))

optimizer = tf.keras.optimizers.SGD(learning_rate)

def train_step(model, inputs, outputs):
    with tf.GradientTape() as t:
        current_cost = cost(model(inputs), outputs)

    grads = t.gradient(current_cost, [model.mu, model.sig])
    optimizer.apply_gradients(zip(grads, [model.mu, model.sig]))

    return current_cost

model = Model()
for epoch in tqdm(range(training_epochs)):
    for i in range(0, len(X_train)):
        _cost = train_step(model, X_train[i], nY_train[i])

mu_val = model.mu
sig_val = model.sig

print(mu_val.numpy())
print(sig_val.numpy())
