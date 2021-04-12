import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

learning_rate = 0.01
training_epochs = 1000

x1 = np.random.normal(-4, 2, 1000)
x2 = np.random.normal(4, 2, 1000)
xs = np.append(x1, x2)
ys = np.asarray([0.]*len(x1) + [1.]*len(x2))
xs = tf.convert_to_tensor(xs, dtype=tf.float32)
ys = tf.convert_to_tensor(ys, dtype=tf.float32)

plt.scatter(xs, ys)

w = tf.Variable([[0.], [0.]], name="parameter")

@tf.function
def model(X, w):
    return tf.math.sigmoid(w[1]*tf.cast(X, tf.float32) + w[0])


@tf.function
def cost(y_predict, y_true):
    return tf.math.reduce_mean(tf.cast(-y_true, tf.float32) * tf.math.log(tf.cast(y_predict, tf.float32)) - \
                               (1 - tf.cast(y_true, tf.float32))*tf.math.log(1 - tf.cast(y_predict, tf.float32)))

optimizer = tf.keras.optimizers.SGD(learning_rate)

@tf.function
def training_step(inputs, w, outputs):
    with tf.GradientTape() as t:
        current_cost = cost(model(inputs, w), outputs)
    grads = t.gradient(current_cost, [w])
    optimizer.apply_gradients(zip(grads, [w]))

    return current_cost

for epoch in tqdm(range(training_epochs)):
    cost = training_step(xs, w, ys)


w_val = w.numpy()
print(w_val)

all_xs = np.linspace(-10, 10, 100)
plt.plot(all_xs, tf.math.sigmoid((all_xs*w_val[1] + w_val[0])), c='r')
plt.show()

