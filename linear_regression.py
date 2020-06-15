import tensorflow as tf
import matplotlib.pyplot as plt


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=None, mask=None):
        return self.dense(inputs)


N = 50
real_a = 0.3
real_b = 0.5
X = tf.random.uniform(shape=(N, 1), minval=0, maxval=1)
Y = real_a * X + real_b

model = Linear()
epoch = 100
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

for e in range(epoch):
    with tf.GradientTape() as tape:
        y = model(X)
        loss = tf.reduce_sum(tf.square(y - Y))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    plt.cla()
    plt.plot(tf.reshape(X, shape=(N,)).numpy(), tf.reshape(Y, shape=(N,)).numpy(), '*')
    plt.plot(tf.reshape(X, shape=(N,)).numpy(), tf.reshape(y, shape=(N,)).numpy(), label='predict')
    plt.legend()
    plt.pause(0.01)
