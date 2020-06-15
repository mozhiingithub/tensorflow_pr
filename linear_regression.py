import tensorflow as tf
import matplotlib.pyplot as plt

real_a = 0.3
real_b = 0.5
X = tf.random.uniform(shape=(50,), minval=0, maxval=1)
Y = real_a * X + real_b

a = tf.Variable(initial_value=-0.2)
b = tf.Variable(initial_value=0.1)
variables = [a, b]

epoch = 200
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

for e in range(epoch):
    with tf.GradientTape() as tape:
        y = a * X + b
        loss = tf.reduce_sum(tf.square(y - Y))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
    plt.cla()
    plt.plot(X.numpy(), Y.numpy(), '*')
    plt.plot(X.numpy(), y.numpy(), label='predict')
    plt.legend()
    plt.pause(0.1)
