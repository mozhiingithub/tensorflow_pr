import tensorflow as tf

x = tf.Variable(initial_value=2.0)
with tf.GradientTape() as tape:
    u = tf.square(x)
    y = tf.math.log(u)
y_grad = tape.gradient(y, x)
print(y)
print(y_grad)
