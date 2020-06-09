import tensorflow as tf
import time

n = 10000
t1 = time.time()
with tf.device("gpu:0"):
    a = tf.random.uniform(shape=(n, n))
    b = tf.random.uniform(shape=(n, n))
    c = tf.matmul(a, b)
t2 = time.time()
print(t2 - t1)
