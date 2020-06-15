import tensorflow as tf
import time
import numpy as np


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist  # 获取数据集
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()  # 获取数据集各部分

        self.x_train = tf.constant(value=self.x_train, dtype=tf.float32)  # 将numpy数组转为tf张量，并将整形转为浮点型
        self.x_train /= 255  # 归一化
        self.x_train: tf.Tensor = tf.expand_dims(input=self.x_train, axis=-1)  # 升维以构建一个颜色通道

        self.x_test = tf.constant(value=self.x_test, dtype=tf.float32)
        self.x_train /= 255
        self.x_test: tf.Tensor = tf.expand_dims(input=self.x_test, axis=-1)

        self.y_train: tf.Tensor = tf.constant(value=self.y_train, dtype=tf.int32)
        self.y_test: tf.Tensor = tf.constant(value=self.y_test, dtype=tf.int32)

        self.train_num: int = self.x_train.shape[0]
        self.test_num: int = self.x_test.shape[0]

    def get_batch(self, batch_size: int):
        index: tf.Tensor = tf.random.uniform(
            shape=(batch_size,),
            maxval=self.train_num,
            dtype=tf.int32,
            seed=time.time_ns())
        return tf.gather(params=self.x_train, indices=index), tf.gather(params=self.y_train, indices=index)


loader: MNISTLoader = MNISTLoader()
