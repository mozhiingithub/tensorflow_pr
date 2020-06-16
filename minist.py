import tensorflow as tf
import time

filter_num = 6


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


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=[3, 3],
            padding='valid',
            strides=(2, 2),
            activation=tf.nn.relu
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=[3, 3],
            padding='valid',
            strides=(2, 2),
            activation=tf.nn.relu
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=[3, 3],
            padding='same',
            strides=(2, 2),
            activation=tf.nn.relu
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=[3, 3],
            padding='valid',
            strides=(1, 1)
        )
        self.flatten = tf.keras.layers.Reshape(target_shape=(10,))

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.flatten(outputs)
        outputs = tf.nn.softmax(outputs)
        return outputs


epoch = 200
batch_size = 32
learning_rate = 1e-3

loader = MNISTLoader()
model = CNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

batch_num = int(loader.train_num / batch_size)
show_num = 1000

sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

for e in range(epoch):
    t1 = time.time()
    print('epoch:', e)
    for batch in range(batch_num):
        # x, y = loader.get_batch(batch_size=batch_size)
        start = batch * batch_size
        end = (batch + 1) * batch_size
        x = loader.x_train[start:end]
        y = loader.y_train[start:end]
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
        if 0 == (batch + 1) % show_num:
            print('batch:', batch + 1, '\tloss:', loss.numpy())
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    t2 = time.time()
    # print('training time:', t2 - t1)
    # print('testing...')
    y_pred = model.predict(x=loader.x_test, batch_size=batch_size)
    sparse_categorical_accuracy.update_state(y_true=loader.y_test, y_pred=y_pred)
    print('test accuracy:', sparse_categorical_accuracy.result().numpy())
