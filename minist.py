import tensorflow as tf
import time


class MNISTLoader:
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

    def get_batch(self, batch_size_: int):
        index: tf.Tensor = tf.random.uniform(
            shape=(batch_size_,),
            maxval=self.train_num,
            dtype=tf.int32,
            seed=time.time_ns())
        return tf.gather(params=self.x_train, indices=index), tf.gather(params=self.y_train, indices=index)


class MyCNN(tf.keras.Model):
    def __init__(self, filters_size):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters_size[0],
            kernel_size=[3, 3],
            padding='valid',
            strides=(2, 2),
            activation=tf.nn.relu,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters_size[1],
            kernel_size=[3, 3],
            padding='valid',
            strides=(2, 2),
            activation=tf.nn.relu,
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filters_size[2],
            kernel_size=[3, 3],
            padding='same',
            strides=(2, 2),
            activation=tf.nn.relu,
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=[3, 3],
            padding='valid',
            strides=(1, 1),
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


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=None, mask=None):  # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(x)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,  # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],  # 感受野大小
            padding='same',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)  # [batch_size, 28, 28, 32]
        x = self.pool1(x)  # [batch_size, 14, 14, 32]
        x = self.conv2(x)  # [batch_size, 14, 14, 64]
        x = self.pool2(x)  # [batch_size, 7, 7, 64]
        x = self.flatten(x)  # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)  # [batch_size, 1024]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


loader = MNISTLoader()

# filters_size_ = [8, 64, 128]
# model = MyCNN(filters_size=filters_size_)

model = CNN()

# model = MLP()

epoch = 100
pause_epoch = -1
batch_size = 32
# show_num = 1000
not_up_count = 0
max_not_up_count = 5
max_acc = 0
batch_num = int(loader.train_num / batch_size)
total_predict_time = 0

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

print('start training...')

for e in range(epoch):
    # t1 = time.time()
    # print('epoch:', e)
    for batch in range(batch_num):
        x, y = loader.get_batch(batch_size_=batch_size)

        # start = batch * batch_size
        # end = (batch + 1) * batch_size
        # x = loader.x_train[start:end]
        # y = loader.y_train[start:end]

        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
        # if 0 == (batch + 1) % show_num:
        #     print('batch:', batch + 1, '\tloss:', loss.numpy())
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    # t2 = time.time()
    # print('training time:', t2 - t1)
    # print('testing...')
    t1 = time.time()
    y_pred = model.predict(x=loader.x_test, batch_size=batch_size)
    t2 = time.time()
    total_predict_time += t2 - t1
    sparse_categorical_accuracy.update_state(y_true=loader.y_test, y_pred=y_pred)
    test_acc = sparse_categorical_accuracy.result().numpy()
    print('epoch:', e + 1, '\ttest accuracy:', test_acc)
    # if test_acc < min_acc:
    #     print('re-init...')
    #     model = CNN2(filters_size=filters_size_)
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #     continue

    if test_acc > max_acc:
        max_acc = test_acc
        not_up_count = 0
    else:
        if not_up_count <= max_not_up_count:
            not_up_count += 1
        else:
            pause_epoch = e + 1
            break

print('the best accuracy:', max_acc)

if pause_epoch == -1:
    pause_epoch = epoch

print('mean predict time:', total_predict_time / pause_epoch)

variables_num = 0
for variable in model.variables:
    variable_num = 1
    for size in variable.shape:
        variable_num *= size
    variables_num += variable_num

print('the total num of the variables:', variables_num)
