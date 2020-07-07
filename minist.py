import tensorflow as tf
import time
import datetime


# mnist数据集加载器
class MNISTLoader:
    def __init__(self):
        mnist = tf.keras.datasets.mnist  # 获取数据集
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()  # 获取数据集各部分

        self.x_train = tf.constant(value=self.x_train, dtype=tf.float32)  # 将numpy数组转为tf张量，并将整形转为浮点型
        self.x_train /= 255  # 归一化
        self.x_train: tf.Tensor = tf.expand_dims(input=self.x_train, axis=-1)  # 升维以构建一个颜色通道

        self.x_test = tf.constant(value=self.x_test, dtype=tf.float32)
        self.x_test /= 255
        self.x_test: tf.Tensor = tf.expand_dims(input=self.x_test, axis=-1)

        self.y_train: tf.Tensor = tf.constant(value=self.y_train, dtype=tf.int32)
        self.y_test: tf.Tensor = tf.constant(value=self.y_test, dtype=tf.int32)

        self.train_num: int = self.x_train.shape[0]
        self.test_num: int = self.x_test.shape[0]

    # 获取一个批次的训练数据
    def get_batch(self, batch_size_: int):
        # 随机生成训练数据的序号集
        index: tf.Tensor = tf.random.uniform(
            shape=(batch_size_,),
            maxval=self.train_num,
            dtype=tf.int32,
            seed=time.time_ns())
        return tf.gather(params=self.x_train, indices=index), tf.gather(params=self.y_train, indices=index)


# 自定义CNN模型结构
class MyCNN(tf.keras.Model):
    def __init__(self, filters_size):
        super().__init__()
        # 输入层规格为28*28*1

        # 第一层卷积，输出规格为13*13*filters_size[0]
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters_size[0],
            kernel_size=[3, 3],
            padding='valid',
            strides=(2, 2),
            activation=tf.nn.relu
        )

        # 第二层卷积，输出规格为6*6*filters_size[1]
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters_size[1],
            kernel_size=[3, 3],
            padding='valid',
            strides=(2, 2),
            activation=tf.nn.relu,
        )

        # 第三层卷积，输出规格为3*3*filters_size[2]
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filters_size[2],
            kernel_size=[3, 3],
            padding='same',
            strides=(2, 2),
            activation=tf.nn.relu,
        )

        # 第四层卷积，输出规格为1*1*10
        self.conv4 = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=[3, 3],
            padding='valid',
            strides=(1, 1),
        )

        # 将1*1*10的张量展成10*1
        self.flatten = tf.keras.layers.Reshape(target_shape=(10,))

    # 前向传播
    def call(self, inputs, training=None, mask=None):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.flatten(outputs)
        outputs = tf.nn.softmax(outputs)
        return outputs


loader = MNISTLoader()  # 初始化数据集加载器

epoch = 1000  # 训练总回合
batch_size = 32  # 批次尺寸，即每次更新数据所用到的样本量
not_up_count = 0  # 统计测试集准确率连续下降次数，用于过拟合时及时终止训练
max_not_up_count = 5  # 最大测试集准确率连续下降次数
max_acc = 0  # 最高测试集准确率标志位
batch_num = int(loader.train_num / batch_size)  # 单次训练回合总批次数

filters_size_ = [4, 8, 16]  # CNN模型每层卷积核个数
model = MyCNN(filters_size=filters_size_)  # 初始化模型

learning_rate = 1e-3  # 学习率
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # 初始化优化器

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)  # 模型及优化器参数保存节点
manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory='./save', max_to_keep=2)  # 节点保存管理器

retrain_flag = False  # 接续训练标志位
if retrain_flag:
    checkpoint.restore(tf.train.latest_checkpoint('./save'))

sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  # 初始化测试集评估工具

print('start training...')

while True:  # 训练回合循环
    for batch in range(batch_num):  # 批次循环
        x, y = loader.get_batch(batch_size_=batch_size)  # 获取当前批次的训练样本
        with tf.GradientTape() as tape:  # 使用tape记录计算图当中的所有运算操作
            y_pred = model(x)  # 获取运算结果
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)  # 计算损失
        grads = tape.gradient(loss, model.variables)  # 损失函数关于模型参数求导，获取梯度
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))  # 优化器优化模型参数
    y_pred = model.predict(x=loader.x_test, batch_size=batch_size)  # 获取测试集运算结果
    sparse_categorical_accuracy.update_state(y_true=loader.y_test, y_pred=y_pred)  # 评估测试集准确率
    test_acc = sparse_categorical_accuracy.result().numpy()  # 获取准确率数值
    print(datetime.datetime.now(), 'test acc:', test_acc)

    # 判断当前训练回合，测试集准确率是否高于此前最值
    if test_acc > max_acc:
        max_acc = test_acc  # 更新最高准确率
        not_up_count = 0  # 过拟合回合数清零
        # 保存节点，节点代号为准确率百分比制式，保留小数点后两位
        manager.save(checkpoint_number=int(test_acc * 1e4))
    else:
        if not_up_count < max_not_up_count:
            not_up_count += 1  # 过拟合次数加1
        else:
            break  # 过拟合次数已经超过上限，退出训练

print('the best accuracy:', max_acc)
